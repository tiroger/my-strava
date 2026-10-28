"""Process Strava webhook events and periodic reconciliation messages."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import boto3

_s3 = boto3.client("s3")
_secrets = boto3.client("secretsmanager")
_STRAVA_API = "https://www.strava.com/api/v3"
_STRAVA_OAUTH = "https://www.strava.com/oauth/token"

# Only these fields are stored; the full Strava payload (segment efforts,
# laps, full-resolution polylines) would bloat the object the site reads.
_STORED_FIELDS = (
    "id",
    "name",
    "type",
    "sport_type",
    "start_date_local",
    "distance",
    "moving_time",
    "elapsed_time",
    "total_elevation_gain",
    "average_speed",
    "max_speed",
    "average_heartrate",
    "max_heartrate",
    "average_watts",
    "weighted_average_watts",
    "kilojoules",
    "kudos_count",
    "gear_id",
    "trainer",
    "year",
    "month",
    "day",
    "map",
)


class StravaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Strava returned HTTP {status}: {body[:500]}")
        self.status = status


def _read_json(key: str, default: Any) -> Any:
    try:
        result = _s3.get_object(Bucket=os.environ["APP_DATA_BUCKET"], Key=key)
    except _s3.exceptions.NoSuchKey:
        return default
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") == "NoSuchKey":
            return default
        raise
    return json.loads(result["Body"].read())


def _write_json(key: str, value: Any) -> None:
    _s3.put_object(
        Bucket=os.environ["APP_DATA_BUCKET"],
        Key=key,
        Body=json.dumps(value, separators=(",", ":")).encode(),
        ContentType="application/json",
        CacheControl="no-cache",
    )


def _credentials() -> str:
    secret = _secrets.get_secret_value(SecretId=os.environ["STRAVA_SECRET_ARN"])
    credentials = json.loads(secret["SecretString"])
    access_token = credentials.get("access_token")
    try:
        expires_at = float(credentials.get("expires_at", 0))
    except (TypeError, ValueError):
        expires_at = 0
    if isinstance(access_token, str) and access_token and expires_at > time.time() + 60:
        return access_token

    data = urllib.parse.urlencode(
        {
            "client_id": credentials["client_id"],
            "client_secret": credentials["client_secret"],
            "refresh_token": credentials["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode()
    request = urllib.request.Request(_STRAVA_OAUTH, data=data, method="POST")
    response = _request_json(request)
    new_access_token = response.get("access_token")
    new_expiration = response.get("expires_at")
    if not isinstance(new_access_token, str) or not new_access_token:
        raise RuntimeError("Strava OAuth response did not contain an access token")
    if not isinstance(new_expiration, (int, float)):
        raise RuntimeError("Strava OAuth response did not contain an expiration time")
    credentials["access_token"] = new_access_token
    credentials["expires_at"] = new_expiration
    if response.get("refresh_token"):
        credentials["refresh_token"] = response["refresh_token"]
    _secrets.put_secret_value(
        SecretId=os.environ["STRAVA_SECRET_ARN"],
        SecretString=json.dumps(credentials),
    )
    return new_access_token


def _request_json(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise StravaHTTPError(error.code, body) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Strava request failed: {error.reason}") from error


def _strava_get(path: str, access_token: str, query: dict[str, Any] | None = None) -> Any:
    suffix = f"?{urllib.parse.urlencode(query)}" if query else ""
    request = urllib.request.Request(
        f"{_STRAVA_API}{path}{suffix}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return _request_json(request)


def _is_public(activity: dict[str, Any]) -> bool:
    return not activity.get("private") and activity.get("visibility") != "only_me"


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    compacted = {key: record[key] for key in _STORED_FIELDS if key in record}
    map_data = record.get("map")
    if isinstance(map_data, dict):
        compacted["map"] = {"summary_polyline": map_data.get("summary_polyline") or ""}
    return compacted


def _to_processed(activity: dict[str, Any]) -> dict[str, Any]:
    start = activity.get("start_date_local") or activity.get("start_date")
    parsed = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    map_data = activity.get("map") or {}
    return _compact({
        **activity,
        "id": str(activity["id"]),
        "start_date_local": start,
        "distance": round(float(activity.get("distance", 0)) * 0.000621371, 1),
        "moving_time": round(float(activity.get("moving_time", 0)) / 3600, 1),
        "elapsed_time": round(float(activity.get("elapsed_time", 0)) / 3600, 1),
        "total_elevation_gain": round(
            float(activity.get("total_elevation_gain", 0)) * 3.28084, 1
        ),
        "average_speed": round(float(activity.get("average_speed", 0)) * 2.23694, 1),
        "max_speed": round(float(activity.get("max_speed", 0)) * 2.23694, 1),
        "year": parsed.year,
        "month": parsed.month,
        "day": parsed.day,
        "map": map_data,
    })


def _reconcile(access_token: str, activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_key = os.getenv("SYNC_STATE_S3_KEY", "sync-state.json")
    state = _read_json(state_key, {})
    # A two-day overlap lets a retry recover events that arrived while Strava
    # was temporarily unavailable. IDs make this safe to replay.
    after = max(0, int(state.get("last_sync_epoch", 0)) - 2 * 24 * 60 * 60)
    by_id = {activity["id"]: activity for activity in activities}
    page = 1
    while True:
        batch = _strava_get(
            "/athlete/activities",
            access_token,
            {"after": after, "per_page": 100, "page": page},
        )
        if not batch:
            break
        for activity in batch:
            activity_id = str(activity["id"])
            if _is_public(activity):
                by_id[activity_id] = _to_processed(activity)
            else:
                by_id.pop(activity_id, None)
        if len(batch) < 100:
            break
        page += 1
        if page > 100:
            raise RuntimeError("Strava pagination limit exceeded during reconciliation")
    return list(by_id.values())


def _sync_bikes(access_token: str) -> None:
    athlete = _strava_get("/athlete", access_token)
    bikes = []
    for bike in athlete.get("bikes", []):
        distance = float(bike.get("distance", 0))
        bikes.append(
            {
                "id": str(bike.get("id", "")),
                "name": bike.get("nickname") or bike.get("name") or "Bike",
                "brand": bike.get("brand_name") or "",
                "model": bike.get("model_name") or "",
                "distance": round(distance * 0.000621371, 1),
                "retired": bool(bike.get("retired", False)),
            }
        )
    _write_json(os.getenv("BIKES_S3_KEY", "bikes.json"), bikes)


def _process_event(event: dict[str, Any], access_token: str, activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply a webhook event using Strava as the source of truth.

    Webhook payloads are unauthenticated, so the event only says which
    activity to look at: deletes are confirmed by a 404, and only the
    configured athlete's public activities are stored.
    """
    athlete_id = os.getenv("STRAVA_ATHLETE_ID", "")
    if not athlete_id:
        raise RuntimeError("STRAVA_ATHLETE_ID is not configured")
    if str(event.get("owner_id")) != athlete_id:
        return activities

    activity_id = str(event["object_id"])
    remaining = [old for old in activities if old["id"] != activity_id]
    try:
        activity = _strava_get(f"/activities/{int(activity_id)}", access_token)
    except StravaHTTPError as error:
        if error.status == 404:
            return remaining
        raise
    if str((activity.get("athlete") or {}).get("id")) != athlete_id:
        return activities
    if not _is_public(activity):
        return remaining
    return remaining + [_to_processed(activity)]


def _records_from_sqs(record: dict[str, Any]) -> dict[str, Any]:
    body = json.loads(record["body"])
    if body.get("source") == "scheduled-reconciliation":
        return {"source": "reconcile"}
    return body


def main(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    messages = event.get("Records", [])
    if messages:
        events = [_records_from_sqs(record) for record in messages]
    else:
        events = [event]

    access_token = _credentials()
    activities_key = os.getenv("ACTIVITIES_S3_KEY", "activities.json")
    state_key = os.getenv("SYNC_STATE_S3_KEY", "sync-state.json")
    activities = _read_json(activities_key, [])

    changed = False
    reconciled = False
    for item in events:
        if item.get("source") == "reconcile":
            activities = _reconcile(access_token, activities)
            changed = True
            reconciled = True
        elif item.get("object_type") == "activity":
            activities = _process_event(item, access_token, activities)
            changed = True

    if changed:
        activities = [_compact(activity) for activity in activities if _is_public(activity)]
        activities.sort(key=lambda activity: activity.get("start_date_local", ""), reverse=True)
        _write_json(activities_key, activities)
    if reconciled:
        _sync_bikes(access_token)
        _write_json(
            state_key,
            {
                "last_sync_epoch": int(time.time()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return {"ok": True, "events": len(events), "activities": len(activities)}
