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

from . import saved_routes
from . import streams as stream_math

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
    "max_watts",
    "device_watts",
    "kilojoules",
    "suffer_score",
    "average_cadence",
    "elev_high",
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


def _delete_object(key: str) -> None:
    _s3.delete_object(Bucket=os.environ["APP_DATA_BUCKET"], Key=key)


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


def _gear_kind(gear_id: str, fallback: str = "bike") -> str:
    # Strava prefixes bike IDs with "b" and shoe IDs with "g".
    if gear_id.startswith("b"):
        return "bike"
    if gear_id.startswith("g"):
        return "shoe"
    return fallback


def _gear_record(gear: dict[str, Any], kind: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    previous = previous or {}
    gear_id = str(gear.get("id", ""))
    return {
        "id": gear_id,
        "name": gear.get("nickname") or gear.get("name") or previous.get("name") or "Gear",
        # Summary gear from /athlete has no brand or model; keep what /gear gave.
        "brand": gear.get("brand_name") or previous.get("brand") or "",
        "model": gear.get("model_name") or previous.get("model") or "",
        "distance": round(float(gear.get("distance") or 0) * 0.000621371, 1),
        "retired": bool(gear.get("retired", previous.get("retired", False))),
        "kind": _gear_kind(gear_id, kind),
    }


def _sync_gear(access_token: str, activities: list[dict[str, Any]], refresh_athlete: bool) -> bool:
    """Keep bikes.json covering every bike and shoe the activities use.

    The athlete profile lists current gear, so reconciliation refreshes
    distances from it. Gear it does not list, such as a bike added since the
    last reconciliation or one now retired, is found through the gear IDs on
    activities and looked up once. Stored gear is never dropped.
    """
    key = os.getenv("BIKES_S3_KEY", "bikes.json")
    stored = {gear["id"]: gear for gear in _read_json(key, []) if gear.get("id")}
    changed = False

    if refresh_athlete:
        athlete = _strava_get("/athlete", access_token)
        for kind, listed in (("bike", athlete.get("bikes") or []), ("shoe", athlete.get("shoes") or [])):
            for gear in listed:
                gear_id = str(gear.get("id", ""))
                if not gear_id:
                    continue
                # Gear first seen here is detailed below like any other new gear.
                if gear_id in stored and "kind" in stored[gear_id]:
                    stored[gear_id] = _gear_record(gear, kind, stored[gear_id])
                    changed = True

    used = {str(activity["gear_id"]) for activity in activities if activity.get("gear_id")}
    if refresh_athlete:
        used |= {
            str(gear.get("id")) for gear in (athlete.get("bikes") or []) + (athlete.get("shoes") or [])
            if gear.get("id")
        }
    # Records without a kind predate gear discovery and lack brand and model.
    unknown = sorted(gear_id for gear_id in used if "kind" not in stored.get(gear_id, {}))
    for gear_id in unknown:
        try:
            gear = _strava_get(f"/gear/{gear_id}", access_token)
        except StravaHTTPError as error:
            if error.status != 404:
                raise
            print(f"Gear {gear_id} is not visible to this athlete; skipped")
            continue
        stored[gear_id] = _gear_record(gear, _gear_kind(gear_id), stored.get(gear_id))
        changed = True

    if changed:
        gear_list = sorted(
            stored.values(),
            key=lambda gear: (gear.get("kind") != "bike", bool(gear.get("retired")), -float(gear.get("distance") or 0)),
        )
        _write_json(key, gear_list)
    return changed


def _streams_key(activity_id: str) -> str:
    return f"{os.getenv('STREAMS_S3_PREFIX', 'streams/')}{activity_id}.json"


def _fetch_streams(activity: dict[str, Any], access_token: str) -> dict[str, Any]:
    """Fetch, store, and summarize one activity's streams; return its index entry."""
    activity_id = str(activity["id"])
    entry: dict[str, Any] = {
        "date": activity.get("start_date_local"),
        "sport_type": activity.get("sport_type") or activity.get("type"),
        "has_streams": False,
    }
    empty_summary = stream_math.summarize({})
    try:
        payload = _strava_get(
            f"/activities/{int(activity_id)}/streams",
            access_token,
            {"keys": ",".join(stream_math.STREAM_KEYS), "key_by_type": "true"},
        )
    except StravaHTTPError as error:
        if error.status != 404:
            raise
        return {**entry, **{key: None for key in empty_summary}, "power_bests": {}}

    data = stream_math.by_type(payload)
    if not data.get("time"):
        return {**entry, **{key: None for key in empty_summary}, "power_bests": {}}
    summary = stream_math.summarize(data)
    _write_json(
        _streams_key(activity_id),
        {
            "id": activity_id,
            "start_date_local": entry["date"],
            "sport_type": entry["sport_type"],
            "samples": stream_math.samples(data),
            "summary": summary,
        },
    )
    return {**entry, "has_streams": True, **summary}


def _fetch_streams_for(
    activities: list[dict[str, Any]],
    access_token: str,
    index: dict[str, Any],
    deadline: float,
    limit: int,
) -> int:
    """Fetch streams for activities in order until the limit, the deadline, or Strava's quota."""
    fetched = 0
    for activity in activities:
        if fetched >= limit or time.monotonic() > deadline:
            break
        try:
            index[str(activity["id"])] = _fetch_streams(activity, access_token)
        except StravaHTTPError as error:
            if error.status == 429:
                print("Strava rate limit reached; remaining streams wait for the next backfill")
                break
            print(f"Streams for {activity['id']} failed with HTTP {error.status}; left for backfill")
            continue
        except RuntimeError as error:
            print(f"Streams for {activity['id']} failed: {error}; left for backfill")
            continue
        fetched += 1
    return fetched


def _saved_route_key(route_id: str) -> str:
    return f"{os.getenv('SAVED_ROUTES_S3_PREFIX', 'saved-routes/')}{route_id}.json"


def _list_saved_routes(access_token: str) -> list[dict[str, Any]]:
    athlete_id = os.getenv("STRAVA_ATHLETE_ID") or str(_strava_get("/athlete", access_token)["id"])
    routes: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _strava_get(f"/athletes/{athlete_id}/routes", access_token, {"page": page, "per_page": 100})
        routes += batch or []
        if not batch or len(batch) < 100:
            return routes
        page += 1
        if page > 50:
            raise RuntimeError("Strava pagination limit exceeded while listing routes")


def _sync_saved_routes(access_token: str, deadline: float) -> int:
    """Store public saved routes and fetch the profiles of new or edited ones."""
    key = os.getenv("SAVED_ROUTES_S3_KEY", "saved-routes.json")
    limit = int(os.getenv("SAVED_ROUTES_PROFILE_LIMIT", "30"))
    previous = {route["id"]: route for route in _read_json(key, [])}
    listed = [route for route in _list_saved_routes(access_token) if not route.get("private")]
    records = {
        saved_routes.route_id(route): saved_routes.summary(
            route, (previous.get(saved_routes.route_id(route)) or {}).get("profile_updated_at")
        )
        for route in listed
    }

    # Routes deleted or made private take their profiles with them.
    for route_id, old in previous.items():
        if route_id not in records and old.get("profile_updated_at"):
            _delete_object(_saved_route_key(route_id))

    stale = sorted(
        (route for route in listed if records[saved_routes.route_id(route)]["profile_updated_at"] != route.get("updated_at")),
        key=lambda route: route.get("updated_at") or "",
        reverse=True,
    )
    fetched = 0
    for route in stale:
        if fetched >= limit or time.monotonic() > deadline:
            break
        route_id = saved_routes.route_id(route)
        try:
            payload = _strava_get(f"/routes/{route_id}/streams", access_token)
        except StravaHTTPError as error:
            if error.status == 429:
                print("Strava rate limit reached; remaining route profiles wait for the next sync")
                break
            print(f"Profile for route {route_id} failed with HTTP {error.status}; skipped")
            continue
        except RuntimeError as error:
            print(f"Profile for route {route_id} failed: {error}; skipped")
            continue
        fetched += 1
        profile = saved_routes.profile(route, payload)
        if profile:
            _write_json(_saved_route_key(route_id), profile)
            records[route_id]["profile_updated_at"] = route.get("updated_at")

    _write_json(
        key,
        sorted(records.values(), key=lambda route: route.get("updated_at") or "", reverse=True),
    )
    return fetched


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
    if body.get("source") == "streams-backfill":
        return {"source": "streams-backfill"}
    if body.get("source") == "routes-sync":
        return {"source": "routes-sync"}
    return body


def _deadline(context: Any) -> float:
    # Leave time to write what was fetched before Lambda stops the invocation.
    remaining_ms = getattr(context, "get_remaining_time_in_millis", lambda: 300_000)()
    return time.monotonic() + max(0.0, remaining_ms / 1000 - 30)


def main(event: dict[str, Any], context: Any) -> dict[str, Any]:
    messages = event.get("Records", [])
    if messages:
        events = [_records_from_sqs(record) for record in messages]
    else:
        events = [event]

    deadline = _deadline(context)
    access_token = _credentials()
    activities_key = os.getenv("ACTIVITIES_S3_KEY", "activities.json")
    state_key = os.getenv("SYNC_STATE_S3_KEY", "sync-state.json")
    index_key = os.getenv("STREAMS_INDEX_S3_KEY", "streams-index.json")
    backfill_limit = int(os.getenv("STREAMS_BACKFILL_LIMIT", "50"))
    activities = _read_json(activities_key, [])
    index = _read_json(index_key, {})

    changed = False
    reconciled = False
    index_changed = False
    stream_targets: list[str] = []
    backfill = False
    routes_sync = False
    for item in events:
        if item.get("source") == "reconcile":
            before = {activity["id"] for activity in activities}
            activities = _reconcile(access_token, activities)
            changed = True
            reconciled = True
            # Only activities this run added; older gaps are the backfill's job.
            stream_targets += [activity["id"] for activity in activities if activity["id"] not in before]
        elif item.get("source") == "streams-backfill":
            backfill = True
        elif item.get("source") == "routes-sync":
            routes_sync = True
        elif item.get("object_type") == "activity":
            activity_id = str(item.get("object_id"))
            activities = _process_event(item, access_token, activities)
            changed = True
            if any(activity["id"] == activity_id for activity in activities):
                stream_targets.append(activity_id)
            elif activity_id in index:
                # Deleted, or now private: its detailed data goes with it.
                _delete_object(_streams_key(activity_id))
                del index[activity_id]
                index_changed = True

    if changed:
        activities = [_compact(activity) for activity in activities if _is_public(activity)]
        activities.sort(key=lambda activity: activity.get("start_date_local", ""), reverse=True)
        _write_json(activities_key, activities)

    _sync_gear(access_token, activities, refresh_athlete=reconciled)

    wanted = set(stream_targets)
    targets = [
        activity for activity in activities
        if activity["id"] not in index and (backfill or activity["id"] in wanted)
    ]
    targets.sort(key=lambda activity: activity.get("start_date_local", ""), reverse=True)
    streams_fetched = _fetch_streams_for(targets, access_token, index, deadline, backfill_limit) if targets else 0
    if streams_fetched or index_changed:
        _write_json(index_key, index)

    route_profiles_fetched = _sync_saved_routes(access_token, deadline) if reconciled or routes_sync else 0

    if reconciled:
        _write_json(
            state_key,
            {
                "last_sync_epoch": int(time.time()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return {
        "ok": True,
        "events": len(events),
        "activities": len(activities),
        "streams_fetched": streams_fetched,
        "route_profiles_fetched": route_profiles_fetched,
    }
