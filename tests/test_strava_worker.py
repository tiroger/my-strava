import json
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

boto3 = types.ModuleType("boto3")
boto3.client = lambda _service: None
sys.modules.setdefault("boto3", boto3)

from services.strava_worker import handler
from services.strava_worker.handler import _to_processed


class NormalizeActivityTests(unittest.TestCase):
    def test_normalizes_units_and_local_date(self):
        activity = _to_processed(
            {
                "id": 123,
                "name": "Morning ride",
                "type": "Ride",
                "start_date_local": "2025-03-04T06:30:00",
                "distance": 16093.44,
                "moving_time": 3600,
                "elapsed_time": 3900,
                "total_elevation_gain": 304.8,
                "average_speed": 5,
                "max_speed": 10,
                "map": {"summary_polyline": "encoded"},
            }
        )

        self.assertEqual(activity["id"], "123")
        self.assertEqual(activity["distance"], 10.0)
        self.assertEqual(activity["moving_time"], 1.0)
        self.assertEqual(activity["elapsed_time"], 1.1)
        self.assertEqual(activity["total_elevation_gain"], 1000.0)
        self.assertEqual(activity["average_speed"], 11.2)
        self.assertEqual(activity["year"], 2025)
        self.assertEqual(activity["map"]["summary_polyline"], "encoded")

    def test_normalizes_utc_activity_date(self):
        activity = _to_processed(
            {
                "id": 456,
                "start_date": datetime(2024, 12, 31, tzinfo=timezone.utc).isoformat(),
            }
        )
        self.assertEqual(activity["year"], 2024)


class StravaRequestTests(unittest.TestCase):
    def test_api_request_uses_bearer_token(self):
        with patch.object(handler, "_request_json", return_value={"ok": True}) as request_json:
            response = handler._strava_get("/athlete", "access-token")

        self.assertEqual(response, {"ok": True})
        request = request_json.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer access-token")

    def test_valid_cached_token_skips_oauth_refresh(self):
        secret = {
            "client_id": "client",
            "client_secret": "secret",
            "refresh_token": "refresh",
            "access_token": "cached-token",
            "expires_at": 2_000_000_000,
        }

        class FakeSecrets:
            def get_secret_value(self, **_kwargs):
                return {"SecretString": json.dumps(secret)}

        with (
            patch.object(handler, "_secrets", FakeSecrets()),
            patch.dict(os.environ, {"STRAVA_SECRET_ARN": "secret-arn"}),
            patch.object(handler, "_request_json") as request_json,
        ):
            access_token = handler._credentials()

        self.assertEqual(access_token, "cached-token")
        request_json.assert_not_called()

    def test_refresh_persists_rotated_credentials_and_expiry(self):
        initial_secret = {
            "client_id": "client",
            "client_secret": "secret",
            "refresh_token": "old-refresh-token",
        }

        class FakeSecrets:
            updated_secret = None

            def get_secret_value(self, **_kwargs):
                return {"SecretString": json.dumps(initial_secret)}

            def put_secret_value(self, **kwargs):
                self.updated_secret = json.loads(kwargs["SecretString"])

        fake_secrets = FakeSecrets()
        oauth_response = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_at": 2_000_000_000,
        }
        with (
            patch.object(handler, "_secrets", fake_secrets),
            patch.dict(os.environ, {"STRAVA_SECRET_ARN": "secret-arn"}),
            patch.object(handler, "_request_json", return_value=oauth_response),
        ):
            access_token = handler._credentials()

        self.assertEqual(access_token, "new-access-token")
        self.assertEqual(fake_secrets.updated_secret["refresh_token"], "new-refresh-token")
        self.assertEqual(fake_secrets.updated_secret["access_token"], "new-access-token")
        self.assertEqual(fake_secrets.updated_secret["expires_at"], 2_000_000_000)


def _strava_activity(activity_id, athlete_id=42, **overrides):
    return {
        "id": activity_id,
        "athlete": {"id": athlete_id},
        "name": "Ride",
        "type": "Ride",
        "start_date_local": "2025-03-04T06:30:00Z",
        "distance": 1609.344,
        "visibility": "everyone",
        "segment_efforts": [{"id": 1}],
        "map": {"summary_polyline": "abc", "polyline": "full-resolution"},
        **overrides,
    }


class ProcessEventTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"STRAVA_ATHLETE_ID": "42"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.existing = [{"id": "1", "name": "Existing"}]

    def _event(self, aspect_type, object_id=1, owner_id=42):
        return {
            "object_type": "activity",
            "aspect_type": aspect_type,
            "object_id": object_id,
            "owner_id": owner_id,
        }

    def test_delete_requires_strava_to_confirm_the_activity_is_gone(self):
        with patch.object(handler, "_strava_get", return_value=_strava_activity(1)):
            result = handler._process_event(self._event("delete"), "token", self.existing)
        self.assertEqual([activity["id"] for activity in result], ["1"])

        missing = handler.StravaHTTPError(404, "Record Not Found")
        with patch.object(handler, "_strava_get", side_effect=missing):
            result = handler._process_event(self._event("delete"), "token", self.existing)
        self.assertEqual(result, [])

    def test_ignores_activities_owned_by_another_athlete(self):
        with patch.object(handler, "_strava_get", return_value=_strava_activity(2, athlete_id=7)):
            result = handler._process_event(self._event("create", object_id=2), "token", self.existing)
        self.assertEqual(result, self.existing)

    def test_ignores_events_for_another_owner_without_calling_strava(self):
        with patch.object(handler, "_strava_get") as strava_get:
            result = handler._process_event(self._event("delete", owner_id=7), "token", self.existing)
        self.assertEqual(result, self.existing)
        strava_get.assert_not_called()

    def test_private_activity_is_removed(self):
        private = _strava_activity(1, visibility="only_me")
        with patch.object(handler, "_strava_get", return_value=private):
            result = handler._process_event(self._event("update"), "token", self.existing)
        self.assertEqual(result, [])

    def test_stores_only_dashboard_fields(self):
        with patch.object(handler, "_strava_get", return_value=_strava_activity(3)):
            result = handler._process_event(self._event("create", object_id=3), "token", [])
        record = result[0]
        self.assertNotIn("segment_efforts", record)
        self.assertNotIn("athlete", record)
        self.assertEqual(record["map"], {"summary_polyline": "abc"})
        self.assertEqual(record["distance"], 1.0)

    def test_other_errors_are_retried(self):
        with patch.object(handler, "_strava_get", side_effect=handler.StravaHTTPError(429, "")):
            with self.assertRaises(handler.StravaHTTPError):
                handler._process_event(self._event("update"), "token", self.existing)


class ReconcileTests(unittest.TestCase):
    def test_upserts_public_and_drops_private_activities(self):
        stored = [{"id": "1", "name": "Old"}, {"id": "2", "name": "Now private"}]
        pages = [[_strava_activity(1, name="Renamed"), _strava_activity(2, private=True)]]
        with (
            patch.object(handler, "_read_json", return_value={}),
            patch.object(handler, "_strava_get", side_effect=pages),
        ):
            result = handler._reconcile("token", stored)
        self.assertEqual([(a["id"], a["name"]) for a in result], [("1", "Renamed")])


if __name__ == "__main__":
    unittest.main()
