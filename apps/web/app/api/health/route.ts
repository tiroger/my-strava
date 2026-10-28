import { loadDataVersion } from "@/lib/activities";
import { getWebSecret } from "@/lib/secrets";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Operational health, and what the deploy workflow's smoke test asserts.
 *
 * Reports whether configuration resolved, never what it resolved to. A 503
 * means the runtime cannot see its configuration at all -- the failure mode
 * when Amplify's variables never reach .env.production. Anything that is
 * merely not set up yet (secret values, first sync) reads as degraded.
 */
export async function GET() {
  const configured = {
    APP_DATA_BUCKET: Boolean(process.env.APP_DATA_BUCKET),
    WEB_SECRET_ARN: Boolean(process.env.WEB_SECRET_ARN),
    STRAVA_EVENTS_QUEUE_URL: Boolean(process.env.STRAVA_EVENTS_QUEUE_URL),
    STRAVA_ATHLETE_ID: Boolean(process.env.STRAVA_ATHLETE_ID),
    STRAVA_SUBSCRIPTION_ID: Boolean(process.env.STRAVA_SUBSCRIPTION_ID),
  };

  const checks: Record<string, string> = {};
  try {
    await getWebSecret();
    checks.web_secret = "ok";
  } catch (error) {
    checks.web_secret = error instanceof Error ? error.message : "unreadable";
  }
  try {
    const version = await loadDataVersion();
    checks.data = version.startsWith("|") ? "activities not synced yet" : "ok";
  } catch (error) {
    checks.data = error instanceof Error ? error.message : "unreadable";
  }

  const runtimeConfigured = configured.APP_DATA_BUCKET && configured.WEB_SECRET_ARN;
  const healthy =
    runtimeConfigured && Object.values(checks).every((value) => value === "ok");
  return NextResponse.json(
    { status: healthy ? "ok" : "degraded", configured, checks },
    {
      status: runtimeConfigured ? 200 : 503,
      headers: { "Cache-Control": "no-store" },
    },
  );
}
