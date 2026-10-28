import { loadActivities, loadSettingsValue, loadStreamsIndex } from "@/lib/activities";
import { calendarDate, localIsoDate } from "@/lib/activity-types";
import { performanceSummary } from "@/lib/metrics/performance";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** FTP history, power curves, the five-point profile, and training load. */
export async function GET() {
  try {
    const [activities, index, settings] = await Promise.all([
      loadActivities(),
      loadStreamsIndex(),
      loadSettingsValue(),
    ]);
    const summary = performanceSummary(
      activities,
      index,
      settings.athlete,
      calendarDate(localIsoDate())!,
    );
    return NextResponse.json(summary, { headers: { "Cache-Control": "private, max-age=60" } });
  } catch (error) {
    console.error("Could not build performance summary", error);
    return NextResponse.json({ error: "Could not load performance data" }, { status: 503 });
  }
}
