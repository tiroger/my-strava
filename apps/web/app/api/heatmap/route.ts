import { loadRoutes } from "@/lib/activities";
import { publicPolyline } from "@/lib/polyline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Every mapped activity's route, ends trimmed, for the heatmap. */
export async function GET() {
  try {
    const routes = (await loadRoutes()).flatMap((activity) => {
      const polyline = publicPolyline(activity.polyline);
      return polyline
        ? [{ id: activity.id, type: activity.type, date: activity.date.slice(0, 10), polyline }]
        : [];
    });
    return NextResponse.json(routes, { headers: { "Cache-Control": "public, max-age=300" } });
  } catch (error) {
    console.error("Could not load heatmap routes", error);
    return NextResponse.json({ error: "Could not load routes" }, { status: 503 });
  }
}
