import { loadSavedRouteProfile } from "@/lib/activities";
import type { RouteProfile } from "@/lib/metrics/route-profile";
import { hideRouteEnds } from "@/lib/polyline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * A saved route's elevation profile and track. Distance and altitude are
 * complete; coordinates near either end are withheld, so the profile still
 * covers the whole route while the map never shows where it starts.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (!/^\d{1,25}$/.test(id)) {
    return NextResponse.json({ error: "Invalid route ID" }, { status: 400 });
  }
  try {
    const raw = await loadSavedRouteProfile(id);
    if (!raw || !raw.distance?.length) {
      return NextResponse.json({ error: "Route profile not synced yet" }, { status: 404 });
    }
    const profile: RouteProfile = {
      id: raw.id,
      updated_at: raw.updated_at,
      distance: raw.distance,
      altitude: raw.altitude,
      latlng: hideRouteEnds(raw.latlng ?? []),
    };
    return NextResponse.json(profile, { headers: { "Cache-Control": "public, max-age=300" } });
  } catch (error) {
    console.error("Could not load route profile", error);
    return NextResponse.json({ error: "Could not load route profile" }, { status: 503 });
  }
}
