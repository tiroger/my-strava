import { loadSavedRoutes } from "@/lib/activities";
import { publicPolyline } from "@/lib/polyline";
import { revalidatedJson } from "@/lib/http";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** The athlete's public saved routes, ends trimmed for privacy. */
export async function GET(request: Request) {
  try {
    const routes = (await loadSavedRoutes()).map((route) => ({
      ...route,
      polyline: publicPolyline(route.polyline),
    }));
    return revalidatedJson(request, routes);
  } catch (error) {
    console.error("Could not load saved routes", error);
    return NextResponse.json({ error: "Could not load saved routes" }, { status: 503 });
  }
}
