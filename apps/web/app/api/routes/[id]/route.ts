import { loadActivityRoute } from "@/lib/activities";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (!/^\d{1,20}$/.test(id)) {
    return NextResponse.json({ error: "Invalid activity ID" }, { status: 400 });
  }
  try {
    const polyline = await loadActivityRoute(id);
    if (!polyline) {
      return NextResponse.json({ error: "Route not found" }, { status: 404 });
    }
    return NextResponse.json(
      { polyline },
      { headers: { "Cache-Control": "public, max-age=300" } },
    );
  } catch (error) {
    console.error("Could not load activity route", error);
    return NextResponse.json({ error: "Could not load route" }, { status: 503 });
  }
}
