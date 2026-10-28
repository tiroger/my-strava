import { loadActivityRecord, loadActivityStream } from "@/lib/activities";
import { hideRouteEnds, publicPolyline } from "@/lib/polyline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Route and second-by-second detail for one activity, ends trimmed for privacy. */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (!/^\d{1,20}$/.test(id)) {
    return NextResponse.json({ error: "Invalid activity ID" }, { status: 400 });
  }
  try {
    const [record, stream] = await Promise.all([loadActivityRecord(id), loadActivityStream(id)]);
    if (!record) return NextResponse.json({ error: "Activity not found" }, { status: 404 });
    const latlng = stream?.samples.latlng;
    const publicStream = stream && latlng
      ? { ...stream, samples: { ...stream.samples, latlng: hideRouteEnds(latlng) } }
      : stream;
    return NextResponse.json(
      { polyline: publicPolyline(record.polyline), stream: publicStream },
      { headers: { "Cache-Control": "public, max-age=300" } },
    );
  } catch (error) {
    console.error("Could not load activity detail", error);
    return NextResponse.json({ error: "Could not load activity" }, { status: 503 });
  }
}
