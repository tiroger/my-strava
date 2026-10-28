"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { formatNumber, getActivityKind, type SportFilter } from "@/lib/activity-types";

const RouteMap = dynamic(() => import("@/components/route-map"), {
  ssr: false,
  loading: () => <div className="route-map map-loading">Loading map…</div>,
});

type HeatRoute = { id: string; type: string; date: string; polyline: string };

/** Every route at once, faint enough that the roads ridden most glow brightest. */
export function HeatmapPanel({
  filter,
  onOpenActivity,
}: {
  filter: SportFilter;
  onOpenActivity: (id: string) => void;
}) {
  const [routes, setRoutes] = useState<HeatRoute[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [year, setYear] = useState("all");

  useEffect(() => {
    let cancelled = false;
    fetch("/api/heatmap")
      .then(async (response) => {
        if (!response.ok) throw new Error("Routes unavailable");
        return (await response.json()) as HeatRoute[];
      })
      .then((value) => !cancelled && setRoutes(value))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, []);

  const years = useMemo(
    () => Array.from(new Set((routes ?? []).map((route) => route.date.slice(0, 4)))).sort().reverse(),
    [routes],
  );
  const visible = useMemo(
    () =>
      (routes ?? []).filter(
        (route) =>
          (year === "all" || route.date.startsWith(year)) &&
          (filter === "All" || getActivityKind(route.type) === filter),
      ),
    [filter, routes, year],
  );

  return (
    <section className="card heatmap-card">
      <div className="card-heading">
        <div>
          <div className="card-kicker">WHERE YOU GO</div>
          <h2>Heatmap · {formatNumber(visible.length)} routes</h2>
        </div>
        <label className="year-select">
          <span>YEAR</span>
          <select value={year} onChange={(event) => setYear(event.target.value)} aria-label="Heatmap year">
            <option value="all">All years</option>
            {years.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
      </div>
      {routes ? (
        <RouteMap
          routes={visible}
          variant="heat"
          height={560}
          onSelect={onOpenActivity}
        />
      ) : (
        <div className="route-map map-loading" style={{ height: 560 }}>
          {failed ? "Routes could not be loaded." : "Loading routes…"}
        </div>
      )}
      <p className="chart-subtitle">
        Brighter lines are roads ridden more often. Click a route to open it. The first and last 500 m
        of every route are hidden to keep home private.
      </p>
    </section>
  );
}
