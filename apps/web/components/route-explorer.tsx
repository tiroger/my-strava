"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { SportIcon } from "@/components/sport-icon";
import { formatCalendarDate, formatNumber, type Activity } from "@/lib/activity-types";

const RouteMap = dynamic(() => import("@/components/route-map"), {
  ssr: false,
  loading: () => <div className="route-map map-loading">Loading map…</div>,
});

export function RouteExplorer({
  activities,
  onOpenActivity,
}: {
  activities: Activity[];
  onOpenActivity: (id: string) => void;
}) {
  const routes = useMemo(() => activities.filter((activity) => activity.hasRoute), [activities]);
  const [selectedId, setSelectedId] = useState(routes[0]?.id ?? "");
  // Polylines by activity ID; null marks a route that is unavailable or hidden.
  const [polylines, setPolylines] = useState<Record<string, string | null>>({});
  const selected = routes.find((activity) => activity.id === selectedId) ?? routes[0];
  const routeId = selected?.id;

  useEffect(() => {
    if (!routeId || routeId in polylines) return;
    let cancelled = false;
    fetch(`/api/routes/${encodeURIComponent(routeId)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Route unavailable");
        return ((await response.json()) as { polyline: string }).polyline;
      })
      .catch(() => null)
      .then((polyline) => {
        if (!cancelled) setPolylines((current) => ({ ...current, [routeId]: polyline }));
      });
    return () => {
      cancelled = true;
    };
  }, [polylines, routeId]);

  if (!selected) {
    return (
      <section className="card route-panel">
        <h2>Routes</h2>
        <p className="chart-subtitle">Route maps appear for activities recorded with GPS.</p>
      </section>
    );
  }

  const polyline = polylines[selected.id];
  return (
    <section className="route-layout">
      <article className="card route-map-card">
        <div className="card-heading">
          <div>
            <div className="card-kicker">GPS ROUTE</div>
            <h2>{selected.name}</h2>
          </div>
          <button className="text-button" type="button" onClick={() => onOpenActivity(selected.id)}>
            Full analysis →
          </button>
        </div>
        {polyline ? (
          <RouteMap routes={[{ id: selected.id, polyline }]} variant="route" height={380} />
        ) : (
          <div className="route-map map-loading">
            {polyline === null ? "This route stays within the hidden area around home." : "Loading route…"}
          </div>
        )}
        <div className="route-stats">
          <span><strong>{formatNumber(selected.distance, 1)}</strong><small>miles</small></span>
          <span><strong>{formatNumber(selected.total_elevation_gain)}</strong><small>ft climbed</small></span>
          <span><strong>{formatNumber(selected.average_speed, 1)}</strong><small>mph average</small></span>
        </div>
        <p className="chart-subtitle">The first and last 500 m of every route are hidden to keep home private.</p>
      </article>
      <article className="card route-list-card">
        <div className="card-kicker">YOUR MAPPED ACTIVITIES</div>
        <h2>{formatNumber(routes.length)} routes</h2>
        <div className="route-list">
          {routes.map((activity) => (
            <button
              className={`route-list-item ${activity.id === selected.id ? "selected" : ""}`}
              key={activity.id}
              onClick={() => setSelectedId(activity.id)}
              type="button"
            >
              <span className="activity-type-icon"><SportIcon type={activity.type} /></span>
              <span><strong>{activity.name}</strong><small>{formatCalendarDate(activity.date)}</small></span>
              <span className="route-distance">{formatNumber(activity.distance, 1)} mi</span>
            </button>
          ))}
        </div>
      </article>
    </section>
  );
}
