"use client";

import { useEffect, useMemo, useState } from "react";
import { formatCalendarDate, type Activity } from "@/lib/activity-types";

function decodePolyline(encoded: string): Array<[number, number]> {
  const coordinates: Array<[number, number]> = [];
  let index = 0;
  let latitude = 0;
  let longitude = 0;

  while (index < encoded.length) {
    let result = 0;
    let shift = 0;
    let byte = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20 && index < encoded.length);
    latitude += result & 1 ? ~(result >> 1) : result >> 1;

    result = 0;
    shift = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20 && index < encoded.length);
    longitude += result & 1 ? ~(result >> 1) : result >> 1;
    coordinates.push([latitude / 1e5, longitude / 1e5]);
  }
  return coordinates;
}

function routePath(encoded: string): string {
  const points = decodePolyline(encoded);
  if (points.length < 2) return "";
  const latitudes = points.map(([latitude]) => latitude);
  const longitudes = points.map(([, longitude]) => longitude);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLon = Math.min(...longitudes);
  const maxLon = Math.max(...longitudes);
  const latSpan = Math.max(maxLat - minLat, 0.00001);
  const lonSpan = Math.max(maxLon - minLon, 0.00001);
  return points
    .map(([latitude, longitude], index) => {
      const x = 20 + ((longitude - minLon) / lonSpan) * 360;
      const y = 20 + ((maxLat - latitude) / latSpan) * 190;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function RouteExplorer({ activities }: { activities: Activity[] }) {
  const routes = useMemo(
    () => activities.filter((activity) => activity.hasRoute),
    [activities],
  );
  const [selectedId, setSelectedId] = useState(routes[0]?.id ?? "");
  // Decoded SVG paths by activity ID; null marks a route that failed to load.
  const [paths, setPaths] = useState<Record<string, string | null>>({});
  const selected = routes.find((activity) => activity.id === selectedId) ?? routes[0];
  const routeId = selected?.id;

  useEffect(() => {
    if (!routeId || routeId in paths) return;
    let cancelled = false;
    fetch(`/api/routes/${encodeURIComponent(routeId)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Route unavailable");
        const { polyline } = (await response.json()) as { polyline: string };
        return routePath(polyline);
      })
      .catch(() => null)
      .then((path) => {
        if (!cancelled) setPaths((current) => ({ ...current, [routeId]: path }));
      });
    return () => {
      cancelled = true;
    };
  }, [paths, routeId]);

  if (!routes.length) {
    return (
      <section className="card route-panel">
        <h2>Route explorer</h2>
        <p className="chart-subtitle">
          Route maps will appear when activities include GPS data.
        </p>
      </section>
    );
  }

  const path = paths[selected.id];
  return (
    <section className="route-layout">
      <article className="card route-map-card">
        <div className="card-kicker">GPS ROUTE</div>
        <h2>{selected.name}</h2>
        <div className="route-canvas">
          <div className="map-grid" />
          <svg viewBox="0 0 400 230" role="img" aria-label={`Route map for ${selected.name}`}>
            {path && <path d={path} className="route-line-shadow" />}
            {path && <path d={path} className="route-line" />}
            {path && <circle cx="20" cy="20" r="4" className="route-start" />}
          </svg>
          <span className="route-map-label">
            {path === undefined
              ? "LOADING ROUTE…"
              : path === null
                ? "ROUTE UNAVAILABLE"
                : "GPS TRACE · NOT TO SCALE"}
          </span>
        </div>
        <div className="route-stats">
          <span><strong>{selected.distance.toFixed(1)}</strong><small>miles</small></span>
          <span><strong>{Math.round(selected.total_elevation_gain).toLocaleString("en-US")}</strong><small>ft climbed</small></span>
          <span><strong>{Math.round(selected.average_speed)}</strong><small>mph avg</small></span>
        </div>
      </article>
      <article className="card route-list-card">
        <div className="card-kicker">YOUR MAPPED ACTIVITIES</div>
        <h2>Routes</h2>
        <div className="route-list">
          {routes.map((activity) => (
            <button
              className={`route-list-item ${activity.id === selected.id ? "selected" : ""}`}
              key={activity.id}
              onClick={() => setSelectedId(activity.id)}
              type="button"
            >
              <span className="activity-type-icon">⌖</span>
              <span><strong>{activity.name}</strong><small>{formatCalendarDate(activity.date)}</small></span>
              <span className="route-distance">{activity.distance.toFixed(1)} mi</span>
            </button>
          ))}
        </div>
      </article>
    </section>
  );
}
