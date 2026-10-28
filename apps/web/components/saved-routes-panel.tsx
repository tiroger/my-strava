"use client";

import { useEffect, useMemo, useState } from "react";
import { RouteAnalysis, TrackStats, type RouteTrack } from "@/components/route-analysis";
import { formatCalendarDate, formatHours, formatNumber } from "@/lib/activity-types";
import type { RouteProfile, SavedRoute } from "@/lib/metrics/route-profile";

const SUB_TYPES: Record<NonNullable<SavedRoute["sub_type"]>, string> = {
  road: "Road",
  mtb: "Mountain bike",
  cx: "Cyclocross",
  trail: "Trail",
  mixed: "Mixed surface",
};

export function SavedRoutesPanel() {
  const [routes, setRoutes] = useState<SavedRoute[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Profiles by route ID; null marks one that is not available yet.
  const [profiles, setProfiles] = useState<Record<string, RouteProfile | null>>({});

  useEffect(() => {
    let cancelled = false;
    fetch("/api/saved-routes")
      .then(async (response) => {
        if (!response.ok) throw new Error("Saved routes unavailable");
        return (await response.json()) as SavedRoute[];
      })
      .then((value) => !cancelled && setRoutes(value))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, []);

  const visible = useMemo(
    () => (routes ?? []).filter((route) => route.name.toLowerCase().includes(query.toLowerCase())),
    [query, routes],
  );
  const selected = visible.find((route) => route.id === selectedId) ?? visible[0];
  const routeId = selected?.id;

  useEffect(() => {
    if (!routeId || routeId in profiles) return;
    let cancelled = false;
    fetch(`/api/saved-routes/${encodeURIComponent(routeId)}`)
      .then(async (response) => (response.ok ? ((await response.json()) as RouteProfile) : null))
      .catch(() => null)
      .then((profile) => !cancelled && setProfiles((current) => ({ ...current, [routeId]: profile })));
    return () => {
      cancelled = true;
    };
  }, [profiles, routeId]);

  if (!routes) {
    return <div className="card empty-state">{failed ? "Saved routes could not be loaded." : "Loading saved routes…"}</div>;
  }
  if (!routes.length) {
    return (
      <div className="card empty-state">
        No saved routes yet. Public routes you create or save on Strava appear after the next daily sync.
      </div>
    );
  }

  return (
    <section className="saved-routes-layout">
      <aside className="card saved-route-list">
        <div className="card-kicker">YOUR ROUTES · {routes.length}</div>
        <label className="search-box saved-route-search">
          <span aria-hidden="true">⌕</span>
          <input aria-label="Search saved routes" placeholder="Search routes" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="route-list">
          {visible.map((route) => (
            <button
              className={`route-list-item ${route.id === selected?.id ? "selected" : ""}`}
              key={route.id}
              type="button"
              onClick={() => setSelectedId(route.id)}
            >
              <span>
                <strong>{route.starred && <span className="star" aria-label="Starred">★ </span>}{route.name}</strong>
                <small>
                  {formatNumber(route.distance, 1)} mi · {formatNumber(route.elevation_gain)} ft
                  {route.sub_type ? ` · ${SUB_TYPES[route.sub_type]}` : ""}
                </small>
              </span>
            </button>
          ))}
          {!visible.length && <div className="empty-state">No routes match.</div>}
        </div>
      </aside>
      {selected && (
        <RouteDetail
          key={selected.id}
          route={selected}
          profile={profiles[selected.id]}
          loading={!(selected.id in profiles)}
        />
      )}
    </section>
  );
}

function RouteDetail({
  route,
  profile,
  loading,
}: {
  route: SavedRoute;
  profile: RouteProfile | null | undefined;
  loading: boolean;
}) {
  return (
    <div className="saved-route-detail">
      <article className="card saved-route-header">
        <div className="card-kicker">
          {route.type === "run" ? "RUN ROUTE" : "RIDE ROUTE"}
          {route.sub_type ? ` · ${SUB_TYPES[route.sub_type].toUpperCase()}` : ""}
          {route.updated_at ? ` · UPDATED ${formatCalendarDate(route.updated_at, { month: "short", day: "numeric", year: "numeric" }).toUpperCase()}` : ""}
        </div>
        <h2 className="saved-route-title">{route.starred && <span className="star">★ </span>}{route.name}</h2>
        {route.description && <p className="chart-subtitle">{route.description}</p>}
        {profile ? <RouteStats route={route} track={profile} /> : <RouteStats route={route} />}
      </article>
      {profile ? (
        <RouteAnalysis track={profile} />
      ) : (
        <article className="card saved-route-profile">
          <div className="map-loading">
            {loading ? "Loading profile…" : "This route’s elevation profile has not synced yet; it arrives with the next daily sync."}
          </div>
        </article>
      )}
    </div>
  );
}

function RouteStats({ route, track }: { route: SavedRoute; track?: RouteTrack }) {
  const stats: Array<[string, string]> = [
    ["Distance", `${formatNumber(route.distance, 1)} mi`],
    ["Elevation gain", `${formatNumber(route.elevation_gain)} ft`],
    ["Climbing rate", route.distance ? `${formatNumber(route.elevation_gain / route.distance)} ft/mi` : "—"],
    ...(route.estimated_moving_time ? ([["Strava estimate", formatHours(route.estimated_moving_time / 3600)]] as Array<[string, string]>) : []),
  ];
  return (
    <div className="detail-stats">
      {stats.map(([label, value]) => (
        <span key={label}><small>{label}</small><strong>{value}</strong></span>
      ))}
      {track && <TrackStats track={track} />}
    </div>
  );
}
