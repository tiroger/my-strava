"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { ElevationProfile } from "@/components/elevation-profile";
import { formatNumber } from "@/lib/activity-types";
import { findClimbs, GRADE_SCALE, smoothedGrades } from "@/lib/metrics/route-profile";

const ProfileMap = dynamic(() => import("@/components/profile-map"), {
  ssr: false,
  loading: () => <div className="route-map map-loading">Loading map…</div>,
});

export type RouteTrack = {
  /** Miles from the start. */
  distance: number[];
  /** Feet. */
  altitude: number[];
  /** Null where hidden near home or not recorded. */
  latlng: Array<[number, number] | null> | null;
};

/** Grade at every point plus a separate half-mile figure for "steepest". */
export function useTrackAnalysis(track: RouteTrack) {
  const grades = useMemo(() => smoothedGrades(track.distance, track.altitude), [track]);
  const climbs = useMemo(() => findClimbs(track.distance, track.altitude, grades), [grades, track]);
  const steepest = useMemo(
    () => Math.max(...smoothedGrades(track.distance, track.altitude, 0.5)),
    [track],
  );
  return { grades, climbs, steepest };
}

/**
 * Elevation profile, gradient map (2D or VeloViewer-style 3D) and climb
 * breakdown for any track, kept in step: hovering the profile moves a cursor
 * on the map, and dragging or picking a climb highlights it on both.
 */
export function RouteAnalysis({ track, mapFallback }: { track: RouteTrack; mapFallback?: React.ReactNode }) {
  const { grades, climbs } = useTrackAnalysis(track);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selection, setSelection] = useState<[number, number] | null>(null);
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const [spinning, setSpinning] = useState(false);
  const hasTrack = !!track.latlng?.some(Boolean);

  return (
    <>
      <article className="card saved-route-profile">
        <div className="card-heading">
          <div className="card-kicker">ELEVATION PROFILE · HOVER TO READ, DRAG TO MEASURE</div>
          {selection && <button className="text-button" type="button" onClick={() => setSelection(null)}>Clear selection</button>}
        </div>
        <ElevationProfile
          distance={track.distance}
          altitude={track.altitude}
          grades={grades}
          hoverIndex={hoverIndex}
          onHover={setHoverIndex}
          selection={selection}
          onSelect={setSelection}
        />
        <div className="grade-legend" aria-label="Gradient colours">
          {GRADE_SCALE.map((band) => (
            <span key={band.label}><i style={{ background: band.color }} />{band.label}</span>
          ))}
        </div>
      </article>

      <div className="detail-grid saved-route-grid">
        <article className="card saved-route-map-card">
          <div className="card-heading">
            <div className="card-kicker">{mode === "3d" ? "3D PROFILE · DRAG WITH RIGHT BUTTON OR CTRL TO ROTATE" : "MAP · COLOURED BY GRADIENT"}</div>
            {hasTrack && (
              <div className="year-metric-tabs" role="group" aria-label="Map view">
                {(["2d", "3d"] as const).map((value) => (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={mode === value}
                    className={mode === value ? "selected" : ""}
                    onClick={() => {
                      setMode(value);
                      if (value === "2d") setSpinning(false);
                    }}
                  >
                    {value.toUpperCase()}
                  </button>
                ))}
                {mode === "3d" && (
                  <button type="button" aria-pressed={spinning} className={spinning ? "selected" : ""} onClick={() => setSpinning((value) => !value)}>
                    Spin
                  </button>
                )}
              </div>
            )}
          </div>
          {hasTrack ? (
            <ProfileMap
              latlng={track.latlng!}
              distance={track.distance}
              altitude={track.altitude}
              grades={grades}
              hoverIndex={hoverIndex}
              selection={selection}
              mode={mode}
              spinning={spinning}
            />
          ) : (
            mapFallback ?? <div className="route-map map-loading">No GPS track for this one.</div>
          )}
          <p className="chart-subtitle">The first and last 500 m are hidden to keep home private.</p>
        </article>
        <article className="card climbs-card">
          <div className="card-kicker">BREAKDOWN · {climbs.length} {climbs.length === 1 ? "CLIMB" : "CLIMBS"}</div>
          {climbs.length ? (
            <table className="climb-table">
              <thead>
                <tr><th>CLIMB</th><th>STARTS</th><th>LENGTH</th><th>AVG</th><th>MAX</th><th>GAIN</th></tr>
              </thead>
              <tbody>
                {climbs.map((climb, index) => (
                  <tr
                    key={climb.startIndex}
                    className={selection?.[0] === climb.startIndex && selection?.[1] === climb.endIndex ? "selected" : ""}
                    onClick={() => setSelection([climb.startIndex, climb.endIndex])}
                  >
                    <td>
                      <span className={`climb-category cat-${climb.category ?? "none"}`}>
                        {climb.category ? (climb.category === "HC" ? "HC" : `Cat ${climb.category}`) : `#${index + 1}`}
                      </span>
                    </td>
                    <td className="numeric-cell">{formatNumber(climb.start, 1)} mi</td>
                    <td className="numeric-cell">{formatNumber(climb.length, 1)} mi</td>
                    <td className="numeric-cell">{climb.averageGrade.toFixed(1)}%</td>
                    <td className="numeric-cell">{climb.maxGrade.toFixed(1)}%</td>
                    <td className="numeric-cell">{formatNumber(climb.gain)} ft</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="empty-state">No sustained climbs: nothing gains 150 ft at 3% or steeper.</p>
          )}
          <p className="chart-subtitle">Click a climb to highlight it on the profile and map. Categories follow Strava’s length × grade scale.</p>
        </article>
      </div>
    </>
  );
}

/** Highest point, steepest half mile and categorized climbs for any track. */
export function TrackStats({ track }: { track: RouteTrack }) {
  const { climbs, steepest } = useTrackAnalysis(track);
  return (
    <>
      <span><small>Highest point</small><strong>{formatNumber(Math.max(...track.altitude))} ft</strong></span>
      <span><small>Steepest half mile</small><strong>{steepest.toFixed(1)}%</strong></span>
      <span><small>Categorized climbs</small><strong>{climbs.filter((climb) => climb.category).length}</strong></span>
    </>
  );
}
