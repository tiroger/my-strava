"use client";

import type { EChartsOption } from "echarts";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { Chart, CHART_COLORS, CHART_TEXT, CHART_TOOLTIP } from "@/components/chart";
import { RouteAnalysis, TrackStats, type RouteTrack } from "@/components/route-analysis";
import { SportIcon } from "@/components/sport-icon";
import {
  formatCalendarDate,
  formatHours,
  formatNumber,
  getActivityKind,
  sportLabel,
  type Activity,
  type Gear,
} from "@/lib/activity-types";
import {
  ftpOn,
  HEART_RATE_ZONES,
  POWER_ZONES,
  rideMetrics,
  timeInZones,
  type ActivityStream,
  type PerformanceSummary,
} from "@/lib/metrics/performance";

const RouteMap = dynamic(() => import("@/components/route-map"), {
  ssr: false,
  loading: () => <div className="route-map map-loading">Loading map…</div>,
});

type Detail = { polyline: string | null; stream: ActivityStream | null };

const PEAK_DURATIONS: Array<[number, string]> = [
  [5, "5 sec"],
  [15, "15 sec"],
  [60, "1 min"],
  [300, "5 min"],
  [1200, "20 min"],
  [3600, "1 hour"],
];

export function ActivityDetail({
  activity,
  gear,
  performance,
  onBack,
}: {
  activity: Activity;
  gear: Gear[];
  performance: PerformanceSummary | null;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/activities/${encodeURIComponent(activity.id)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Activity unavailable");
        return (await response.json()) as Detail;
      })
      .then((value) => !cancelled && setDetail(value))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [activity.id]);

  const stream = detail?.stream ?? null;
  const samples = useMemo(() => stream?.samples ?? {}, [stream]);
  const track = useMemo<RouteTrack | null>(() => {
    const { distance, altitude, latlng } = samples;
    if (!distance || !altitude) return null;
    const keep = distance.flatMap((miles, index) =>
      miles !== null && altitude[index] !== null && altitude[index] !== undefined ? [index] : [],
    );
    if (keep.length < 10) return null;
    return {
      distance: keep.map((index) => distance[index]!),
      altitude: keep.map((index) => altitude[index]!),
      latlng: latlng ? keep.map((index) => latlng[index] ?? null) : null,
    };
  }, [samples]);
  const summary = stream?.summary ?? null;
  const ftp = performance
    ? performance.ftp?.source === "override"
      ? performance.ftp.ftp
      : ftpOn(performance.ftpHistory, activity.date.slice(0, 10))
    : null;
  const heartRate = performance?.heartRate ?? { restingHr: 49, maxHr: 190 };
  const metrics = summary ? rideMetrics(summary, ftp, heartRate) : null;
  const load = performance?.loads[activity.id];
  const bike = gear.find((item) => item.id === activity.gear_id);
  const kind = getActivityKind(activity.type);
  const weight = performance?.weightKg ?? null;

  const stats: Array<[string, string]> = [
    ...(activity.distance > 0 ? ([["Distance", `${formatNumber(activity.distance, 1)} mi`]] as Array<[string, string]>) : []),
    ["Moving time", formatHours(activity.moving_time)],
    ...(activity.total_elevation_gain > 0 ? ([["Elevation", `${formatNumber(activity.total_elevation_gain)} ft`]] as Array<[string, string]>) : []),
    ...(activity.average_speed > 0 && kind !== "Pickleball" ? ([["Average speed", `${formatNumber(activity.average_speed, 1)} mph`]] as Array<[string, string]>) : []),
    ...(activity.average_heartrate ? ([["Heart rate", `${Math.round(activity.average_heartrate)} avg · ${Math.round(activity.max_heartrate ?? 0)} max`]] as Array<[string, string]>) : []),
    ...(activity.average_watts ? ([["Power", `${Math.round(activity.average_watts)} W avg`]] as Array<[string, string]>) : []),
    ...(activity.kilojoules ? ([["Energy", `${formatNumber(activity.kilojoules)} kJ ≈ ${formatNumber(activity.kilojoules)} kcal`]] as Array<[string, string]>) : []),
    ...(load ? ([["Training load", `${Math.round(load.tss)} (${load.source})`]] as Array<[string, string]>) : []),
    ...(activity.suffer_score ? ([["Relative Effort", formatNumber(activity.suffer_score)]] as Array<[string, string]>) : []),
    ["Kudos", formatNumber(activity.kudos_count)],
  ];

  const analysis: Array<[string, string, string]> = metrics
    ? [
        ...(summary?.normalized_power ? ([["Normalized Power", `${Math.round(summary.normalized_power)} W`, "30-second rolling power, weighted for physiological cost"]] as Array<[string, string, string]>) : []),
        ...(metrics.intensity ? ([["Intensity Factor", metrics.intensity.toFixed(2), `Normalized Power ÷ FTP (${ftp} W on this date)`]] as Array<[string, string, string]>) : []),
        ...(metrics.tss ? ([["Training Stress Score", Math.round(metrics.tss).toString(), "An hour at FTP scores 100; uses moving time only"]] as Array<[string, string, string]>) : []),
        ...(metrics.variabilityIndex ? ([["Variability Index", metrics.variabilityIndex.toFixed(2), "Normalized ÷ average power; near 1.0 means steady"]] as Array<[string, string, string]>) : []),
        ...(metrics.efficiencyFactor ? ([["Efficiency Factor", metrics.efficiencyFactor.toFixed(2), "Normalized Power per heartbeat; rises as aerobic fitness improves"]] as Array<[string, string, string]>) : []),
        ...(metrics.decoupling !== null ? ([["Aerobic decoupling", `${metrics.decoupling.toFixed(1)}%`, "Drop in power per heartbeat from first to second half; under 5% suggests good endurance"]] as Array<[string, string, string]>) : []),
        ...(metrics.trimp ? ([["TRIMP", Math.round(metrics.trimp).toString(), `Banister heart-rate impulse, resting ${heartRate.restingHr} / max ${heartRate.maxHr} bpm`]] as Array<[string, string, string]>) : []),
      ]
    : [];

  const time = samples.time ?? [];
  const minutes = time.map((seconds) => (seconds === null ? null : Math.round((seconds / 60) * 10) / 10));
  const profile: EChartsOption | null =
    samples.distance && samples.altitude
      ? {
          grid: { top: 12, left: 46, right: 12, bottom: 26 },
          tooltip: { ...CHART_TOOLTIP, trigger: "axis", valueFormatter: (value) => `${formatNumber(Number(value))} ft` },
          xAxis: {
            type: "value",
            axisLabel: { ...CHART_TEXT, formatter: (value: number) => `${value} mi` },
            splitLine: { show: false },
            max: "dataMax",
          },
          yAxis: { type: "value", scale: true, axisLabel: CHART_TEXT, splitLine: { lineStyle: { color: CHART_COLORS.grid } } },
          series: [{
            type: "line",
            showSymbol: false,
            data: samples.distance.map((distance, index) => [distance, samples.altitude![index]]),
            lineStyle: { color: "#9aa5ab", width: 1.4 },
            areaStyle: { color: "rgba(154, 165, 171, 0.12)" },
          }],
        }
      : null;

  const effort: EChartsOption | null =
    samples.watts || samples.heartrate
      ? {
          grid: { top: 34, left: 42, right: 42, bottom: 26 },
          legend: { top: 0, left: "center", textStyle: CHART_TEXT, itemWidth: 12, itemHeight: 3 },
          tooltip: { ...CHART_TOOLTIP, trigger: "axis" },
          xAxis: {
            type: "value",
            max: "dataMax",
            axisLabel: { ...CHART_TEXT, formatter: (value: number) => `${Math.round(value)} min` },
            splitLine: { show: false },
          },
          yAxis: [
            { type: "value", name: "W", nameTextStyle: CHART_TEXT, axisLabel: CHART_TEXT, splitLine: { lineStyle: { color: CHART_COLORS.grid } } },
            { type: "value", name: "bpm", nameTextStyle: CHART_TEXT, axisLabel: CHART_TEXT, splitLine: { show: false }, scale: true },
          ],
          series: [
            ...(samples.watts
              ? [{
                  name: "Power",
                  type: "line" as const,
                  showSymbol: false,
                  sampling: "lttb" as const,
                  data: samples.watts.map((watts, index) => [minutes[index], watts]),
                  lineStyle: { color: CHART_COLORS.orange, width: 1 },
                  itemStyle: { color: CHART_COLORS.orange },
                  markLine: ftp
                    ? {
                        symbol: "none",
                        silent: true,
                        lineStyle: { color: "#7d898f", type: "dashed" as const },
                        label: { formatter: `FTP ${ftp} W`, color: "#aab4b8", fontSize: 9 },
                        data: [{ yAxis: ftp }],
                      }
                    : undefined,
                }]
              : []),
            ...(samples.heartrate
              ? [{
                  name: "Heart rate",
                  type: "line" as const,
                  yAxisIndex: 1,
                  showSymbol: false,
                  data: samples.heartrate.map((bpm, index) => [minutes[index], bpm]),
                  lineStyle: { color: CHART_COLORS.pink, width: 1.2 },
                  itemStyle: { color: CHART_COLORS.pink },
                }]
              : []),
          ],
        }
      : null;

  const powerZones = samples.watts && ftp ? timeInZones(time, samples.watts, ftp, POWER_ZONES) : null;
  const hrZones = samples.heartrate ? timeInZones(time, samples.heartrate, heartRate.maxHr, HEART_RATE_ZONES) : null;
  const bests = summary?.power_bests ?? {};

  return (
    <section className="activity-detail">
      <div className="detail-heading card">
        <button className="text-button back-button" type="button" onClick={onBack}>← Back</button>
        <div className="detail-title">
          <span className={`activity-avatar detail-icon`}><SportIcon type={activity.type} /></span>
          <div>
            <h2>{activity.name}</h2>
            <p>
              {formatCalendarDate(activity.date, { weekday: "long", month: "long", day: "numeric", year: "numeric" })}
              {" · "}{sportLabel(activity.type)}
              {bike && ` · ${bike.name}`}
            </p>
          </div>
        </div>
        <div className="detail-stats">
          {stats.map(([label, value]) => (
            <span key={label}><small>{label}</small><strong>{value}</strong></span>
          ))}
          {track && <TrackStats track={track} />}
        </div>
      </div>

      {track && (
        <RouteAnalysis
          track={track}
          mapFallback={
            detail?.polyline ? (
              <RouteMap routes={[{ id: activity.id, polyline: detail.polyline }]} variant="route" height={460} />
            ) : undefined
          }
        />
      )}

      <div className="detail-grid">
        {activity.hasRoute && !track && (
          <article className="card detail-map">
            <div className="card-kicker">ROUTE</div>
            {detail?.polyline ? (
              <RouteMap routes={[{ id: activity.id, polyline: detail.polyline }]} variant="route" height={330} />
            ) : (
              <div className="route-map map-loading">{detail ? "Route is within the hidden area around home." : failed ? "Route unavailable." : "Loading map…"}</div>
            )}
            <p className="chart-subtitle">The first and last 500 m are hidden to keep home private.</p>
          </article>
        )}
        <article className="card detail-analysis">
          <div className="card-kicker">ANALYSIS</div>
          {!detail && !failed && <p className="empty-state">Loading detail…</p>}
          {detail && !stream && (
            <p className="empty-state">
              Second-by-second data for this activity has not been synced yet. The worker backfills older
              activities a few dozen at a time.
            </p>
          )}
          {analysis.length > 0 && (
            <dl className="analysis-list">
              {analysis.map(([label, value, help]) => (
                <div key={label} title={help}>
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                  <small>{help}</small>
                </div>
              ))}
            </dl>
          )}
          {Object.keys(bests).length > 0 && (
            <table className="peak-table">
              <thead>
                <tr><th>PEAK POWER</th><th>WATTS</th>{weight && <th>W/KG</th>}{ftp && <th>% FTP</th>}</tr>
              </thead>
              <tbody>
                {PEAK_DURATIONS.filter(([seconds]) => bests[String(seconds)]).map(([seconds, label]) => {
                  const watts = bests[String(seconds)];
                  return (
                    <tr key={seconds}>
                      <td>{label}</td>
                      <td className="numeric-cell">{watts}</td>
                      {weight && <td className="numeric-cell">{(watts / weight).toFixed(2)}</td>}
                      {ftp && <td className="numeric-cell">{Math.round((watts / ftp) * 100)}%</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </article>
      </div>

      {(effort || (profile && !track)) && (
        <div className="detail-grid">
          {effort && (
            <article className="card detail-chart">
              <div className="card-kicker">EFFORT</div>
              <Chart option={effort} height={240} label="Power and heart rate over the activity" />
            </article>
          )}
          {profile && !track && (
            <article className="card detail-chart">
              <div className="card-kicker">ELEVATION PROFILE</div>
              <Chart option={profile} height={240} label="Elevation over distance" />
            </article>
          )}
        </div>
      )}

      {(powerZones || hrZones) && (
        <div className="detail-grid">
          {powerZones && ftp && (
            <ZoneBars
              title={`POWER ZONES · FTP ${ftp} W`}
              zones={POWER_ZONES.map((zone) => zone.name)}
              seconds={powerZones}
            />
          )}
          {hrZones && (
            <ZoneBars
              title={`HEART RATE ZONES · MAX ${heartRate.maxHr} BPM`}
              zones={HEART_RATE_ZONES.map((zone) => zone.name)}
              seconds={hrZones}
            />
          )}
        </div>
      )}
    </section>
  );
}

function ZoneBars({ title, zones, seconds }: { title: string; zones: string[]; seconds: number[] }) {
  const total = seconds.reduce((sum, value) => sum + value, 0) || 1;
  const top = seconds.indexOf(Math.max(...seconds));
  return (
    <article className="card zone-card">
      <div className="card-kicker">{title}</div>
      <ul className="zone-list">
        {zones.map((zone, index) => (
          <li key={zone} className={index === top ? "top" : ""}>
            <span>Z{index + 1} {zone}</span>
            <span className="zone-track"><i style={{ width: `${(seconds[index] / total) * 100}%` }} /></span>
            <b>{formatHours(seconds[index] / 3600)}</b>
          </li>
        ))}
      </ul>
    </article>
  );
}
