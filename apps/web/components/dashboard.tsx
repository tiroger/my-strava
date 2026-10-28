"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ActiveDays } from "@/components/active-days";
import { ActivityDetail } from "@/components/activity-detail";
import { GaragePanel } from "@/components/garage-panel";
import { GoalsCard } from "@/components/goals-card";
import { HeatmapPanel } from "@/components/heatmap-panel";
import { PerformancePanel } from "@/components/performance-panel";
import { ProgressionChart } from "@/components/progression-chart";
import { RouteExplorer } from "@/components/route-explorer";
import { SavedRoutesPanel } from "@/components/saved-routes-panel";
import { SportIcon, SPORT_TONES } from "@/components/sport-icon";
import { YearReview } from "@/components/year-review";
import {
  activityYear,
  calendarDate,
  formatCalendarDate,
  formatHours,
  formatMetric,
  formatNumber,
  getActivityKind,
  isDistanceSport,
  matchesFilter,
  METRIC_LABELS,
  metricsFor,
  SPORT_LABELS,
  sportLabel,
  type Activity,
  type Gear,
  type MaintenanceRecord,
  type MetricKey,
  type SportFilter,
  type SportKind,
} from "@/lib/activity-types";
import type { PerformanceSummary } from "@/lib/metrics/performance";
import { progression } from "@/lib/metrics/progression";
import type { Settings } from "@/lib/settings";
import { ownerRequest, useAdmin, type AdminSession } from "@/lib/use-admin";
import { useToday } from "@/lib/use-today";

type Section =
  | "Overview"
  | "Activities"
  | "Routes"
  | "Saved routes"
  | "Heatmap"
  | "Performance"
  | "Garage"
  | "Year in review";

const SECTIONS: Array<{ name: Section; icon: string }> = [
  { name: "Overview", icon: "◫" },
  { name: "Activities", icon: "↗" },
  { name: "Routes", icon: "⌖" },
  { name: "Saved routes", icon: "△" },
  { name: "Heatmap", icon: "◍" },
  { name: "Performance", icon: "ϟ" },
  { name: "Garage", icon: "⚙" },
  { name: "Year in review", icon: "▥" },
];

/** Sections that show every sport the same way have no sport filter. */
const UNFILTERED: Section[] = ["Saved routes", "Performance", "Garage"];

export function Dashboard({
  dataVersion,
  initialActivities,
  initialAdmin,
  initialGear,
  initialMaintenance,
  initialSettings,
  serverToday,
}: {
  dataVersion: string;
  initialActivities: Activity[];
  initialAdmin: boolean;
  initialGear: Gear[];
  initialMaintenance: MaintenanceRecord[];
  initialSettings: Settings;
  serverToday: string;
}) {
  const router = useRouter();
  const admin = useAdmin(initialAdmin);
  const todayIso = useToday(serverToday);
  const today = calendarDate(todayIso) ?? { year: 1970, month: 0, day: 1 };
  const [section, setSection] = useState<Section>("Overview");
  const [openActivityId, setOpenActivityId] = useState<string | null>(null);
  const [filter, setFilter] = useState<SportFilter>("All");
  const [settings, setSettings] = useState(initialSettings);
  const [performance, setPerformance] = useState<PerformanceSummary | null>(null);
  const [performanceError, setPerformanceError] = useState("");

  // Poll a small version check and re-render only when the data changed.
  useEffect(() => {
    const timer = window.setInterval(async () => {
      if (document.visibilityState !== "visible") return;
      try {
        const response = await fetch("/api/data-version", { cache: "no-store" });
        if (!response.ok) return;
        const { version } = (await response.json()) as { version?: string };
        if (version && version !== dataVersion) router.refresh();
      } catch {
        // Offline or a transient error; try again on the next tick.
      }
    }, 60_000);
    return () => window.clearInterval(timer);
  }, [dataVersion, router]);

  const needsPerformance = section === "Performance" || section === "Activities" || openActivityId !== null;
  useEffect(() => {
    if (!needsPerformance || performance) return;
    let cancelled = false;
    fetch("/api/performance")
      .then(async (response) => {
        if (!response.ok) throw new Error("Performance data is unavailable right now.");
        return (await response.json()) as PerformanceSummary;
      })
      .then((value) => !cancelled && setPerformance(value))
      .catch((error: unknown) => {
        if (!cancelled) setPerformanceError(error instanceof Error ? error.message : "Unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [needsPerformance, performance]);

  const saveSettings = useCallback(async (next: Settings) => {
    const saved = await ownerRequest<Settings>("/api/settings", "PUT", next);
    setSettings(saved);
    setPerformance(null);
  }, []);

  const saveGoal = useCallback(
    async (key: string, value: number | null) => {
      const goals = { ...settings.goals };
      if (value === null) delete goals[key];
      else goals[key] = value;
      await saveSettings({ ...settings, goals });
    },
    [saveSettings, settings],
  );

  const kinds = useMemo(() => {
    const counts = new Map<SportKind, number>();
    for (const activity of initialActivities) {
      const kind = getActivityKind(activity.type);
      counts.set(kind, (counts.get(kind) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([kind]) => kind);
  }, [initialActivities]);

  const filtered = useMemo(
    () => initialActivities.filter((activity) => matchesFilter(activity, filter)),
    [filter, initialActivities],
  );
  const latest = initialActivities.reduce<Activity | null>(
    (top, activity) => (!top || activity.date > top.date ? activity : top),
    null,
  );
  const openActivity = initialActivities.find((activity) => activity.id === openActivityId);

  const open = (id: string) => {
    setOpenActivityId(id);
    window.scrollTo({ top: 0 });
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <Link className="brand" href="/" aria-label="My Strava home">
          <span className="brand-mark">S</span>
          <span>
            <strong>MY STRAVA</strong>
            <small>RIDE INTELLIGENCE</small>
          </span>
        </Link>
        <div className="side-label">WORKSPACE</div>
        <nav className="navigation" aria-label="Main navigation">
          {SECTIONS.map((item) => (
            <button
              className={`nav-item ${section === item.name && !openActivity ? "active" : ""}`}
              key={item.name}
              aria-current={section === item.name && !openActivity ? "page" : undefined}
              onClick={() => {
                setSection(item.name);
                setOpenActivityId(null);
              }}
              type="button"
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              {item.name}
              {item.name === "Activities" && <span className="nav-count">{formatNumber(initialActivities.length)}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sync-status">
            <span className="status-dot" />
            <span>
              <strong>Synced from Strava</strong>
              <small>{latest ? `Latest activity ${formatCalendarDate(latest.date)}` : "Waiting for the first sync"}</small>
            </span>
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="crumb-slash">/</span>
            {openActivity ? (
              <>
                <button className="crumb-link" type="button" onClick={() => setOpenActivityId(null)}>{section}</button>
                <span className="crumb-slash">/</span>
                <strong>{openActivity.name}</strong>
              </>
            ) : (
              <strong>{section}</strong>
            )}
          </div>
          <div className="topbar-actions">
            <span className="topbar-date">
              {formatCalendarDate(todayIso, { weekday: "short", month: "short", day: "numeric" })}
            </span>
            <OwnerControl admin={admin} />
            <a className="strava-connect" href="https://www.strava.com/athletes/644338" rel="noreferrer" target="_blank">
              <span className="strava-glyph">⌁</span>
              View on Strava
            </a>
          </div>
        </header>

        <div className="content">
          {!openActivity && (
            <SectionHeader
              section={section}
              filter={filter}
              kinds={kinds}
              onFilter={setFilter}
              showFilter={!UNFILTERED.includes(section)}
            />
          )}

          {openActivity ? (
            <ActivityDetail
              activity={openActivity}
              gear={initialGear}
              performance={performance}
              onBack={() => setOpenActivityId(null)}
            />
          ) : section === "Overview" ? (
            <Overview
              activities={filtered}
              filter={filter}
              today={today}
              goals={settings.goals}
              canEdit={admin.authenticated}
              onSaveGoal={saveGoal}
              onOpenActivity={open}
              onShowAll={() => setSection("Activities")}
            />
          ) : section === "Activities" ? (
            <ActivityTable activities={filtered} loads={performance?.loads} onOpenActivity={open} expanded />
          ) : section === "Routes" ? (
            <RouteExplorer activities={filtered} onOpenActivity={open} />
          ) : section === "Saved routes" ? (
            <SavedRoutesPanel />
          ) : section === "Heatmap" ? (
            <HeatmapPanel filter={filter} onOpenActivity={open} />
          ) : section === "Performance" ? (
            <PerformancePanel
              performance={performance}
              error={performanceError}
              settings={settings}
              canEdit={admin.authenticated}
              onSaveSettings={saveSettings}
            />
          ) : section === "Garage" ? (
            <GaragePanel
              gear={initialGear}
              activities={initialActivities}
              initialRecords={initialMaintenance}
              todayIso={todayIso}
              canEdit={admin.authenticated}
            />
          ) : (
            <YearReview activities={filtered} filter={filter} today={today} onOpenActivity={open} />
          )}
          <footer className="page-footer">
            <span>MY STRAVA · PERSONAL TRAINING SPACE · POWERED BY STRAVA</span>
            <span>New activities arrive from Strava within a minute of upload.</span>
          </footer>
        </div>
      </section>
    </main>
  );
}

function SectionHeader({
  section,
  filter,
  kinds,
  onFilter,
  showFilter,
}: {
  section: Section;
  filter: SportFilter;
  kinds: SportKind[];
  onFilter: (filter: SportFilter) => void;
  showFilter: boolean;
}) {
  const overview = section === "Overview";
  return (
    <section className={`welcome-row ${overview ? "" : "compact"}`}>
      <div>
        <div className="eyebrow">
          <span className="eyebrow-line" />
          {overview ? "YOUR TRAINING AT A GLANCE" : section.toUpperCase()}
        </div>
        {overview ? (
          <>
            <h1>Ride with <em>intention.</em></h1>
            <p className="welcome-copy">Your training story, translated into a clearer picture.</p>
          </>
        ) : (
          <h1 className="section-title">{section}</h1>
        )}
      </div>
      {showFilter && (
        <label className="sport-filter">
          <span>SPORT</span>
          <select aria-label="Select activity sport" value={filter} onChange={(event) => onFilter(event.target.value as SportFilter)}>
            <option value="All">{SPORT_LABELS.All}</option>
            {kinds.map((kind) => <option key={kind} value={kind}>{SPORT_LABELS[kind]}</option>)}
          </select>
        </label>
      )}
    </section>
  );
}

function OwnerControl({ admin }: { admin: AdminSession }) {
  const [open, setOpen] = useState(false);
  const [passphrase, setPassphrase] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (admin.authenticated) {
    return (
      <button
        className="owner-button active"
        type="button"
        onClick={() => void admin.lock().catch((reason: unknown) => setError(String(reason)))}
        title="Owner mode is on: goals, settings and maintenance are editable. Click to lock."
      >
        ● Owner · Lock
      </button>
    );
  }
  return (
    <div className="owner-control">
      <button className="owner-button" type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        Owner
      </button>
      {open && (
        <form
          className="owner-popover card"
          onSubmit={async (event) => {
            event.preventDefault();
            setBusy(true);
            setError("");
            try {
              await admin.unlock(passphrase);
              setPassphrase("");
              setOpen(false);
            } catch (reason) {
              setError(reason instanceof Error ? reason.message : "Could not unlock");
            } finally {
              setBusy(false);
            }
          }}
        >
          <label htmlFor="owner-passphrase">Owner passphrase</label>
          <div className="admin-input-row">
            <input
              autoComplete="current-password"
              autoFocus
              id="owner-passphrase"
              onChange={(event) => setPassphrase(event.target.value)}
              type="password"
              value={passphrase}
            />
            <button className="primary-button" disabled={busy || !passphrase} type="submit">
              {busy ? "Checking…" : "Unlock"}
            </button>
          </div>
          <p>The dashboard is public. Owner mode edits goals, settings and the maintenance log.</p>
          {error && <p className="form-error">{error}</p>}
        </form>
      )}
    </div>
  );
}

function Overview({
  activities,
  filter,
  today,
  goals,
  canEdit,
  onSaveGoal,
  onOpenActivity,
  onShowAll,
}: {
  activities: Activity[];
  filter: SportFilter;
  today: { year: number; month: number; day: number };
  goals: Record<string, number>;
  canEdit: boolean;
  onSaveGoal: (key: string, value: number | null) => Promise<void>;
  onOpenActivity: (id: string) => void;
  onShowAll: () => void;
}) {
  const metrics = metricsFor(filter);
  const [metricChoice, setMetric] = useState<MetricKey>(metrics[0]);
  const metric = metrics.includes(metricChoice) ? metricChoice : metrics[0];
  const result = useMemo(() => progression(activities, metric, today), [activities, metric, today]);
  const headline = useMemo(() => progression(activities, metrics[0], today), [activities, metrics, today]);
  const distanceSport = isDistanceSport(filter);

  const totals = activities.reduce(
    (sum, activity) => ({
      distance: sum.distance + activity.distance,
      hours: sum.hours + activity.moving_time,
      elevation: sum.elevation + activity.total_elevation_gain,
      longest: Math.max(sum.longest, activity.moving_time),
    }),
    { distance: 0, hours: 0, elevation: 0, longest: 0 },
  );
  const yearToDate = headline.current?.total ?? 0;
  const change =
    headline.lastYearToDate && headline.lastYearToDate > 0
      ? Math.round(((yearToDate - headline.lastYearToDate) / headline.lastYearToDate) * 100)
      : null;
  const recent = [...activities].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 5);

  return (
    <>
      <section className="metrics-grid" aria-label="Training totals">
        <MetricCard
          label={distanceSport ? "TOTAL DISTANCE" : "SESSIONS"}
          value={distanceSport ? formatNumber(totals.distance) : formatNumber(activities.length)}
          unit={distanceSport ? "mi" : ""}
          detail={`${formatNumber(activities.length)} ${filter === "All" ? "activities" : SPORT_LABELS[filter].toLowerCase()} since ${activities.length ? activityYear(activities.reduce((a, b) => (a.date < b.date ? a : b))) : "—"}`}
          icon="↗"
          accent="orange"
        />
        <MetricCard
          label="ACTIVE TIME"
          value={formatHours(totals.hours)}
          detail={`Average ${formatHours(activities.length ? totals.hours / activities.length : 0)} per activity`}
          icon="◷"
          accent="blue"
        />
        <MetricCard
          label={distanceSport ? "ELEVATION GAIN" : "LONGEST SESSION"}
          value={distanceSport ? formatNumber(totals.elevation) : formatHours(totals.longest)}
          unit={distanceSport ? "ft" : ""}
          detail={distanceSport ? `${formatNumber(totals.elevation / 29_032, 1)} times up Everest` : "Personal best duration"}
          icon="⌃"
          accent="green"
        />
        <MetricCard
          label={`${today.year} SO FAR`}
          value={metrics[0] === "time" ? formatHours(yearToDate) : formatNumber(yearToDate)}
          unit={metrics[0] === "distance" ? "mi" : metrics[0] === "sessions" ? "sessions" : ""}
          detail={
            change === null
              ? headline.onPace
                ? `On pace for ${formatMetric(headline.onPace, metrics[0])}`
                : "Nothing recorded yet this year"
              : `${change >= 0 ? "+" : ""}${change}% vs ${today.year - 1} by this date`
          }
          icon="◉"
          accent="violet"
        />
      </section>

      <section className="dashboard-grid">
        <article className="card volume-card">
          <div className="card-heading">
            <div>
              <div className="card-kicker">YEAR OVER YEAR</div>
              <h2>Cumulative {METRIC_LABELS[metric].toLowerCase()} by day of the year</h2>
            </div>
            <div className="year-metric-tabs" role="group" aria-label="Progression metric">
              {metrics.map((value) => (
                <button
                  aria-pressed={metric === value}
                  className={metric === value ? "selected" : ""}
                  key={value}
                  onClick={() => setMetric(value)}
                  type="button"
                >
                  {METRIC_LABELS[value]}
                </button>
              ))}
            </div>
          </div>
          <p className="chart-subtitle">
            <span className="legend-swatch orange" /> {today.year}
            <span className="legend-swatch light" /> {today.year - 1}
            <span className="legend-swatch faint" /> earlier years
            <span className="legend-swatch bar" /> {today.year} by month
          </p>
          <ProgressionChart progression={result} metric={metric} highlightYear={today.year} today={today} />
        </article>
        <GoalsCard
          activities={activities}
          filter={filter}
          today={today}
          goals={goals}
          canEdit={canEdit}
          onSaveGoal={onSaveGoal}
        />
      </section>

      <section className="dashboard-grid">
        <article className="card active-days-card">
          <div className="card-kicker">ACTIVE DAYS · {today.year}</div>
          <ActiveDays activities={activities} year={today.year} today={today} compact />
        </article>
        <article className="card recent-card">
          <div className="card-heading">
            <div>
              <div className="card-kicker">KEEP THE MOMENTUM</div>
              <h2>Recent {filter === "All" ? "activity" : SPORT_LABELS[filter].toLowerCase()}</h2>
            </div>
            <button className="text-button" type="button" onClick={onShowAll}>
              View all <span aria-hidden="true">→</span>
            </button>
          </div>
          <div className="activity-list">
            {recent.map((activity) => (
              <button className="activity-row" key={activity.id} type="button" onClick={() => onOpenActivity(activity.id)}>
                <span className={`activity-avatar ${SPORT_TONES[getActivityKind(activity.type)]}`}>
                  <SportIcon type={activity.type} />
                </span>
                <span className="activity-description">
                  <strong>{activity.name}</strong>
                  <small>{formatCalendarDate(activity.date, { month: "short", day: "numeric" })} · {sportLabel(activity.type)}</small>
                </span>
                <span className="activity-distance">
                  {activity.distance > 0 ? <>{formatNumber(activity.distance, 1)} <small>mi</small></> : formatHours(activity.moving_time)}
                </span>
              </button>
            ))}
            {!recent.length && <div className="empty-state">No activities yet. They appear after the first sync.</div>}
          </div>
        </article>
      </section>

      <ActivityTable activities={activities} onOpenActivity={onOpenActivity} />
    </>
  );
}

function MetricCard({
  label,
  value,
  unit,
  detail,
  icon,
  accent,
}: {
  label: string;
  value: string;
  unit?: string;
  detail: string;
  icon: string;
  accent: string;
}) {
  return (
    <article className="card metric-card">
      <div className={`metric-icon ${accent}`}>{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span>{unit}</span>}
      </div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}

function ActivityTable({
  activities,
  loads,
  onOpenActivity,
  expanded = false,
}: {
  activities: Activity[];
  loads?: PerformanceSummary["loads"];
  onOpenActivity: (id: string) => void;
  expanded?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [sortNewest, setSortNewest] = useState(true);
  const [showAll, setShowAll] = useState(expanded);
  const rows = useMemo(() => {
    const needle = query.toLowerCase();
    return activities
      .filter((activity) => `${activity.name} ${sportLabel(activity.type)}`.toLowerCase().includes(needle))
      .sort((a, b) => (sortNewest ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date)));
  }, [activities, query, sortNewest]);
  const visible = showAll ? rows : rows.slice(0, 10);

  return (
    <section className="card activity-table-card">
      <div className="table-card-heading">
        <div>
          <div className="card-kicker">YOUR HISTORY</div>
          <h2>Activity log</h2>
        </div>
        <div className="table-actions">
          <label className="search-box">
            <span aria-hidden="true">⌕</span>
            <input aria-label="Search activities" placeholder="Search activities" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <button className="sort-button" type="button" onClick={() => setSortNewest((value) => !value)}>
            {sortNewest ? "Newest first ↓" : "Oldest first ↑"}
          </button>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>ACTIVITY</th>
              <th>DATE</th>
              <th>TYPE</th>
              <th>DISTANCE</th>
              <th>TIME</th>
              <th>ELEVATION</th>
              {loads && <th title="Training load: TSS from power, or heart-rate based">LOAD</th>}
            </tr>
          </thead>
          <tbody>
            {visible.map((activity) => {
              const kind = getActivityKind(activity.type);
              const load = loads?.[activity.id];
              return (
                <tr key={activity.id}>
                  <td>
                    <button className="activity-name-cell" type="button" onClick={() => onOpenActivity(activity.id)}>
                      <span className={`activity-type-icon ${SPORT_TONES[kind]}`}><SportIcon kind={kind} /></span>
                      <span>{activity.name}</span>
                    </button>
                  </td>
                  <td>{formatCalendarDate(activity.date)}</td>
                  <td><span className="type-pill">{sportLabel(activity.type)}</span></td>
                  <td className="numeric-cell">{activity.distance > 0 ? `${formatNumber(activity.distance, 1)} mi` : "—"}</td>
                  <td>{formatHours(activity.moving_time)}</td>
                  <td className="numeric-cell">{activity.total_elevation_gain > 0 ? `${formatNumber(activity.total_elevation_gain)} ft` : "—"}</td>
                  {loads && (
                    <td className="numeric-cell" title={load ? `From ${load.source}` : undefined}>
                      {load ? Math.round(load.tss) : "—"}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length && <div className="empty-state">No activities match.</div>}
        {rows.length > 10 && (
          <button className="load-more" onClick={() => setShowAll((value) => !value)} type="button">
            {showAll ? "Show fewer activities" : `Showing 10 of ${formatNumber(rows.length)} activities`}
            <span>{showAll ? "Collapse ↑" : "Show all →"}</span>
          </button>
        )}
      </div>
    </section>
  );
}
