"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  activityYear,
  activityDay,
  activityMonth,
  calendarDate,
  formatCalendarDate,
  getActivityKind,
  type Activity,
  type Bike,
  type ComparisonMetric,
  type MaintenanceRecord,
} from "@/lib/activity-types";
import { YearComparisonChart } from "@/components/year-comparison-chart";
import { RouteExplorer } from "@/components/route-explorer";
import { GaragePanel } from "@/components/garage-panel";
import { YearReview } from "@/components/year-review";
import { useToday } from "@/lib/use-today";

type Section = "Overview" | "Activities" | "Routes" | "Garage" | "Year in review";

const sections: Section[] = [
  "Overview",
  "Activities",
  "Routes",
  "Garage",
  "Year in review",
];

const icons: Record<Section, string> = {
  Overview: "◫",
  Activities: "↗",
  Routes: "⌖",
  Garage: "⚙",
  "Year in review": "▥",
};

function formatDistance(miles: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(
    miles,
  );
}

function formatDuration(hours: number): string {
  const minutes = Math.round(hours * 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h ? `${h}h ${m}m` : `${m}m`;
}

export function Dashboard({
  dataVersion,
  initialActivities,
  initialAdmin,
  initialBikes,
  initialMaintenance,
  serverToday,
}: {
  dataVersion: string;
  initialActivities: Activity[];
  initialAdmin: boolean;
  initialBikes: Bike[];
  initialMaintenance: MaintenanceRecord[];
  serverToday: string;
}) {
  const [section, setSection] = useState<Section>("Overview");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All activities");
  const [sortNewest, setSortNewest] = useState(true);
  const [comparisonMetric, setComparisonMetric] =
    useState<ComparisonMetric>("distance");
  const [showAll, setShowAll] = useState(false);
  const router = useRouter();
  const todayIso = useToday(serverToday);
  const today = calendarDate(todayIso) ?? { year: 1970, month: 0, day: 1 };

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

  const activities = useMemo(() => {
    const filtered = initialActivities.filter((activity) => {
      const matchesText = `${activity.name} ${activity.type}`
        .toLowerCase()
        .includes(query.toLowerCase());
      const kind = getActivityKind(activity.type);
      return (
        matchesText &&
        (filter === "All activities" || kind === filter)
      );
    });
    return filtered.sort((a, b) =>
      sortNewest
        ? b.date.localeCompare(a.date)
        : a.date.localeCompare(b.date),
    );
  }, [filter, initialActivities, query, sortNewest]);

  const categories = useMemo(
    () =>
      Array.from(
        new Set(initialActivities.map((activity) => getActivityKind(activity.type))),
      ).sort(),
    [initialActivities],
  );
  const selectedActivities = useMemo(
    () =>
      initialActivities.filter(
        (activity) =>
          filter === "All activities" ||
          getActivityKind(activity.type) === filter,
      ),
    [filter, initialActivities],
  );
  const isDistanceSport = filter !== "Pickleball";
  const metric = isDistanceSport ? comparisonMetric : comparisonMetric === "distance" ? "sessions" : comparisonMetric;

  const summary = useMemo(() => {
    const totalDistance = selectedActivities.reduce(
      (sum, activity) => sum + activity.distance,
      0,
    );
    const totalClimbing = selectedActivities.reduce(
      (sum, activity) => sum + activity.total_elevation_gain,
      0,
    );
    const totalHours = selectedActivities.reduce(
      (sum, activity) => sum + activity.moving_time,
      0,
    );
    const lastDate = selectedActivities.reduce(
      (latest, activity) =>
        activity.date > latest ? activity.date : latest,
      "",
    );
    return {
      totalDistance,
      totalClimbing,
      totalHours,
      lastDate,
      count: selectedActivities.length,
    };
  }, [selectedActivities]);

  const yearToDate = selectedActivities.filter(
    (activity) => activityYear(activity) === today.year,
  );
  const yearToDateDistance = yearToDate.reduce(
    (sum, activity) => sum + activity.distance,
    0,
  );
  const yearToDateHours = yearToDate.reduce(
    (sum, activity) => sum + activity.moving_time,
    0,
  );
  const averageSession = summary.count ? summary.totalHours / summary.count : 0;
  const longestSession = Math.max(
    0,
    ...selectedActivities.map((activity) => activity.moving_time),
  );
  const previousYearCount = selectedActivities.filter((activity) => {
    if (activityYear(activity) !== today.year - 1) return false;
    const month = activityMonth(activity);
    const day = activityDay(activity);
    return (
      month !== undefined &&
      day !== undefined &&
      (month < today.month || (month === today.month && day <= today.day))
    );
  }).length;
  const yearCountChange =
    yearToDate.length && previousYearCount
      ? Math.round(((yearToDate.length - previousYearCount) / previousYearCount) * 100)
      : null;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <Link className="brand" href="/" aria-label="My Strava home">
          <span className="brand-mark">S</span>
          <span>
            <strong>STRAVA</strong>
            <small>RIDE INTELLIGENCE</small>
          </span>
        </Link>
        <div className="side-label">WORKSPACE</div>
        <nav className="navigation" aria-label="Main navigation">
          {sections.map((item) => (
            <button
              className={`nav-item ${section === item ? "active" : ""}`}
              key={item}
              onClick={() => {
                setSection(item);
                if (item === "Activities") setShowAll(true);
              }}
              type="button"
            >
              <span className="nav-icon" aria-hidden="true">
                {icons[item]}
              </span>
              {item}
              {item === "Activities" && (
                <span className="nav-count">{initialActivities.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sync-status">
            <span className="status-dot" />
            <span>
              <strong>Data up to date</strong>
              <small>
                {summary.lastDate
                  ? `Last activity ${formatCalendarDate(summary.lastDate)}`
                  : "Waiting for activity data"}
              </small>
            </span>
          </div>
          <div className="profile-chip">
            <span className="avatar">RL</span>
            <span>
              <strong>Rider profile</strong>
              <small>Personal dashboard</small>
            </span>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="crumb-slash">/</span>
            <strong>{section}</strong>
          </div>
          <div className="topbar-actions">
            <span className="topbar-date">
              {formatCalendarDate(todayIso, {
                weekday: "short",
                month: "short",
                day: "numeric",
              })}
            </span>
            <button
              className="icon-button"
              type="button"
              aria-label="Refresh activities"
              title="Activity data refreshes automatically from Strava"
              onClick={() => window.location.reload()}
            >
              ↻
            </button>
            <a className="strava-connect" href="https://www.strava.com/">
              <span className="strava-glyph">⌁</span>
              Connected to Strava
            </a>
          </div>
        </header>

        <div className="content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                YOUR TRAINING AT A GLANCE
              </div>
              <h1>
                Ride with <em>intention.</em>
              </h1>
              <p className="welcome-copy">
                Your training story, translated into a clearer picture.
              </p>
            </div>
            <div className="welcome-tools">
              <label className="sport-filter">
                <span>SPORT</span>
                <select
                  aria-label="Select activity sport"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                >
                  <option value="All activities">All activities</option>
                  {categories.map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </label>
              <div className="welcome-meta">
                <span className="live-dot" />
                LIVE ACTIVITY SYNC
              </div>
            </div>
          </section>

          {section === "Garage" ? (
            <GaragePanel
              bikes={initialBikes}
              initialAuthenticated={initialAdmin}
              initialRecords={initialMaintenance}
            />
          ) : section === "Routes" ? (
            <RouteExplorer activities={selectedActivities} />
          ) : section === "Year in review" ? (
            <YearReview activities={selectedActivities} category={filter} />
          ) : (
            <>
          <section className="metrics-grid" aria-label="Ride statistics">
            <MetricCard
              label={isDistanceSport ? "TOTAL DISTANCE" : "ACTIVITY SESSIONS"}
              value={isDistanceSport ? formatDistance(summary.totalDistance) : summary.count.toLocaleString("en-US")}
              unit={isDistanceSport ? "mi" : ""}
              detail={`${summary.count.toLocaleString("en-US")} ${filter === "Pickleball" ? "pickleball sessions" : "selected activities"}`}
              icon="↗"
              accent="orange"
            />
            <MetricCard
              label="TOTAL ACTIVE TIME"
              value={formatDuration(summary.totalHours)}
              detail={`Average ${formatDuration(averageSession)} per activity`}
              icon="◷"
              accent="blue"
            />
            <MetricCard
              label={filter === "Pickleball" ? "LONGEST SESSION" : "ELEVATION GAIN"}
              value={filter === "Pickleball" ? formatDuration(longestSession) : formatDistance(summary.totalClimbing)}
              unit={filter === "Pickleball" ? "" : "ft"}
              detail={filter === "Pickleball" ? "Personal best duration" : "Total climbing"}
              icon="⌃"
              accent="green"
            />
            <MetricCard
              label="YEAR TO DATE"
              value={isDistanceSport ? formatDistance(yearToDateDistance) : yearToDate.length.toLocaleString("en-US")}
              unit={isDistanceSport ? "mi" : "sessions"}
              detail={yearToDate.length === 0 ? "No activities recorded yet" : yearCountChange === null ? `${formatDuration(yearToDateHours)} active time` : `${yearCountChange >= 0 ? "+" : ""}${yearCountChange}% vs last year by this date`}
              icon="◉"
              accent="violet"
            />
          </section>

          <section className="dashboard-grid">
            <article className="card volume-card">
              <div className="card-heading">
                <div>
                  <div className="card-kicker">PERSONAL PROGRESS</div>
                  <h2>Year over year</h2>
                </div>
                <select
                  className="filter-select chart-metric-select"
                  aria-label="Comparison metric"
                  value={metric}
                  onChange={(event) => setComparisonMetric(event.target.value as ComparisonMetric)}
                >
                  {isDistanceSport && <option value="distance">Distance</option>}
                  <option value="duration">Active hours</option>
                  <option value="sessions">Sessions</option>
                </select>
              </div>
              <p className="chart-subtitle">
                Monthly totals, matched to today&apos;s date in each year
                {metric === "distance" && filter === "All activities" &&
                  " · activities with distance data"}
              </p>
              <YearComparisonChart
                activities={selectedActivities}
                highlightYear={today.year}
                metric={metric}
                throughDay={today.day}
                throughMonth={today.month}
              />
            </article>

            <article className="card recent-card">
              <div className="card-heading">
                <div>
                  <div className="card-kicker">KEEP THE MOMENTUM</div>
                  <h2>Recent {filter === "All activities" ? "activity" : filter.toLowerCase()}</h2>
                </div>
                <button
                  className="text-button"
                  type="button"
                  onClick={() => setSection("Activities")}
                >
                  View all <span aria-hidden="true">→</span>
                </button>
              </div>
              <div className="activity-list">
                {activities.slice(0, 4).map((activity, index) => (
                  <ActivityRow
                    activity={activity}
                    index={index}
                    key={activity.id}
                  />
                ))}
                {activities.length === 0 && (
                  <div className="empty-state">
                    No activities found. New Strava activities will appear
                    after sync.
                  </div>
                )}
              </div>
            </article>
          </section>

          <section className="card activity-table-card">
            <div className="table-card-heading">
              <div>
                <div className="card-kicker">YOUR HISTORY</div>
                <h2>{section === "Overview" ? "Activity log" : section}</h2>
              </div>
              <div className="table-actions">
                <label className="search-box">
                  <span aria-hidden="true">⌕</span>
                  <input
                    aria-label="Search activities"
                    placeholder="Search activities"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                </label>
                <button
                  className="sort-button"
                  type="button"
                  onClick={() => setSortNewest((value) => !value)}
                  title="Toggle date sort"
                >
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
                  </tr>
                </thead>
                <tbody>
                  {activities.slice(0, showAll ? activities.length : 10).map((activity) => (
                    <tr key={activity.id}>
                      <td className="activity-name-cell">
                        <span className="activity-type-icon">↗</span>
                        <span>{activity.name}</span>
                      </td>
                      <td>{formatCalendarDate(activity.date)}</td>
                      <td>
                        <span className="type-pill">
                          {getActivityKind(activity.type)}
                        </span>
                      </td>
                      <td className="numeric-cell">
                        {activity.distance > 0 ? `${formatDistance(activity.distance)} mi` : "—"}
                      </td>
                      <td>{formatDuration(activity.moving_time)}</td>
                      <td className="numeric-cell">
                        {activity.total_elevation_gain > 0 ? `${formatDistance(activity.total_elevation_gain)} ft` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {activities.length > 10 && (
                <button
                  className="load-more"
                  onClick={() => setShowAll((value) => !value)}
                  type="button"
                >
                  {showAll ? "Show fewer activities" : `Showing 10 of ${activities.length.toLocaleString("en-US")} activities`}
                  <span>{showAll ? "Collapse ↑" : "Show all activities →"}</span>
                </button>
              )}
            </div>
          </section>
            </>
          )}
          <footer className="page-footer">
            <span>MY STRAVA · PERSONAL TRAINING SPACE</span>
            <span>New activities sync automatically; this dashboard checks for updates every minute.</span>
          </footer>
        </div>
      </section>
    </main>
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

function ActivityRow({
  activity,
  index,
}: {
  activity: Activity;
  index: number;
}) {
  return (
    <div className="activity-row">
      <span className={`activity-avatar avatar-${index % 4}`}>
        {getActivityKind(activity.type) === "Ride" ? "↗" : "⌁"}
      </span>
      <span className="activity-description">
        <strong>{activity.name}</strong>
        <small>
          {formatCalendarDate(activity.date, { month: "short", day: "numeric" })} ·{" "}
          {getActivityKind(activity.type)}
        </small>
      </span>
      <span className="activity-distance">
        {formatDistance(activity.distance)} <small>mi</small>
      </span>
    </div>
  );
}
