"use client";

import { useMemo, useState } from "react";
import { YearComparisonChart } from "@/components/year-comparison-chart";
import {
  activityDay,
  activityMonth,
  activityYear,
  type Activity,
  type ComparisonMetric,
} from "@/lib/activity-types";

export function YearReview({
  activities,
  category,
}: {
  activities: Activity[];
  category: string;
}) {
  const [metric, setMetric] = useState<ComparisonMetric>(
    category === "Pickleball" ? "sessions" : "distance",
  );
  const years = useMemo(
    () =>
      Array.from(
        new Set(
          activities
            .map(activityYear)
            .filter((year): year is number => year !== undefined),
        ),
      ).sort((a, b) => b - a),
    [activities],
  );
  const [selectedYear, setSelectedYear] = useState<number | undefined>();
  const year = selectedYear ?? years[0];
  const yearActivities = activities.filter(
    (activity) => activityYear(activity) === year,
  );
  const latestDate = yearActivities.reduce(
    (latest, activity) => (activity.date > latest ? activity.date : latest),
    "",
  );
  const cutoffActivity = yearActivities.find(
    (activity) => activity.date === latestDate,
  );
  const cutoffMonth = cutoffActivity ? activityMonth(cutoffActivity) ?? 11 : 11;
  const cutoffDay = cutoffActivity ? activityDay(cutoffActivity) ?? 31 : 31;
  const metrics: ComparisonMetric[] =
    category === "Pickleball"
      ? ["duration", "sessions"]
      : ["distance", "duration", "sessions"];
  const personalBests = [
    {
      label: "Longest session",
      value: `${Math.max(0, ...yearActivities.map((activity) => activity.moving_time)).toFixed(1)} hrs`,
      icon: "◷",
    },
    {
      label: category === "Pickleball" ? "Most active day" : "Longest distance",
      value:
        category === "Pickleball"
          ? `${Math.max(
              0,
              ...Object.values(
                yearActivities.reduce<Record<string, number>>((days, activity) => {
                  days[activity.date.slice(0, 10)] =
                    (days[activity.date.slice(0, 10)] ?? 0) + 1;
                  return days;
                }, {}),
              ),
            )} sessions`
          : `${Math.max(0, ...yearActivities.map((activity) => activity.distance)).toFixed(1)} mi`,
      icon: "↗",
    },
    {
      label: "Most elevation",
      value: `${Math.round(Math.max(0, ...yearActivities.map((activity) => activity.total_elevation_gain))).toLocaleString("en-US")} ft`,
      icon: "⌃",
    },
    {
      label: "Fastest average speed",
      value: `${Math.max(0, ...yearActivities.map((activity) => activity.average_speed)).toFixed(1)} mph`,
      icon: "⌁",
    },
  ];

  if (!years.length) {
    return <div className="card year-review-empty">No activity history is available yet.</div>;
  }

  return (
    <section className="year-review-layout">
      <article className="card year-chart-card">
        <div className="card-heading">
          <div><div className="card-kicker">YOUR OWN BASELINE</div><h2>Compare your seasons</h2></div>
          <label className="year-select">
            <span>YEAR VIEW</span>
            <select
              aria-label="Select year for review"
              onChange={(event) => setSelectedYear(Number(event.target.value))}
              value={year}
            >
              {years.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
        </div>
        <p className="chart-subtitle">
          {category} · same-month and same-day comparison across years
        </p>
        <div className="year-metric-tabs" role="group" aria-label="Comparison metric">
          {metrics.map((value) => (
            <button
              aria-pressed={metric === value}
              className={metric === value ? "selected" : ""}
              key={value}
              onClick={() => setMetric(value)}
              type="button"
            >
              {value === "distance" ? "Distance" : value === "duration" ? "Active time" : "Sessions"}
            </button>
          ))}
        </div>
        <YearComparisonChart
          activities={activities}
          highlightYear={year}
          metric={metric}
          throughDay={cutoffDay}
          throughMonth={cutoffMonth}
        />
      </article>
      <article className="card year-summary-card">
        <div className="card-kicker">SEASON SNAPSHOT</div>
        <h2>{year} in numbers</h2>
        <div className="season-total">
          <strong>{yearActivities.length.toLocaleString("en-US")}</strong>
          <span>activities</span>
        </div>
        <div className="season-stats">
          <span><strong>{yearActivities.reduce((sum, activity) => sum + activity.moving_time, 0).toFixed(1)}</strong><small>active hours</small></span>
          <span><strong>{category === "Pickleball" ? yearActivities.length : yearActivities.reduce((sum, activity) => sum + activity.distance, 0).toFixed(0)}</strong><small>{category === "Pickleball" ? "sessions" : "distance miles"}</small></span>
          <span><strong>{Math.round(yearActivities.reduce((sum, activity) => sum + activity.total_elevation_gain, 0)).toLocaleString("en-US")}</strong><small>climbing feet</small></span>
        </div>
        <div className="personal-best-heading">PERSONAL BESTS · {year}</div>
        {personalBests.map((best) => (
          <div className="best-row" key={best.label}>
            <span className="best-icon">{best.icon}</span>
            <span>{best.label}</span>
            <strong>{best.value}</strong>
          </div>
        ))}
      </article>
    </section>
  );
}
