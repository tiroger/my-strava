"use client";

import type { EChartsOption } from "echarts";
import { Chart, CHART_COLORS, CHART_TEXT, CHART_TOOLTIP } from "@/components/chart";
import {
  formatCalendarDate,
  formatHours,
  formatNumber,
  SPORT_LABELS,
  type Activity,
  type CalendarDate,
  type SportKind,
} from "@/lib/activity-types";
import { activeDays } from "@/lib/metrics/active-days";

const KIND_COLORS: Record<SportKind, string> = {
  Ride: CHART_COLORS.orange,
  Run: CHART_COLORS.blue,
  Pickleball: CHART_COLORS.green,
  Walk: CHART_COLORS.gold,
  Swim: "#4fb3c8",
  Training: CHART_COLORS.violet,
  Other: CHART_COLORS.muted,
};

/** A year as a calendar: shaded by active hours, with streaks and the sport mix. */
export function ActiveDays({
  activities,
  year,
  today,
  compact = false,
}: {
  activities: Activity[];
  year: number;
  today: CalendarDate;
  compact?: boolean;
}) {
  const summary = activeDays(activities, year, today);
  const maxHours = Math.max(1, ...summary.days.map((day) => day.hours));

  const calendar: EChartsOption = {
    tooltip: {
      ...CHART_TOOLTIP,
      formatter: (item) => {
        const [date, hours, count] = (item as unknown as { value: [string, number, number] }).value;
        return `<b>${formatCalendarDate(date, { weekday: "short", month: "short", day: "numeric" })}</b><br/>${count} ${count === 1 ? "activity" : "activities"} · ${formatHours(hours)}`;
      },
    },
    visualMap: {
      show: false,
      min: 0,
      max: maxHours,
      inRange: { color: ["#3a2a24", "#8a3a22", CHART_COLORS.orange, "#ffb08f"] },
    },
    calendar: {
      top: 22,
      left: 30,
      right: 6,
      bottom: 4,
      range: String(year),
      cellSize: ["auto", 13],
      splitLine: { show: false },
      itemStyle: { color: "#161c20", borderColor: "#0f1417", borderWidth: 2 },
      dayLabel: { ...CHART_TEXT, firstDay: 1, nameMap: ["S", "M", "T", "W", "T", "F", "S"] },
      monthLabel: { ...CHART_TEXT },
      yearLabel: { show: false },
    },
    series: [
      {
        type: "heatmap",
        coordinateSystem: "calendar",
        data: summary.days.map((day) => [day.date, Math.round(day.hours * 100) / 100, day.count]),
      },
    ],
  };

  const pie: EChartsOption = {
    tooltip: {
      ...CHART_TOOLTIP,
      formatter: (item) => {
        const entry = item as { name: string; value: number; percent: number };
        return `${entry.name}: <b>${entry.value}</b> (${entry.percent}%)`;
      },
    },
    series: [
      {
        type: "pie",
        radius: ["58%", "82%"],
        padAngle: 2,
        itemStyle: { borderRadius: 3 },
        label: { show: false },
        data: summary.byKind.map((entry) => ({
          name: SPORT_LABELS[entry.kind],
          value: entry.count,
          itemStyle: { color: KIND_COLORS[entry.kind] },
        })),
      },
    ],
  };

  const percent = summary.daysElapsed ? summary.activeDays / summary.daysElapsed : 0;
  return (
    <div className={`active-days ${compact ? "compact" : ""}`}>
      <div className="active-days-stats">
        <span>
          <strong>{summary.activeDays}</strong>
          <small>of {summary.daysElapsed} days active ({Math.round(percent * 100)}%)</small>
        </span>
        <span>
          <strong>{summary.longestStreak}</strong>
          <small>day longest streak</small>
        </span>
        {year === today.year && (
          <span>
            <strong>{summary.currentStreak}</strong>
            <small>day current streak</small>
          </span>
        )}
      </div>
      <div className="active-days-body">
        <Chart option={calendar} height={128} label={`Calendar of active days in ${year}`} />
        {summary.byKind.length > 1 && (
          <div className="sport-mix">
            <Chart option={pie} height={112} label={`Activities by sport in ${year}`} />
            <ul>
              {summary.byKind.slice(0, 5).map((entry) => (
                <li key={entry.kind}>
                  <i style={{ background: KIND_COLORS[entry.kind] }} />
                  {SPORT_LABELS[entry.kind]}
                  <b>{formatNumber(entry.count)}</b>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
