"use client";

import { useState } from "react";
import {
  formatMetric,
  METRIC_LABELS,
  metricsFor,
  type Activity,
  type CalendarDate,
  type MetricKey,
  type SportFilter,
} from "@/lib/activity-types";
import { defaultGoal, goalProgress, progression } from "@/lib/metrics/progression";

function goalMetrics(filter: SportFilter): MetricKey[] {
  return metricsFor(filter).slice(0, 2);
}

/**
 * Annual goals for the selected sport: progress against an even pace, what is
 * needed per week from here, and the best year on record. Goals default to
 * last year plus ten percent until the owner sets one.
 */
export function GoalsCard({
  activities,
  filter,
  today,
  goals,
  canEdit,
  onSaveGoal,
}: {
  activities: Activity[];
  filter: SportFilter;
  today: CalendarDate;
  goals: Record<string, number>;
  canEdit: boolean;
  onSaveGoal: (key: string, value: number | null) => Promise<void>;
}) {
  return (
    <article className="card goals-card">
      <div className="card-kicker">{today.year} GOALS</div>
      <h2>On track?</h2>
      {goalMetrics(filter).map((metric) => (
        <GoalRow
          key={`${filter}:${metric}`}
          activities={activities}
          metric={metric}
          today={today}
          stored={goals[`${filter}:${metric}`]}
          canEdit={canEdit}
          onSave={(value) => onSaveGoal(`${filter}:${metric}`, value)}
        />
      ))}
    </article>
  );
}

function GoalRow({
  activities,
  metric,
  today,
  stored,
  canEdit,
  onSave,
}: {
  activities: Activity[];
  metric: MetricKey;
  today: CalendarDate;
  stored: number | undefined;
  canEdit: boolean;
  onSave: (value: number | null) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const result = progression(activities, metric, today);
  const lastYear = result.series.find((entry) => entry.year === today.year - 1)?.total;
  const fallback = defaultGoal(lastYear, metric);
  const goal = stored ?? fallback;
  const actual = result.current?.total ?? 0;

  async function save(value: number | null) {
    setError("");
    try {
      await onSave(value);
      setEditing(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save the goal");
    }
  }

  const label = METRIC_LABELS[metric].toLowerCase();
  const header = (
    <div className="goal-header">
      <span>
        <strong>{METRIC_LABELS[metric]}</strong>
        <small>
          {goal ? `Goal ${formatMetric(goal, metric)}` : "No goal yet"}
          {!stored && goal ? " · last year +10%" : ""}
        </small>
      </span>
      {canEdit && !editing && (
        <button
          className="text-button"
          type="button"
          onClick={() => {
            setDraft(goal ? String(Math.round(goal)) : "");
            setEditing(true);
          }}
        >
          Set goal
        </button>
      )}
    </div>
  );

  const editor = editing && (
    <form
      className="goal-editor"
      onSubmit={(event) => {
        event.preventDefault();
        const value = Number(draft);
        if (Number.isFinite(value) && value > 0) void save(metric === "time" ? value : Math.round(value));
      }}
    >
      <input
        aria-label={`${today.year} ${label} goal`}
        inputMode="decimal"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <button className="primary-button" type="submit">Save</button>
      {stored !== undefined && (
        <button className="text-button" type="button" onClick={() => void save(null)}>
          Use default
        </button>
      )}
      <button className="text-button" type="button" onClick={() => setEditing(false)}>Cancel</button>
    </form>
  );

  if (!goal) {
    return (
      <div className="goal-row">
        {header}
        {editor}
        <p className="goal-note">
          {formatMetric(actual, metric)} so far. Set a goal to track pace against it.
        </p>
        {error && <p className="form-error">{error}</p>}
      </div>
    );
  }

  const progress = goalProgress(goal, actual, result.todayPosition);
  const percent = (value: number) => `${Math.min(Math.max((value / goal) * 100, 0), 100)}%`;
  const ahead = progress.delta >= 0;
  return (
    <div className="goal-row">
      {header}
      {editor}
      <div className="goal-track" aria-hidden="true">
        <span className={`goal-fill ${ahead ? "ahead" : "behind"}`} style={{ width: percent(actual) }} />
        <span className="goal-pace" style={{ left: percent(progress.expected) }} title="Even pace" />
      </div>
      <div className="goal-figures">
        <span>
          <strong>{formatMetric(actual, metric)}</strong> of {formatMetric(goal, metric)}
        </span>
        <span className={ahead ? "goal-ahead" : "goal-behind"}>
          {formatMetric(Math.abs(progress.delta), metric)} {ahead ? "ahead of" : "behind"} pace
        </span>
      </div>
      <p className="goal-note">
        {actual >= goal
          ? `Goal reached. On pace for ${formatMetric(progress.projected, metric)}.`
          : `Needs ${formatMetric(progress.neededPerWeek, metric)} a week from here · on pace for ${formatMetric(progress.projected, metric)}.`}
        {result.best && ` Best year: ${result.best.year}, ${formatMetric(result.best.total, metric)}.`}
      </p>
      {error && <p className="form-error">{error}</p>}
    </div>
  );
}
