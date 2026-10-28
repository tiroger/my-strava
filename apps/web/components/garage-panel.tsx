"use client";

import { useMemo, useState } from "react";
import {
  calendarDate,
  formatCalendarDate,
  formatNumber,
  localIsoDate,
  type Activity,
  type Gear,
  type MaintenanceRecord,
} from "@/lib/activity-types";
import {
  BIKE_PHOTOS,
  bikeRecords,
  scheduleStatus,
  SERVICE_CATALOG,
} from "@/lib/maintenance";
import { ownerRequest } from "@/lib/use-admin";

type GearUsage = { yearMiles: number; lastUsed: string | null; activities: number };

function usageByGear(activities: Activity[], year: number): Map<string, GearUsage> {
  const usage = new Map<string, GearUsage>();
  for (const activity of activities) {
    if (!activity.gear_id) continue;
    const entry = usage.get(activity.gear_id) ?? { yearMiles: 0, lastUsed: null, activities: 0 };
    entry.activities += 1;
    if (calendarDate(activity.date)?.year === year) entry.yearMiles += activity.distance;
    if (!entry.lastUsed || activity.date > entry.lastUsed) entry.lastUsed = activity.date;
    usage.set(activity.gear_id, entry);
  }
  return usage;
}

export function GaragePanel({
  gear,
  activities,
  initialRecords,
  todayIso,
  canEdit,
}: {
  gear: Gear[];
  activities: Activity[];
  initialRecords: MaintenanceRecord[];
  todayIso: string;
  canEdit: boolean;
}) {
  const year = calendarDate(todayIso)?.year ?? new Date().getFullYear();
  const usage = useMemo(() => usageByGear(activities, year), [activities, year]);
  const bikes = gear.filter((item) => item.kind === "bike");
  const active = bikes.filter((bike) => !bike.retired);
  const retired = bikes.filter((bike) => bike.retired);
  const shoes = gear.filter((item) => item.kind === "shoe" && !item.retired);
  const [showRetired, setShowRetired] = useState(false);

  return (
    <section className="garage-layout">
      <div className="garage-gear">
        <article className="card garage-bike-card">
          <div className="card-heading">
            <div><div className="card-kicker">YOUR EQUIPMENT</div><h2>The garage</h2></div>
            <span className="period-pill">{active.length} ACTIVE</span>
          </div>
          <div className="bike-grid">
            {active.map((bike) => (
              <GearCard key={bike.id} item={bike} usage={usage.get(bike.id)} year={year} />
            ))}
            {!active.length && (
              <div className="empty-state">
                No bikes yet. They appear after the first Strava sync, including any added later.
              </div>
            )}
          </div>
          {retired.length > 0 && (
            <>
              <button className="text-button retired-toggle" type="button" onClick={() => setShowRetired((value) => !value)}>
                {showRetired ? "Hide" : "Show"} {retired.length} retired {retired.length === 1 ? "bike" : "bikes"}
              </button>
              {showRetired && (
                <div className="bike-grid">
                  {retired.map((bike) => (
                    <GearCard key={bike.id} item={bike} usage={usage.get(bike.id)} year={year} />
                  ))}
                </div>
              )}
            </>
          )}
        </article>
        {shoes.length > 0 && (
          <article className="card garage-bike-card">
            <div className="card-kicker">SHOES</div>
            <div className="shoe-list">
              {shoes.map((shoe) => (
                <span key={shoe.id}>
                  <strong>{shoe.name}</strong>
                  <small>{[shoe.brand, shoe.model].filter(Boolean).join(" ") || "Strava gear"}</small>
                  <b>{formatNumber(shoe.distance)} mi</b>
                </span>
              ))}
            </div>
          </article>
        )}
      </div>
      <MaintenanceCard bikes={active.length ? active : bikes} initialRecords={initialRecords} todayIso={todayIso} canEdit={canEdit} />
    </section>
  );
}

function GearCard({ item, usage, year }: { item: Gear; usage: GearUsage | undefined; year: number }) {
  const photo = BIKE_PHOTOS[item.id];
  return (
    <article className={`bike-card ${item.retired ? "retired" : ""}`}>
      {photo ? (
        // eslint-disable-next-line @next/next/no-img-element -- static, pre-sized photos
        <img className="bike-photo" src={photo} alt="" loading="lazy" width={900} height={675} />
      ) : (
        <div className="bike-photo placeholder" aria-hidden="true">⌁</div>
      )}
      <strong>{item.name}</strong>
      <small>{[item.brand, item.model].filter(Boolean).join(" ") || "Strava gear"}</small>
      <div className="bike-figures">
        <span><b>{formatNumber(item.distance)}</b><small>mi total</small></span>
        <span><b>{formatNumber(usage?.yearMiles ?? 0)}</b><small>mi in {year}</small></span>
      </div>
      {usage?.lastUsed && <em>Last ridden {formatCalendarDate(usage.lastUsed, { month: "short", day: "numeric", year: "numeric" })}</em>}
    </article>
  );
}

function MaintenanceCard({
  bikes,
  initialRecords,
  todayIso,
  canEdit,
}: {
  bikes: Gear[];
  initialRecords: MaintenanceRecord[];
  todayIso: string;
  canEdit: boolean;
}) {
  const [records, setRecords] = useState(initialRecords);
  const [bikeId, setBikeId] = useState(bikes[0]?.id ?? "");
  const [part, setPart] = useState("Chain");
  const [service, setService] = useState(SERVICE_CATALOG.Chain[0]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ error?: string; notice?: string }>({});
  const bike = bikes.find((item) => item.id === bikeId) ?? bikes[0];
  const history = bike ? bikeRecords(records, bike) : [];
  const schedules = bike ? scheduleStatus(records, bike, todayIso) : [];

  async function addRecord() {
    if (!bike) return;
    setBusy(true);
    setStatus({});
    try {
      const record = await ownerRequest<MaintenanceRecord>("/api/maintenance", "POST", {
        bikeId: bike.id,
        bike: bike.name,
        part,
        service,
        date: localIsoDate(),
        mileage: bike.distance,
      });
      setRecords((current) => [record, ...current]);
      setStatus({ notice: `${part} · ${service} logged for ${bike.name}.` });
    } catch (reason) {
      setStatus({ error: reason instanceof Error ? reason.message : "Could not save service" });
    } finally {
      setBusy(false);
    }
  }

  async function deleteRecord(id: string) {
    setBusy(true);
    setStatus({});
    try {
      await ownerRequest(`/api/maintenance?id=${encodeURIComponent(id)}`, "DELETE");
      setRecords((current) => current.filter((record) => record.id !== id));
    } catch (reason) {
      setStatus({ error: reason instanceof Error ? reason.message : "Could not delete record" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="card maintenance-card">
      <div className="card-heading">
        <div><div className="card-kicker">SERVICE</div><h2>Maintenance</h2></div>
        {!canEdit && <span className="private-pill">OWNER EDITS</span>}
      </div>

      <label className="bike-select">
        <span>Bike</span>
        <select value={bike?.id ?? ""} onChange={(event) => setBikeId(event.target.value)}>
          {bikes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </label>

      {bike && (
        <div className="schedule-list">
          <div className="personal-best-heading">NEXT DUE · {formatNumber(bike.distance)} MI ON THE ODOMETER</div>
          {schedules.length ? (
            schedules.map((entry) => {
              const overdue = entry.remaining < 0;
              const soon = !overdue && entry.remaining <= (entry.unit === "mi" ? 25 : 14);
              return (
                <div className={`schedule-row ${overdue ? "overdue" : soon ? "soon" : ""}`} key={entry.schedule.label}>
                  <span>
                    <strong>{entry.schedule.label}</strong>
                    <small>Last {formatCalendarDate(entry.last.date, { month: "short", day: "numeric", year: "numeric" })} · due {entry.dueLabel}</small>
                  </span>
                  <b>
                    {overdue
                      ? `${formatNumber(Math.abs(entry.remaining))} ${entry.unit} overdue`
                      : `in ${formatNumber(entry.remaining)} ${entry.unit}`}
                  </b>
                </div>
              );
            })
          ) : (
            <p className="goal-note">
              Log a chain or tire service to start tracking when the next one is due.
            </p>
          )}
        </div>
      )}

      {canEdit && bike && (
        <div className="service-form">
          <label>Part
            <select value={part} onChange={(event) => {
              setPart(event.target.value);
              setService(SERVICE_CATALOG[event.target.value][0]);
            }}>
              {Object.keys(SERVICE_CATALOG).map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label>Service
            <select value={service} onChange={(event) => setService(event.target.value)}>
              {SERVICE_CATALOG[part].map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <button className="primary-button" disabled={busy} onClick={() => void addRecord()} type="button">
            {busy ? "Saving…" : "＋ Log service"}
          </button>
        </div>
      )}
      {(status.error || status.notice) && (
        <p className={status.error ? "form-error" : "form-notice"} role="status">{status.error ?? status.notice}</p>
      )}

      <div className="maintenance-list">
        <div className="personal-best-heading">HISTORY</div>
        {history.slice(0, 12).map((record) => (
          <div className="maintenance-row" key={record.id}>
            <span className="service-icon">✓</span>
            <span><strong>{record.part} · {record.service}</strong><small>{formatCalendarDate(record.date, { month: "short", day: "numeric", year: "numeric" })}</small></span>
            <span className="maintenance-mileage">{formatNumber(record.mileage)} mi</span>
            {canEdit && <button aria-label={`Delete ${record.part} ${record.service} record`} className="delete-button" disabled={busy} onClick={() => void deleteRecord(record.id)} type="button">×</button>}
          </div>
        ))}
        {!history.length && <div className="empty-state">No service records for this bike yet.</div>}
      </div>
    </article>
  );
}
