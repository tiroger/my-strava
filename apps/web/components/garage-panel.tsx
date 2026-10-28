"use client";

import { useState } from "react";
import {
  formatCalendarDate,
  localIsoDate,
  type Bike,
  type MaintenanceRecord,
} from "@/lib/activity-types";

const serviceOptions: Record<string, string[]> = {
  Chain: ["Clean", "Lube", "Replace", "Hot wax"],
  Cassette: ["Replace", "Clean"],
  Tires: ["Replace", "Sealant top-up"],
  "Brake pads": ["Replace"],
  Wheels: ["True", "Service"],
  Other: ["Inspect", "Service", "Replace"],
};

export function GaragePanel({
  bikes,
  initialAuthenticated,
  initialRecords,
}: {
  bikes: Bike[];
  initialAuthenticated: boolean;
  initialRecords: MaintenanceRecord[];
}) {
  const [records, setRecords] = useState(initialRecords);
  const [authenticated, setAuthenticated] = useState(initialAuthenticated);
  const [passphrase, setPassphrase] = useState("");
  const [selectedBike, setSelectedBike] = useState(bikes.find((bike) => !bike.retired)?.name ?? "");
  const [part, setPart] = useState("Chain");
  const [service, setService] = useState(serviceOptions.Chain[0]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  async function unlock() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/admin/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passphrase }),
      });
      if (!response.ok) {
        const result = (await response.json()) as { error?: string };
        throw new Error(result.error ?? "Could not unlock admin mode");
      }
      setAuthenticated(true);
      setPassphrase("");
      setNotice("Admin mode enabled for this browser.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not unlock");
    } finally {
      setBusy(false);
    }
  }

  async function addRecord() {
    const bike = bikes.find((item) => item.name === selectedBike);
    if (!bike) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/maintenance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bike: bike.name,
          part,
          service,
          date: localIsoDate(),
          mileage: bike.distance,
        }),
      });
      const result = (await response.json()) as MaintenanceRecord | { error?: string };
      if (!response.ok || !("id" in result)) {
        throw new Error(("error" in result && result.error) || "Could not save service");
      }
      setRecords((current) => [result, ...current]);
      setNotice(`${service} logged for ${bike.name}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save service");
    } finally {
      setBusy(false);
    }
  }

  async function deleteRecord(id: string) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/maintenance?id=${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const result = (await response.json()) as { error?: string };
        throw new Error(result.error ?? "Could not delete service record");
      }
      setRecords((current) => current.filter((record) => record.id !== id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete record");
    } finally {
      setBusy(false);
    }
  }

  async function lock() {
    setError("");
    try {
      const response = await fetch("/api/admin/session", { method: "DELETE" });
      if (!response.ok) throw new Error("Could not lock admin mode");
      setAuthenticated(false);
      setNotice("Admin mode disabled.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not lock admin mode");
    }
  }

  return (
    <section className="garage-layout">
      <article className="card garage-bike-card">
        <div className="card-heading">
          <div><div className="card-kicker">YOUR EQUIPMENT</div><h2>The garage</h2></div>
          <span className="period-pill">{bikes.filter((bike) => !bike.retired).length} ACTIVE</span>
        </div>
        <div className="bike-grid">
          {bikes.filter((bike) => !bike.retired).map((bike) => (
            <article className="bike-card" key={bike.id}>
              <div className="bike-symbol">⌁</div>
              <strong>{bike.name}</strong>
              <small>{[bike.brand, bike.model].filter(Boolean).join(" ") || "Strava gear"}</small>
              <span>{Math.round(bike.distance).toLocaleString("en-US")} <small>mi tracked</small></span>
            </article>
          ))}
          {!bikes.some((bike) => !bike.retired) && (
            <div className="empty-state">No gear is available yet. Run the initial Strava sync to load your bikes.</div>
          )}
        </div>
      </article>
      <article className="card maintenance-card">
        <div className="card-heading">
          <div><div className="card-kicker">SERVICE HISTORY</div><h2>Maintenance log</h2></div>
          {authenticated ? (
            <button className="text-button" type="button" onClick={lock}>Lock admin mode</button>
          ) : (
            <span className="private-pill">PRIVATE EDITS</span>
          )}
        </div>
        {!authenticated ? (
          <form
            className="admin-unlock"
            onSubmit={(event) => {
              event.preventDefault();
              void unlock();
            }}
          >
            <label htmlFor="admin-passphrase">Owner passphrase</label>
            <div className="admin-input-row">
              <input
                autoComplete="current-password"
                id="admin-passphrase"
                onChange={(event) => setPassphrase(event.target.value)}
                type="password"
                value={passphrase}
              />
              <button className="primary-button" disabled={busy || !passphrase} type="submit">
                {busy ? "Checking…" : "Unlock"}
              </button>
            </div>
            <p>Dashboard data is public. Only the owner can change service records.</p>
          </form>
        ) : (
          <div className="service-form">
            <label>Bike
              <select value={selectedBike} onChange={(event) => setSelectedBike(event.target.value)}>
                {bikes.filter((bike) => !bike.retired).map((bike) => <option key={bike.id}>{bike.name}</option>)}
              </select>
            </label>
            <label>Part
              <select value={part} onChange={(event) => {
                setPart(event.target.value);
                setService(serviceOptions[event.target.value][0]);
              }}>
                {Object.keys(serviceOptions).map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
            <label>Service
              <select value={service} onChange={(event) => setService(event.target.value)}>
                {serviceOptions[part].map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
            <button className="primary-button" disabled={busy || !selectedBike} onClick={() => void addRecord()} type="button">
              {busy ? "Saving…" : "＋ Log service"}
            </button>
          </div>
        )}
        {(error || notice) && <p className={error ? "form-error" : "form-notice"} role="status">{error || notice}</p>}
        <div className="maintenance-list">
          {records.slice(0, 8).map((record) => (
            <div className="maintenance-row" key={record.id}>
              <span className="service-icon">✓</span>
              <span><strong>{record.service} · {record.part}</strong><small>{record.bike} · {formatCalendarDate(record.date)}</small></span>
              <span className="maintenance-mileage">{Math.round(record.mileage).toLocaleString("en-US")} mi</span>
              {authenticated && <button aria-label={`Delete ${record.service} record`} className="delete-button" disabled={busy} onClick={() => void deleteRecord(record.id)} type="button">×</button>}
            </div>
          ))}
          {!records.length && <div className="empty-state">No service records yet.</div>}
        </div>
      </article>
    </section>
  );
}
