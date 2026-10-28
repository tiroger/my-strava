import "server-only";

import { GetObjectCommand, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { parse } from "csv-parse/sync";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type {
  Activity,
  Bike,
  MaintenanceRecord,
} from "@/lib/activity-types";

const s3 = new S3Client({});

// Route polylines stay on the server and are fetched per activity, which
// keeps the dashboard payload small.
export type ActivityRecord = Omit<Activity, "hasRoute"> & {
  polyline: string | null;
};

type StoredArray<T> = { etag: string; value: T[] };

export class MaintenanceConflictError extends Error {
  constructor() {
    super("Maintenance records changed during the update");
  }
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function objectRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} entry must be an object`);
  }
  return value as Record<string, unknown>;
}

function textOrFallback(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const text = value.trim();
    if (text && !["nan", "none", "null"].includes(text.toLowerCase())) {
      return text;
    }
  }
  return "Activity";
}

function mapValue(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0) return null;
  const match = value.match(/summary_polyline['"]?\s*:\s*['"]([^'"]*)/);
  return match?.[1] || null;
}

export function normalizeDate(value: unknown): string {
  const text = String(value ?? "").trim();
  // The legacy CSV snapshot stores dates as MM-DD-YYYY.
  const legacy = /^(\d{2})-(\d{2})-(\d{4})/.exec(text);
  if (legacy) return `${legacy[3]}-${legacy[1]}-${legacy[2]}`;
  // Strava's start_date_local is wall-clock time despite its trailing "Z".
  return text.replace(/(Z|[+-]\d{2}:?\d{2})$/, "");
}

export function normalizeActivity(row: Record<string, unknown>): ActivityRecord {
  return {
    id: String(row.id ?? ""),
    name: String(row.name ?? "Untitled activity"),
    type: textOrFallback(row.sport_type, row.type),
    date: normalizeDate(row.start_date_local ?? row.date),
    distance: numberOrNull(row.distance) ?? 0,
    moving_time: numberOrNull(row.moving_time) ?? 0,
    total_elevation_gain: numberOrNull(row.total_elevation_gain) ?? 0,
    average_speed: numberOrNull(row.average_speed) ?? 0,
    average_heartrate: numberOrNull(row.average_heartrate),
    weighted_average_watts: numberOrNull(row.weighted_average_watts),
    kudos_count: numberOrNull(row.kudos_count) ?? 0,
    polyline:
      typeof row.map === "string"
        ? mapValue(row.map)
        : typeof row.map === "object" && row.map !== null
          ? String(
              (row.map as { summary_polyline?: string }).summary_polyline ?? "",
            ) || null
          : null,
  };
}

function normalizeBike(row: Record<string, unknown>): Bike {
  return {
    id: textOrFallback(row.id, ""),
    name: textOrFallback(row.name, row.nickname, "Bike"),
    brand: textOrFallback(row.brand, row.brand_name, ""),
    model: textOrFallback(row.model, row.model_name, ""),
    distance: numberOrNull(row.distance) ?? 0,
    retired: row.retired === true,
  };
}

function normalizeMaintenance(row: Record<string, unknown>): MaintenanceRecord {
  return {
    id: String(row.id ?? ""),
    bike: String(row.bike ?? ""),
    part: String(row.part ?? ""),
    service: String(row.service ?? ""),
    date: String(row.date ?? ""),
    mileage: numberOrNull(row.mileage) ?? 0,
  };
}

function dataBucket(): string | undefined {
  const bucket = process.env.APP_DATA_BUCKET;
  if (!bucket && process.env.NODE_ENV === "production") {
    throw new Error("APP_DATA_BUCKET is not configured");
  }
  return bucket;
}

const objectKeys = {
  activities: () => process.env.ACTIVITIES_S3_KEY ?? "activities.json",
  bikes: () => process.env.BIKES_S3_KEY ?? "bikes.json",
  maintenance: () => process.env.MAINTENANCE_S3_KEY ?? "maintenance.json",
};

function httpStatus(error: unknown): number | undefined {
  return (error as { $metadata?: { httpStatusCode?: number } } | null)?.$metadata
    ?.httpStatusCode;
}

const objectCache = new Map<string, StoredArray<unknown>>();

// Reads a JSON array from S3, reusing the parsed value while its ETag is
// unchanged. A missing object (before the first sync) reads as empty.
async function readArray<T>(
  bucket: string,
  key: string,
  label: string,
  normalize: (row: Record<string, unknown>) => T,
): Promise<StoredArray<T>> {
  const cached = objectCache.get(key) as StoredArray<T> | undefined;
  try {
    const response = await s3.send(
      new GetObjectCommand({ Bucket: bucket, Key: key, IfNoneMatch: cached?.etag }),
    );
    if (!response.Body) throw new Error(`${label} object has an empty body`);
    const rows = JSON.parse(await response.Body.transformToString()) as unknown;
    if (!Array.isArray(rows)) {
      throw new Error(`${label} object must contain a JSON array`);
    }
    const stored = {
      etag: response.ETag ?? "",
      value: rows.map((row) => normalize(objectRecord(row, label))),
    };
    objectCache.set(key, stored);
    return stored;
  } catch (error) {
    if (cached && httpStatus(error) === 304) return cached;
    if (httpStatus(error) === 404) {
      objectCache.delete(key);
      return { etag: "", value: [] };
    }
    throw error;
  }
}

async function readLocalCsv(file: string): Promise<Record<string, unknown>[]> {
  const contents = await readFile(path.resolve(process.cwd(), "../../data", file), "utf8");
  const rows = parse(contents, { columns: true, skip_empty_lines: true }) as
    | Record<string, unknown>[]
    | undefined;
  return rows ?? [];
}

async function loadActivityRecords(): Promise<StoredArray<ActivityRecord>> {
  const bucket = dataBucket();
  if (bucket) {
    return readArray(bucket, objectKeys.activities(), "Activity", normalizeActivity);
  }
  const rows = await readLocalCsv("processed_data.csv");
  return { etag: "local", value: rows.map(normalizeActivity) };
}

async function loadBikes(): Promise<StoredArray<Bike>> {
  const bucket = dataBucket();
  if (bucket) return readArray(bucket, objectKeys.bikes(), "Bike", normalizeBike);
  const rows = await readLocalCsv("bike_data.csv");
  return {
    etag: "local",
    value: rows.map((row) => ({
      id: String(row.id ?? ""),
      name: String(row.nickname || row.name || row.id || "Bike"),
      brand: String(row.brand_name ?? ""),
      model: String(row.model_name ?? ""),
      distance: numberOrNull(row.converted_distance) ?? 0,
      retired: row.retired === true || row.retired === "True",
    })),
  };
}

async function loadMaintenance(): Promise<StoredArray<MaintenanceRecord>> {
  const bucket = dataBucket();
  if (!bucket) return { etag: "local", value: [] };
  return readArray(bucket, objectKeys.maintenance(), "Maintenance", normalizeMaintenance);
}

export async function loadDashboardData(): Promise<{
  activities: Activity[];
  bikes: Bike[];
  maintenance: MaintenanceRecord[];
  version: string;
}> {
  const [activities, bikes, maintenance] = await Promise.all([
    loadActivityRecords(),
    loadBikes(),
    loadMaintenance(),
  ]);
  return {
    activities: activities.value.map(({ polyline, ...activity }) => ({
      ...activity,
      hasRoute: !!polyline,
    })),
    bikes: bikes.value,
    maintenance: maintenance.value,
    version: [activities.etag, bikes.etag, maintenance.etag].join("|"),
  };
}

// Cheap when nothing changed: cached objects are revalidated with ETags.
export async function loadDataVersion(): Promise<string> {
  const objects = await Promise.all([
    loadActivityRecords(),
    loadBikes(),
    loadMaintenance(),
  ]);
  return objects.map((object) => object.etag).join("|");
}

export async function loadActivityRoute(id: string): Promise<string | null> {
  const { value } = await loadActivityRecords();
  return value.find((activity) => activity.id === id)?.polyline ?? null;
}

export async function loadMaintenanceRecords(): Promise<MaintenanceRecord[]> {
  return (await loadMaintenance()).value;
}

// Writes are conditional on the version that was read, so two concurrent
// edits cannot silently overwrite each other.
export async function updateMaintenanceRecords(
  update: (records: MaintenanceRecord[]) => MaintenanceRecord[],
): Promise<void> {
  const bucket = process.env.APP_DATA_BUCKET;
  if (!bucket) throw new Error("APP_DATA_BUCKET is not configured");
  const key = objectKeys.maintenance();
  const current = await readArray(bucket, key, "Maintenance", normalizeMaintenance);
  try {
    await s3.send(
      new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: JSON.stringify(update(current.value)),
        ContentType: "application/json",
        CacheControl: "no-cache",
        ...(current.etag ? { IfMatch: current.etag } : { IfNoneMatch: "*" }),
      }),
    );
  } catch (error) {
    if (httpStatus(error) === 412 || httpStatus(error) === 409) {
      objectCache.delete(key);
      throw new MaintenanceConflictError();
    }
    throw error;
  }
}
