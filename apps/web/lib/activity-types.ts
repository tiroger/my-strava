export type Activity = {
  id: string;
  name: string;
  type: string;
  date: string;
  distance: number;
  moving_time: number;
  total_elevation_gain: number;
  average_speed: number;
  average_heartrate: number | null;
  weighted_average_watts: number | null;
  kudos_count: number;
  hasRoute: boolean;
};

export type Bike = {
  id: string;
  name: string;
  brand: string;
  model: string;
  distance: number;
  retired: boolean;
};

export type MaintenanceRecord = {
  id: string;
  bike: string;
  part: string;
  service: string;
  date: string;
  mileage: number;
};

export type ComparisonMetric = "distance" | "duration" | "sessions";

export function getActivityKind(type: string): string {
  const normalized = type.toLowerCase().replace(/[^a-z]/g, "");
  if (normalized.includes("pickleball")) return "Pickleball";
  if (
    normalized.includes("ride") ||
    normalized.includes("cycling") ||
    normalized.includes("bike")
  ) {
    return "Ride";
  }
  if (normalized.includes("run")) return "Run";
  if (normalized.includes("walk")) return "Walk";
  if (normalized.includes("swim")) return "Swim";
  return type || "Activity";
}

export type CalendarDate = { year: number; month: number; day: number };

// Activity and maintenance dates are wall-clock dates ("YYYY-MM-DD..."), so
// they are read from the string rather than through the viewer's timezone.
export function calendarDate(date: string): CalendarDate | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(date);
  if (!match) return undefined;
  return {
    year: Number(match[1]),
    month: Number(match[2]) - 1,
    day: Number(match[3]),
  };
}

export function formatCalendarDate(
  date: string,
  options: Intl.DateTimeFormatOptions = {},
): string {
  const parts = calendarDate(date);
  if (!parts) return "—";
  return new Intl.DateTimeFormat("en-US", { ...options, timeZone: "UTC" }).format(
    Date.UTC(parts.year, parts.month, parts.day),
  );
}

export function localIsoDate(date: Date = new Date()): string {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

export function activityYear(activity: Activity): number | undefined {
  return calendarDate(activity.date)?.year;
}

export function activityMonth(activity: Activity): number | undefined {
  return calendarDate(activity.date)?.month;
}

export function activityDay(activity: Activity): number | undefined {
  return calendarDate(activity.date)?.day;
}

export function comparisonValue(
  activity: Activity,
  metric: ComparisonMetric,
): number {
  if (metric === "distance") return activity.distance;
  if (metric === "duration") return activity.moving_time;
  return 1;
}

export function compareByYear(
  activities: Activity[],
  metric: ComparisonMetric,
  monthsThrough: number,
  dayThrough: number,
): { years: number[]; months: string[]; values: Record<number, (number | null)[]> } {
  const years = Array.from(
    new Set(
      activities
        .map(activityYear)
        .filter((year): year is number => year !== undefined),
    ),
  ).sort((a, b) => a - b);
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const values: Record<number, (number | null)[]> = {};
  for (const year of years) {
    values[year] = months.map((_, month) => {
      if (month > monthsThrough) return null;
      return activities
        .filter(
          (activity) =>
            activityYear(activity) === year &&
            activityMonth(activity) === month &&
            (month !== monthsThrough || (activityDay(activity) ?? 0) <= dayThrough),
        )
        .reduce((sum, activity) => sum + comparisonValue(activity, metric), 0);
    });
  }
  return { years, months, values };
}
