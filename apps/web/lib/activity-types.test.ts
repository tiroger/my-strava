import { describe, expect, it } from "vitest";
import {
  activityDay,
  activityMonth,
  activityYear,
  compareByYear,
  formatCalendarDate,
  getActivityKind,
  localIsoDate,
  type Activity,
} from "./activity-types";

const activity = (
  id: string,
  date: string,
  distance: number,
): Activity => ({
  id,
  name: "Activity",
  type: "Pickleball",
  date,
  distance,
  moving_time: 1,
  total_elevation_gain: 0,
  average_speed: 0,
  average_heartrate: null,
  weighted_average_watts: null,
  kudos_count: 0,
  hasRoute: false,
});

describe("activity types and year comparison", () => {
  it("recognizes Pickleball even when Strava labels the base activity Workout", () => {
    expect(getActivityKind("Pickleball")).toBe("Pickleball");
    expect(getActivityKind("VirtualRide")).toBe("Ride");
    expect(getActivityKind("Run")).toBe("Run");
  });

  it("compares the same month and day across years", () => {
    const result = compareByYear(
      [
        activity("1", "2024-10-25T10:00:00", 3),
        activity("2", "2024-10-27T10:00:00", 8),
        activity("3", "2025-10-24T10:00:00", 5),
      ],
      "sessions",
      9,
      26,
    );
    expect(result.values[2024]?.[9]).toBe(1);
    expect(result.values[2025]?.[9]).toBe(1);
    expect(result.values[2024]?.[10]).toBeNull();
  });

  it("reads wall-clock dates without shifting them into the viewer's timezone", () => {
    // The test script runs in America/Los_Angeles, where parsing this as UTC
    // would move the ride to December 31 of the previous year.
    const ride = activity("4", "2026-01-01T06:30:00", 20);
    expect(activityYear(ride)).toBe(2026);
    expect(activityMonth(ride)).toBe(0);
    expect(activityDay(ride)).toBe(1);
    expect(formatCalendarDate(ride.date)).toBe("1/1/2026");
    expect(formatCalendarDate("2026-10-26", { month: "short", day: "numeric" })).toBe(
      "Oct 26",
    );
    expect(formatCalendarDate("not a date")).toBe("—");
  });

  it("formats today's local date for new records", () => {
    expect(localIsoDate(new Date(2026, 9, 26, 23, 30))).toBe("2026-10-26");
  });
});
