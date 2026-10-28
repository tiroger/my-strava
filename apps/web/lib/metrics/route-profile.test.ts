import { describe, expect, it } from "vitest";
import { findClimbs, gradeColor, GRADE_SCALE, smoothedGrades, summarizeRange } from "./route-profile";

// A mile of flat, two miles climbing 5% (528 ft), then a mile descending 5%.
const distance = Array.from({ length: 401 }, (_, index) => index * 0.01);
const altitude = distance.map((miles) =>
  miles <= 1 ? 100 : miles <= 3 ? 100 + (miles - 1) * 264 : 628 - (miles - 3) * 264,
);

describe("route profile", () => {
  const grades = smoothedGrades(distance, altitude);

  it("measures grade across a distance window", () => {
    expect(grades[50]).toBeCloseTo(0);
    expect(grades[200]).toBeCloseTo(5);
    expect(grades[350]).toBeCloseTo(-5);
  });

  it("colours gradients from the VeloViewer-style scale", () => {
    expect(gradeColor(0)).toBe(GRADE_SCALE[3].color);
    expect(gradeColor(6)).toBe(GRADE_SCALE[6].color);
    expect(gradeColor(-10)).toBe(GRADE_SCALE[0].color);
  });

  it("finds the sustained climb and categorizes it", () => {
    const climbs = findClimbs(distance, altitude, grades);
    expect(climbs).toHaveLength(1);
    expect(climbs[0].start).toBeCloseTo(1);
    expect(climbs[0].length).toBeCloseTo(2);
    expect(climbs[0].gain).toBeCloseTo(528);
    expect(climbs[0].averageGrade).toBeCloseTo(5);
    // 3219 m x 5% = 16,093: category 3.
    expect(climbs[0].category).toBe("3");
  });

  it("summarizes a selected range like VeloViewer", () => {
    const range = summarizeRange(distance, altitude, 300, 100);
    expect(range.distance).toBeCloseTo(2);
    expect(range.gain).toBeCloseTo(528);
    expect(range.grade).toBeCloseTo(5);
  });
});
