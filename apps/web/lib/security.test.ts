import { describe, expect, it } from "vitest";
import { safeEqual, sameOrigin } from "./security";

const request = (origin?: string) =>
  new Request("https://dashboard.example/api/admin/session", {
    method: "POST",
    headers: origin ? { origin } : {},
  });

describe("request security helpers", () => {
  it("accepts only requests from the same host", () => {
    expect(sameOrigin(request("https://dashboard.example"))).toBe(true);
    expect(sameOrigin(request("https://attacker.example"))).toBe(false);
    expect(sameOrigin(request())).toBe(false);
  });

  it("rejects opaque origins instead of throwing", () => {
    expect(sameOrigin(request("null"))).toBe(false);
  });

  it("compares secrets of different lengths safely", () => {
    expect(safeEqual("secret", "secret")).toBe(true);
    expect(safeEqual("secret", "secret-but-longer")).toBe(false);
  });
});
