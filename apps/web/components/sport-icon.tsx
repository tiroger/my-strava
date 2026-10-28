import { getActivityKind, type SportKind } from "@/lib/activity-types";

const PATHS: Record<SportKind, string> = {
  Ride: "M5 17a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm14 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM5 14l4-7h5l5 7M9 7l3 7h7M13 4h3",
  Run: "M14 4.5a1.5 1.5 0 1 0 0 .01M7 20l3-6 3 2v5M10 14l1-5 4 3h3M8 9l3-1",
  Pickleball: "M9 4a5 5 0 0 1 5 5c0 2-1 3.4-2.4 4.2L10 19H8l.6-6A5 5 0 0 1 9 4Zm9 11a2 2 0 1 0 0 .01",
  Walk: "M12 4.5a1.5 1.5 0 1 0 0 .01M10 21l1.5-6L14 17v4M11.5 15l.5-6 3 2.5M9 12l3-3",
  Swim: "M3 15c2 0 2-1.5 4.5-1.5S10 15 12 15s2-1.5 4.5-1.5S19 15 21 15M3 19c2 0 2-1.5 4.5-1.5S10 19 12 19s2-1.5 4.5-1.5S19 19 21 19M8 11l4-4 3 3M16 7.5a1.5 1.5 0 1 0 0 .01",
  Training: "M4 9v6M7 7v10M17 7v10M20 9v6M7 12h10",
  Other: "M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0",
};

/** Small line icon for an activity's sport, sized by the surrounding font. */
export function SportIcon({ type, kind }: { type?: string; kind?: SportKind }) {
  const sport = kind ?? getActivityKind(type ?? "");
  return (
    <svg
      className="sport-icon"
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={PATHS[sport]} />
    </svg>
  );
}

export const SPORT_TONES: Record<SportKind, string> = {
  Ride: "tone-orange",
  Run: "tone-blue",
  Pickleball: "tone-green",
  Walk: "tone-gold",
  Swim: "tone-blue",
  Training: "tone-violet",
  Other: "tone-muted",
};
