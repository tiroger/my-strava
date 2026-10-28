# Agent notes

- **Next.js is pinned to 15.** AWS Amplify Hosting compute supports Next.js 12 through 15; do not upgrade to 16 until Amplify's supported matrix includes it (see `amplify.yml`). ESLint therefore uses `eslint-config-next` 15 through `FlatCompat`.
- Server runtime configuration reaches the SSR bundle only through `.env.production`, written in `amplify.yml`; add any new runtime variable to that list and to the Terraform `web` module.
- Activity dates are wall-clock strings. Read them with the helpers in `lib/activity-types.ts`, never with `new Date(...)` getters, which shift them into the viewer's timezone.
- Route coordinates near either end are private: every API that returns them goes through `lib/polyline.ts` (`publicPolyline`, `hideRouteEnds`).
- `npm test` runs under `TZ=America/Los_Angeles` on purpose.
