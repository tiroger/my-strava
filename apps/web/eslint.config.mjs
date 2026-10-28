import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { FlatCompat } from "@eslint/eslintrc";
import { defineConfig, globalIgnores } from "eslint/config";

// eslint-config-next 15 ships a legacy eslintrc config with no flat-config
// subpath exports, unlike 16. FlatCompat bridges it.
const compat = new FlatCompat({
  baseDirectory: dirname(fileURLToPath(import.meta.url)),
});

export default defineConfig([
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  globalIgnores([".next/**", "node_modules/**", ".npm/**", "next-env.d.ts", "public/maplibre-gl-*.mjs"]),
]);
