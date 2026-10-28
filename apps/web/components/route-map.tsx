"use client";

import { LngLatBounds, Map as MapLibreMap, NavigationControl, setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef } from "react";
import { decodePolyline } from "@/lib/polyline";

/** CARTO's dark basemap: MapLibre-native vector tiles, no API key required. */
const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// MapLibre parses tiles in a Web Worker that the bundler cannot resolve from
// the package, so it is served from /public (see scripts-sync-maplibre.mjs).
if (typeof window !== "undefined") setWorkerUrl("/maplibre-gl-worker.mjs");

export type MapRoute = { id: string; polyline: string };

type FeatureCollection = {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: { id: string };
    geometry: { type: "LineString"; coordinates: number[][] };
  }>;
};

const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/**
 * Bounds covering the routes, or for a heatmap only those starting within
 * ~150 km of the median start, so a few rides abroad don't zoom out to the world.
 */
function toGeoJson(
  routes: MapRoute[],
  focusOnHome: boolean,
): { data: FeatureCollection; bounds: LngLatBounds | null } {
  let bounds: LngLatBounds | null = null;
  const lines = routes.map((route) => ({
    route,
    coordinates: decodePolyline(route.polyline).map(([lat, lng]) => [lng, lat]),
  }));
  const starts = lines.filter((line) => line.coordinates.length).map((line) => line.coordinates[0]);
  const center = focusOnHome && starts.length > 2
    ? [median(starts.map(([lng]) => lng)), median(starts.map(([, lat]) => lat))]
    : null;
  const features = lines.flatMap(({ route, coordinates }) => {
    if (coordinates.length < 2) return [];
    const [lng0, lat0] = coordinates[0];
    const nearby = !center || (Math.abs(lng0 - center[0]) < 1.8 && Math.abs(lat0 - center[1]) < 1.4);
    if (nearby) {
      for (const [lng, lat] of coordinates) {
        bounds = bounds ? bounds.extend([lng, lat]) : new LngLatBounds([lng, lat], [lng, lat]);
      }
    }
    return [{
      type: "Feature" as const,
      properties: { id: route.id },
      geometry: { type: "LineString" as const, coordinates },
    }];
  });
  return { data: { type: "FeatureCollection", features }, bounds };
}

/**
 * One route drawn boldly, or many drawn faintly so that overlapping rides
 * build into a heatmap. Route ends arrive already trimmed for privacy.
 */
export default function RouteMap({
  routes,
  variant,
  onSelect,
  height = 420,
}: {
  routes: MapRoute[];
  variant: "route" | "heat";
  onSelect?: (id: string) => void;
  height?: number;
}) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const onSelectRef = useRef(onSelect);
  const geo = useMemo(() => toGeoJson(routes, variant === "heat"), [routes, variant]);
  const geoRef = useRef(geo);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!container.current) return;
    const instance = new MapLibreMap({
      container: container.current,
      style: STYLE,
      center: [-122.27, 37.8],
      zoom: 9,
      attributionControl: { compact: true },
    });
    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.current = instance;
    // The container is often measured before layout settles (and changes with
    // the viewport), so keep the canvas matched to it.
    const resize = new ResizeObserver(() => instance.resize());
    resize.observe(container.current);
    const heat = variant === "heat";

    instance.on("load", () => {
      instance.addSource("routes", { type: "geojson", data: geoRef.current.data });
      if (!heat) {
        instance.addLayer({
          id: "route-casing",
          type: "line",
          source: "routes",
          layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-color": "#fc6737", "line-opacity": 0.25, "line-width": 9 },
        });
      }
      instance.addLayer({
        id: "routes",
        type: "line",
        source: "routes",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: heat
          ? {
              "line-color": "#ff7a45",
              "line-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0.16, 13, 0.3],
              "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1.1, 14, 2.6],
            }
          : { "line-color": "#fc6737", "line-width": 3.4 },
      });
      if (heat) {
        // A wider invisible layer makes thin lines easy to click.
        instance.addLayer({
          id: "routes-hit",
          type: "line",
          source: "routes",
          paint: { "line-color": "#000", "line-opacity": 0, "line-width": 10 },
        });
        instance.on("click", "routes-hit", (event) => {
          const id = event.features?.[0]?.properties?.id;
          if (typeof id === "string") onSelectRef.current?.(id);
        });
        instance.on("mouseenter", "routes-hit", () => (instance.getCanvas().style.cursor = "pointer"));
        instance.on("mouseleave", "routes-hit", () => (instance.getCanvas().style.cursor = ""));
      }
      if (geoRef.current.bounds) {
        instance.fitBounds(geoRef.current.bounds, { padding: 36, duration: 0, maxZoom: 14 });
      }
    });

    return () => {
      resize.disconnect();
      instance.remove();
      map.current = null;
    };
  }, [variant]);

  useEffect(() => {
    geoRef.current = geo;
    const instance = map.current;
    const source = instance?.getSource("routes");
    if (!instance || !source || source.type !== "geojson") return;
    (source as unknown as { setData: (data: FeatureCollection) => void }).setData(geo.data);
    if (geo.bounds) instance.fitBounds(geo.bounds, { padding: 36, duration: 400, maxZoom: 14 });
  }, [geo]);

  return <div ref={container} className="route-map" style={{ height }} />;
}
