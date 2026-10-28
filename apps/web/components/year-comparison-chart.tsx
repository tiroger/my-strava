"use client";

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { compareByYear } from "@/lib/activity-types";
import type { Activity, ComparisonMetric } from "@/lib/activity-types";

const palette = ["#fc6737", "#58a6d2", "#b296e8", "#66bb8a", "#e7b554", "#db6d9d"];

function displayValue(value: number, metric: ComparisonMetric): string {
  if (metric === "distance") return `${Math.round(value).toLocaleString("en-US")} mi`;
  if (metric === "duration") return `${value.toFixed(1)} hrs`;
  return `${Math.round(value).toLocaleString("en-US")} sessions`;
}

export function YearComparisonChart({
  activities,
  highlightYear,
  metric,
  throughMonth,
  throughDay,
}: {
  activities: Activity[];
  highlightYear: number;
  metric: ComparisonMetric;
  throughMonth: number;
  throughDay: number;
}) {
  const comparison = compareByYear(
    activities,
    metric,
    throughMonth,
    throughDay,
  );
  const option: EChartsOption = {
    animationDuration: 500,
    aria: {
      enabled: true,
      decal: { show: true },
      label: {
        description: `Monthly ${metric} comparison for ${comparison.years.join(", ")}.`,
      },
    },
    color: palette,
    grid: { top: 28, left: 52, right: 18, bottom: 56 },
    legend: {
      type: "scroll",
      bottom: 5,
      textStyle: { color: "#a9b2b7", fontSize: 10 },
      pageTextStyle: { color: "#a9b2b7" },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: "#68747b" } },
      backgroundColor: "#20272b",
      borderColor: "#3d484e",
      textStyle: { color: "#edf0f1", fontSize: 11 },
      formatter: (items) => {
        if (!Array.isArray(items)) return "";
        const title = items[0]?.name ?? "";
        const lines = items
          .filter((item) => typeof item.value === "number")
          .map(
            (item) =>
              `${item.marker} ${item.seriesName}: ${displayValue(Number(item.value), metric)}`,
          );
        return [title, ...lines].join("<br/>");
      },
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: comparison.months,
      axisLine: { lineStyle: { color: "#384248" } },
      axisLabel: { color: "#849097", fontSize: 9 },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      min: 0,
      splitLine: { lineStyle: { color: "#242c30", type: "dashed" } },
      axisLabel: {
        color: "#849097",
        fontSize: 9,
        formatter: (value: number) => value.toLocaleString("en-US"),
      },
    },
    dataZoom: [
      { type: "inside", zoomOnMouseWheel: false, moveOnMouseMove: true },
      {
        type: "slider",
        height: 14,
        bottom: 32,
        borderColor: "#313a3f",
        backgroundColor: "#1a2024",
        fillerColor: "#fc67372b",
        handleStyle: { color: "#fc6737" },
        textStyle: { color: "#7f8a90", fontSize: 8 },
      },
    ],
    series: comparison.years.map((year, index) => ({
      name: String(year),
      type: "line",
      smooth: 0.25,
      connectNulls: false,
      showSymbol: false,
      symbolSize: 6,
      lineStyle: { width: year === highlightYear ? 3 : 1.8 },
      emphasis: { focus: "series", scale: 1.3 },
      areaStyle:
        year === highlightYear
          ? { opacity: 0.08 }
          : undefined,
      data: comparison.values[year],
      itemStyle: { color: palette[index % palette.length] },
    })),
  };

  if (!comparison.years.length) {
    return <div className="chart-empty">Add activities to see your year-over-year trend.</div>;
  }

  return (
    <div className="echart-frame">
      <ReactECharts
        option={option}
        style={{ height: 248, width: "100%" }}
        notMerge
        lazyUpdate
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
