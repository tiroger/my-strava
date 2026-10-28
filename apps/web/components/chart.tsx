"use client";

import ReactEChartsCore from "echarts-for-react/lib/core";
import { BarChart, HeatmapChart, LineChart, PieChart, RadarChart } from "echarts/charts";
import {
  AriaComponent,
  CalendarComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  RadarComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import type { EChartsOption } from "echarts";
import { CanvasRenderer } from "echarts/renderers";

// Only the chart types the dashboard draws; the full ECharts bundle is several
// times larger.
echarts.use([
  AriaComponent,
  BarChart,
  CalendarComponent,
  CanvasRenderer,
  GridComponent,
  HeatmapChart,
  LegendComponent,
  LineChart,
  MarkLineComponent,
  PieChart,
  RadarChart,
  RadarComponent,
  TooltipComponent,
  VisualMapComponent,
]);

export const CHART_TEXT = { color: "#849097", fontSize: 9 };
export const CHART_TOOLTIP = {
  backgroundColor: "#20272b",
  borderColor: "#3d484e",
  textStyle: { color: "#edf0f1", fontSize: 11 },
};
export const CHART_COLORS = {
  orange: "#fc6737",
  blue: "#58a6d2",
  violet: "#b296e8",
  green: "#66bb8a",
  gold: "#e7b554",
  pink: "#db6d9d",
  muted: "#5b666c",
  grid: "#242c30",
};

export function Chart({
  option,
  height,
  label,
}: {
  option: EChartsOption;
  height: number;
  label: string;
}) {
  return (
    <div className="echart-frame" role="img" aria-label={label}>
      <ReactEChartsCore
        echarts={echarts}
        option={{ aria: { enabled: true, label: { description: label } }, ...option }}
        style={{ height, width: "100%" }}
        notMerge
        lazyUpdate
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
