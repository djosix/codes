<template lang="pug">
  div
    .inline.fields
      label Variables
      .field(v-for='fuzVar in Object.values(fuzVars)')
        .ui.radio.checkbox
          input(
            :id='`fs_var_plot_${fuzVar.symbol}`'
            name='fs_var_plot'
            type='radio'
            @click='plot(fuzVar.symbol)')
          label(:for='`fs_var_plot_${fuzVar.symbol}`') {{ fuzVar.name }}
      i (drag on the plot to adjust)

    canvas(width='516' height='185' ref='canvas')
</template>


<script lang="ts">
import { getDefaultVars } from "/lib/mods/fs/defaults";
import { FuzzyVar } from "~lib/mods/fs/fuzzy";
import { Vec, helpers } from "/lib/math";

import _Chart from "chart.js";
const Chart = <any>_Chart;

import $ from "jquery";

/**
 * Get default Chart.js settings
 */
function getChartConfig() {
  return {
    type: "line",
    data: {
      datasets: []
    },
    options: {
      elements: {
        point: {
          radius: 0,
          hitRadius: 0,
          hoverRadius: 0
        }
      },
      tooltips: {
        enabled: false
      }
    }
  };
}

/**
 * Handle canvas mouse drag tracking
 * @param that vue component instance (this)
 */
function patchForMouseDrag(that) {
  // internal state
  let active = false;

  // internal state
  let track = {
    position: null,
    vector: null,
    fuzSet: null,
    mean: null,
    std: null
  };

  let canvas = that.$refs.canvas;

  function clientXToChartX(chart, clientX) {
    let labels = chart.config.data.labels;
    let chartLeft = labels[0];
    let chartWidth = labels[labels.length - 1] - chartLeft;

    let meta = chart.getDatasetMeta(0);
    let chartLeftPx = meta.data[0]._model.x;
    let chartWidthPx = meta.data[meta.data.length - 1]._model.x - chartLeftPx;

    return (chartWidth * (clientX - chartLeftPx)) / chartWidthPx + chartLeft;
  }

  function start(e) {
    active = true;

    let position = Chart.helpers.getRelativePosition(e, that.chart);
    track.position = Vec.fromObject(position);
    let chartX = clientXToChartX(that.chart, position.x);

    let minDist = Infinity;
    let activePlot: VarPlot = that.varPlots[that.activeSym];

    for (let fuzSet of activePlot.fuzVar.sets) {
      let dist = Math.abs(fuzSet.mean - chartX);
      if (dist < minDist) {
        track.fuzSet = fuzSet;
        minDist = dist;
      }
    }

    // Save original fuzzy variable properties
    track.mean = track.fuzSet.mean;
    track.std = track.fuzSet.std;
  }

  function move(e) {
    if (!active) {
      return;
    }

    let activePlot: VarPlot = that.varPlots[that.activeSym];
    let fuzVar = activePlot.fuzVar;

    let position = Chart.helpers.getRelativePosition(e, that.chart);
    track.vector = Vec.fromObject(position).subtract(track.position);

    let mean = track.mean + track.vector.x * fuzVar.range.size * 0.002;
    let std = track.std - track.vector.y * fuzVar.range.size * 0.0015;

    track.fuzSet.mean = helpers.clip(mean, fuzVar.range.min, fuzVar.range.max);
    track.fuzSet.std = helpers.clip(std, 1, fuzVar.range.size / 2);

    activePlot.update();
    that.chart.update();
  }

  function stop(e) {
    active = false;
  }

  canvas.onmousedown = start;
  canvas.onmousemove = move;
  canvas.onmouseup = stop;
  canvas.onmouseleave = stop;
  canvas.onmouseup = stop;
  canvas.onmouseout = stop;
}


class VarPlot {
  private readonly colors = {
    X: ["rgb(255, 160, 160)", "rgb(255, 120, 120)", "rgb(255, 80, 80)"],
    Y: ["rgb(160, 200, 160)", "rgb(80, 200, 80)", "rgb(20, 200, 20)"],
    Z: ["rgb(160, 160, 255)", "rgb(120, 120, 255)", "rgb(80, 80, 255)"]
  };

  fuzVar: FuzzyVar;
  labels: number[];
  datasets: any[];

  constructor(fuzVar: FuzzyVar) {
    this.fuzVar = fuzVar;
    this.labels = helpers.range(fuzVar.range.min, fuzVar.range.max, 1);
    this.datasets = [];

    for (let i = 0; i < fuzVar.sets.length; i++) {
      let label = fuzVar.sets[i].name;
      let borderColors = this.colors[fuzVar.symbol][i];
      let backgroundColor = borderColors.replace(")", ", 0.3)");

      this.datasets.push({
        label, data: [],
        borderColors, backgroundColor, borderWidth: 1
      });
    }
  }

  update() {
    for (let i = 0; i < this.fuzVar.sets.length; i++) {
      let data = this.datasets[i].data;
      for (let j = 0; j < this.labels.length; j++) {
        data[j] = this.fuzVar.sets[i].mf(this.labels[j]);
      }
    }
  }

  updateChart(chart: any) {
    chart.config.data.labels = this.labels;
    chart.config.data.datasets = this.datasets;
    chart.update();
  }
}


/**
 * Initialize variable plotting sessions
 */
function initVarPlots(fuzVars: {[s: string]: FuzzyVar}) {
  let varPlots = {};

  for (let [symbol, fuzVar] of Object.entries(fuzVars)) {
    varPlots[fuzVar.symbol] = new VarPlot(fuzVar);
  }

  return varPlots;
}

export default {
  methods: {
    plot(varSym: string) {
      this.activeSym = varSym;
      this.varPlots[varSym].update();
      this.varPlots[varSym].updateChart(this.chart);
    },

    get() {
      return this.fuzVars;
    },

    reset() {
      this.fuzVars = getDefaultVars();
      this.varPlots = initVarPlots(this.fuzVars);
      this.plot(this.activeSym);
    }
  },

  created() {
    this.fuzVars = getDefaultVars();
    this.varPlots = initVarPlots(this.fuzVars);
    this.activeSym = null;
  },

  mounted() {
    // @ts-ignore
    this.chart = new Chart(this.$refs.canvas, getChartConfig());

    // Add event listeners to the canvas
    patchForMouseDrag(this);

    // Click on the first radio
    $("[name=fs_var_plot]:first").click();
  }
};
</script>
