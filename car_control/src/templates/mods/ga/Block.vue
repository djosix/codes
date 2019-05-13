<template lang="pug">
  .ui.form
    .ui.grid
      .row
        .column
          .ui.grid
            .two.columns.row
              .column
                .field
                  label Population Size
                    span {{ appliedConfig.populationSize }}
                  input(
                    v-model='config.populationSize' type='number'
                    :min='limits.populationSize[0]'
                    :max='limits.populationSize[1]')

                .field
                  label Selection
                    span {{ appliedConfig.selectionName }}
                  select.conf.ui.dropdown(v-model='config.selectionName')
                    option(value='roulette_wheel') Routlette Wheel
                    option(value='tournament') Tournament
                  
                .field
                  label Mutation Prob.
                    span {{ appliedConfig.mutationProb }}
                  input(
                    v-model='config.mutationProb'
                    type='number' step='0.01'
                    :min='limits.mutationProb[0]'
                    :max='limits.mutationProb[1]')

              .column
                .field
                  label RBF Kernels
                    span {{ appliedConfig.networkSize }}
                  input(
                    v-model='config.networkSize' type='number'
                    :min='limits.networkSize[0]' :max='limits.networkSize[1]')

                .field
                  label Crossover Prob.
                    span {{ appliedConfig.crossoverProb }}
                  input(
                    v-model='config.crossoverProb'
                    type='number' step='0.01'
                    :min='limits.crossoverProb[0]'
                    :max='limits.crossoverProb[1]')
                
                .field
                  label Muatation Scale
                    span {{ appliedConfig.mutationScale }}
                  input(
                    v-model='config.mutationScale'
                    type='number' step='0.0001'
                    :min='limits.mutationScale[0]'
                    :max='limits.mutationScale[1]')

      .row
        .column
          .inline.fields
            .field
              .ui.button(
                @click='initialize()'
                v-bind:class='{ disabled: running }'
              ) Init
            .field
              .ui.button(
                @click='optimize()'
                v-bind:class='{ disabled: running || !ga }'
              ) Optimize
            .field
              .ui.button(
                @click='pause()'
                v-bind:class='{ disabled: !running || !ga }'
              ) Pause
            .field
              .ui.button(@click='reset()') Reset
          .field
            .ui.action.input(@click='$refs.datasetInput.click()')
              input(
                type='text' readonly :value='config.dataset' placeholder='Dataset')
              input.noshow(
                type='file' accept='.txt' ref='datasetInput'
                @input='e => { config.dataset = e.target.value; }')
              .ui.icon.button
                i.cloud.file.icon


      .row
        .column
          b Statistics
          pre.ui.segment {{ stat }}
          

          
</template>


<script>
import context from '/lib/context';
import mod from '/lib/mods/ga/module';
import math from 'mathjs';

import { getDefaultSettings } from '/lib/mods/ga/defaults';
import { helpers } from '/lib/math';

import $ from 'jquery';

export default {
  data() {
    return {
      config: getDefaultSettings(),
      appliedConfig: {},
      running: false,
      stopped: false,
      limits: {
        populationSize: [5, 200],
        networkSize: [1, 10],
        crossoverProb: [0, 1],
        mutationProb: [0, 1],
        mutationScale: [0, 1],
      },
      stat: 'nothing to display',
      ga: null,
    };
  },
  methods: {
    collectConfig() {
      for (let key of Object.keys(this.limits)) {
        this.config[key] = helpers.clip(
          Number(this.config[key]),
          ...this.limits[key]
        );
      }
      if (this.config.populationSize % 2 == 1) {
        this.config.populationSize += 1;
      }
      return Object.assign({}, this.config);
    },

    reset() {
      this.config = getDefaultSettings();
      context.emit('logInfo', 'Preset loaded');
    },

    initialize() {
      let config = Object.assign({}, this.config);
      delete config.dataset;
      context.emit('setLoading', true, `Initializing GA with ${JSON.stringify(config, null, 1)}`);
      for (let key of Object.keys(config)) {
        config[key] = ' = ' + config[key];
      }

      setTimeout(() => {
        mod.initialize();
        this.ga = mod.ga;
        this.appliedConfig = config;
        this.stat = 'start optimizing to display statistics';
        context.emit('setLoading', false);
        context.emit('logInfo', `GA initialized`);
      }, 100);
    },

    optimize() {
      context.emit('logInfo', 'Starting GA optimization');

      setTimeout(() => {
        this.stopped = false;
        this.optimizeStep();
      }, 100);
    },

    optimizeStep() {
      if (this.stopped) {
        this.running = false;
        return;
      }

      this.running = true;
      this.ga.step();

      this.stat = JSON.stringify(this.ga.statistics(), null, 2);

      setTimeout(() => {
        this.optimizeStep();
      }, 1);
    },
    pause() {
      this.stopped = true;
      this.running = false;
      context.emit('logInfo', 'GA optimization paused');
    },
  },
  components: {
  }
};
</script>


