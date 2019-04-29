<template lang="pug">
  .ui.form
    .ui.grid
      .row
        .column
          .ui.grid
            .two.columns.row
              .column
                .field
                  label Swarm Size
                    span {{ appliedConfig.swarmSize }}
                  input(
                    v-model='config.swarmSize' type='number'
                    :min='limits.swarmSize[0]'
                    :max='limits.swarmSize[1]')

                .field
                  label Inertia Weight
                    span {{ appliedConfig.inertiaWeight }}
                  input(
                    v-model='config.inertiaWeight'
                    type='number' step='0.01'
                    :min='limits.inertiaWeight[0]'
                    :max='limits.inertiaWeight[1]')
                    
                .field
                  label Cognitive Constant
                    span {{ appliedConfig.cognitiveConstant }}
                  input(
                    v-model='config.cognitiveConstant'
                    type='number' step='0.01'
                    :min='limits.cognitiveConstant[0]'
                    :max='limits.cognitiveConstant[1]')

              .column
                .field
                  label RBF Kernels
                    span {{ appliedConfig.networkSize }}
                  input(
                    v-model='config.networkSize' type='number'
                    :min='limits.networkSize[0]' :max='limits.networkSize[1]')

                .field
                  label Max Velocity
                    span {{ appliedConfig.maxVelocity }}
                  input(
                    v-model='config.maxVelocity'
                    type='number' step='0.01'
                    :min='limits.maxVelocity[0]'
                    :max='limits.maxVelocity[1]')

                .field
                  label Social Constant
                    span {{ appliedConfig.socialConstant }}
                  input(
                    v-model='config.socialConstant'
                    type='number' step='0.01'
                    :min='limits.socialConstant[0]'
                    :max='limits.socialConstant[1]')

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
                v-bind:class='{ disabled: running || !pso }'
              ) Opt.
            .field
              .ui.button(
                @click='pause()'
                v-bind:class='{ disabled: !running || !pso }'
              ) Pause
            .field
              .ui.button(
                v-bind:class='{ disabled: !(pso && pso.iteration) }'
                @click='$refs.saveModel.click()'
              ) Save
            .field
              .ui.button(
                v-bind:class='{ disabled: !pso }'
                @click='$refs.loadModel.click()'
              ) Load
            .field
              .ui.button(@click='reset()') Reset
          .field
            .ui.action.input(@click='$refs.datasetInput.value = ""; $refs.datasetInput.click()')
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
    
    input.noshow(
      @input='e => loadModel(e.target.value)'
      type='file' ref='loadModel')

    input.noshow(
      @input='e => saveModel(e.target.value)'
      type='file' nwsaveas='RBFN model params.txt' ref='saveModel')

          
</template>


<script>
import context from '/lib/context';
import mod from '/lib/mods/pso/module';
import math from 'mathjs';

import { getDefaultSettings } from '/lib/mods/pso/defaults';
import { helpers } from '/lib/math';

import $ from 'jquery';
import { writeTextFile, readTextFile } from '../../../lib/fs';

export default {
  data() {
    return {
      config: getDefaultSettings(),
      appliedConfig: {},
      running: false,
      stopped: false,
      limits: {
        swarmSize: [2, 200],
        networkSize: [1, 10],
        inertiaWeight: [0, 5],
        cognitiveConstant: [0, 10],
        socialConstant: [0, 10],
        maxVelocity: [0, 1],
      },
      stat: 'nothing to display',
      pso: null,
    };
  },
  methods: {
    collectConfig() {
      for (let key of Object.keys(this.limits)) {
        this.config[key] = helpers.clip(Number(this.config[key]), ...this.limits[key]);
      }
      return Object.assign({}, this.config);
    },

    loadModel(path) {
      if (path) {
        mod.importModel(readTextFile(path));
        context.emit('logInfo', 'PSO model loaded from ' + path);
      }
    },

    saveModel(path) {
      writeTextFile(path, mod.exportModel());
      context.emit('logInfo', 'PSO model saved to ' + path);
    },

    reset() {
      this.config = getDefaultSettings();
      context.emit('logInfo', 'Preset loaded');
    },

    initialize() {
      let config = Object.assign({}, this.config);
      delete config.dataset;
      context.emit('setLoading', true, `Initializing PSO with ${JSON.stringify(config, null, 1)}`);
      for (let key of Object.keys(config)) {
        config[key] = ' = ' + config[key];
      }

      setTimeout(() => {
        mod.initialize();
        this.pso = mod.pso;
        this.appliedConfig = config;
        mod.hardLoaded = false;
        this.stat = 'start optimizing to display statistics';
        context.emit('setLoading', false);
        context.emit('logInfo', `PSO initialized`);
      }, 100);
    },

    optimize() {
      this.stopped = false;
      context.emit('logInfo', 'Starting PSO');

      setTimeout(() => {
        this.optimizeStep();
      }, 1);
    },

    optimizeStep() {
      if (this.stopped) {
        this.running = false;
        return;
      }

      mod.hardLoaded = false;
      this.running = true;
      this.pso.step();

      this.stat = JSON.stringify(this.pso.statistics(), null, 2);

      setTimeout(() => {
        this.optimizeStep();
      }, 1);
    },

    pause() {
      this.stopped = true;
      this.running = false;
      context.emit('logInfo', 'PSO optimization paused');
    },
  },
  components: {
  }
};
</script>


