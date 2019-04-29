<template lang="pug">
  div
    .ui.top.attached.tabular.menu
      a.item(
        v-for='m in modules'
        :data-tab='m.slug'
        @click='setModule(m.slug)'
      ) {{ m.name }}

    .ui.bottom.attached.tab.blurring.segment(
      v-for='m in modules'
      :data-tab='m.slug'
    )
      component(v-bind:is='m.slug' :ref='m.slug')
      .ui.inverted.dimmer(v-bind:class='{ active: isLoading }')
        .ui.text.loader {{ loadingMessage }}
</template>


<script>
import context from "/lib/context";

//===================================================
// Register modules and components

import test from '/lib/mods/test/module';
import TestBlock from './test/Block.vue';
test.setComponentType(TestBlock);
context.registerModule(test);

import fuzzySystem from '/lib/mods/fs/module';
import FuzzySystemBlock from './fs/Block.vue';
fuzzySystem.setComponentType(FuzzySystemBlock);
context.registerModule(fuzzySystem);

import geneticAlgorithm from '/lib/mods/ga/module';
import geneticAlgorithmBlock from './ga/Block.vue';
geneticAlgorithm.setComponentType(geneticAlgorithmBlock);
context.registerModule(geneticAlgorithm);

import particleSwarmOptimization from '/lib/mods/pso/module';
import particleSwarmOptimizationBlock from './pso/Block.vue';
particleSwarmOptimization.setComponentType(particleSwarmOptimizationBlock);
context.registerModule(particleSwarmOptimization);

//===================================================

let modules = Object.values(context.modules);
let components = {};
modules.forEach(m => {
  components[m.slug] = m.componentType;
});

import $ from "jquery";

export default {
  data: () => {
    return {
      modules,
      isLoading: false,
      loadingMessage: '',
    };
  },
  methods: {
    setModule(name) {
      context.setModule(name);
    },

    changeModuleTab(name) {
      $(`.menu .item[data-tab=${name}]`).click();
    }
  },
  components,
  mounted() {
    $(".menu .item").tab();

    modules.forEach(m => {
      m.setComponent(this.$refs[m.slug][0]);
    });

    context.register('changeModuleTab', this.changeModuleTab);
    context.register('setLoading', (isLoading, message) => {
      this.isLoading = isLoading;
      this.loadingMessage = message || '';
    });
  }
};
</script>

