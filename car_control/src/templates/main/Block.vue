<template lang="pug">
  div
    Canvas
    Monitor
    Control
    Console
</template>


<script lang="ts">
import $ from 'jquery';

import context from '/lib/context';

import Console from "./Console.vue";
import Canvas from "./Canvas.vue";
import Monitor from "./Monitor.vue";
import Control from "./Control.vue";

export default {
  components: {
    Canvas,
    Monitor,
    Control,
    Console
  },
};
</script>
