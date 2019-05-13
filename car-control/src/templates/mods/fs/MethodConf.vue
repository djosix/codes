<template lang="pug">
  .field
    label Method
    table.ui.very.compact.structured.celled.center.aligned.table
      tr(v-for='funcSet in funcLib.funcSets')
        td {{ funcSet.name }}
        td
          select(:name='funcSet.slug' v-model='config[funcSet.slug]')
            option(v-for='func in funcSet.funcs' :value='func.slug' v-html='func.name')
</template>


<script lang="ts">
import { getDefaultMethods } from '/lib/mods/fs/defaults';
import funcLib from '/lib/mods/fs/funcs';


export default {
  data: () => {
    return {
      funcLib,
      config: getDefaultMethods(),
    }
  },
  methods: {
    get() {
      return this.config;
    },
    reset() {
      this.config = getDefaultMethods();
    }
  }
};
</script>

