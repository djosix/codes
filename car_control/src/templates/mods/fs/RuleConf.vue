<template lang="pug">
  .row
    .column
      .field
        label Rules
        table.ui.very.compact.structured.celled.center.aligned.table
          tr
            td.collapsing {{ fuzzyVars.Y.name }} \ {{ fuzzyVars.X.name }}
            td(v-for='xSet in fuzzyVars.X.sets')
              i {{ xSet.name }}
          tr(v-for='ySet in fuzzyVars.Y.sets')
            td
              i {{ ySet.name }}
            td(v-for='xSet in fuzzyVars.X.sets')
              select.rule(v-model='config.data[xSet.symbol][ySet.symbol]')
                option(v-for='zSet in fuzzyVars.Z.sets' :value='zSet.symbol') {{ zSet.name }}
</template>


<script lang="ts">

import { getDefaultRuleSet, getDefaultVars } from '/lib/mods/fs/defaults';

export default {
  data() {
    return {
      config: getDefaultRuleSet(),
      fuzzyVars: getDefaultVars(),
    };
  },
  methods: {
    get() {
      return this.config;
    },
    reset() {
      this.config = getDefaultRuleSet();
    }
  },
}
</script>

