<template lang="pug">
  .ui.form
    .ui.grid
      .row
        .column
          MethodConf(ref='methods')
      .row
        .column
          RuleConf(ref='rules')
      .row
        .column
          VarConf(ref='vars')
      .row
        .column
          .ui.button(@click='reset') Load Preset
</template>


<script>
import context from '/lib/context';

import MethodConf from "./MethodConf.vue";
import RuleConf from "./RuleConf.vue";
import VarConf from "./VarConf.vue";

export default {
  methods: {
    collectConfig() {
      return {
        methods: this.$refs.methods.get(),
        rules: this.$refs.rules.get(),
        vars: this.$refs.vars.get(),
      };
    },
    reset() {
      context.emit('logInfo', 'Preset loaded');
      this.$refs.methods.reset();
      this.$refs.rules.reset();
      this.$refs.vars.reset();
    }
  },
  components: {
    MethodConf,
    RuleConf,
    VarConf
  }
};
</script>


