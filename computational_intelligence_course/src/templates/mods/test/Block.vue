<template lang="pug">
  div
    .ui.form
      .field
        label Steering Angle
        textarea.code(type='text' v-model='steeringAngle')
      .field
        .ui.button.preset-btn(@click='loadPreset') Load Preset
</template>

<style scoped lang="less">
.code {
  font-family: monospace;
}
</style>

<script>
import context from '/lib/context';
import test from '/lib/mods/test/module';

let defaultCode = `/* empty semicolon will return */

let [x, y, z] = [4.9, 1.43, 2];
(right - left - x) > 0
  ? (right - left - x) / y
  : (right - left - x) / z`;

export default {
  data() {
    return {
      steeringAngle: defaultCode
    };
  },
  methods: {
    loadPreset() {
      this.steeringAngle = defaultCode;
      context.emit('logInfo', 'Preset loaded');
    },
  },
};
</script>

