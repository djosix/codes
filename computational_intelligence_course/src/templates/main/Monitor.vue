<template lang="pug">
  table.ui.very.compact.table
    tr
      td.four.wide: b Position
      td.four.wide {{ position }}
      td.five.wide: b Radar (Front)
      td.three.wide {{ front }}
    tr
      td.four.wide: b Angle
      td.four.wide {{ angle }}
      td.five.wide: b Radar (Left)
      td.three.wide {{ left }}
    tr
      td.four.wide: b Steering
      td.four.wide {{ steer }}
      td.five.wide: b Radar (Right)
      td.three.wide {{ right }} 
</template>


<script lang="ts">
import context from '/lib/context';
import { helpers } from "/lib/math";

function getDefaultData() {
  return {
    position: "-",
    angle: "-",
    steer: "-",
    front: "-",
    left: "-",
    right: "-"
  };
}

export default {
  data: getDefaultData,
  methods: {
    reset() {
      Object.assign(this.$data, getDefaultData());
    },
    update(status) {
      status = Object.assign({}, status);
      this.position = status.position.clone().unfloat();
      this.angle = helpers.round(status.angle, 2);
      this.steer = helpers.round(status.steer, 2);
      this.front = helpers.round(status.front, 2);
      this.left = helpers.round(status.left, 2);
      this.right = helpers.round(status.right, 2);
    }
  },
  mounted() {
    context.register('updateMonitor', this.update);
  }
};
</script>
