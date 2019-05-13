<template lang="pug">
  canvas(width='420' height='460' ref='canvas')
</template>


<script lang="ts">
import context from '/lib/context';

export default {
  mounted() {
    context.setCanvas(this.$refs.canvas);
  }
};
</script>
