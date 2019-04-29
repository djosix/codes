<template lang="pug">
  .ui.grid
    .seven.wide.column
      MainBlock
    .nine.wide.column
      ModBlock
</template>

<script>
import MainBlock from "./main/Block";
import ModBlock from "./mods/Block";

export default {
  methods: {},
  components: {
    MainBlock,
    ModBlock
  }
};
</script>
