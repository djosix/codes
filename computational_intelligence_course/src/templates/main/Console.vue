<template lang="pug">
  .ui.message.logs(ref='logs')
    .entry(v-for='entry in entries' v-bind:class='entry.classList') {{ entry.text }}
</template>

<style scoped lang="less">
.logs {
  min-height: 100px;
  max-height: 100px;
  overflow: scroll;
  overflow-x: hidden;
  padding: 10px;
}

.logs > .entry {
  display: block;
  margin: 0px;
  padding: 0px;
  // word-wrap: break-word;
  // overflow-wrap: break-word;
  overflow-wrap: anywhere;

  &:last-child {
    font-weight: bold;
  }

  &.error {
    color: #dc143c;
  }

  &.success {
    color: #126d12;
  }

  &.warning {
    color: #927c01;
  }

  &.info {
    color: #2f4f4f;
  }
}
</style>

<script lang="ts">
import context from "/lib/context";

function formatLog(text) {
  let time = new Date().toLocaleTimeString();
  return `${time} | ${text}`;
}

export default {
  data: () => {
    return {
      entries: []
    };
  },
  methods: {
    reset() {
      this.entries = [];
    },
    addEntry(text, classList = []) {
      this.entries.push({
        text,
        classList
      });
    },
    log(text, classList = []) {
      this.addEntry(formatLog(text), classList);
    },
    logError(text) {
      this.log(text, ["error"]);
    },
    logInfo(text) {
      this.log(text, ["info"]);
    },
    logSuccess(text) {
      this.log(text, ["success"]);
    },
    logWarning(text) {
      this.log(text, ["warning"]);
    }
  },
  updated() {
    this.$refs.logs.scrollTo(0, this.$refs.logs.scrollHeight);
  },
  mounted() {
    context.register("logRaw", this.addEntry);
    context.register("log", this.log);
    context.register("logInfo", this.logInfo);
    context.register("logError", this.logError);
    context.register("logSuccess", this.logSuccess);
    context.register("logWarning", this.logWarning);
    context.register("resetLogs", this.reset);
  }
};
</script>
