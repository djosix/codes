<template lang="pug">
  .ui.form
    .inline.fields
      .field
        .ui.action.input(@click='selectPath')
          input(
            type='text'
            readonly
            :value='pathDisplay'
            placeholder='Env Path'
            style='width: 100px')
          input.noshow(
            type='file'
            accept='.txt'
            @input='e => changePath(e.target.value)'
            ref='fileInput')
          .ui.icon.button
            i.cloud.file.icon

      .field
        .ui.small.icon.button(@click='start' v-bind:class='{ disabled: config.start.disabled }')
          i.ui.play.icon

      .field(v-if='config.stop.show')
        .ui.small.icon.button(@click='stop' v-bind:class='{ disabled: config.stop.disabled }')
          i.ui.stop.icon

      .field(v-if='config.reset.show')
        .ui.small.icon.button(@click='reset' v-bind:class='{ disabled: config.reset.disabled }')
          i.ui.redo.icon

      .field
        .ui.small.right.labeled.input
          input(
            type='number' 
            name='fps'
            min='1' max='120' value='30'
            placeholder='fps'
            style='width: 70px'
            ref='fps'
            @input='e => changeFPS(e.target.value)')
          .ui.basic.label FPS

      .field(v-if='config.save.show')
        .ui.small.icon.button(@click='save' v-bind:class='{ disabled: config.save.disabled }')
          i.ui.save.icon
          input.noshow(
            type='file'
            nwdirectory
            @input='saveHist'
            ref='fileOutput')
</template>


<script lang="ts">
import context from "/lib/context";
import { helpers } from "/lib/math";

let btnStates = {
  initialized: {
    start: { show: true, disabled: false },
    stop: { show: true, disabled: true },
    reset: { show: false, disabled: true },
    save: { show: true, disabled: true }
  },
  started: {
    start: { show: true, disabled: true },
    stop: { show: true, disabled: false },
    reset: { show: false, disabled: true },
    save: { show: true, disabled: true }
  },
  stopped: {
    start: { show: true, disabled: false },
    stop: { show: false, disabled: true },
    reset: { show: true, disabled: false },
    save: { show: true, disabled: true }
  },
  collided: {
    start: { show: true, disabled: false },
    stop: { show: false, disabled: true },
    reset: { show: true, disabled: false },
    save: { show: true, disabled: true }
  },
  finished: {
    start: { show: true, disabled: false },
    stop: { show: false, disabled: true },
    reset: { show: true, disabled: false },
    save: { show: true, disabled: false }
  },
  locked: {
    start: { show: true, disabled: true },
    stop: { show: true, disabled: true },
    reset: { show: false, disabled: true },
    save: { show: true, disabled: true }
  }
};


export default {
  data() {
    return {
      config: btnStates.initialized,
      path: '',
    };
  },
  computed: {
    pathDisplay() {
      return this.path.replace(/.*\//, '');
    }
  },
  methods: {
    start() {
      context.start();
    },
    stop() {
      context.stop();
    },
    reset() {
      context.reset();
    },
    save() {
      this.$refs.fileOutput.click();
    },
    saveHist() {
      let path = this.$refs.fileOutput.value;
      context.env.saveHist(path);
    },
    changeFPS(value) {
      value = helpers.clip(value, this.$refs.fps.min, this.$refs.fps.max);
      context.fps = this.$refs.fps.value = value;
    },
    selectPath() {
      this.$refs.fileInput.click();
    },
    changePath(path) {
      try {
        context.setEnv(path);
      } catch (e) {
        if (this.path) {
          context.emit('logError', e);
          context.emit('logWarning', `Fallback to old env: ${this.path}`);
        }
      }
    },
    setConfig(state: string) {
      if (state in btnStates) {
        this.config = btnStates[state];
      }
    }
  },
  mounted() {
    context.register("updateControl", this.setConfig);
    context.register("changePathDisplay", (path) => { this.path = path; });
    this.changeFPS(this.$refs.fps.value);

    document.addEventListener('keydown', (e) => {
      if (e.keyCode == 32) {
        switch (context.status) {
          case 'collided':
          case 'finished':
          case 'stopped':
            context.reset();
            break;
          case 'initialized':
            context.start();
            break;
          case 'started':
            context.stop();
            break;
        }
      }
    });
  },
};
</script>
