import { Module } from './module';
import { Env } from './env';
import { Renderer } from './render';
import { EventManager } from './event';

class Context extends EventManager {
  env: Env;
  renderer: Renderer;
  module: Module = null;
  modules: Module[] = [];
  status: string;
  fps: number;

  constructor() {
    super();
  }

  // Modules

  registerModule(mod: Module) {
    this.modules[mod.slug] = mod;
  }

  setModule(slug: string) {
    if (slug in this.modules) {
      this.module = this.modules[slug];
    } else {
      alert(`"${slug}" is not registered as an module`);
    }
  }

  setCanvas(canvas: HTMLCanvasElement) {
    this.renderer = new Renderer(canvas);
  }

  setEnv(path: string) {
    try {
      this.env = new Env(path);
    } catch (e) {
      this.emit('logError', `Cannot set env due to an error`);
      throw e;
    }
    this.emit('logSuccess', `Loaded: ${path}`);
    this.emit('changePathDisplay', path);
    this.renderer.fitRange(this.env.mapRange);
    this.reset();
  }

  stop() {
    this.status = 'stopped';
    this.emit('updateControl', this.status);
  }

  reset() {
    // Check if env is set
    if (!this.env || !this.renderer) {
      this.emit('logError', 'Env not set, cannot reset');
      return;
    }

    // Reset env and render
    this.env.reset();
    this.renderer.render(this.env);
    this.status = 'initialized';
    this.emit('logInfo', 'Initialized');
    this.emit('updateControl', this.status);
  }

  start() {
    // Check if env is set
    if (!this.env || !this.renderer) {
      this.emit('logError', 'Env not set, cannot start');
      return;
    }
    
    // Check if the module is set
    if (!this.module) {
      this.emit('logError', 'Module not set, cannot start');
      return;
    }

    if (this.status != 'initialized') {
      this.reset();
    }

    let mod = this.module;
    
    // Initialize module config
    try {
      mod.init();
    } catch (e) {
      this.emit('logError', `Cannot start: ${e.message}`);
      return;
    }

    // Start a Env coroutine
    let envCor = this.env.start();
    envCor.next();
    
    this.status = 'started';
    this.emit('logInfo', `Driving... [module: ${mod.name}]`);
    this.emit('updateControl', this.status);

    let step = () => {
      let steerDeg: number;
      try {
        steerDeg = mod.predict(this.env.state.radarVals);
      } catch (e) {
        this.status = 'stopped';
        this.emit('logError', `Stopped: ${e}`);
        this.emit('updateControl', this.status);
        // return;
        throw e;
      }

      let out = envCor.next(steerDeg);

      if (out.done) {
        // Coroutine loop is breaked
        if (this.env.state.result == 'collided') {
          this.status = 'collided';
          this.emit('logInfo', `Car hit the wall [step: ${this.env.state.step}]`);
          this.emit('updateControl', this.status);
        } else if (this.env.state.result == 'finished') {
          this.status = 'finished';
          this.emit('logSuccess', `Car reached the goal [step: ${this.env.state.step}]`)
          this.emit('updateControl', this.status);
        } else {
          this.status = 'stopped';
          this.emit('logWarning', `Stopped by an unknown exception [step: ${this.env.state.step}]`)
          this.emit('updateControl', this.status);
        }
        return;
      }

      if (this.status == 'stopped') {
        this.emit('logInfo', `Stopped by user [step: ${this.env.state.step}]`)
        this.emit('updateControl', this.status);
        return;
      } else if (this.status != 'started') {
        this.status = 'stopped';
        this.emit('logWarning', `Stopped by an unknown exception [step: ${this.env.state.step}]`)
        this.emit('updateControl', this.status);
        return;
      }

      this.emit('updateMonitor', {
        position: this.env.car.position,
        angle: this.env.car.angleDeg,
        steer: steerDeg,
        ...this.env.state.radarVals
      });

      this.renderer.render(this.env);

      setTimeout(step, 1000 / this.fps);
    };

    // Actually start
    step();
  }
}

export { Context };
export default new Context();
