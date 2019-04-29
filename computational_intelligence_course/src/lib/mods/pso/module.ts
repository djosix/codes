import { Module } from '/lib/module';
import { PSO } from './lib';
import { loadDataset } from '../ga/dataset';
import { Network } from '../ga/rbfnet';
import { normalizeInput, denormalizeOutput } from '../ga/normalize';


class ParticleSwarmOptimization extends Module {
  name = 'Particle Swarm Opt.';
  slug = 'particle_swarm_optimization';

  pso: PSO;
  func: (X: number[]) => number;

  hardLoaded = false;

  initialize() {
    let config = this.component.collectConfig();
    this.pso = new PSO(
      config.swarmSize,
      config.networkSize,
      config.inertiaWeight,
      config.maxVelocity,
      config.cognitiveConstant,
      config.socialConstant,
      loadDataset(config.dataset));
  }

  init() {
    if (!this.pso) {
      throw new Error('PSO not initialized yet');
    }

    if (!this.hardLoaded)
      this.func = this.pso.getBestFunc();
  }

  predict(inputs: {front: number, left: number, right: number}): number {
    let X = [inputs.front, inputs.left, inputs.right];
    return this.func(X);
  }

  importModel(text: string) {
    let network = new Network(this.pso.networkSize, 3);
    let lines = text.split('\n');
    
    network.bias = Number(lines.splice(0, 1)[0]);
    console.log(lines);
    for (let i = 0; i < lines.length; i++) {
      let nums = lines[i].split(' ').map(Number);
      network.weights[i] = nums.splice(0, 1)[0];
      network.centers[i] = nums.splice(0, network.dim);
      network.sigmas[i] = nums.splice(0, 1)[0];
    }
    
    this.func = (X: number[]) => {
      X = normalizeInput(X);
      let y = network.forward(X);
      y = denormalizeOutput(y);
      return y;
    };

    this.hardLoaded = true;
  }

  exportModel(): string {
    let network = new Network(this.pso.networkSize, 3);
    network.setParams(this.pso.overallBestPosition);
    let lines = [];
    lines.push(network.bias);
    for (let i = 0; i < network.size; i++) {
      let neuronParams = [
        network.weights[i],
        ...network.centers[i],
        network.sigmas[i]];
      lines.push(neuronParams.join(' '));
    }
    return lines.join('\n');
  }
}

export default new ParticleSwarmOptimization();
export { ParticleSwarmOptimization };
