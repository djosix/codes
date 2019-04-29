import { Module } from '/lib/module';
import { GA } from './lib';
import { loadDataset } from './dataset';


class GeneticAlgorithm extends Module {
  name = 'Genetic Algorithm';
  slug = 'genetic_algorithm';

  ga: GA;
  func: (X: number[]) => number;

  initialize() {
    let config = this.component.collectConfig();
    this.ga = new GA(
        config.networkSize, 3,
        config.populationSize,
        config.selectionName,
        config.crossoverProb,
        config.mutationProb,
        config.mutationScale,
        loadDataset(config.dataset));
  }

  init() {
    if (!this.ga) {
      throw new Error('GA not initialized yet');
    }

    this.func = this.ga.getBestFunc();
  }

  predict(inputs: {front: number, left: number, right: number}): number {
    let X = [inputs.front, inputs.left, inputs.right];
    return this.func(X);
  }
}

export default new GeneticAlgorithm();
export { GeneticAlgorithm };
