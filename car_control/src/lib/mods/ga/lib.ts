import math from 'mathjs';
import { helpers } from '/lib/math';
import { normalizeInput, normalizeOutput, denormalizeOutput } from './normalize';
import { getParamsSize, Network } from './rbfnet';

//=====================================================================
// Individual
//=====================================================================

export class Individual {
  networkPrototype: any;
  chromosome: number[];
  fitness: number;

  constructor(size: number, dim: number, init = true) {
    this.networkPrototype = {size, dim};

    if (init) {
      // size * (center[dim] + sigma + weight) + bias
      this.chromosome = helpers.uniformArray(getParamsSize(size, dim));
    }
  }

  network() {
    let network = new Network(
      this.networkPrototype.size,
      this.networkPrototype.dim);
    
    network.setParams([...this.chromosome]);
    return network;
  }

  clone() {
    let individual = new Individual(null, null, false);
    individual.networkPrototype = this.networkPrototype;
    individual.chromosome = [...this.chromosome];
    individual.fitness = this.fitness;
    return individual;
  }
}

//=====================================================================
// Selection
//=====================================================================

function softmax(X: number[]): number[] {
  X = math.exp(X) as number[];
  return math.dotDivide(X, math.sum(X)) as number[];
}

function rouletteWheelSelection(population: Individual[]): Individual[] {
  let probs = softmax(population.map(ind => ind.fitness));
  let accProbs = [0];

  for (let prob of probs) {
    accProbs.push(accProbs[accProbs.length - 1] + prob);
  }

  let selected = [];

  while (selected.length < population.length) {
    let k = Math.random();
    for (let i = 0; i < population.length; i++) {
      if (accProbs[i] <= k && k < accProbs[i + 1]) {
        selected.push(population[i].clone());    
        break;
      }
    }
  }

  return selected;
}

function tournamentSelection(population: Individual[]): Individual[] {
  let selected = [];

  while (selected.length < population.length) {
    let winner: Individual;
    let bestFit = -Infinity;

    helpers.randomDraw(population.length, 2)
      .map(idx => population[idx])
      .forEach(ind => {
        if (ind.fitness < bestFit)
          return;
        winner = ind;
        bestFit = ind.fitness;
      });

    selected.push(winner.clone());
  }

  return selected;
}

class Selection {
  method: (p: Individual[]) => Individual[];

  constructor(name: string) {
    switch (name) {
      case 'roulette_wheel':
        this.method = rouletteWheelSelection;
        break;
      case 'tournament':
        this.method = tournamentSelection;
        break;
      default:
        throw Error(`no such method: ${name}`);
    }
  }

  perform(population: Individual[]) {
    return this.method(population);
  }
}

//=====================================================================
// Mutation
//=====================================================================

class Mutation {
  prob: number;
  scale: number;

  constructor(prob: number, scale: number) {
    this.prob = prob;
    this.scale = scale;
  }

  perform(ind: Individual) {
    let dirty = false;

    ind = ind.clone();
    
    if (Math.random() > this.prob) {
      return ind;
    }

    for (let i = 0; i < ind.chromosome.length; i++) {
      ind.chromosome[i] += helpers.uniform(-this.scale, this.scale);
    }

    // make recompute fitness
    ind.fitness = null;

    return ind;
  }
}


//=====================================================================
// Crossover
//=====================================================================

function cxReal(n1: number, n2: number): number[] {
  let d = Math.random();
  return [
    d * (n2 - n1) + n1,
    d * (n1 - n2) + n2,
  ];
}

class Crossover {
  prob: number;

  constructor(prob: number) {
    this.prob = prob;
  }

  perform(ind1: Individual, ind2: Individual) {
    console.assert(ind1.chromosome.length == ind2.chromosome.length);

    let chromosomeLength = ind1.chromosome.length;
    ind1 = ind1.clone(), ind2 = ind2.clone();

    if (Math.random() > this.prob) {
      return [ind1, ind2];
    }

    for (let i = 0; i < chromosomeLength; i++) {
      [ind1.chromosome[i], ind2.chromosome[i]] =
        cxReal(ind1.chromosome[i], ind2.chromosome[i]);
    }

    ind1.fitness = ind2.fitness = null;

    return [ind1, ind2];
  }
}

//=====================================================================
// Fitness
//=====================================================================

export class Fitness {
  n = 0;
  Xs = [] as number[][];
  ys = [] as number[];

  constructor(dataset: number[][]) {
    dataset.forEach(row => {
      this.n++;
      this.Xs.push(normalizeInput(row.slice(0, 3)));
      this.ys.push(normalizeOutput(row[3]));
    });
  }

  compute(ind: Individual) {
    if (typeof ind.fitness == 'number') {
      return;
    }
    ind.fitness = this.computeFitnessForNetwork(ind.network());
  }

  computeFitnessForNetwork(network: Network) {
    let outs = this.Xs.map(X => network.forward(X));
    let mse = math.mean(math.dotPow(math.subtract(this.ys, outs), 2));
    return 1 - mse;
  }
}

//=====================================================================
// Genetic Algorithm
//=====================================================================

function shuffled(items: any[]) {
  return helpers
    .randomDraw(items.length, items.length)
    .map(index => items[index]);
}

function pairwise(items: any[]) {
  let pairs = [];
  for (let i = 0; i < items.length; i += 2) {
    pairs.push([items[i], items[i + 1]]);
  }
  return pairs;
}

export class GA {
  generation = 0;
  population = [] as Individual[];
  bestInd: Individual;

  selection: Selection;
  crossover: Crossover;
  mutation: Mutation;
  fitness: Fitness;

  constructor(
      networkSize: number,
      networkDim: number,
      populationSize: number,
      selectionName: string,
      crossoverProb: number,
      mutationProb: number,
      mutationScale: number,
      dataset: number[][])
  {

    this.selection = new Selection(selectionName);
    this.crossover = new Crossover(crossoverProb);
    this.mutation = new Mutation(mutationProb, mutationScale);
    this.fitness = new Fitness(dataset);

    for (let i = 0; i < populationSize; i++) {
      let ind = new Individual(networkSize, networkDim);
      this.population.push(ind);
    }

    // Compute fitnesses
    this.population.forEach(ind => this.fitness.compute(ind));
    this.bestInd = this.getBestInd();
  }

  step() {
    // Parent selection
    let parents = this.selection.perform(this.population);

    // Crossover
    let childs = pairwise(shuffled(parents))
      .map(pair => this.crossover.perform(pair[0], pair[1]))
      .reduce((a1, a2) => a1.concat(a2));

    // Mutation
    this.population = childs.map(ind => this.mutation.perform(ind));

    // Compute fitnesses
    this.population.forEach(ind => this.fitness.compute(ind));
    let currentBestInd = this.getBestInd();
    if (currentBestInd.fitness > this.bestInd.fitness) {
      this.bestInd = currentBestInd;
    }

    this.generation++;
  }

  statistics() {
    let fitnessValues = this.population.map(ind => ind.fitness);
    let max = math.max(fitnessValues);
    let min = math.min(fitnessValues);
    return {
      generation: this.generation,
      fitness: {
        overallMax: this.bestInd.fitness,
        max, min, range: max - min,
        mean: math.mean(fitnessValues),
        std: math.std(fitnessValues),
      },
    };
  }

  getBestInd() {
    let bestFit = -Infinity;
    let bestInd = null;
    for (let ind of this.population) {
      if (ind.fitness > bestFit) {
        bestInd = ind;
        bestFit = ind.fitness;
      }
    }
    return bestInd;
  }

  getBestFunc() {
    let network = this.bestInd.network();
    return (X: number[]) => {
      X = normalizeInput(X);
      let y = network.forward(X);
      y = denormalizeOutput(y);
      return y;
    };
  }

  exportBestParams() {
    let network = this.bestInd.network();
    let lines = [];
    lines.push(network.bias);
    for (let i = 0; i < network.size; i++) {
      let neuronParams = network.centers[i]
        .concat([network.sigmas[i], network.weights[i]]);
      lines.push(neuronParams.join(' '));
    }
    return lines.join('\n');
  }
}
