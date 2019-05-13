import { Network, getParamsSize } from '../ga/rbfnet';
import { normalizeInput, normalizeOutput, denormalizeOutput } from '../ga/normalize';
import { Fitness } from '../ga/lib';
import math from 'mathjs';


function shapedUniformArray(shape: number[], scale = 1): any {
  return math.add(math.dotMultiply(math.random(shape), scale * 2), -scale);
}

export class PSO {

  swarmSize: number;
  networkSize: number;
  paramsSize: number;

  maxVelocity: number;
  inertiaWeight: number;
  cognitiveContstant: number;
  socialContstant: number;

  fitness: Fitness;
  
  iteration = 0;

  // particles
  positions: number[][];
  velocities: number[][];
  fitnesses = [] as number[];
  bestPositions = [] as number[][];
  bestFitnesses: number[];

  currentBestPosition: number[];
  currentBestFitness = -Infinity;

  overallBestPosition: number[];
  overallBestFitness = -Infinity;

  constructor(
      swarmSize: number,
      networkSize: number,
      inertiaWeight: number,
      maxVelocity: number,
      cognitiveConstant: number,
      socialConstant: number,
      dataset: number[][])
  {
    this.swarmSize = swarmSize;
    this.networkSize = networkSize;
    this.paramsSize = getParamsSize(networkSize, 3);
    this.inertiaWeight = inertiaWeight;
    this.maxVelocity = maxVelocity;
    this.cognitiveContstant = cognitiveConstant;
    this.socialContstant = socialConstant;

    this.fitness = new Fitness(dataset);

    this.positions = shapedUniformArray([swarmSize, this.paramsSize]) as number[][];
    this.velocities = shapedUniformArray([swarmSize, this.paramsSize], ) as number[][];
    this.bestFitnesses = this.positions.map(_ => -Infinity);
    this.computeFitnessesAndUpdateBests();
  }

  computeFitnessesAndUpdateBests() {
    this.currentBestFitness = -Infinity;

    for (let i = 0; i < this.swarmSize; i++) {
      // Compute fitness
      let network = new Network(this.networkSize, 3);
      network.setParams(this.positions[i]);
      this.fitnesses[i] = this.fitness.computeFitnessForNetwork(network);

      // Update best position for each particle
      if (this.fitnesses[i] > this.bestFitnesses[i]) {
        this.bestPositions[i] = this.positions[i];
        this.bestFitnesses[i] = this.fitnesses[i];
      }

      // Update best position for this iteration
      if (this.fitnesses[i] > this.currentBestFitness) {
        this.currentBestPosition = this.positions[i];
        this.currentBestFitness = this.fitnesses[i];
      }
    }

    // Update overall best position
    if (this.currentBestFitness > this.overallBestFitness) {
      this.overallBestPosition = this.currentBestPosition;
      this.overallBestFitness = this.currentBestFitness;
    }
  }

  step() {
    for (let i = 0; i < this.swarmSize; i++) {
      this.velocities[i] = math.add(
        math.dotMultiply(this.velocities[i], this.inertiaWeight),
        math.add(
          math.dotMultiply(
            math.subtract(this.bestPositions[i], this.positions[i]),
            this.cognitiveContstant * Math.random()
          ),
          math.dotMultiply(
            math.subtract(this.currentBestPosition, this.positions[i]),
            this.socialContstant * Math.random()
          ),
        ),
      ) as number[];
    }

    this.positions = math.add(this.positions, this.velocities) as number[][];

    this.computeFitnessesAndUpdateBests();

    this.iteration++;
  }

  statistics() {
    let currentMinFitness = math.min(this.fitnesses);
    return {
      iteration: this.iteration,
      fitness: {
        overallMax: this.overallBestFitness,
        max: this.currentBestFitness,
        min: currentMinFitness,
        range: this.currentBestFitness - currentMinFitness,
        mean: math.mean(this.fitnesses),
        std: math.std(this.fitnesses),
      },
    };
  }
  
  getBestFunc() {
    let network = new Network(this.networkSize, 3);
    network.setParams(this.overallBestPosition);

    return (X: number[]) => {
      X = normalizeInput(X);
      let y = network.forward(X);
      y = denormalizeOutput(y);
      return y;
    };
  }

}
