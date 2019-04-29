import math from "mathjs";
import { helpers } from "/lib/math";

export function getParamsSize(size: number, dim: number): number {
  // bias + size * (weight + sigma + center[dim])
  return 1 + size * (2 + dim);
}

export class Network {
  size: number;
  dim: number;
  
  bias = 0;
  sigmas = [] as number[];
  weights = [] as number[];
  centers = [] as number[][];

  constructor(size: number, dim: number) {
    this.size = size;
    this.dim = dim;
  }

  forward(X: number[]): number {
    let dots = this.centers.map(center => {
      let d = math.subtract(X, center) as number[];
      return math.dot(d, d);
    });
    let vars = math.dotMultiply(this.sigmas, this.sigmas);
    let zs = math.dotDivide(dots, math.multiply(vars, -2)) as number[];
    let outs = math.dotMultiply(math.exp(zs), this.weights) as number[];
    return math.sum(outs) + this.bias;
  }

  setParams(params: number[]) {
    if (params.length != getParamsSize(this.size, this.dim)) {
      throw Error(`size mismatch: ${params.length}`);
    }
    params = [...params];
    this.bias = params.splice(0, 1)[0];
    this.sigmas = params.splice(0, this.size).map(n => helpers.clip((n + 1) / 2, 0.01, 10));
    this.weights = params.splice(0, this.size);
    this.centers = math.reshape(params, [this.size, this.dim]) as number[][];
  }
}