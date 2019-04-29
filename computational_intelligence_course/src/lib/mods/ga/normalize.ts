import { helpers } from "/lib/math";

export function normalizeInput(X: number[]): number[] {
  // [0, 80] => [-1, 1]
  return X.map(x => helpers.clip(x, 0, 80) / 40 - 1);
}

export function normalizeOutput(y: number): number {
  // [-40, 40] => [-1, 1]
  return helpers.clip(y, -40, 40) / 40;
}

export function denormalizeOutput(y: number): number {
  // [-1, 1] => [-40, 40]
  return -helpers.clip(y, -1.0, 1.0) * 40;
}
