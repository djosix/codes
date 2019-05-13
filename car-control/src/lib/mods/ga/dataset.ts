import { readTextFile } from '/lib/fs';

export function loadDataset(path: string) {
  return readTextFile(path)
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length)
    .map(line => line.split(' ').map(Number));
};
