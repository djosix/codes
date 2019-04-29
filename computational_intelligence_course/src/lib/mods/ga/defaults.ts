
function getDefaultSettings() {
  return {
    populationSize: 10,
    networkSize: 2,
    selectionName: 'tournament',
    crossoverProb: 0.5,
    mutationProb: 0.5,
    mutationScale: 0.1,
    dataset: './data/ga/train4dAll.txt',
  };
}

export { getDefaultSettings };
