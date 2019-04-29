export function getDefaultSettings() {
  return {
    swarmSize: 10,
    networkSize: 2,
    inertiaWeight: 0.8,
    cognitiveConstant: 1.5,
    socialConstant: 1,
    maxVelocity: 0.1,
    dataset: './data/ga/train4dAll.txt',
  };
}
