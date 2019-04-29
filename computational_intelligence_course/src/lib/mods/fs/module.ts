import { Module } from '/lib/module';
import { FuzzyVar, FuzzySet, FuzzyRuleSet } from './fuzzy';
import funcLib from '/lib/mods/fs/funcs';
import math from 'mathjs';


class FuzzySystem extends Module {
  name = 'Fuzzy System';
  slug = 'fuzzy_system';

  compVars: Function;
  imply: Function;
  compRules: Function;
  defuzz: Function;
  fuzVars: {X: FuzzyVar, Y: FuzzyVar, Z: FuzzyVar};
  rules: FuzzyRuleSet;

  constructor() {
    super();
  }

  init() {
    // @ts-ignore
    this.config = this.component.collectConfig();

    this.defuzz =
      funcLib
        .find('defuzzifier')
        .find(this.config.methods.defuzzifier)
        .f;

    this.imply =
      funcLib
        .find('implication')
        .find(this.config.methods.implication)
        .f;

    this.compVars =
      funcLib
        .find('variable_composition')
        .find(this.config.methods.variable_composition)
        .f;

    this.compRules =
      funcLib
        .find('rule_composition')
        .find(this.config.methods.rule_composition)
        .f;

    this.fuzVars = this.config.vars;
    console.assert(this.fuzVars);

    this.rules = this.config.rules;
    console.assert(this.rules);
  }

  predict(inputs: {front: number, left: number, right: number}) {
    let { front, left, right } = inputs;

    let x = front;
    let y = right - left;

    let {X, Y, Z} = this.fuzVars;

    let zOuts = [];
    for (let [symX, muX] of X.sets.map(s => [s.symbol, s.mf(x)])) {
      for (let [symY, muY] of Y.sets.map(s => [s.symbol, s.mf(y)])) {
        // Rule inference
        let symZ = this.rules.get([symX as string, symY as string]);
        let setZ = Z.find(symZ);

        // Variable composition
        let alpha = this.compVars(muX, muY);

        // Implication
        zOuts.push((z: number) => this.imply(alpha, setZ.mf(z)));
      }
    }

    // @ts-ignore
    let zIdxs = <number[]>math
      .range(Z.range.min, Z.range.max + Z.int, Z.int)
      .toArray();

    let zVals = zIdxs.map(z =>
      zOuts
        .map(f => f(z))
        // @ts-ignore
        // Rule composition
        .reduce(this.compRules)
    );

    // Defuzzification
    let z = this.defuzz(zIdxs, zVals);

    return z;
  }
}

export default new FuzzySystem();
export { FuzzySystem };
