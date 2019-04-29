import math from 'mathjs';


class FuncLib {
  funcSets: FuncSet[];

  constructor(funcSets: FuncSet[]) {
    this.funcSets = funcSets;
  }

  find(slug: string): FuncSet {
    let funcSet = this.funcSets.find(s => (s.slug === slug));
    if (!(funcSet instanceof FuncSet)) {
      throw Error(`cannot find function set: ${slug}`);
    }
    return funcSet;
  }
}


class FuncSet {
  name: string;
  slug: string;
  funcs: Func[];

  constructor(name: string, slug: string, funcs: Func[]) {
    this.name = name;
    this.slug = slug;
    this.funcs = funcs;
  }

  find(slug: string): Func {
    let func = this.funcs.find(f => (f.slug === slug));
    if (!(func instanceof Func)) {
      throw Error(`cannot find function: ${slug}`);
    }
    return func;
  }
}


class Func {
  name: string;
  slug: string;
  f: Function;

  constructor(name: string, slug: string, f: Function) {
    this.name = name;
    this.slug = slug;
    this.f = f;
  }

  call(...args: any[]): number {
    return this.f(...args);
  }
}


let funcLib = new FuncLib([

  new FuncSet('Implication', 'implication', [
    new Func(
      'Dienes-Rescher (Kleene-Dienes)',
      'dienes_rescher',
      (x: number, y: number) => Math.max(1 - x, y)
    ),
    new Func(
      '&Lstrok;ukasieweicz',
      'lukasieweicz',
      (x: number, y: number) => Math.min(1, 1 - x + y)
    ),
    new Func(
      'Kleene-Dienes-&Lstrok;uk',
      'kleene_dienes_luk',
      (x: number, y: number) => 1 - x + x * y
    ),
    new Func(
      'Zadel',
      'zadel',
      (x: number, y: number) => Math.max(Math.min(x, y), 1 - x)
    ),
    new Func(
      'G&ouml;del',
      'godel',
      (x: number, y: number) => x <= y ? 1 : y
    ),
    new Func(
      'Gaines',
      'gaines',
      (x: number, y: number) => x <= y ? 1 : y / x
    ),
    new Func(
      'Goguen',
      'goguen',
      (x: number, y: number) => x < y ? 1 : y / x
    ),
    new Func(
      'Mamdani',
      'mamdani',
      (x: number, y: number) => Math.min(x, y)
    ),
    new Func(
      'Larsen (Product)',
      'larsen',
      (x: number, y: number) => x * y
    ),
    new Func(
      'Standard Strict',
      'standard_strict',
      (x: number, y: number) => x <= y ? 1 : 0
    ),
  ]),

  new FuncSet('Defuzzifier', 'defuzzifier', [
    new Func(
      'Center of Gravity',
      'center_of_gravity',
      (idxs: number[], vals: number[]) => {
        return math.dot(vals, idxs) / math.sum(vals);
      }
    ),

    new Func(
      'First of Maxima',
      'first_of_maxima',
      (idxs: number[], vals: number[]) => {
        var idx = 0;
        var max = -Infinity;

        for (let i = 0; i < idxs.length; i++) {
          let val = vals[i];
          if (val > max) {
            idx = i;
            max = val;
          }
        }

        return idxs[idx];
      }
    ),

    new Func(
      'Last of Maxima',
      'last_of_maxima',
      (idxs: number[], vals: number[]) => {
        var idx = 0;
        var max = -Infinity;

        for (let i = 0; i < idxs.length; i++) {
          let val = vals[i];
          if (val >= max) {
            idx = i;
            max = val;
          }
        }

        return idxs[idx];
      }
    ),

    new Func(
      'Mean of Maxima',
      'mean_of_maxima',
      (idxs: number[], vals: number[]) => {
        var is = [];
        var max = -Infinity;

        for (let i = 0; i < idxs.length; i++) {
          let val = vals[i];
          if (val > max) {
            is = [idxs[i]];
            max = val;
          } else if (val == max) {
            is.push(idxs[i]);
          }
        }

        return math.mean(is);
      }
    ),

    new Func(
      'Modified Mean of Maxima',
      'modified_mean_of_maxima',
      (idxs: number[], vals: number[]) => {
        let max = Math.max(...vals);
        let maxIdxs = idxs.filter((_, i) => vals[i] == max);
        return (maxIdxs[0] + maxIdxs[maxIdxs.length - 1]) / 2;
      }
    ),
  ]),

  new FuncSet('Variable Composition', 'variable_composition', [
    new Func(
      'Minimum (G&ouml;del)',
      'minimum',
      (x: number, y: number) => Math.min(x, y)),
    new Func(
      'Product',
      'product',
      (x: number, y: number) => x * y),
    new Func(
      '&Lstrok;ukasiewicz',
      'lukasiewicz',
      (x: number, y: number) => Math.max(0, x + y - 1)),
    new Func(
      'Drastic',
      'drastic',
      (x: number, y: number) => x == 1 ? y : (y == 1 ? x : 0)),
    new Func(
      'Nilpotent Minimum',
      'nilpotent_minimum',
      (x: number, y: number) => x + y > 1 ? Math.min(x, y) : 0),
    new Func(
      'Hamacher Minimum',
      'hamacher_product',
      (x: number, y: number) => x == 0 && y == 0 ? 0 : (x * y) / (x + y - x * y)),
  ]),

  new FuncSet('Rule Composition', 'rule_composition', [
    new Func(
      'Maximum',
      'maximum',
      (x: number, y: number) => Math.max(x, y)),
    new Func(
      'Probabilistic Sum',
      'probabilistic_sum',
      (x: number, y: number) => x + y - x * y),
    new Func(
      'Bounded Sum',
      'bounded_sum',
      (x: number, y: number) => Math.min(x + y, 1)),
    new Func(
      'Drastic',
      'drastic',
      (x: number, y: number) => x == 0 ? y : y == 0 ? x : 1),
    new Func(
      'Nilpotent Maximum',
      'nilpotent_maximum',
      (x: number, y: number) => x + y < 1 ? Math.max(x, y) : 1),
    new Func(
      'Einstein Sum',
      'einstein_sum',
      (x: number, y: number) => (x + y) / (1 + x * y)),
  ]),
]);

export default funcLib;
