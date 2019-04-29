import { Range } from '/lib/math';


class FuzzySet {
  symbol: string;
  name: string;
  mean: number;
  std: number;
  ext: number;

  constructor(symbol: string, name: string, mean: number, std: number, ext: number) {
    this.symbol = symbol;
    this.name = name;
    this.mean = mean;
    this.std = std;
    this.ext = ext;
  }

  /**
   * RBF membership function
   * @param x fuzzy singelton
   */
  mf(x: number): number {
    if ((x - this.mean) * this.ext > 0) {
      // Entended
      return 1;
    } else {
      // RBF
      return Math.exp(- Math.pow(x - this.mean, 2) / (2 * this.std * this.std));
    }
  }
}


class FuzzyVar {
  symbol: string;
  name: string;
  sets: FuzzySet[];
  range: Range;
  int: number;

  constructor(symbol: string, name: string, range: number[], int: number, sets: FuzzySet[]) {
    this.symbol = symbol;
    this.name = name;
    this.range = new Range(range);
    this.int = int;
    this.sets = sets;
  }

  find(symbol: string): FuzzySet {
    let fuzSet = this.sets.find(s => (s.symbol == symbol));
    if (!(fuzSet instanceof FuzzySet)) {
      throw Error(`cannot find fuzzy set: ${symbol}`);
    }
    return fuzSet;
  }
}


class FuzzyRuleSet {
  data: any;
  vars: any;

  constructor(inVars: FuzzyVar[], outVar: FuzzyVar) {
    this.vars = {};
    for (let fuzVar of inVars) {
      this.vars[fuzVar.symbol] = fuzVar;
    }
    this.vars[outVar.symbol] = outVar;

    let construct = (fuzVars: FuzzyVar[]) => {
      if (fuzVars.length > 0) {
        let data = {};
        let fuzVar = fuzVars.shift();
        for (let fuzSet of fuzVar.sets) {
          data[fuzSet.symbol] = construct([...fuzVars]);
        }
        return data;
      }
      return null;
    };

    this.data = construct([...inVars]);
  }

  set(inSyms: string[], outSym: string) {
    let recurSet = (data: any, inSyms: string[], outSym: string) => {
      if (inSyms.length > 1) {
        let inSym = inSyms.shift();
        recurSet(data[inSym], [...inSyms], outSym);
      } else if (inSyms.length == 1) {
        data[inSyms[0]] = outSym;
      }
    };
    
    recurSet(this.data, [...inSyms], outSym);
  }

  get(inSyms: string[]) {
    let recurGet = (data: any, inSyms: string[]) => {
      if (inSyms.length > 1) {
        let inSym = inSyms.shift();
        return recurGet(data[inSym], [...inSyms]);
      } else if (inSyms.length == 1) {
        return data[inSyms[0]];
      }
    };

    return recurGet(this.data, [...inSyms]);
  }
}


export { FuzzySet, FuzzyVar, FuzzyRuleSet };
