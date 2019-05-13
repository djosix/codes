import { FuzzySet, FuzzyVar, FuzzyRuleSet } from './fuzzy';

export function getDefaultVars() {
  return {
    X: new FuzzyVar('X', 'Front', [0, 40], 0.1, [
      new FuzzySet('S', 'Small', 0, 7.960000000000001, -1),
      new FuzzySet('M', 'Medium', 13.520000000000001, 16.759999999999998, 0),
      new FuzzySet('L', 'Large', 36.88000000000001, 20, 1),
    ]),
  
    Y: new FuzzyVar('Y', 'Right-Left', [-40, 40], 0.1, [
      new FuzzySet('S', 'Small', -19.479999999999997, 14.120000000000006, -1),
      new FuzzySet('M', 'Medium', 0.32000000000000384, 17.2, 0),
      new FuzzySet('L', 'Large', 4.7600000000000025, 3.800000000000001, 1),
    ]),
  
    Z: new FuzzyVar('Z', 'Steer', [-40, 40], 0.1, [
      new FuzzySet('S', 'Left', -18.159999999999997, 21.68, -1),
      new FuzzySet('M', 'Front', -5.828670879282072e-16, 4.24, 0),
      new FuzzySet('L', 'Right', 14.8, 16.039999999999992, 1),
    ]),
  };
}

export function getDefaultRuleSet() {
  let {X, Y, Z} = getDefaultVars();
  let rules = new FuzzyRuleSet([X, Y], Z);
  rules.set(['S', 'S'], 'S');
  rules.set(['S', 'M'], 'M');
  rules.set(['S', 'L'], 'L');
  rules.set(['M', 'S'], 'S');
  rules.set(['M', 'M'], 'M');
  rules.set(['M', 'L'], 'L');
  rules.set(['L', 'S'], 'S');
  rules.set(['L', 'M'], 'M');
  rules.set(['L', 'L'], 'L');
  return rules;
}

export function getDefaultMethods() {
  return {
    'implication': 'mamdani',
    'defuzzifier': 'center_of_gravity',
    'variable_composition': 'minimum',
    'rule_composition': 'maximum',
  };
}
