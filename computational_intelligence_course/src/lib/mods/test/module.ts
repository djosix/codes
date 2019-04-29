import { Module } from '/lib/module';

class Test extends Module {
  name = 'Test';
  slug = 'test';

  constructor() {
    super();
  }

  init() {
    this.config = {
      // @ts-ignore
      steeringAngle: this.component.steeringAngle
    };
  }

  predict(inputs: {front: number, left: number, right: number}) {
    let {front, left, right} = inputs;
    // @ts-ignore
    let steerDeg = eval(this.config.steeringAngle);
    return steerDeg;
  }
}

export default new Test();
export { Test };
