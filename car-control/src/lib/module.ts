// @ts-ignore
import Vue from 'vue';

abstract class Module {
  abstract name: string;
  abstract slug: string;
  component: Vue.Component;
  componentType: any;
  config: any = {};

  setComponent(c: Vue.Component) {
    this.component = c;
  }

  setComponentType(c: any) {
    this.componentType = c;
  }

  abstract init(): void;
  abstract predict(input: {front: number, left: number, right: number}): number;
}

export { Module };
