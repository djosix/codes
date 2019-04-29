import Vec from 'victor';
import math from 'mathjs';

class Line {
  p1: Vec;
  p2: Vec;

  constructor(p1: Vec, p2: Vec) {
    this.p1 = p1;
    this.p2 = p2;
  }

  clone(): Line {
    return new Line(this.p1.clone(), this.p2.clone());
  }

  getLength(): number {
    return this.p1.distance(this.p2);
  }

  intersectWithRay(p: Vec, v: Vec): Vec {
    let v1 = this.p1.clone().subtract(p);
    let v2 = this.p2.clone().subtract(p);

    let div = v.x * (v2.y - v1.y) - v.y * (v2.x - v1.x);
    if (Math.abs(div) < 1e-10) {
      // Considered to be divided by zero
      return null;
    }

    // p_i = n * v
    let n = (v1.x * (v2.y - v1.y) - v1.y * (v2.x - v1.x)) / div;
    if (n < 0) {
      // Ray has reverse direction to the line
      return null;
    }

    // p_i = m * (v2 - v1)
    let m: number;

    if (v2.x != v1.x) {
      m = (n * v.x - v1.x) / (v2.x - v1.x);
    } else if (v2.y != v1.y) {
      m = (n * v.y - v1.y) / (v2.y - v1.y);
    } else {
      // Divided by zero
      return null;
    }

    if (m < 0 || m > 1) {
      // Out of the line
      return null;
    }

    return new Vec(v.x * n, v.y * n);
  }

  collidedWithCircle(center: Vec, radius: number): boolean {
    let v1 = this.p1.clone().subtract(center);
    let v2 = this.p2.clone().subtract(center);
    let v12 = this.p2.clone().subtract(this.p1);

    let lenSq1 = v1.lengthSq();
    let lenSq2 = v2.lengthSq();
    let lenSq12 = v12.lengthSq();
    let radiusSq = radius * radius;

    if (v1.dot(v12) >= 0) {
      return lenSq1 < radiusSq;
    } else if (v2.dot(v12) <= 0) {
      return lenSq2 < radiusSq;
    }

    let num = lenSq12 - lenSq1 + lenSq2;
    let den = 2 * Math.sqrt(lenSq12);
    let pow = Math.pow(num / den, 2);
    let chordDistanceSq = lenSq2 - pow;

    return chordDistanceSq < radiusSq;
  }
}

class Range {
  min: number = Infinity;
  max: number = -Infinity;

  constructor(range?: number[]) {
    if (range) {
      [this.min, this.max] = range;
    }
  }

  get size(): number {
    return this.max - this.min;
  }

  update(n: number): void {
    this.min = Math.min(this.min, n);
    this.max = Math.max(this.max, n);
  }
}

class Range2D {
  x = new Range();
  y = new Range();

  get size(): Vec {
    return new Vec(this.x.size, this.y.size);
  }

  update(vec: Vec) {
    this.x.update(vec.x);
    this.y.update(vec.y);
  }
}

class Transform {
  scale: number = 1;
  origin: number = 0;
  mirror: boolean = false;

  apply(value: number): number {
    let sign = this.mirror ? -1 : 1;
    return sign * this.scale * value + this.origin;
  }
}

class Transform2D {
  x: Transform = new Transform();
  y: Transform = new Transform();

  get origin(): Vec {
    return new Vec(this.x.origin, this.y.origin);
  }

  set origin(origin: Vec) {
    this.x.origin = origin.x;
    this.y.origin = origin.y;
  }

  set scale(scale: number) {
    this.x.scale = scale;
    this.y.scale = scale;
  }

  apply(v: Vec): Vec {
    return new Vec(this.x.apply(v.x), this.y.apply(v.y));
  }
}


let helpers = {
  degToRad(deg: number): number {
    return Math.PI * deg / 180;
  },

  radToDeg(rad: number): number {
    return 180 * rad / Math.PI;
  },

  clip(value: number, min: number, max: number): number {
    return Math.max(Math.min(value, max), min);
  },

  round(value: number, precision: number): number {
    let pow10 = Math.pow(10, precision);
    return Math.round(value * pow10) / pow10;
  },

  
  range(start: number, stop: number, interval: number) {
    let array = [];
    for (let i = start; i < stop + interval; i += interval) {
      array.push(i);
    }
    return array;
  },

  uniform(low = -1, high = 1): number {
    return Math.random() * (high - low) + low;
  },
  
  uniformArray(size: number, low = -1, high = 1): number[] {
    return math.add(math.multiply(math.random([size]), high - low), low) as number[];
  },

  randomDraw(size: number, k: number): number[] {
    if (k > size) throw Error('k must be less than or equal to size');

    let candidates = [];
    let selected = [];

    for (let i = 0; i < size; i++) {
      candidates.push(i);
    }

    while (selected.length < k) {
      let index = Math.random() * candidates.length;
      selected.push(candidates.splice(index, 1));
    }

    return selected;
  }
};

export {
  Vec, Line,
  Range, Range2D,
  Transform, Transform2D,
  helpers
}
