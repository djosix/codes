import { Vec, Line, Range2D, helpers } from './math';
import { readTextFile, writeTextFile, createDirectory } from './fs';
import context from './context';

class Car {
  public position: Vec;
  public angleDeg: number;

  public readonly radius = 3;
  public readonly radarRelDegs = { front: 0, left: 45, right: -45 };

  constructor(position: Vec, angleDeg: number) {
    this.position = position;
    this.angleDeg = angleDeg;
  }

  public getRadarDegs(): Object {
    let absDegs = {};
    for (let [name, relDeg] of Object.entries(this.radarRelDegs)) {
      absDegs[name] = this.angleDeg + relDeg;
    }
    return absDegs;
  }

  public step(steerDeg: number): void {
    steerDeg = helpers.clip(steerDeg, -40, 40);

    let angleRad = helpers.degToRad(this.angleDeg);
    let steerRad = helpers.degToRad(steerDeg);

    this.position.x += Math.cos(angleRad + steerRad);
    this.position.x += Math.sin(steerRad) * Math.sin(angleRad);

    this.position.y += Math.sin(angleRad + steerRad);
    this.position.y -= Math.sin(steerRad) * Math.cos(angleRad);

    this.angleDeg -= helpers.radToDeg(Math.asin(2 * Math.sin(steerRad) / this.radius));
  }
}


class Env {

  path: string;

  data: {
    initialState: {
      position: Vec,
      angleDeg: number,
    },
    walls: Line[],
    wallPts: Vec[],
    goalArea: {
      p1: Vec,
      p2: Vec,
      edges: Line[],
    }
  };

  state: {
    step: number;
    radarVals: {
      front: number,
      left: number,
      right: number
    },
    radarRelPts: {
      front: Vec,
      left: Vec,
      right: Vec
    },
    trajectory: Vec[],
    result: string,
  };

  car: Car;
  mapRange = new Range2D();
  history = [];

  constructor(path: string) {
    this.loadData(path);
    this.reset();
  }

  public reset(): void {
    this.car = new Car(
      this.data.initialState.position.clone(),
      this.data.initialState.angleDeg
    );

    this.state = {
      step: 0,
      radarVals: {
        front: 0,
        left: 0,
        right: 0,
      },
      radarRelPts: {
        front: null,
        left: null,
        right: null,
      },
      trajectory: [],
      result: null,
    };

    this.history = [];
  }

  public *start() {
    this.state.result = null;

    while (true) {
      this.state.trajectory.push(this.car.position.clone());
      this.updateRadarStatus();

      if (this.isCollided()) {
        this.state.result = 'collided';
        break;
      }

      if (this.isFinished()) {
        this.state.result = 'finished';
        break;
      }

      let steerDeg = yield

      this.history.push({
        front: this.state.radarVals.front,
        left: this.state.radarVals.left,
        right: this.state.radarVals.right,
        steer: steerDeg,
        x: this.car.position.x,
        y: this.car.position.y
      });

      this.car.step(steerDeg);

      this.state.step++;
    }
  }

  private isCollided(): boolean {
    for (let line of this.data.walls) {
      if (line.collidedWithCircle(this.car.position, this.car.radius)) {
        return true
      }
    }
    return false;
  }

  private isFinished(): boolean {
    for (let line of this.data.goalArea.edges) {
      if (line.collidedWithCircle(this.car.position, this.car.radius)) {
        return true;
      }
    }
    return false;
  }

  private updateRadarStatus(): void {
    let radarDegs = this.car.getRadarDegs();
    for (let [key, deg] of Object.entries(radarDegs)) {
      let radarRad = helpers.degToRad(deg);
      let radarVec = new Vec(Math.cos(radarRad), Math.sin(radarRad));
      let nearest: Vec = null;
      let minLenSq: number = Infinity;

      for (let line of this.data.walls) {
        let intersectPt = line.intersectWithRay(this.car.position, radarVec);
        if (intersectPt != null) {
          let lenSq = intersectPt.lengthSq();
          if (lenSq < minLenSq) {
            nearest = intersectPt;
            minLenSq = lenSq;
          }
        }
      }

      this.state.radarVals[key] = Math.sqrt(minLenSq);
      this.state.radarRelPts[key] = nearest;
    }
  }

  private loadData(path: string) {
    this.path = path;

    let data: any = {};

    let tuples: number[][] =
      readTextFile(path)
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map(line => line.split(',').map(Number));

    // Car initial position and angle

    let [x, y, angleDeg] = tuples.shift();
    let p = new Vec(x, y);

    data.initialState = {
      position: p,
      angleDeg: angleDeg
    };

    // Goal area
    let p1 = Vec.fromArray(tuples.shift());
    let p2 = Vec.fromArray(tuples.shift());

    let r = new Range2D();
    r.update(p1);
    r.update(p2);

    data.goalArea = {
      p1: p1,
      p2: p2,
      edges: [
        new Line(new Vec(r.x.min, r.y.min), new Vec(r.x.max, r.y.min)),
        new Line(new Vec(r.x.min, r.y.min), new Vec(r.x.min, r.y.max)),
        new Line(new Vec(r.x.max, r.y.min), new Vec(r.x.max, r.y.max)),
        new Line(new Vec(r.x.min, r.y.max), new Vec(r.x.max, r.y.max)),
      ]
    };

    // Walls

    data.walls = []
    data.wallPts = [];
    
    let lastPoint: Vec = null;

    while (tuples.length > 0) {
      let currPoint = Vec.fromArray(tuples.shift());
      this.mapRange.update(currPoint);

      data.wallPts.push(currPoint);
      if (lastPoint != null) {
        data.walls.push(new Line(lastPoint, currPoint));
      }
      lastPoint = currPoint;
    }
    
    this.data = data;
  }

  public saveHist(path: string) {
    // train4D
    let train4D =
      this.history
        .map(item => `${item.front} ${item.right} ${item.left} ${item.steer}`)
        .reduce((s1, s2) => `${s1}\n${s2}`);
      
    // train6D
    let train6D =
      this.history
        .map(item => `${item.x} ${item.y} ${item.front} ${item.right} ${item.left} ${item.steer}`)
        .reduce((s1, s2) => `${s1}\n${s2}`);

    writeTextFile(`${path}/train4D.txt`, train4D);
    writeTextFile(`${path}/train6D.txt`, train6D);

    context.emit('logInfo', `History saved: ${path}`);
  }
}

export {
  Car,
  Env,
};
