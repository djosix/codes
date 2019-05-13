import { Vec, Range2D, Transform2D, helpers } from './math';
import { Env } from './env';


class Renderer {
  protected canvas: HTMLCanvasElement;
  protected transform: Transform2D;
  protected scale: number = 1;

  public ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.ctx.save();

    this.transform = new Transform2D();
    this.transform.y.mirror = true;
  }

  public fitRange(range: Range2D, padding: number = 10) {
    let rangeWidth = range.x.size;
    let rangeHeight = range.y.size;
    let canvasWidth = this.canvas.clientWidth;
    let canvasHeight = this.canvas.clientHeight;

    if (rangeHeight / rangeWidth > canvasHeight / canvasWidth) {
      // Fit range height
      let newHeight = canvasHeight - padding * 2;
      this.scale = newHeight / rangeHeight;
      let paddingX = (canvasWidth - this.scale * rangeWidth) / 2;
      this.transform.origin = new Vec(
        paddingX - range.x.min * this.scale,
        padding + range.y.max * this.scale
      );
      this.transform.scale = this.scale;
    } else {
      // Fit range width
      let newWidth = canvasWidth - padding * 2;
      this.scale = newWidth / rangeWidth;
      let paddingY = (canvasHeight - this.scale * rangeHeight) / 2;
      this.transform.origin = new Vec(
        padding - range.x.min * this.scale,
        paddingY + range.y.max * this.scale
      );
      this.transform.scale = this.scale;
    }
  }

  t(p: Vec): Vec {
    // Convert virtual position into canvas position
    return this.transform.apply(p);
  }

  render(env: Env) {
    this.clear();
    this.renderAxes();
    this.renderGoal(env);
    this.renderWalls(env);
    this.renderTraj(env);
    this.renderCar(env);
  }

  clear() {
    // Clear the canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  renderAxes() {
    // Render X and Y axes
    let center = this.transform.origin;
    this.ctx.save();
    this.ctx.strokeStyle = '#ddd';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(0, center.y);
    this.ctx.lineTo(this.canvas.width, center.y);
    this.ctx.moveTo(center.x, 0);
    this.ctx.lineTo(center.x, this.canvas.height);
    this.ctx.stroke();
    this.ctx.restore();
  }

  renderWalls(env: Env) {
    // Render walls
    this.ctx.save();
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    let begin = true;
    for (let p of env.data.wallPts) {
      p = this.t(p)
      if (begin) {
        this.ctx.moveTo(p.x, p.y);
        begin = false;
      } else {
        this.ctx.lineTo(p.x, p.y);
      }
    }
    this.ctx.stroke();
    this.ctx.restore();

    // Render radar dots
    if (env.state.radarRelPts) {
      this.ctx.save();
      // @ts-ignore
      for (let relPt of Object.values(env.state.radarRelPts)) {
        if (!relPt) {
          continue;
        }
        let p = env.car.position.clone().add(relPt);
        if (!p) {
          continue;
        }
        p = this.t(p);
        this.ctx.fillStyle = 'black';
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
        this.ctx.fill();
      }
      this.ctx.restore();
    }
  }

  renderGoal(env: Env) {
    // Render goal rect
    this.ctx.save();
    let [p1, p2] = [
      this.t(env.data.goalArea.p1),
      this.t(env.data.goalArea.p2)
    ];
    let [w, h] = [p2.x - p1.x, p2.y - p1.y];
    this.ctx.beginPath();
    this.ctx.lineWidth = 0;
    this.ctx.fillStyle = '#ddd';
    this.ctx.rect(p1.x, p1.y, w, h);
    this.ctx.fill();
    this.ctx.restore();
  }

  renderCar(env: Env) {
    // Render car
    this.ctx.save();
    let p = this.t(env.car.position);
    let frontAngle = helpers.degToRad(-env.car.angleDeg);
    this.ctx.lineWidth = 1;
    this.ctx.strokeStyle = 'black';
    this.ctx.beginPath();
    this.ctx.arc(p.x, p.y, env.car.radius * this.scale, frontAngle, frontAngle + 2 * Math.PI);
    this.ctx.lineTo(p.x, p.y);
    this.ctx.stroke();
    this.ctx.restore();

    // Render radar lines
    if (env.state.radarRelPts) {
      // @ts-ignore
      for (let relPt of Object.values(env.state.radarRelPts)) {
        let pL = relPt;
        if (!pL) {
          continue;
        }
        pL = this.t(pL.clone().add(env.car.position));
        this.ctx.setLineDash([4, 4]);
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = 'black'
        this.ctx.beginPath();
        this.ctx.moveTo(p.x, p.y);
        this.ctx.lineTo(pL.x, pL.y);
        this.ctx.stroke();
        this.ctx.setLineDash([1, 0]);
      }
    }
  }

  renderTraj(env: Env) {
    this.ctx.save();
    this.ctx.lineWidth = 0.5;
    this.ctx.strokeStyle = 'gray';
    this.ctx.setLineDash([2, 2]);
    this.ctx.beginPath();
    let begin = true;
    for (let p of env.state.trajectory) {
      p = this.t(p);
      if (begin) {
        this.ctx.moveTo(p.x, p.y)
        begin = false;
      } else {
        this.ctx.lineTo(p.x, p.y);
      }
    }
    this.ctx.stroke();
    this.ctx.restore();
  }
}

export { Renderer };
