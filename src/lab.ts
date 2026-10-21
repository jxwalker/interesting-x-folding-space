/**
 * The live experiment, still with no rendering: one dataset, one network,
 * training in small per-frame batches, and the per-layer "views" the renderers
 * draw (bead and space-grid positions in display coordinates, smoothed so the
 * picture drifts rather than jumps), plus the meters (linear probe, directions
 * used) and the decision sheet.
 */
import {
  DEFAULT_NOISE,
  MLP,
  Probe,
  covariance,
  eigSym,
  makeDataset,
  makeGrid,
  participationRatio,
  project,
  projectionBasis,
  type Act,
  type Dataset,
  type NetConfig,
  type ShapeId,
  type Trace,
} from "./nn";

/** One keyframe of the walk: a layer's linear part ("pre") or after the bend ("post"). */
export interface Stage {
  kind: "pre" | "post";
  /** View index: 0 = input, 1..depth = hidden layers (depth = "Output"). */
  view: number;
  /** Position on the layer slider. */
  s: number;
  /** Neurons in this layer. */
  dim: number;
  /** Bead positions, n × 3, in display units (fit to [-1, 1]³). */
  data: Float32Array;
  /** Space-grid sample points, G × 3, display units (may leave the cube). */
  grid: Float32Array;
  // smoothed display normalisation: display = (basis·h − c) ∘ k
  c: Float64Array;
  k: Float64Array;
  basis: Float64Array | null;
  fresh: boolean;
}

export interface Plane {
  /** Unit normal and offset in output-view display coords: n·p + d = 0. */
  n: [number, number, number];
  d: number;
  ok: boolean;
}

export const STEPS_PER_SECOND = 60;
/** Auto-pause after this many steps; the picture has settled long before. */
export const SETTLE_STEPS = 2400;
/** Early stopping once every bead is right and the loss is below this. */
export const EARLY_STOP_LOSS = 0.06;
export const GRID_LINES = 7;
/** Per-axis fit: a squashed axis may be magnified up to 1/ANISO× relative to the widest one. */
export const ANISO = 0.12;
export const GRID_SAMPLES = 21;

export class Lab {
  shape: ShapeId = "twisted";
  cfg: NetConfig = { width: 3, depth: 4, act: "tanh" };
  noise = DEFAULT_NOISE;
  lr = 0.02;
  seed = 1;
  ds!: Dataset;
  net!: MLP;
  readonly grid = makeGrid(GRID_LINES, GRID_SAMPLES, 1);
  stages: Stage[] = [];
  /** Index into `stages` of each view's "post" keyframe. */
  viewStage: number[] = [];
  training = true;
  loss = Math.log(2);
  acc = 0.5;
  /** Probability of teal for every bead / every grid point (output). */
  pred = new Float32Array(0);
  gridPred = new Float32Array(0);
  /** Linear-probe accuracy per view, participation ratio per view, normalised eigenvalues per view. */
  probe: number[] = [];
  dims: number[] = [];
  eig: number[][] = [];
  lossHist: number[] = [];
  plane: Plane = { n: [1, 0, 0], d: 0, ok: false };
  /** Bumped on every rebuild (new data / architecture) so renderers can reallocate. */
  version = 0;
  /** Lab time (s) at the last scramble (start over / new shape). */
  scrambleAt = -10;
  time = 0;
  /** Highest accuracy reached since the last reset, and the step it was first > 95 %. */
  best = 0;
  solvedAt = -1;
  private probes: Probe[] = [];
  /** Why training stopped by itself: "solved" (early stopping) or "budget" (step cap). */
  stopReason: "" | "solved" | "budget" = "";
  private earlyStop = true;
  private budgetFrom = 0;
  private stepCarry = 0;
  private probeT = 0;
  private traceD: Trace | null = null;
  private traceG: Trace | null = null;
  private dirty = true;

  constructor() {
    this.rebuild();
  }

  get depth(): number {
    return this.cfg.depth;
  }

  get steps(): number {
    return this.net.steps;
  }

  get settled(): boolean {
    return this.net.steps - this.budgetFrom >= SETTLE_STEPS;
  }

  /** Resume after an automatic stop: a fresh step budget, and no early stop this time. */
  resume(): void {
    if (this.stopReason) {
      this.budgetFrom = this.net.steps;
      this.earlyStop = false;
    }
    this.stopReason = "";
    this.training = true;
  }

  /** New data + new random network (keeps settings). */
  rebuild(seed = this.seed): void {
    this.seed = seed;
    this.ds = makeDataset(this.shape, seed, this.noise);
    this.net = new MLP(this.cfg, seed);
    this.stages = [];
    this.viewStage = [];
    const sizes = this.net.sizes;
    const n = this.ds.n, G = this.grid.count;
    for (let v = 0; v <= this.depth; v++) {
      const dim = sizes[v];
      if (v > 0 && dim === 3) this.stages.push(this.makeStage("pre", v, v - 0.5, dim, n, G));
      this.viewStage.push(this.stages.length);
      this.stages.push(this.makeStage("post", v, v, dim, n, G));
    }
    this.probes = sizes.slice(0, this.depth + 1).map((d) => new Probe(d));
    this.pred = new Float32Array(n);
    this.gridPred = new Float32Array(G);
    this.lossHist = [];
    this.traceD = this.traceG = null;
    this.stepCarry = 0;
    this.stopReason = "";
    this.earlyStop = true;
    this.budgetFrom = 0;
    this.best = 0;
    this.solvedAt = -1;
    this.scrambleAt = this.time;
    this.version++;
    this.dirty = true;
    this.refresh(true);
    this.updateMeters(60);
  }

  private makeStage(kind: "pre" | "post", view: number, s: number, dim: number, n: number, G: number): Stage {
    return { kind, view, s, dim, data: new Float32Array(n * 3), grid: new Float32Array(G * 3), c: new Float64Array(3), k: new Float64Array([1, 1, 1]), basis: null, fresh: true };
  }

  setShape(shape: ShapeId): void {
    this.shape = shape;
    this.training = true;
    this.rebuild(this.seed);
  }

  setConfig(patch: Partial<NetConfig>): void {
    this.cfg = { ...this.cfg, ...patch };
    this.training = true;
    this.rebuild(this.seed);
  }

  setNoise(noise: number): void {
    this.noise = noise;
    // same network, new beads: it keeps learning from where it was
    this.ds = makeDataset(this.shape, this.seed, noise);
    this.traceD = null;
    this.dirty = true;
    this.refresh(false);
  }

  setAct(act: Act): void {
    this.setConfig({ act });
  }

  /** Start over: a fresh random network (and fresh jitter on the beads). */
  restart(): void {
    this.training = true;
    this.rebuild(this.seed + 1);
  }

  /**
   * Advance lab time. `fixedSteps` (stepped/record mode) trains exactly that
   * many steps; otherwise the step count follows wall time so training keeps
   * pace even when rendering is slow.
   */
  update(dt: number, fixedSteps?: number): void {
    this.time += dt;
    let k = 0;
    if (this.training) {
      if (fixedSteps !== undefined) k = fixedSteps;
      else {
        this.stepCarry += Math.min(dt, 0.5) * STEPS_PER_SECOND;
        k = Math.min(40, Math.floor(this.stepCarry));
        this.stepCarry -= k;
      }
      for (let i = 0; i < k; i++) {
        const r = this.net.step(this.ds, this.lr);
        if (!Number.isFinite(r.loss)) {
          // should never happen (clipping + stable BCE), but never show NaN: restart cleanly
          this.rebuild(this.seed + 1);
          return;
        }
        this.loss = r.loss;
        this.acc = r.acc;
        if (this.net.steps % 4 === 0) {
          this.lossHist.push(r.loss);
          if (this.lossHist.length > 240) this.lossHist.shift();
        }
        if (r.acc > this.best) this.best = r.acc;
        if (this.solvedAt < 0 && r.acc >= 0.95) this.solvedAt = this.net.steps;
        if (this.settled) {
          this.training = false;
          this.stopReason = "budget";
          break;
        }
        // early stopping: every bead right and confidently so; more steps would only
        // saturate the layers further without changing a single answer
        if (this.earlyStop && r.acc >= 0.995 && r.loss < EARLY_STOP_LOSS) {
          this.training = false;
          this.stopReason = "solved";
          break;
        }
      }
      if (k > 0) this.dirty = true;
    }
    if (this.dirty) this.refresh(false);
    this.probeT += dt;
    if (this.probeT >= 0.5) {
      this.probeT = 0;
      this.updateMeters(this.training ? 30 : 60);
    }
  }

  /** Recompute every stage's display positions (and output predictions). */
  refresh(snap: boolean): void {
    this.dirty = false;
    const net = this.net, ds = this.ds, g = this.grid;
    const n = ds.n, G = g.count, L = net.layers;
    const trD = (this.traceD = net.forward(ds.X, n, this.traceD));
    const trG = (this.traceG = net.forward(g.pts, G, this.traceG));
    const zD = trD.post[L], zG = trG.post[L];
    let correct = 0, loss = 0;
    for (let i = 0; i < n; i++) {
      const p = 1 / (1 + Math.exp(-zD[i]));
      this.pred[i] = p;
      if ((p > 0.5 ? 1 : 0) === ds.y[i]) correct++;
      loss += Math.max(zD[i], 0) - ds.y[i] * zD[i] + Math.log1p(Math.exp(-Math.abs(zD[i])));
    }
    this.acc = correct / n;
    this.loss = loss / n;
    for (let i = 0; i < G; i++) this.gridPred[i] = 1 / (1 + Math.exp(-zG[i]));

    for (const st of this.stages) {
      const d = st.dim;
      const H = st.kind === "post" ? trD.post[st.view] : trD.pre[st.view - 1];
      const HG = st.kind === "post" ? trG.post[st.view] : trG.pre[st.view - 1];
      const { basis } = projectionBasis(H, n, d, st.basis);
      st.basis = basis;
      const P = project(H, n, d, basis);
      const PG = project(HG, G, d, basis);
      // fit to the cube: centre on the bead cloud's box, per-axis scale with limited anisotropy
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
          const v = P[i * 3 + a];
          if (v < lo[a]) lo[a] = v;
          if (v > hi[a]) hi[a] = v;
        }
      }
      const half = [0, 1, 2].map((a) => Math.max(1e-6, (hi[a] - lo[a]) / 2));
      const hmax = Math.max(...half);
      const blend = snap || st.fresh ? 1 : 0.18;
      st.fresh = false;
      for (let a = 0; a < 3; a++) {
        const cT = (hi[a] + lo[a]) / 2;
        const kT = 0.9 / Math.max(half[a], hmax * ANISO, 1e-3);
        st.c[a] += (cT - st.c[a]) * blend;
        st.k[a] = Math.exp(Math.log(st.k[a]) + (Math.log(kT) - Math.log(st.k[a])) * blend);
        // never let the smoothing push beads out of the cube: shrink at once if needed
        const reach = Math.max(hi[a] - st.c[a], st.c[a] - lo[a], 1e-9) * st.k[a];
        if (reach > 0.97) st.k[a] *= 0.97 / reach;
      }
      // a < 3-wide layer: keep the unused axes flat at the centre
      for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) st.data[i * 3 + a] = (P[i * 3 + a] - st.c[a]) * st.k[a];
      for (let i = 0; i < G; i++) for (let a = 0; a < 3; a++) st.grid[i * 3 + a] = (PG[i * 3 + a] - st.c[a]) * st.k[a];
    }

    // decision sheet: w·h + b = 0 in the final hidden layer (always 3 wide, identity basis)
    const out = this.stages[this.viewStage[this.depth]];
    const w = net.W[L - 1], b = net.b[L - 1][0];
    const nx = w[0] / out.k[0], ny = w[1] / out.k[1], nz = w[2] / out.k[2];
    const off = w[0] * out.c[0] + w[1] * out.c[1] + w[2] * out.c[2] + b;
    const len = Math.hypot(nx, ny, nz);
    this.plane = len > 1e-9 ? { n: [nx / len, ny / len, nz / len], d: off / len, ok: true } : { n: [1, 0, 0], d: 0, ok: false };
  }

  /** Linear probe + covariance spectrum per view. */
  updateMeters(probeSteps: number): void {
    const tr = this.traceD ?? this.net.forward(this.ds.X, this.ds.n);
    const n = this.ds.n;
    this.probe = [];
    this.dims = [];
    this.eig = [];
    for (let v = 0; v <= this.depth; v++) {
      const d = this.net.sizes[v];
      const H = tr.post[v];
      this.probe.push(this.probes[v].fit(H, n, this.ds.y, probeSteps));
      const { cov } = covariance(H, n, d);
      const { vals } = eigSym(cov, d);
      this.dims.push(participationRatio(vals));
      const top = vals[0] || 1;
      this.eig.push(Array.from(vals, (x) => x / top));
    }
  }

  /** Slider range: 0 (input) … depth (output). */
  stageAt(s: number): { a: Stage; b: Stage; f: number } {
    const st = this.stages;
    if (s <= st[0].s) return { a: st[0], b: st[0], f: 0 };
    for (let i = 0; i < st.length - 1; i++) {
      if (s <= st[i + 1].s) {
        const f = (s - st[i].s) / (st[i + 1].s - st[i].s);
        return { a: st[i], b: st[i + 1], f: f * f * (3 - 2 * f) };
      }
    }
    const last = st[st.length - 1];
    return { a: last, b: last, f: 0 };
  }

  /** Interpolated bead and grid positions at slider position s. */
  sample(s: number, outData: Float32Array, outGrid: Float32Array | null): void {
    const { a, b, f } = this.stageAt(s);
    const g = 1 - f;
    for (let i = 0; i < outData.length; i++) outData[i] = a.data[i] * g + b.data[i] * f;
    if (outGrid) for (let i = 0; i < outGrid.length; i++) outGrid[i] = a.grid[i] * g + b.grid[i] * f;
  }

  /** The keyframe positions of bead i through every stage (for tracing its journey). */
  journey(i: number): [number, number, number][] {
    return this.stages.map((st) => [st.data[i * 3], st.data[i * 3 + 1], st.data[i * 3 + 2]]);
  }

  viewName(v: number): string {
    return v === 0 ? "Input" : v === this.depth ? "Output" : `Layer ${v}`;
  }

  finite(): boolean {
    if (!this.net.finite() || !Number.isFinite(this.loss)) return false;
    for (const st of this.stages) {
      for (const x of st.data) if (!Number.isFinite(x)) return false;
    }
    return true;
  }
}
