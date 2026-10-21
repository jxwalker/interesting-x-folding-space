/**
 * The science, with no rendering: seeded RNG, the three datasets, a small
 * multilayer perceptron (forward, backprop, Adam, Xavier init, gradient
 * clipping), symmetric eigen-decomposition (Jacobi), PCA with frame-to-frame
 * sign/order alignment, a warm-started linear probe, the participation ratio,
 * the space-grid lattice and a headless self-test.
 *
 * Everything here is plain TypeScript over typed arrays so it can run in a
 * browser frame loop, a worker or Node.
 */

export type Act = "tanh" | "relu" | "leaky";
export type ShapeId = "twisted" | "rings" | "shell";

// ───────────────────────── random numbers ─────────────────────────

/** mulberry32: tiny, fast, seedable PRNG returning floats in [0, 1). */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0 || 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rng: () => number): number {
  let u = 0;
  while (u < 1e-12) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ───────────────────────── datasets ─────────────────────────

export interface Dataset {
  shape: ShapeId;
  n: number;
  /** n × 3, row-major, inside roughly [-1, 1]³. */
  X: Float64Array;
  /** 0 = amber (class A), 1 = teal (class B). */
  y: Uint8Array;
  /** Position along its strand / ring in [0, 1) (used to order beads for tracing). */
  t: Float64Array;
}

/** Geometry knobs, tuned by measurement (see selfTest and README). */
export const GEOM = {
  helix: { radius: 0.52, halfLength: 0.9, turns: 1.25 },
  rings: { radius: 0.5, offset: 0.5, tube: 0.22 },
  shell: { ball: 0.36, shell: 0.86 },
};

export const DEFAULT_NOISE = 0.02;
export const DEFAULT_N = 400;

export function makeDataset(shape: ShapeId, seed: number, noise = DEFAULT_NOISE, n = DEFAULT_N): Dataset {
  const rng = makeRng(seed * 7919 + 17);
  const X = new Float64Array(n * 3);
  const y = new Uint8Array(n);
  const t = new Float64Array(n);
  const half = n >> 1;
  for (let i = 0; i < n; i++) {
    const cls = i < half ? 0 : 1;
    const k = cls ? i - half : i;
    const m = cls ? n - half : half;
    // evenly spaced along the curve with a little jitter, so strands read as bead strings
    const u = (k + 0.5 + (rng() - 0.5) * 0.6) / m;
    let px = 0, py = 0, pz = 0;
    if (shape === "twisted") {
      const { radius, halfLength, turns } = GEOM.helix;
      const a = u * turns * 2 * Math.PI + (cls ? Math.PI : 0);
      px = -halfLength + 2 * halfLength * u;
      py = radius * Math.cos(a);
      pz = radius * Math.sin(a);
    } else if (shape === "rings") {
      // Two chain links: A lies in the xy-plane, B in the xz-plane; each passes through the other's hole.
      const { radius, offset } = GEOM.rings;
      const { tube } = GEOM.rings;
      const a = u * 2 * Math.PI;
      // a solid tube cross-section (uniform disc) around the ring's centre line
      const rr = tube * Math.sqrt(rng()), ph = rng() * 2 * Math.PI;
      const R = radius + rr * Math.cos(ph), off = rr * Math.sin(ph);
      if (cls === 0) {
        px = -offset / 2 + R * Math.cos(a);
        py = R * Math.sin(a);
        pz = off;
      } else {
        px = offset / 2 - R * Math.cos(a);
        py = off;
        pz = R * Math.sin(a);
      }
    } else {
      // A solid ball (A) inside a closed spherical shell (B); Fibonacci-sphere spacing.
      const { ball, shell } = GEOM.shell;
      const g = Math.PI * (3 - Math.sqrt(5));
      const zz = 1 - 2 * u;
      const rr = Math.sqrt(Math.max(0, 1 - zz * zz));
      const r = cls === 0 ? ball * Math.cbrt(0.08 + 0.92 * rng()) : shell;
      px = r * rr * Math.cos(g * k);
      py = r * zz;
      pz = r * rr * Math.sin(g * k);
    }
    X[i * 3] = px + noise * gauss(rng);
    X[i * 3 + 1] = py + noise * gauss(rng);
    X[i * 3 + 2] = pz + noise * gauss(rng);
    y[i] = cls;
    t[i] = u;
  }
  return { shape, n, X, y, t };
}

// ───────────────────────── the network ─────────────────────────

export interface NetConfig {
  /** Neurons per hidden layer ("room to move"), 2–8. */
  width: number;
  /** Hidden layers ("folds"), 2–6. The last one is always 3 wide. */
  depth: number;
  act: Act;
}

/** Layer sizes: 3 → W × (depth−1) → 3 → 1. */
export function layerSizes(c: NetConfig): number[] {
  const s = [3];
  for (let i = 0; i < c.depth - 1; i++) s.push(c.width);
  s.push(3, 1);
  return s;
}

const LEAK = 0.05;

function actF(a: Act, z: number): number {
  if (a === "tanh") return Math.tanh(z);
  if (a === "relu") return z > 0 ? z : 0;
  return z > 0 ? z : LEAK * z;
}

/** Derivative given pre-activation z and post-activation h. */
function actD(a: Act, z: number, h: number): number {
  if (a === "tanh") return 1 - h * h;
  if (a === "relu") return z > 0 ? 1 : 0;
  return z > 0 ? 1 : LEAK;
}

export interface Trace {
  n: number;
  /** pre[l]: n × sizes[l+1] (W·h + b), l = 0..L−1. */
  pre: Float64Array[];
  /** post[0] = input; post[l+1] = act(pre[l]) for hidden layers; the last entry is the logit. */
  post: Float64Array[];
}

export interface StepResult {
  loss: number;
  acc: number;
  gradNorm: number;
}

export class MLP {
  readonly sizes: number[];
  readonly act: Act;
  readonly W: Float64Array[] = [];
  readonly b: Float64Array[] = [];
  private mW: Float64Array[] = [];
  private vW: Float64Array[] = [];
  private mb: Float64Array[] = [];
  private vb: Float64Array[] = [];
  private gW: Float64Array[] = [];
  private gb: Float64Array[] = [];
  private t = 0;
  steps = 0;
  private trace: Trace | null = null;

  constructor(cfg: NetConfig, seed: number) {
    this.sizes = layerSizes(cfg);
    this.act = cfg.act;
    const rng = makeRng(seed * 104729 + 3);
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const fin = this.sizes[l], fout = this.sizes[l + 1];
      const lim = Math.sqrt(6 / (fin + fout)); // Xavier/Glorot uniform
      const w = new Float64Array(fout * fin);
      for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * lim;
      const b = new Float64Array(fout);
      for (let i = 0; i < fout; i++) b[i] = (rng() * 2 - 1) * 0.1;
      this.W.push(w);
      this.b.push(b);
      for (const arr of [this.mW, this.vW, this.gW]) arr.push(new Float64Array(fout * fin));
      for (const arr of [this.mb, this.vb, this.gb]) arr.push(new Float64Array(fout));
    }
  }

  get layers(): number {
    return this.sizes.length - 1;
  }

  /** Forward pass over n points (rows of X, 3 wide). Pass a previous Trace to reuse its buffers. */
  forward(X: Float64Array, n: number, cache: Trace | null = null): Trace {
    const L = this.layers;
    let tr = cache;
    if (!tr || tr.n !== n || tr.pre.length !== L || tr.pre.some((p, l) => p.length !== n * this.sizes[l + 1])) {
      tr = { n, pre: [], post: [X] };
      for (let l = 0; l < L; l++) {
        tr.pre.push(new Float64Array(n * this.sizes[l + 1]));
        tr.post.push(l === L - 1 ? tr.pre[l] : new Float64Array(n * this.sizes[l + 1]));
      }
    }
    tr.post[0] = X;
    for (let l = 0; l < L; l++) {
      const fin = this.sizes[l], fout = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l];
      const hin = tr.post[l], z = tr.pre[l], hout = tr.post[l + 1];
      const last = l === L - 1;
      for (let i = 0; i < n; i++) {
        const io = i * fin, oo = i * fout;
        for (let o = 0; o < fout; o++) {
          let s = b[o];
          const wo = o * fin;
          for (let k = 0; k < fin; k++) s += W[wo + k] * hin[io + k];
          z[oo + o] = s;
          if (!last) hout[oo + o] = actF(this.act, s);
        }
      }
    }
    return tr;
  }

  /** Logits for n points. */
  logits(X: Float64Array, n: number): Float64Array {
    return this.forward(X, n).post[this.layers].slice();
  }

  /** One full-batch Adam step on binary cross-entropy. */
  step(ds: Dataset, lr: number, clip = 2): StepResult {
    const L = this.layers;
    const n = ds.n;
    const tr = (this.trace = this.forward(ds.X, n, this.trace));
    for (let l = 0; l < L; l++) {
      this.gW[l].fill(0);
      this.gb[l].fill(0);
    }
    const logit = tr.post[L];
    let loss = 0, correct = 0;
    // delta at the output layer: dL/dz = σ(z) − y, averaged
    let delta = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const z = logit[i], yy = ds.y[i];
      // numerically stable BCE: softplus(z) − y·z
      loss += Math.max(z, 0) - yy * z + Math.log1p(Math.exp(-Math.abs(z)));
      const p = 1 / (1 + Math.exp(-z));
      delta[i] = (p - yy) / n;
      if ((z > 0 ? 1 : 0) === yy) correct++;
    }
    for (let l = L - 1; l >= 0; l--) {
      const fin = this.sizes[l], fout = this.sizes[l + 1];
      const hin = tr.post[l];
      const gW = this.gW[l], gb = this.gb[l], W = this.W[l];
      for (let i = 0; i < n; i++) {
        const io = i * fin, oo = i * fout;
        for (let o = 0; o < fout; o++) {
          const d = delta[oo + o];
          if (d === 0) continue;
          gb[o] += d;
          const wo = o * fin;
          for (let k = 0; k < fin; k++) gW[wo + k] += d * hin[io + k];
        }
      }
      if (l > 0) {
        const prev = new Float64Array(n * fin);
        const zp = tr.pre[l - 1], hp = tr.post[l];
        for (let i = 0; i < n; i++) {
          const io = i * fin, oo = i * fout;
          for (let k = 0; k < fin; k++) {
            let s = 0;
            for (let o = 0; o < fout; o++) s += W[o * fin + k] * delta[oo + o];
            prev[io + k] = s * actD(this.act, zp[io + k], hp[io + k]);
          }
        }
        delta = prev;
      }
    }
    // global-norm gradient clipping
    let g2 = 0;
    for (let l = 0; l < L; l++) {
      for (const v of this.gW[l]) g2 += v * v;
      for (const v of this.gb[l]) g2 += v * v;
    }
    const gn = Math.sqrt(g2);
    const scale = gn > clip ? clip / gn : 1;
    // Adam
    this.t++;
    this.steps++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 / (1 - Math.pow(b1, this.t)), c2 = 1 / (1 - Math.pow(b2, this.t));
    for (let l = 0; l < L; l++) {
      adam(this.W[l], this.gW[l], this.mW[l], this.vW[l], scale, lr, b1, b2, c1, c2, eps);
      adam(this.b[l], this.gb[l], this.mb[l], this.vb[l], scale, lr, b1, b2, c1, c2, eps);
    }
    return { loss: loss / n, acc: correct / n, gradNorm: gn };
  }

  /** Accuracy and loss without updating (used for reporting). */
  evaluate(ds: Dataset): { loss: number; acc: number } {
    const z = this.forward(ds.X, ds.n).post[this.layers];
    let loss = 0, correct = 0;
    for (let i = 0; i < ds.n; i++) {
      const yy = ds.y[i];
      loss += Math.max(z[i], 0) - yy * z[i] + Math.log1p(Math.exp(-Math.abs(z[i])));
      if ((z[i] > 0 ? 1 : 0) === yy) correct++;
    }
    return { loss: loss / ds.n, acc: correct / ds.n };
  }

  finite(): boolean {
    for (let l = 0; l < this.layers; l++) {
      for (const v of this.W[l]) if (!Number.isFinite(v)) return false;
      for (const v of this.b[l]) if (!Number.isFinite(v)) return false;
    }
    return true;
  }
}

function adam(p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array, scale: number, lr: number, b1: number, b2: number, c1: number, c2: number, eps: number): void {
  for (let i = 0; i < p.length; i++) {
    const gi = g[i] * scale;
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= (lr * (m[i] * c1)) / (Math.sqrt(v[i] * c2) + eps);
  }
}

// ───────────────────────── linear algebra ─────────────────────────

/** Mean and covariance of n rows × d columns. */
export function covariance(H: Float64Array, n: number, d: number): { mean: Float64Array; cov: Float64Array } {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += H[i * d + k];
  for (let k = 0; k < d; k++) mean[k] /= n;
  const cov = new Float64Array(d * d);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) {
      const da = H[i * d + a] - mean[a];
      for (let b = a; b < d; b++) cov[a * d + b] += da * (H[i * d + b] - mean[b]);
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      cov[a * d + b] /= Math.max(1, n - 1);
      cov[b * d + a] = cov[a * d + b];
    }
  }
  return { mean, cov };
}

/**
 * Cyclic Jacobi eigen-decomposition of a symmetric d×d matrix.
 * Returns eigenvalues sorted descending and eigenvectors as rows (vecs[k*d + j]).
 */
export function eigSym(A: Float64Array, d: number): { vals: Float64Array; vecs: Float64Array } {
  const a = Float64Array.from(A);
  const v = new Float64Array(d * d);
  for (let i = 0; i < d; i++) v[i * d + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p * d + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p * d + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * d + q] - a[p * d + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = a[k * d + p], akq = a[k * d + q];
          a[k * d + p] = c * akp - s * akq;
          a[k * d + q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = a[p * d + k], aqk = a[q * d + k];
          a[p * d + k] = c * apk - s * aqk;
          a[q * d + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < d; k++) {
          const vkp = v[k * d + p], vkq = v[k * d + q];
          v[k * d + p] = c * vkp - s * vkq;
          v[k * d + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: d }, (_, i) => i).sort((i, j) => a[j * d + j] - a[i * d + i]);
  const vals = new Float64Array(d);
  const vecs = new Float64Array(d * d);
  order.forEach((src, k) => {
    vals[k] = Math.max(0, a[src * d + src]);
    for (let j = 0; j < d; j++) vecs[k * d + j] = v[j * d + src];
  });
  return { vals, vecs };
}

/** Participation ratio (Σλ)² / Σλ²: the effective number of directions the data spreads along. */
export function participationRatio(vals: ArrayLike<number>): number {
  let s = 0, s2 = 0;
  for (let i = 0; i < vals.length; i++) {
    s += vals[i];
    s2 += vals[i] * vals[i];
  }
  return s2 > 1e-18 ? (s * s) / s2 : 0;
}

const PERMS3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/**
 * A d → 3 projection basis (rows). For d = 3 it is the identity, for d < 3 it
 * pads, for d > 3 it is the top-3 principal directions, re-ordered and sign-
 * flipped to best match `prev` so the view doesn't flip between frames.
 */
export function projectionBasis(H: Float64Array, n: number, d: number, prev: Float64Array | null): { basis: Float64Array; vals: Float64Array } {
  const { cov } = covariance(H, n, d);
  const { vals, vecs } = eigSym(cov, d);
  const basis = new Float64Array(3 * d);
  if (d <= 3) {
    for (let k = 0; k < Math.min(3, d); k++) basis[k * d + k] = 1;
    return { basis, vals };
  }
  for (let k = 0; k < 3; k++) for (let j = 0; j < d; j++) basis[k * d + j] = vecs[k * d + j];
  if (prev && prev.length === basis.length) {
    const dot = (a: number, b: number) => {
      let s = 0;
      for (let j = 0; j < d; j++) s += prev[a * d + j] * basis[b * d + j];
      return s;
    };
    let best = PERMS3[0], bestS = -1;
    for (const p of PERMS3) {
      // only allow swaps of nearly-equal variance directions to keep "top 3" honest
      const sc = Math.abs(dot(0, p[0])) + Math.abs(dot(1, p[1])) + Math.abs(dot(2, p[2]));
      if (sc > bestS + 1e-9) {
        bestS = sc;
        best = p;
      }
    }
    const out = new Float64Array(3 * d);
    for (let k = 0; k < 3; k++) {
      const sgn = dot(k, best[k]) < 0 ? -1 : 1;
      for (let j = 0; j < d; j++) out[k * d + j] = sgn * basis[best[k] * d + j];
    }
    return { basis: out, vals };
  }
  return { basis, vals };
}

/** rows × d → rows × 3 through a 3×d basis (no centring). */
export function project(H: Float64Array, rows: number, d: number, basis: Float64Array, out?: Float64Array): Float64Array {
  const o = out && out.length === rows * 3 ? out : new Float64Array(rows * 3);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < 3; k++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += basis[k * d + j] * H[i * d + j];
      o[i * 3 + k] = s;
    }
  }
  return o;
}

// ───────────────────────── linear probe ─────────────────────────

/**
 * Logistic regression on one layer's activations ("could one flat sheet split
 * the beads here?"). Features are standardised; weights warm-start between
 * calls so a few dozen steps per update are enough to track training.
 */
export class Probe {
  w: Float64Array;
  readonly d: number;
  constructor(d: number) {
    this.d = d;
    this.w = new Float64Array(d + 1);
  }

  fit(H: Float64Array, n: number, y: Uint8Array, steps = 40, lr = 0.8): number {
    const d = this.d;
    const { mean, cov } = covariance(H, n, d);
    const sd = new Float64Array(d);
    for (let k = 0; k < d; k++) sd[k] = Math.sqrt(cov[k * d + k]) + 1e-6;
    const g = new Float64Array(d + 1);
    const m = new Float64Array(d + 1);
    const w = this.w;
    for (let s = 0; s < steps; s++) {
      g.fill(0);
      for (let i = 0; i < n; i++) {
        let z = w[d];
        for (let k = 0; k < d; k++) z += w[k] * ((H[i * d + k] - mean[k]) / sd[k]);
        const r = 1 / (1 + Math.exp(-z)) - y[i];
        for (let k = 0; k < d; k++) g[k] += r * ((H[i * d + k] - mean[k]) / sd[k]);
        g[d] += r;
      }
      // heavy-ball momentum; standardised features keep this well conditioned
      for (let k = 0; k <= d; k++) {
        m[k] = 0.8 * m[k] + g[k] / n;
        w[k] -= lr * m[k];
        if (!Number.isFinite(w[k])) w[k] = 0;
      }
    }
    let correct = 0;
    for (let i = 0; i < n; i++) {
      let z = w[d];
      for (let k = 0; k < d; k++) z += w[k] * ((H[i * d + k] - mean[k]) / sd[k]);
      if ((z > 0 ? 1 : 0) === y[i]) correct++;
    }
    return correct / n;
  }
}

// ───────────────────────── space grid ─────────────────────────

/**
 * A lattice of `lines`³-ish grid lines through [-ext, ext]³: for each axis,
 * lines × lines lines along it, each sampled at `samples` points. Returns the
 * sample points (P × 3) and line-segment index pairs.
 */
export function makeGrid(lines = 9, samples = 17, ext = 1): { pts: Float64Array; segs: Uint32Array; count: number } {
  const per = lines * lines * samples;
  const pts = new Float64Array(per * 3 * 3);
  const segs = new Uint32Array(lines * lines * (samples - 1) * 3 * 2);
  let p = 0, s = 0;
  const coord = (i: number, m: number) => -ext + (2 * ext * i) / (m - 1);
  for (let axis = 0; axis < 3; axis++) {
    for (let i = 0; i < lines; i++) {
      for (let j = 0; j < lines; j++) {
        const start = p;
        for (let k = 0; k < samples; k++) {
          const c = [0, 0, 0];
          c[axis] = coord(k, samples);
          c[(axis + 1) % 3] = coord(i, lines);
          c[(axis + 2) % 3] = coord(j, lines);
          pts[p * 3] = c[0];
          pts[p * 3 + 1] = c[1];
          pts[p * 3 + 2] = c[2];
          if (k > 0) {
            segs[s++] = p - 1;
            segs[s++] = p;
          }
          p++;
        }
        void start;
      }
    }
  }
  return { pts, segs, count: p };
}

// ───────────────────────── headless experiments ─────────────────────────

export interface RunResult {
  acc: number;
  loss: number;
  finite: boolean;
  /** probe accuracy per view (input, hidden 1..depth). */
  probe: number[];
}

/** Train from scratch, headless, and report. */
export function trainRun(shape: ShapeId, cfg: NetConfig, seed: number, steps: number, lr: number, noise = DEFAULT_NOISE, withProbe = false): RunResult {
  const ds = makeDataset(shape, seed, noise);
  const net = new MLP(cfg, seed);
  for (let s = 0; s < steps; s++) net.step(ds, lr);
  const ev = net.evaluate(ds);
  const probe: number[] = [];
  if (withProbe) {
    const tr = net.forward(ds.X, ds.n);
    for (let v = 0; v < net.layers; v++) {
      const d = net.sizes[v];
      probe.push(new Probe(d).fit(tr.post[v], ds.n, ds.y, 300, 0.8));
    }
  }
  return { acc: ev.acc, loss: ev.loss, finite: net.finite() && Number.isFinite(ev.loss), probe };
}

/** Training budget used by the live demo's settle point and the self-test. */
export const BUDGET = { steps: 1800, lr: 0.03 };
export const DEFAULT_CFG: NetConfig = { width: 3, depth: 4, act: "tanh" };

export interface SelfTest {
  twisted: { acc: number };
  ringsNarrow: { width: number; accs: number[]; mean: number };
  ringsWide: { width: number; accs: number[]; mean: number };
  probe: number[];
  probeRises: boolean;
  noNaN: boolean;
  ms: number;
  pass: boolean;
}

export function selfTest(seeds = [1, 2, 3]): SelfTest {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const { steps, lr } = BUDGET;
  const tw = trainRun("twisted", DEFAULT_CFG, 1, steps, lr, DEFAULT_NOISE, true);
  const narrow = seeds.map((s) => trainRun("rings", { ...DEFAULT_CFG, width: 3 }, s, steps, lr));
  const wide = seeds.map((s) => trainRun("rings", { ...DEFAULT_CFG, width: 6 }, s, steps, lr));
  const mean = (a: RunResult[]) => a.reduce((s, r) => s + r.acc, 0) / a.length;
  const nm = mean(narrow), wm = mean(wide);
  const probeRises = tw.probe[tw.probe.length - 1] > tw.probe[0];
  const noNaN = [tw, ...narrow, ...wide].every((r) => r.finite);
  const r3 = (x: number) => Math.round(x * 1000) / 1000;
  const pass = tw.acc >= 0.95 && wm >= 0.95 && nm <= wm - 0.05 && probeRises && noNaN;
  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  return {
    twisted: { acc: r3(tw.acc) },
    ringsNarrow: { width: 3, accs: narrow.map((r) => r3(r.acc)), mean: r3(nm) },
    ringsWide: { width: 6, accs: wide.map((r) => r3(r.acc)), mean: r3(wm) },
    probe: tw.probe.map(r3),
    probeRises,
    noNaN,
    ms: Math.round(t1 - t0),
    pass,
  };
}
