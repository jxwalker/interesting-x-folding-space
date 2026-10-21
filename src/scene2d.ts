/**
 * Canvas2D fallback ("simple mode"): the same lab, drawn as a rotatable
 * isometric cube with depth-sorted beads, the space grid, the decision sheet,
 * the traced journey and a clickable filmstrip of layers.
 */
import type { Lab } from "./lab";
import { planeBox } from "./geom";
import { AMBER, CYAN, TEAL, type Layout, type PickResult, type UiState, type View } from "./view";

type V3 = [number, number, number];

export class Scene2D implements View {
  readonly kind = "2d" as const;
  readonly canvas: HTMLCanvasElement;
  private g: CanvasRenderingContext2D;
  private w = 1;
  private h = 1;
  private dpr = 1;
  private layout: Layout = { cx: 640, cy: 360, size: 400, phone: false, film: true };
  private yaw = -0.5;
  private pitch = 0.32;
  private yawV = 0;
  private zoomF = 1;
  private idle = 10;
  private time = 0;
  private dragging = false;
  private pos = new Float32Array(0);
  private grid = new Float32Array(0);
  private screen = new Float32Array(0);
  private order: number[] = [];
  private sprites: Record<string, HTMLCanvasElement> = {};
  private filmRects: { x: number; y: number; s: number; view: number }[] = [];

  constructor(container: HTMLElement) {
    const c = document.createElement("canvas");
    c.id = "canvas";
    c.setAttribute("aria-label", "Cube showing the beads at the current layer (simple mode). Tap a bead to trace it; drag to turn.");
    container.appendChild(c);
    this.canvas = c;
    this.g = c.getContext("2d")!;
    for (const [k, col] of [["a", AMBER], ["b", TEAL], ["t", "#ffffff"]] as const) this.sprites[k] = sprite(col);
  }

  resize(w: number, h: number, layout: Layout): void {
    this.w = w;
    this.h = h;
    this.layout = layout;
    this.dpr = Math.min(2, devicePixelRatio || 1);
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
  }

  orbit(dx: number, dy: number): void {
    this.idle = 0;
    this.dragging = true;
    this.yaw -= dx * 0.006;
    this.pitch = Math.max(0.05, Math.min(0.9, this.pitch + dy * 0.004));
    this.yawV = -dx * 0.006 * 60 * 0.5 + this.yawV * 0.5;
  }

  release(): void {
    this.dragging = false;
  }

  zoom(f: number): void {
    this.zoomF = Math.max(0.65, Math.min(1.5, this.zoomF * f));
  }

  poke(): void {
    this.idle = 0;
  }

  private scale(): number {
    return (this.layout.size / 2.9) * this.zoomF;
  }

  /** display coords → screen (x, y, depth) */
  private proj(x: number, y: number, z: number, cx: number, cy: number, k: number): V3 {
    const cyw = Math.cos(this.yaw + this.drift()), syw = Math.sin(this.yaw + this.drift());
    const x1 = x * cyw + z * syw, z1 = -x * syw + z * cyw;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    return [cx + x1 * k, cy - y2 * k, z2];
  }

  private drift(): number {
    const d = Math.min(1, Math.max(0, (this.idle - 4) / 3));
    return d * 0.17 * Math.sin(this.time * 0.16);
  }

  render(dt: number, lab: Lab, ui: UiState): void {
    this.time += dt;
    this.idle += dt;
    if (!this.dragging) {
      this.yaw += this.yawV * dt;
      this.yawV *= Math.exp(-dt * 2.4);
    }
    const g = this.g, W = this.w, H = this.h;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const bg = g.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, "#0b1322");
    bg.addColorStop(1, "#050608");
    g.fillStyle = bg;
    g.fillRect(0, 0, W, H);

    const n = lab.ds.n;
    if (this.pos.length !== n * 3) {
      this.pos = new Float32Array(n * 3);
      this.screen = new Float32Array(n * 3);
      this.order = Array.from({ length: n }, (_, i) => i);
    }
    if (this.grid.length !== lab.grid.count * 3) this.grid = new Float32Array(lab.grid.count * 3);
    lab.sample(ui.s, this.pos, ui.showGrid ? this.grid : null);
    const wb = ui.wobble * 0.05;
    const k = this.scale() * 0.8, cx = this.layout.cx, cy = this.layout.cy;
    const ky = k * (1 + wb);
    const P = (x: number, y: number, z: number) => this.proj(x, (y * ky) / k, z, cx, cy, k);

    // plinth shadow + cube
    const L = 1.25;
    const corners: V3[] = [];
    for (let i = 0; i < 8; i++) corners.push(P(i & 1 ? L : -L, i & 2 ? L : -L, i & 4 ? L : -L));
    g.fillStyle = "rgba(0,0,0,0.45)";
    g.beginPath();
    const base = [0, 1, 5, 4].map((i) => P(i & 1 ? L * 1.35 : -L * 1.35, -L - 0.02, i & 4 ? L * 1.35 : -L * 1.35));
    base.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.closePath();
    g.fill();
    const edges: [number, number][] = [];
    for (let i = 0; i < 8; i++) for (const b of [1, 2, 4]) if (!(i & b)) edges.push([i, i | b]);
    const drawEdges = (front: boolean) => {
      for (const [a, b] of edges) {
        const zmid = (corners[a][2] + corners[b][2]) / 2;
        if ((zmid > 0) !== front) continue;
        g.strokeStyle = front ? "rgba(210,235,255,0.55)" : "rgba(210,235,255,0.16)";
        g.lineWidth = front ? 1.4 : 1;
        g.beginPath();
        g.moveTo(corners[a][0], corners[a][1]);
        g.lineTo(corners[b][0], corners[b][1]);
        g.stroke();
      }
    };
    drawEdges(false);

    // grid
    const D = lab.depth;
    const outW = smooth(ui.s, D - 0.6, D);
    if (ui.showGrid) {
      const segs = lab.grid.segs;
      g.lineWidth = 0.7;
      const lim = 1.12;
      for (let s = 0; s < segs.length; s += 2) {
        const a = segs[s], b = segs[s + 1];
        const ax = this.grid[a * 3], ay = this.grid[a * 3 + 1], az = this.grid[a * 3 + 2];
        const bx = this.grid[b * 3], by = this.grid[b * 3 + 1], bz = this.grid[b * 3 + 2];
        if (Math.max(Math.abs(ax), Math.abs(ay), Math.abs(az), Math.abs(bx), Math.abs(by), Math.abs(bz)) > lim) continue;
        const p = lab.gridPred[a];
        const tint = outW * Math.min(1, Math.abs(p - 0.5) * 2);
        g.strokeStyle = tint > 0.05 ? (p < 0.5 ? `rgba(255,154,60,${0.1 + 0.2 * tint})` : `rgba(47,209,197,${0.1 + 0.2 * tint})`) : "rgba(200,210,225,0.12)";
        const pa = P(ax, ay, az), pb = P(bx, by, bz);
        g.beginPath();
        g.moveTo(pa[0], pa[1]);
        g.lineTo(pb[0], pb[1]);
        g.stroke();
      }
    }

    // sheet
    const sheetA = outW * smooth(lab.acc, 0.72, 0.93) * ui.sheetIn * (lab.plane.ok ? 1 : 0);
    if (sheetA > 0.01) {
      const [nx, ny, nz] = lab.plane.n;
      const pts = planeBox(nx, ny, nz, lab.plane.d - (1 - ui.sheetIn) * 1.4, 1.22);
      if (pts.length >= 3) {
        g.beginPath();
        pts.forEach((p, i) => {
          const q = P(p[0], p[1], p[2]);
          if (i) g.lineTo(q[0], q[1]);
          else g.moveTo(q[0], q[1]);
        });
        g.closePath();
        g.fillStyle = `rgba(200,240,255,${0.12 * sheetA + ui.flash * 0.1})`;
        g.fill();
        g.strokeStyle = `rgba(111,231,255,${0.85 * sheetA})`;
        g.lineWidth = 2;
        g.shadowColor = CYAN;
        g.shadowBlur = 12;
        g.stroke();
        g.shadowBlur = 0;
      }
    }

    // beads, depth-sorted
    const scr = Math.max(0, 1 - (lab.time - lab.scrambleAt) / 0.9);
    const burst = scr * scr * 0.9;
    for (let i = 0; i < n; i++) {
      let x = this.pos[i * 3], y = this.pos[i * 3 + 1], z = this.pos[i * 3 + 2];
      if (burst > 0) {
        x += Math.sin(i * 2.399 + lab.time * 9) * burst;
        y += Math.cos(i * 3.1 + lab.time * 7) * burst;
        z += Math.sin(i * 1.7 + lab.time * 8) * burst;
      }
      const p = P(x, y, z);
      this.screen[i * 3] = p[0];
      this.screen[i * 3 + 1] = p[1];
      this.screen[i * 3 + 2] = p[2];
    }
    this.order.sort((a, b) => this.screen[a * 3 + 2] - this.screen[b * 3 + 2]);
    const r = Math.max(2.5, k * 0.06);
    for (const i of this.order) {
      const sx = this.screen[i * 3], sy = this.screen[i * 3 + 1], sz = this.screen[i * 3 + 2];
      const rr = r * Math.max(0.5, 1 + sz * 0.12) * (i === ui.traced ? 1.6 : 1);
      const spr = i === ui.traced ? this.sprites.t : lab.ds.y[i] ? this.sprites.b : this.sprites.a;
      g.drawImage(spr, sx - rr, sy - rr, rr * 2, rr * 2);
      const wrong = (lab.pred[i] > 0.5 ? 1 : 0) !== lab.ds.y[i];
      if (wrong && outW > 0.3) {
        g.strokeStyle = `rgba(255,255,255,${0.3 + 0.3 * Math.sin(this.time * 7 + i)})`;
        g.lineWidth = 1;
        g.beginPath();
        g.arc(sx, sy, rr * 1.15, 0, Math.PI * 2);
        g.stroke();
      }
    }

    // trace
    if (ui.traced >= 0 && ui.traced < n) {
      const pts = lab.journey(ui.traced).map((p) => P(p[0], p[1], p[2]));
      g.strokeStyle = "rgba(230,250,255,0.85)";
      g.lineWidth = 2;
      g.shadowColor = "#bff";
      g.shadowBlur = 8;
      g.beginPath();
      pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
      g.stroke();
      g.shadowBlur = 0;
      g.fillStyle = "#fff";
      for (const p of pts) {
        g.beginPath();
        g.arc(p[0], p[1], 2.5, 0, Math.PI * 2);
        g.fill();
      }
    }
    drawEdges(true);

    // filmstrip
    this.filmRects = [];
    if (this.layout.film) {
      const count = D + 1;
      const s = Math.max(16, Math.min(70, (this.layout.size * 1.1) / count - 10));
      const total = count * (s + 10) - 10;
      const fy = Math.min(H - s - 12, cy + this.layout.size * 0.62);
      const fx = cx - total / 2;
      const active = Math.round(ui.s);
      for (let v = 0; v < count; v++) {
        const x = fx + v * (s + 10);
        this.filmRects.push({ x, y: fy, s, view: v });
        g.fillStyle = v === active ? "rgba(111,231,255,0.12)" : "rgba(255,255,255,0.04)";
        g.strokeStyle = v === active ? "rgba(111,231,255,0.8)" : "rgba(255,255,255,0.22)";
        g.lineWidth = 1;
        g.beginPath();
        g.roundRect(x, fy, s, s, 6);
        g.fill();
        g.stroke();
        const st = lab.stages[lab.viewStage[v]];
        if (!st) continue;
        const ks = s * 0.32;
        for (let i = 0; i < n; i += 2) {
          const p = this.proj(st.data[i * 3], st.data[i * 3 + 1], st.data[i * 3 + 2], x + s / 2, fy + s / 2, ks);
          g.fillStyle = i === ui.traced ? "#fff" : lab.ds.y[i] ? TEAL : AMBER;
          g.fillRect(p[0] - 1, p[1] - 1, 2, 2);
        }
      }
    }
  }

  pick(clientX: number, clientY: number): PickResult {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    for (const f of this.filmRects) if (x >= f.x && x <= f.x + f.s && y >= f.y && y <= f.y + f.s) return { kind: "film", view: f.view };
    let best = -1, bestD = this.layout.phone ? 22 : 14;
    for (let i = 0; i < this.screen.length / 3; i++) {
      const d = Math.hypot(this.screen[i * 3] - x, this.screen[i * 3 + 1] - y) - this.screen[i * 3 + 2] * 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best >= 0 ? { kind: "bead", index: best } : { kind: "none" };
  }

  dispose(): void {
    this.canvas.remove();
  }
}

function smooth(x: number, a: number, b: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function sprite(col: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 48;
  const g = c.getContext("2d")!;
  const gr = g.createRadialGradient(18, 17, 2, 24, 24, 23);
  gr.addColorStop(0, "#ffffff");
  gr.addColorStop(0.25, col);
  gr.addColorStop(0.85, shade(col));
  gr.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = gr;
  g.beginPath();
  g.arc(24, 24, 23, 0, Math.PI * 2);
  g.fill();
  return c;
}

function shade(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, gg = (n >> 8) & 255, b = n & 255;
  return `rgb(${Math.round(r * 0.45)},${Math.round(gg * 0.45)},${Math.round(b * 0.45)})`;
}
