/** Procedural canvas textures: veined dark marble, soft light pools, dust sprites. */
import { makeRng } from "./nn";

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")!];
}

/** Tileable-ish 2D value noise with fbm. */
function noise2(seed: number): (x: number, y: number) => number {
  const r = makeRng(seed);
  const N = 256;
  const t = Float32Array.from({ length: N * N }, () => r());
  const at = (i: number, j: number) => t[(j & (N - 1)) * N + (i & (N - 1))];
  return (x: number, y: number) => {
    const i = Math.floor(x), j = Math.floor(y);
    const fx = x - i, fy = y - j;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
}

/** Dark veined marble: colour + roughness maps. */
export function marble(size = 1024, seed = 3): { color: HTMLCanvasElement; rough: HTMLCanvasElement } {
  const W = size, H = size;
  const [c, g] = canvas(W, H);
  const [r, rg] = canvas(W, H);
  const img = g.createImageData(W, H);
  const rim = rg.createImageData(W, H);
  const n = noise2(seed), m = noise2(seed + 9);
  const fbm = (x: number, y: number, f: (x: number, y: number) => number) => {
    let s = 0, a = 0.5, fr = 1;
    for (let o = 0; o < 5; o++) {
      s += a * f(x * fr, y * fr);
      a *= 0.5;
      fr *= 2.03;
    }
    return s;
  };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = (x / W) * 6, v = (y / H) * 6;
      const t1 = fbm(u, v, n), t2 = fbm(u * 1.7 + 11, v * 1.7 + 3, m);
      // two families of veins: thin bright cracks + broad cloudy bands
      const vein = Math.pow(1 - Math.abs(Math.sin((u * 0.9 + v * 0.55 + t1 * 5.5) * 2.1)), 22);
      const vein2 = Math.pow(1 - Math.abs(Math.sin((u * -0.4 + v * 1.1 + t2 * 6.5) * 2.6)), 34);
      const cloud = t2 * 0.55 + t1 * 0.25;
      const o = (y * W + x) * 4;
      const base = 13 + cloud * 22;
      const vv = vein * 0.85 + vein2 * 0.6;
      img.data[o] = base + vv * 110 + vein2 * 30;
      img.data[o + 1] = base + 2 + vv * 108 + vein2 * 16;
      img.data[o + 2] = base + 5 + vv * 108;
      img.data[o + 3] = 255;
      const ro = 40 + cloud * 60 + vv * 40;
      rim.data[o] = rim.data[o + 1] = rim.data[o + 2] = ro;
      rim.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  rg.putImageData(rim, 0, 0);
  return { color: c, rough: r };
}

/** Soft radial light pool (white; tint in the material). */
export function pool(size = 256): HTMLCanvasElement {
  const [c, g] = canvas(size, size);
  const gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gr.addColorStop(0, "rgba(255,255,255,1)");
  gr.addColorStop(0.25, "rgba(255,255,255,0.55)");
  gr.addColorStop(0.6, "rgba(255,255,255,0.14)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, size, size);
  // faint caustic ripples
  const rng = makeRng(7);
  g.globalCompositeOperation = "lighter";
  for (let k = 0; k < 26; k++) {
    const a = rng() * Math.PI * 2, rr = (0.1 + rng() * 0.32) * size;
    g.strokeStyle = `rgba(255,255,255,${0.04 + rng() * 0.06})`;
    g.lineWidth = 1 + rng() * 2.5;
    g.beginPath();
    g.ellipse(size / 2 + Math.cos(a) * rr * 0.3, size / 2 + Math.sin(a) * rr * 0.3, rr, rr * (0.5 + rng() * 0.5), rng() * 3, 0, Math.PI * 1.4);
    g.stroke();
  }
  return c;
}

/** Tiny round sprite for dust motes. */
export function dot(size = 64): HTMLCanvasElement {
  const [c, g] = canvas(size, size);
  const gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gr.addColorStop(0, "rgba(255,255,255,1)");
  gr.addColorStop(0.35, "rgba(255,255,255,0.35)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, size, size);
  return c;
}
