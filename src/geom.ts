/** Small geometry helpers shared by both renderers (no three.js). */

/** Convex polygon where the plane n·p + d = 0 cuts the cube [-h, h]³, ordered around its centre. */
export function planeBox(nx: number, ny: number, nz: number, d: number, h: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const corner = (i: number): [number, number, number] => [i & 1 ? h : -h, i & 2 ? h : -h, i & 4 ? h : -h];
  const f = (p: [number, number, number]) => nx * p[0] + ny * p[1] + nz * p[2] + d;
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue;
      const a = corner(i), b = corner(i | bit);
      const fa = f(a), fb = f(b);
      if ((fa < 0) !== (fb < 0)) {
        const t = fa / (fa - fb);
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
      }
    }
  }
  if (out.length < 3) return out;
  // order around the centroid in the plane's own 2D basis
  let cx = 0, cy = 0, cz = 0;
  for (const p of out) {
    cx += p[0];
    cy += p[1];
    cz += p[2];
  }
  cx /= out.length;
  cy /= out.length;
  cz /= out.length;
  const ref = Math.abs(nx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let ux = ny * ref[2] - nz * ref[1], uy = nz * ref[0] - nx * ref[2], uz = nx * ref[1] - ny * ref[0];
  const ul = Math.hypot(ux, uy, uz) || 1;
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
  const ang = (p: [number, number, number]) => Math.atan2((p[0] - cx) * vx + (p[1] - cy) * vy + (p[2] - cz) * vz, (p[0] - cx) * ux + (p[1] - cy) * uy + (p[2] - cz) * uz);
  return out.sort((a, b) => ang(a) - ang(b)).slice(0, 6);
}
