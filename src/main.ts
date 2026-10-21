import "./styles.css";
import { Lab, STEPS_PER_SECOND } from "./lab";
import { selfTest, type Act, type ShapeId } from "./nn";
import { Scene2D } from "./scene2d";
import type { Layout, UiState, View } from "./view";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const params = new URLSearchParams(location.search);
const recording = params.get("record") === "1";
if (recording) document.documentElement.classList.add("ix-record");
const force2d = params.get("renderer") === "2d";
const hq = params.get("hq") === "1" || recording;
/** The seed whose untangling reads best on first load (and in the video). */
const WOW_SEED = 12;

const lab = new Lab();
lab.rebuild(Number(params.get("seed")) || WOW_SEED);
const stage = $("stage");
let view: View;
let frames = 0;
let fps = 60;

const ui: UiState = { s: lab.depth, traced: -1, showGrid: true, flash: 0, sheetIn: 0, wobble: 0 };
/** Where the slider wants the view to be; ui.s eases toward it. */
let sTarget = lab.depth;
let wobbleV = 0;
let lastLayer = Math.round(ui.s);

// ───────────────────────── renderer selection ─────────────────────────

async function makeWebGL(): Promise<View> {
  const probe = document.createElement("canvas");
  const ctx = probe.getContext("webgl2") || probe.getContext("webgl");
  if (!ctx) throw new Error("WebGL unavailable");
  (ctx.getExtension("WEBGL_lose_context") as { loseContext?: () => void } | null)?.loseContext?.();
  const { Scene3D } = await import("./scene3d");
  return new Scene3D(stage, { hq });
}

function switchTo2D(reason: string): void {
  if (view?.kind === "2d") return;
  console.warn(`[folding-space] using Canvas2D fallback: ${reason}`);
  try {
    view?.dispose();
  } catch {
    /* ignore */
  }
  stage.querySelectorAll("canvas").forEach((c) => c.remove());
  view = new Scene2D(stage);
  $("badge").hidden = false;
  bindCanvas(view.canvas);
  onResize();
}

async function boot(): Promise<void> {
  if (force2d) {
    switchTo2D("forced by ?renderer=2d");
  } else {
    try {
      view = await makeWebGL();
      view.canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        switchTo2D("webglcontextlost");
      });
      bindCanvas(view.canvas);
      onResize();
      view.render(1 / 60, lab, ui); // compile now so init errors surface here
    } catch (err) {
      switchTo2D(`WebGL init failed: ${(err as Error)?.message ?? err}`);
    }
  }
  window.__ixRecord.ready = true;
  requestAnimationFrame(loop);
}

// ───────────────────────── layout ─────────────────────────

const dock = $("dock");
const adv = $("adv");
const intro = $("intro");

function computeLayout(w: number, h: number): Layout {
  const phone = w <= 760;
  if (phone) {
    const top = intro.getBoundingClientRect().bottom + 4;
    const bottom = dock.getBoundingClientRect().top - 6;
    const avail = Math.max(140, bottom - top);
    return { cx: w / 2, cy: top + avail * 0.47, size: Math.min(avail * 0.6, w * 0.6), phone: true, film: avail > 300 };
  }
  const left = dock.getBoundingClientRect().right + 10;
  const right = adv.hidden ? w - 10 : adv.getBoundingClientRect().left - 10;
  const regionW = Math.max(220, right - left);
  return {
    cx: left + regionW * 0.5,
    cy: h * 0.36,
    size: Math.min(h * 0.44, regionW * 0.5),
    phone: false,
    film: true,
  };
}

function onResize(): void {
  const w = innerWidth, h = innerHeight;
  if (!adv.hidden) {
    const phoneAdv = w <= 760;
    if (dock.classList.contains("mini") !== phoneAdv || phoneAdv) {
      dock.classList.toggle("mini", phoneAdv);
      dock.style.bottom = phoneAdv ? `${adv.offsetHeight}px` : "";
    }
  }
  view?.resize(w, h, computeLayout(w, h));
}
addEventListener("resize", onResize);

// ───────────────────────── pointer: orbit / tap ─────────────────────────

let drag: { id: number; x0: number; y0: number; x: number; y: number; moved: boolean } | null = null;
const touches = new Map<number, { x: number; y: number }>();
let pinch0 = 0;

function bindCanvas(c: HTMLCanvasElement): void {
  c.addEventListener("pointerdown", (e) => {
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    view.poke();
    if (touches.size === 2) {
      const [a, b] = [...touches.values()];
      pinch0 = Math.hypot(a.x - b.x, a.y - b.y);
      drag = null;
      return;
    }
    c.setPointerCapture(e.pointerId);
    drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, moved: false };
  });
  c.addEventListener("pointermove", (e) => {
    if (touches.has(e.pointerId)) touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (touches.size === 2 && pinch0 > 0) {
      const [a, b] = [...touches.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      view.zoom(pinch0 / Math.max(10, d));
      pinch0 = d;
      return;
    }
    if (!drag || drag.id !== e.pointerId) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 6) {
      drag.moved = true;
      c.classList.add("grabbing");
    }
    if (drag.moved) view.orbit(e.clientX - drag.x, e.clientY - drag.y);
    drag.x = e.clientX;
    drag.y = e.clientY;
  });
  const end = (e: PointerEvent) => {
    touches.delete(e.pointerId);
    if (touches.size < 2) pinch0 = 0;
    if (!drag || drag.id !== e.pointerId) return;
    if (!drag.moved && e.type === "pointerup") tap(e.clientX, e.clientY);
    view.release();
    c.classList.remove("grabbing");
    drag = null;
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    view.zoom(Math.exp(e.deltaY * 0.001));
  }, { passive: false });
}

let tracedAt = -99;
function tap(x: number, y: number): void {
  const hit = view.pick(x, y);
  if (hit.kind === "bead") {
    ui.traced = hit.index;
    tracedAt = clock;
  } else if (hit.kind === "film") {
    jumpTo(hit.view);
  } else {
    ui.traced = -1;
  }
}

// ───────────────────────── wow sequence ─────────────────────────

/**
 * First visit: learn at the output view; when it's solved the glass sheet
 * slides in with a flash; then the view rewinds and walks Input → Output once.
 */
type Phase = "learn" | "sheet" | "hold" | "rewind" | "walk" | "done";
let phase: Phase = "learn";
let phaseT = 0;
let clock = 0;
let walkedAt = -99;

function cancelAuto(): void {
  phase = "done";
}

function runWow(dt: number): void {
  phaseT += dt;
  switch (phase) {
    case "learn":
      if ((lab.acc >= 0.95 && lab.steps > 30) || phaseT > 6) {
        phase = "sheet";
        phaseT = 0;
      }
      break;
    case "sheet":
      if (phaseT >= 1.1) {
        phase = "hold";
        phaseT = 0;
      }
      break;
    case "hold":
      if (phaseT >= 0.4) {
        phase = "rewind";
        phaseT = 0;
      }
      break;
    case "rewind": {
      const f = Math.min(1, phaseT / 0.6);
      setView(lab.depth * (1 - f * f * (3 - 2 * f)), true);
      if (f >= 1) {
        phase = "walk";
        phaseT = 0;
      }
      break;
    }
    case "walk": {
      const dur = 3.2;
      const f = Math.min(1, phaseT / dur);
      setView(lab.depth * f, true);
      walkedAt = clock;
      if (f >= 1) phase = "done";
      break;
    }
  }
}

// sheet slide-in: once per trained network, as soon as it's mostly right (with a soft flash)
let sheetVersion = lab.version;
let sheetGo = false;
let flashed = false;
function updateSheet(dt: number): void {
  if (lab.version !== sheetVersion) {
    sheetVersion = lab.version;
    ui.sheetIn = 0;
    sheetGo = flashed = false;
  }
  if (phase === "learn") {
    ui.sheetIn = 0;
    return;
  }
  if (!sheetGo && (phase === "sheet" || lab.acc >= 0.9)) sheetGo = true;
  if (!sheetGo) return;
  ui.sheetIn = Math.min(1, ui.sheetIn + dt / 1.0);
  if (!flashed && ui.sheetIn > 0.85 && lab.acc >= 0.9) {
    flashed = true;
    ui.flash = 1;
  }
}

// ───────────────────────── controls ─────────────────────────

const layer = $<HTMLInputElement>("layer");
const widthEl = $<HTMLInputElement>("width");
const depthEl = $<HTMLInputElement>("depth");
const actEl = $<HTMLSelectElement>("act");
const lrEl = $<HTMLInputElement>("lr");
const noiseEl = $<HTMLInputElement>("noise");
const gridEl = $<HTMLInputElement>("grid");
const learnBtn = $("btn-learn");
const ticks = $("ticks");

function paintRange(el: HTMLInputElement): void {
  const p = ((+el.value - +el.min) / (+el.max - +el.min)) * 100;
  el.style.setProperty("--p", `${p}%`);
}

function setView(s: number, auto = false): void {
  sTarget = Math.max(0, Math.min(lab.depth, s));
  if (auto) ui.s = sTarget;
  layer.value = String(sTarget);
  paintRange(layer);
}

function jumpTo(v: number): void {
  cancelAuto();
  setView(v);
  walkedAt = clock;
}

function buildTicks(): void {
  ticks.innerHTML = "";
  layer.max = String(lab.depth);
  for (let v = 0; v <= lab.depth; v++) {
    const t = document.createElement("span");
    t.textContent = v === 0 ? "Input" : v === lab.depth ? "Output" : String(v);
    t.addEventListener("click", () => jumpTo(v));
    ticks.appendChild(t);
  }
}

function syncControls(): void {
  widthEl.value = String(lab.cfg.width);
  depthEl.value = String(lab.cfg.depth);
  actEl.value = lab.cfg.act;
  lrEl.value = String(Math.log10(lab.lr));
  noiseEl.value = String(lab.noise);
  gridEl.checked = ui.showGrid;
  $("width-val").textContent = String(lab.cfg.width);
  $("depth-val").textContent = String(lab.cfg.depth);
  $("lr-val").textContent = lab.lr.toFixed(3);
  $("noise-val").textContent = lab.noise.toFixed(3);
  [widthEl, depthEl, lrEl, noiseEl, layer].forEach(paintRange);
  for (const s of ["twisted", "rings", "shell"] as const) {
    const b = $(`shape-${s}`);
    b.classList.toggle("on", lab.shape === s);
    b.setAttribute("aria-pressed", String(lab.shape === s));
  }
  const presetOn = (id: string, on: boolean) => $(id).classList.toggle("on", on);
  presetOn("preset-helix", lab.shape === "twisted" && lab.cfg.width === 3);
  presetOn("preset-stuck", lab.shape === "rings" && lab.cfg.width === 3);
  presetOn("preset-freed", lab.shape === "rings" && lab.cfg.width === 6);
  syncLearn();
}

function syncLearn(): void {
  $("learn-label").textContent = lab.training ? "Pause learning" : "Keep learning";
  learnBtn.setAttribute("aria-pressed", String(lab.training));
  learnBtn.classList.toggle("paused", !lab.training);
}

/** After any retrain: keep the current view position if it still exists. */
function afterRebuild(): void {
  buildTicks();
  setView(Math.min(sTarget, lab.depth), true);
  ui.traced = ui.traced < lab.ds.n ? ui.traced : -1;
  lastLayer = Math.round(ui.s);
  syncControls();
  kick();
}

function setShape(s: ShapeId): void {
  cancelAuto();
  lab.setShape(s);
  afterRebuild();
}

for (const s of ["twisted", "rings", "shell"] as const) $(`shape-${s}`).addEventListener("click", () => setShape(s));

learnBtn.addEventListener("click", () => {
  if (lab.training) lab.training = false;
  else lab.resume();
  if (lab.training) cancelAuto();
  syncLearn();
});
$("btn-reset").addEventListener("click", () => {
  cancelAuto();
  lab.restart();
  afterRebuild();
});

layer.addEventListener("input", () => {
  cancelAuto();
  sTarget = +layer.value;
  walkedAt = clock;
  paintRange(layer);
});
// arrow keys step stage by stage (stretch, then bend, then the next layer)
layer.addEventListener("keydown", (e) => {
  const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
  if (!dir) return;
  e.preventDefault();
  cancelAuto();
  const ss = lab.stages.map((st) => st.s);
  const next = dir > 0 ? ss.find((x) => x > sTarget + 1e-3) : [...ss].reverse().find((x) => x < sTarget - 1e-3);
  if (next !== undefined) setView(next);
  walkedAt = clock;
});
layer.addEventListener("change", () => {
  // settle on the nearest stage (half steps are the "stretch" stage of 3-wide layers)
  const st = lab.stages.reduce((a, b) => (Math.abs(b.s - sTarget) < Math.abs(a.s - sTarget) ? b : a));
  setView(st.s);
});

widthEl.addEventListener("input", () => {
  cancelAuto();
  lab.setConfig({ width: +widthEl.value });
  afterRebuild();
});
depthEl.addEventListener("input", () => {
  cancelAuto();
  const atOut = sTarget >= lab.depth - 0.01;
  lab.setConfig({ depth: +depthEl.value });
  afterRebuild();
  if (atOut) setView(lab.depth, true);
});
actEl.addEventListener("change", () => {
  cancelAuto();
  lab.setAct(actEl.value as Act);
  afterRebuild();
});
lrEl.addEventListener("input", () => {
  lab.lr = Math.pow(10, +lrEl.value);
  // a network that stopped on its own picks up again so the new step size shows
  if (!lab.training && lab.stopReason) lab.resume();
  syncControls();
});
noiseEl.addEventListener("input", () => {
  lab.setNoise(+noiseEl.value);
  if (!lab.training) lab.resume();
  syncControls();
});
gridEl.addEventListener("change", () => {
  ui.showGrid = gridEl.checked;
});

function preset(shape: ShapeId, width: number): void {
  cancelAuto();
  lab.shape = shape;
  lab.cfg = { width, depth: 4, act: "tanh" };
  lab.noise = 0.02;
  lab.lr = 0.02;
  lab.training = true;
  lab.rebuild(lab.seed);
  afterRebuild();
  setView(lab.depth, true);
}
$("preset-helix").addEventListener("click", () => preset("twisted", 3));
$("preset-stuck").addEventListener("click", () => preset("rings", 3));
$("preset-freed").addEventListener("click", () => preset("rings", 6));

const advBtn = $("btn-advanced");
function setAdv(open: boolean): void {
  adv.hidden = !open;
  advBtn.setAttribute("aria-expanded", String(open));
  placeSheets();
  onResize();
}

/** Phone: with Advanced open, the dock shrinks to the live caption and sits on top of the sheet. */
function placeSheets(): void {
  const phoneAdv = innerWidth <= 760 && !adv.hidden;
  dock.classList.toggle("mini", phoneAdv);
  dock.style.bottom = phoneAdv ? `${adv.offsetHeight}px` : "";
}
advBtn.addEventListener("click", () => setAdv(adv.hidden));
$("btn-adv-close").addEventListener("click", () => setAdv(false));

function kick(): void {
  wobbleV = Math.min(12, wobbleV + 9);
}

// ───────────────────────── intro card ─────────────────────────

const explain = $("explain");
const helpBtn = $("btn-help");
const INTRO_KEY = "folding-space:intro-seen";

function setExplain(open: boolean, remember = true): void {
  if (explain.hidden === !open) return;
  explain.hidden = !open;
  helpBtn.setAttribute("aria-expanded", String(open));
  if (open) $("btn-explain-ok").focus({ preventScroll: true });
  else if (remember) {
    try {
      localStorage.setItem(INTRO_KEY, "1");
    } catch {
      /* private mode */
    }
  }
}
helpBtn.addEventListener("click", () => setExplain(explain.hidden));
$("btn-explain-x").addEventListener("click", () => setExplain(false));
$("btn-explain-ok").addEventListener("click", () => setExplain(false));
addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !explain.hidden) setExplain(false);
});
document.addEventListener("pointerdown", (e) => {
  const t = e.target as Node;
  if (!explain.hidden && !explain.contains(t) && !helpBtn.contains(t)) setExplain(false);
}, true);
{
  let seen = false;
  try {
    seen = localStorage.getItem(INTRO_KEY) === "1";
  } catch {
    /* ignore */
  }
  if (!seen && !recording && params.get("intro") !== "0") setExplain(true, false);
}

// ───────────────────────── live caption ─────────────────────────

const caption = $("caption");
const captionText = $("caption-text");
let capKey = "";

const NOUN: Record<ShapeId, { things: string; apart: string }> = {
  twisted: { things: "strands", apart: "the strands drift apart" },
  rings: { things: "rings", apart: "the rings pull apart" },
  shell: { things: "ball and shell", apart: "the ball and shell separate" },
};

const pct = (x: number) => `${Math.round(x * 100)}%`;
/** "up from 61%" / "down from 61%" / "the same as" — mid layers can dip before the network climbs out */
const vs = (x: number, x0: number) => {
  const d = Math.round(x * 100) - Math.round(x0 * 100);
  return d > 0 ? `up from ${pct(x0)}` : d < 0 ? `down from ${pct(x0)}` : `the same as`;
};

function narrate(): [key: string, text: string, tone: "" | "busy" | "good" | "stuck"] {
  const D = lab.depth;
  const v = Math.round(ui.s);
  const acc = lab.acc;
  const W = lab.cfg.width;
  const noun = NOUN[lab.shape];
  if (ui.traced >= 0 && clock - tracedAt < 6) return ["trace", `Following one bead through ${D} layers: watch where each fold carries it.`, "good"];
  if (lab.time - lab.scrambleAt < 1.3 && lab.steps < 40) return ["random", `A random network: space is scrambled, but the ${noun.things} are still tangled.`, ""];
  const walking = clock - walkedAt < 1.6 || (phase === "walk" || phase === "rewind");
  if (walking || (phase === "done" && Math.abs(ui.s - D) > 0.05)) {
    const p = lab.probe[v] ?? 0.5, p0 = lab.probe[0] ?? 0.5;
    const wide = lab.net.sizes[v] > 3 ? ` (${lab.net.sizes[v]} neurons; showing the 3 main directions)` : "";
    if (v === 0) return [`walk0`, `Input: the raw tangle. One flat sheet could split only ${pct(p0)} of the beads here.`, "busy"];
    if (v === D) return [`walk${v}`, `Output: one flat glass sheet splits ${pct(acc)} of the beads, ${vs(acc, p0)} at the input.`, acc > 0.97 ? "good" : "busy"];
    return [`walk${v}`, `Layer ${v} of ${D}${wide}: a flat sheet could split ${pct(p)} of the beads here, ${vs(p, p0)} at the input.`, "busy"];
  }
  const knot = lab.shape !== "twisted";
  const stuck = knot && W <= 3 && lab.steps >= 150 && acc < 0.985;
  if (stuck) {
    const why = lab.shape === "rings" ? "can't pass one ring through the other" : "can't pull the ball out through the closed shell";
    return ["stuck", `Stuck at ${pct(acc)}: with only ${W} neurons per layer, the network can bend space but ${why}.`, "stuck"];
  }
  if (acc >= 0.985) {
    if (knot && W >= 4) {
      const what = lab.shape === "rings" ? "lifted one ring clear" : "lifted the ball out of the shell";
      return ["solved4", `With room for a 4th direction, the network ${what}: ${pct(acc)} correct.`, "good"];
    }
    if (knot) return ["squeezed", `${pct(acc)} correct, but only by crushing part of a ${lab.shape === "rings" ? "ring" : "shell"} flat: 3 neurons can't lift it clear.`, "good"];
    return ["solved", `Untangled: one flat glass sheet now splits amber from teal (${pct(acc)} correct).`, "good"];
  }
  if (!lab.training && lab.stopReason) return ["finished", `Done learning at ${pct(acc)} correct. Press Keep learning to train longer.`, ""];
  if (!lab.training) return ["paused", `Paused at ${pct(acc)} correct. Press Keep learning to continue.`, ""];
  if (knot && W >= 4) return ["learn4", `Learning with ${W} neurons per layer: room for a 4th direction (${pct(acc)} correct).`, "busy"];
  return ["learn", `Learning… every step nudges the folds so ${noun.apart} (${pct(acc)} correct).`, "busy"];
}

function updateCaption(): void {
  const [key, text, tone] = narrate();
  if (captionText.textContent === text) return;
  caption.dataset.tone = tone;
  captionText.textContent = text;
  if (key !== capKey) {
    capKey = key;
    captionText.style.animation = "none";
    void captionText.offsetWidth;
    captionText.style.animation = "";
  }
}

// ───────────────────────── HUD ─────────────────────────

const mAcc = $("m-acc"), fAcc = $("f-acc"), mLoss = $("m-loss"), mProbe = $("m-probe"), fProbe = $("f-probe"), mDims = $("m-dims");
const pips = $("dims-pips");
const layerVal = $("layer-val"), viewNote = $("view-note");
let hudT = 0;

function setupCanvas(c: HTMLCanvasElement): [CanvasRenderingContext2D, number, number] {
  const r = c.getBoundingClientRect();
  const d = Math.min(2, devicePixelRatio || 1);
  const w = Math.max(10, Math.round(r.width * d)), h = Math.max(10, Math.round(r.height * d));
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const g = c.getContext("2d")!;
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  return [g, r.width, r.height];
}

function drawLoss(c: HTMLCanvasElement, axes: boolean): void {
  const [g, W, H] = setupCanvas(c);
  const hst = lab.lossHist;
  if (hst.length < 2) return;
  const max = Math.max(0.7, ...hst);
  const X = (i: number) => (i / (hst.length - 1)) * (W - 2) + 1;
  const Y = (l: number) => 2 + (1 - Math.sqrt(l / max)) * (H - 4);
  if (axes) {
    g.fillStyle = "rgba(214,226,238,0.5)";
    g.font = "10px ui-monospace, Menlo, monospace";
    g.fillText(`${hst[hst.length - 1].toFixed(3)}`, W - 42, 12);
  }
  const grd = g.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, "rgba(111,231,255,0.25)");
  grd.addColorStop(1, "rgba(111,231,255,0)");
  g.fillStyle = grd;
  g.beginPath();
  g.moveTo(X(0), H);
  hst.forEach((l, i) => g.lineTo(X(i), Y(l)));
  g.lineTo(X(hst.length - 1), H);
  g.fill();
  g.strokeStyle = "#6fe7ff";
  g.lineWidth = 1.5;
  g.beginPath();
  hst.forEach((l, i) => (i ? g.lineTo(X(i), Y(l)) : g.moveTo(X(i), Y(l))));
  g.stroke();
}

function drawEig(v: number): void {
  const [g, W, H] = setupCanvas($<HTMLCanvasElement>("plot-eig"));
  const ev = lab.eig[v] ?? [];
  const n = ev.length;
  if (!n) return;
  const pad = 10, bw = Math.min(34, (W - pad * 2) / n - 6);
  g.font = "10px ui-monospace, Menlo, monospace";
  ev.forEach((x, i) => {
    const bh = Math.max(1, x * (H - 26));
    const bx = pad + i * (bw + 6);
    const grd = g.createLinearGradient(0, H - 14 - bh, 0, H - 14);
    grd.addColorStop(0, "#bff6ff");
    grd.addColorStop(1, "rgba(111,231,255,0.35)");
    g.fillStyle = grd;
    g.fillRect(bx, H - 14 - bh, bw, bh);
    g.fillStyle = "rgba(214,226,238,0.55)";
    g.fillText(`λ${i + 1}`, bx + bw / 2 - 7, H - 3);
  });
  $("eig-note").textContent = `${(lab.dims[v] ?? 0).toFixed(1)} of ${n} used`;
}

function drawProbe(v: number): void {
  const [g, W, H] = setupCanvas($<HTMLCanvasElement>("plot-probe"));
  const p = lab.probe;
  if (p.length < 2) return;
  const pl = 30, pr = 12, pt = 8, pb = 16;
  const X = (i: number) => pl + (i / (p.length - 1)) * (W - pl - pr);
  const Y = (a: number) => pt + (1 - (a - 0.4) / 0.6) * (H - pt - pb);
  g.font = "10px ui-monospace, Menlo, monospace";
  g.strokeStyle = "rgba(255,255,255,0.08)";
  g.fillStyle = "rgba(214,226,238,0.5)";
  for (const a of [0.5, 1]) {
    g.beginPath();
    g.moveTo(pl, Y(a));
    g.lineTo(W - pr, Y(a));
    g.stroke();
    g.fillText(`${a * 100}%`, 2, Y(a) + 3);
  }
  g.strokeStyle = "#6fe7ff";
  g.lineWidth = 2;
  g.beginPath();
  p.forEach((a, i) => (i ? g.lineTo(X(i), Y(a)) : g.moveTo(X(i), Y(a))));
  g.stroke();
  p.forEach((a, i) => {
    g.fillStyle = i === v ? "#ffffff" : "#6fe7ff";
    g.beginPath();
    g.arc(X(i), Y(a), i === v ? 4.5 : 3, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = "rgba(214,226,238,0.55)";
    g.fillText(i === 0 ? "in" : i === p.length - 1 ? "out" : String(i), X(i) - 6, H - 3);
  });
}

function updateHud(dt: number): void {
  const v = Math.round(ui.s);
  const D = lab.depth;
  mAcc.textContent = pct(lab.acc);
  fAcc.style.width = pct(lab.acc);
  mLoss.textContent = lab.loss.toFixed(3);
  const p = lab.probe[v] ?? 0.5;
  mProbe.textContent = pct(p);
  fProbe.style.width = pct(Math.max(0, (p - 0.5) * 2));
  const dim = lab.net.sizes[v];
  mDims.textContent = `${(lab.dims[v] ?? 0).toFixed(1)} / ${dim}`;
  if (pips.childElementCount !== dim) {
    pips.innerHTML = "";
    for (let i = 0; i < dim; i++) pips.appendChild(document.createElement("i")).appendChild(document.createElement("b"));
  }
  const ev = lab.eig[v] ?? [];
  Array.from(pips.children).forEach((el, i) => ((el.firstChild as HTMLElement).style.transform = `scaleX(${Math.max(0.02, ev[i] ?? 0)})`));
  // slider label: which layer, which stage
  const st = lab.stageAt(ui.s);
  const near = st.f < 0.5 ? st.a : st.b;
  layerVal.textContent = near.kind === "pre" ? `${lab.viewName(near.view)} · stretch` : lab.viewName(near.view);
  Array.from(ticks.children).forEach((t, i) => t.classList.toggle("on", i === v));
  const dimHere = lab.net.sizes[near.view];
  const actName = lab.cfg.act === "tanh" ? "tanh" : lab.cfg.act === "relu" ? "ReLU" : "leaky ReLU";
  viewNote.textContent =
    near.view === 0
      ? "The beads as they are: 3 numbers each (x, y, z)."
      : dimHere > 3
        ? `${dimHere} neurons: showing this layer's 3 main directions (PCA) of ${dimHere}.`
        : dimHere < 3
          ? `${dimHere} neurons: space is squashed flat onto ${dimHere === 2 ? "a plane" : "a line"}.`
          : `3 neurons: first a stretch (W·h + b), then a bend (${actName}).${near.view === D ? " The glass sheet is the final w·h + b = 0." : ""}`;
  hudT += dt;
  if (hudT > 0.12) {
    hudT = 0;
    drawLoss($<HTMLCanvasElement>("spark-loss"), false);
    if (!adv.hidden) {
      drawEig(v);
      drawProbe(v);
      drawLoss($<HTMLCanvasElement>("plot-loss"), true);
      $("steps-val").textContent = `${lab.steps} steps`;
    }
  }
  if (learnBtn.classList.contains("paused") === lab.training) syncLearn();
  updateCaption();
}

// ───────────────────────── loop ─────────────────────────

let last = performance.now();
let fpsT = 0, fpsN = 0;
let stepped = false;
let simMs = 0, renderMs = 0;

function loop(now: number): void {
  const dt = Math.min(0.5, Math.max(0, (now - last) / 1000));
  last = now;
  if (!stepped) frame(dt, undefined);
  requestAnimationFrame(loop);
}

function frame(dtRaw: number, fixedSteps: number | undefined, draw = true): void {
  const dt = Math.min(0.1, dtRaw);
  clock += dt;
  const t0 = performance.now();
  // training follows wall time (up to 0.5 s per frame) so it keeps learning at low fps
  lab.update(fixedSteps === undefined ? dtRaw : dt, fixedSteps);
  runWow(dt);
  updateSheet(dt);
  // ease the view toward the slider; spring wobble when crossing into a new layer
  if (phase === "done") ui.s += (sTarget - ui.s) * (1 - Math.exp(-dt * 10));
  const layerNow = Math.round(ui.s);
  if (layerNow !== lastLayer) {
    lastLayer = layerNow;
    kick();
  }
  // damped spring, semi-implicit with small sub-steps so it stays stable at any frame rate
  for (let left = dt; left > 1e-6; left -= 1 / 120) {
    const h = Math.min(1 / 120, left);
    wobbleV += (-260 * ui.wobble - 11 * wobbleV) * h;
    ui.wobble = Math.max(-1, Math.min(1, ui.wobble + wobbleV * h));
  }
  ui.flash = Math.max(0, ui.flash - dt * 1.1);
  const t1 = performance.now();
  try {
    if (draw) view.render(dt, lab, ui);
  } catch (err) {
    if (view.kind === "webgl") switchTo2D(`render error: ${(err as Error)?.message ?? err}`);
    else throw err;
  }
  const t2 = performance.now();
  simMs += (t1 - t0 - simMs) * 0.1;
  renderMs += (t2 - t1 - renderMs) * 0.1;
  updateHud(dt);
  frames++;
  const now = performance.now();
  if (!fpsT) fpsT = now;
  fpsN++;
  if (now - fpsT >= 500) {
    fps = (fpsN * 1000) / (now - fpsT);
    fpsT = now;
    fpsN = 0;
  }
}

// ───────────────────────── test hooks ─────────────────────────

interface FoldInfo {
  renderer: "webgl" | "2d";
  shape: ShapeId;
  width: number;
  depth: number;
  act: Act;
  layerView: number;
  step: number;
  loss: number;
  acc: number;
  probe: number[];
  dims: number[];
  frames: number;
  fps: number;
  training: boolean;
  phase: string;
  traced: number;
  sheetIn: number;
  nan: boolean;
  simMs: number;
  renderMs: number;
}

declare global {
  interface Window {
    __foldInfo: () => FoldInfo;
    __foldSelfTest: () => ReturnType<typeof selfTest>;
    __foldAdvance: (seconds: number) => FoldInfo;
    __ixRecord: { ready: boolean; setStepped: (on: boolean) => void; stepFrame: (dt: number) => void };
  }
}
const r3 = (x: number) => Math.round(x * 1000) / 1000;
window.__foldInfo = () => ({
  renderer: view?.kind ?? "webgl",
  shape: lab.shape,
  width: lab.cfg.width,
  depth: lab.cfg.depth,
  act: lab.cfg.act,
  layerView: r3(ui.s),
  step: lab.steps,
  loss: r3(lab.loss),
  acc: r3(lab.acc),
  probe: lab.probe.map(r3),
  dims: lab.dims.map((d) => Math.round(d * 100) / 100),
  frames,
  fps: Math.round(fps * 10) / 10,
  training: lab.training,
  phase,
  traced: ui.traced,
  sheetIn: r3(ui.sheetIn),
  nan: !lab.finite() || !Number.isFinite(ui.s),
  simMs: Math.round(simMs * 100) / 100,
  renderMs: Math.round(renderMs * 100) / 100,
});
window.__foldSelfTest = () => selfTest();
// Test hook: advance training + the auto-play deterministically without drawing (30 fps steps).
window.__foldAdvance = (seconds: number) => {
  for (let t = 0; t < seconds; t += 1 / 30) frame(1 / 30, 2, false);
  return window.__foldInfo();
};
// Deterministic fixed-dt stepping for scripts/record-demo.mjs ("capture": "frames").
window.__ixRecord = {
  ready: false,
  setStepped(on: boolean) {
    stepped = on;
    if (on) setExplain(false, false);
    view?.setStepped?.(on);
  },
  stepFrame(dt: number) {
    const d = Math.min(0.1, Math.max(1 / 240, dt || 1 / 30));
    frame(d, Math.round(d * STEPS_PER_SECOND));
  },
};

buildTicks();
setView(lab.depth, true);
syncControls();
void boot();
