/** Renderer-agnostic view interface (WebGL scene or Canvas2D fallback). */
import type { Lab } from "./lab";

export const AMBER = "#ff9a3c";
export const TEAL = "#2fd1c5";
export const CYAN = "#6fe7ff";

/** Where the hero cube should land on screen (CSS px). */
export interface Layout {
  cx: number;
  cy: number;
  /** Desired on-screen height of the hero cube. */
  size: number;
  phone: boolean;
  /** Show the filmstrip of per-layer cubes. */
  film: boolean;
}

/** Per-frame UI state the renderers read (the lab holds the science). */
export interface UiState {
  /** Slider position 0 (input) … depth (output), already eased. */
  s: number;
  /** Bead being traced, or −1. */
  traced: number;
  showGrid: boolean;
  /** 0..1 flash when accuracy first crosses 95 %. */
  flash: number;
  /** 0..1 sheet slide-in progress (1 = in place). */
  sheetIn: number;
  /** 0..1 spring kick on each layer change. */
  wobble: number;
}

export type PickResult = { kind: "bead"; index: number } | { kind: "film"; view: number } | { kind: "none" };

export interface View {
  readonly kind: "webgl" | "2d";
  readonly canvas: HTMLCanvasElement;
  resize(w: number, h: number, layout: Layout): void;
  render(dt: number, lab: Lab, ui: UiState): void;
  pick(clientX: number, clientY: number): PickResult;
  /** Deterministic stepping (recorder/tests): disables wall-clock adaptive quality. */
  setStepped?(on: boolean): void;
  /** Drag to turn the view (px deltas); release lets a flick coast. */
  orbit(dx: number, dy: number): void;
  release(): void;
  zoom(f: number): void;
  /** Called on user interaction (pauses idle camera drift). */
  poke(): void;
  dispose(): void;
}
