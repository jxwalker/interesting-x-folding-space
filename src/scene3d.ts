/**
 * WebGL scene (three.js): a bevelled acrylic display cube on a two-tier dark
 * marble plinth. Inside, glossy glass beads (amber / teal) sit wherever the
 * current layer of the network puts them, threaded by a silver wire lattice
 * (the space grid pushed through the same layers). At the output a frosted
 * glass sheet with a cyan rim is the classifier's exact plane. A row of small
 * glass cubes on the lower tier is a live filmstrip of every layer.
 * ACES tone mapping, bloom, PCF soft shadows, vignette and grain.
 */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { Lab } from "./lab";
import { AMBER, CYAN, TEAL, type Layout, type PickResult, type UiState, type View } from "./view";
import { dot, marble, pool } from "./textures";
import { planeBox } from "./geom";

/** Hero cube side, and the half-size of the display volume inside it. */
const CUBE = 2.0;
const IN = 0.8;
const HERO_Y = CUBE / 2 + 0.005;
const TIER1_Y = 0; // top of the upper tier
const TIER2_Y = -0.34; // top of the lower tier
const BEAD_R = 0.058; // world units
const MINI = 0.56;

const cAmber = new THREE.Color(AMBER);
const cTeal = new THREE.Color(TEAL);
const cCyan = new THREE.Color(CYAN);

const FINISH_SHADER = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uVignette: { value: 1.0 },
    uGrain: { value: 0.035 },
  },
  vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float uTime; uniform float uVignette; uniform float uGrain;
    varying vec2 vUv;
    float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      vec2 q = vUv - 0.5;
      float v = smoothstep(0.85, 0.2, length(q * vec2(1.05, 1.15)));
      c.rgb *= mix(1.0 - uVignette * 0.62, 1.0, v);
      float g = hash(vUv * 1024.0 + fract(uTime) * 91.0) - 0.5;
      c.rgb += g * uGrain;
      gl_FragColor = c;
    }`,
};

const BACKDROP_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = p.xyww;
}`;
const BACKDROP_FRAG = /* glsl */ `
varying vec3 vDir;
void main() {
  float h = vDir.y;
  vec3 navy = vec3(0.014, 0.022, 0.040);
  vec3 coal = vec3(0.010, 0.011, 0.014);
  vec3 col = mix(coal, navy, smoothstep(-0.1, 0.6, h));
  // a soft cool glow behind the cube, warmer low on the left
  float glow = pow(max(0.0, dot(vDir, normalize(vec3(0.35, 0.25, -1.0)))), 6.0);
  col += vec3(0.020, 0.034, 0.055) * glow;
  float warm = pow(max(0.0, dot(vDir, normalize(vec3(-1.0, 0.1, -0.6)))), 5.0);
  col += vec3(0.028, 0.018, 0.010) * warm;
  gl_FragColor = vec4(col, 1.0);
}`;

const LATTICE_VERT = /* glsl */ `
attribute float aPred;
varying vec3 vP;
varying float vPred;
void main() {
  vP = position;
  vPred = aPred;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const LATTICE_FRAG = /* glsl */ `
uniform float uOpacity;
uniform float uTint;
uniform float uLim;
uniform vec3 uSilver;
uniform vec3 uAmber;
uniform vec3 uTeal;
varying vec3 vP;
varying float vPred;
void main() {
  vec3 a = abs(vP);
  float m = max(a.x, max(a.y, a.z));
  if (m > uLim) discard;
  float edge = smoothstep(uLim, uLim - 0.12, m);
  float side = clamp((vPred - 0.5) * 2.0, -1.0, 1.0);
  vec3 tint = side < 0.0 ? uAmber : uTeal;
  vec3 col = mix(uSilver, tint, uTint * smoothstep(0.05, 0.9, abs(side)));
  // at the boundary itself the lattice stays bright silver
  col += uTint * vec3(0.5, 0.75, 0.85) * (1.0 - smoothstep(0.0, 0.12, abs(side))) * 0.2;
  gl_FragColor = vec4(col, uOpacity * edge);
}`;

interface MiniCube {
  group: THREE.Group;
  beads: THREE.InstancedMesh;
  edges: THREE.Mesh;
  pad: THREE.Mesh;
  glass: THREE.Mesh;
  view: number;
  lit: number;
  size: number;
}

export class Scene3D implements View {
  readonly kind = "webgl" as const;
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private finish: ShaderPass;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.1, 120);
  private w = 1;
  private h = 1;
  private dpr = 1;
  private mobile: boolean;
  private hq: boolean;
  private stepped = false;
  private layout: Layout = { cx: 640, cy: 360, size: 400, phone: false, film: true };
  /** Eased copy of the layout so panels opening/closing glide the cube. */
  private lay = { cx: 0, cy: 0, size: 0 };
  // camera rig (yaw/pitch around the hero cube, with inertia and an idle drift)
  private yaw = -0.5;
  private pitch = 0.4;
  private yawV = 0;
  private pitchV = 0;
  private zoomF = 1;
  private zoomT = 1;
  private idle = 10;
  private time = 0;
  private dragging = false;
  private lastDrag = 0;
  // hero
  private hero = new THREE.Group();
  private inner = new THREE.Group();
  private beads!: THREE.InstancedMesh;
  private beadGlow!: THREE.InstancedBufferAttribute;
  private lattice!: THREE.LineSegments;
  private latPos!: THREE.BufferAttribute;
  private latPred!: THREE.BufferAttribute;
  private latU = {
    uOpacity: { value: 0.12 },
    uTint: { value: 0 },
    uLim: { value: 1.12 },
    uSilver: { value: new THREE.Color(0.78, 0.82, 0.88) },
    uAmber: { value: cAmber.clone().multiplyScalar(1.1) },
    uTeal: { value: cTeal.clone().multiplyScalar(1.1) },
  };
  private sheet!: THREE.Mesh;
  private sheetMat!: THREE.MeshPhysicalMaterial;
  private rim: THREE.Mesh[] = [];
  private rimMat!: THREE.MeshBasicMaterial;
  private trace: THREE.Mesh | null = null;
  private traceKey = "";
  private traceMat!: THREE.MeshBasicMaterial;
  private traceDots!: THREE.InstancedMesh;
  private pools: THREE.Mesh[] = [];
  private innerLight!: THREE.PointLight;
  private dust!: THREE.Points;
  private dustBase!: Float32Array;
  // filmstrip
  private film = new THREE.Group();
  private minis: MiniCube[] = [];
  private filmKey = "";
  private glassGeo = new Map<number, THREE.BufferGeometry>();
  private edgeGeo = new Map<number, THREE.BufferGeometry>();
  private glassMat!: THREE.MeshPhysicalMaterial;
  private glassBackMat!: THREE.MeshPhysicalMaterial;
  private miniBeadGeo = new THREE.IcosahedronGeometry(1, 0);
  private miniBeadMat = new THREE.MeshStandardMaterial({ roughness: 0.2, metalness: 0, emissive: 0xffffff, emissiveIntensity: 0.0 });
  private poolTex!: THREE.Texture;
  // scratch
  private pos: Float32Array = new Float32Array(0);
  private gridPos: Float32Array = new Float32Array(0);
  private version = -1;
  private m4 = new THREE.Matrix4();
  private v3 = new THREE.Vector3();
  private s3 = new THREE.Vector3();
  private col = new THREE.Color();
  private ray = new THREE.Raycaster();
  private fpsAcc = 0;
  private fpsFrames = 0;
  private lowFor = 0;

  constructor(container: HTMLElement, opts: { hq?: boolean } = {}) {
    this.hq = !!opts.hq;
    this.mobile = matchMedia("(pointer: coarse)").matches || Math.min(innerWidth, innerHeight) < 600;
    const canvas = document.createElement("canvas");
    canvas.id = "canvas";
    canvas.setAttribute("aria-label", "Acrylic cube showing the beads at the current layer. Tap a bead to trace its journey; drag to turn the view.");
    container.appendChild(canvas);
    this.canvas = canvas;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance", alpha: false, preserveDrawingBuffer: false });
    } catch (e) {
      canvas.remove();
      throw e;
    }
    this.renderer = renderer;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    this.dpr = Math.min(window.devicePixelRatio || 1, this.mobile ? 1.5 : 2);
    renderer.setPixelRatio(this.dpr);

    this.scene.fog = new THREE.FogExp2(0x06090f, 0.035);
    this.buildEnvironment();
    this.buildPlinth();
    this.buildHero();
    this.buildFx();
    this.scene.add(this.film);

    const dbg0 = new URLSearchParams(location.search).get("dbg") ?? "";
    if (dbg0.includes("noshadow")) renderer.shadowMap.enabled = false;
    const samples = this.mobile || dbg0.includes("nomsaa") ? 0 : 4;
    const rt = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples });
    this.composer = new EffectComposer(renderer, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.4, 0.45, 0.9);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.finish = new ShaderPass(FINISH_SHADER);
    this.composer.addPass(this.finish);

    // debug knobs for look-dev: ?dbg=noenv,norim,nokey,nofill,nobloom
    const dbg = new URLSearchParams(location.search).get("dbg") ?? "";
    if (dbg.includes("noenv")) this.scene.environmentIntensity = 0;
    if (dbg.includes("nobloom")) this.bloom.enabled = false;
    if (dbg.includes("nodust")) this.dust.visible = false;
    if (dbg.includes("nofilm")) this.film.visible = false;
    if (dbg.includes("nohero")) this.inner.visible = false;
    if (dbg.includes("nopoint")) this.innerLight.visible = false;
    this.scene.traverse((o) => {
      if ((o as THREE.SpotLight).isSpotLight && dbg.includes("nokey")) o.visible = false;
      if ((o as THREE.DirectionalLight).isDirectionalLight && dbg.includes(o.name)) o.visible = false;
    });
  }

  // ───────────────────────────── build ─────────────────────────────

  private buildEnvironment(): void {
    // a dark studio with a few softboxes, placed (high, to the sides and behind the
    // camera) so they streak the acrylic and beads but don't glare off the plinth
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const env = new THREE.Scene();
    env.add(new THREE.Mesh(new THREE.SphereGeometry(20, 24, 12), new THREE.MeshBasicMaterial({ color: 0x07090c, side: THREE.BackSide })));
    const box = (w: number, h: number, c: number, k: number, pos: [number, number, number]) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(k), side: THREE.DoubleSide }));
      m.position.set(...pos);
      m.lookAt(0, 0, 0);
      env.add(m);
    };
    box(10, 6, 0xfff3e6, 2.6, [0, 14, 2]); // overhead
    box(2.2, 9, 0xdfeaff, 2.2, [11, 6, 4]); // right strip
    box(2.2, 9, 0xffd6a8, 1.8, [-11, 6, 1]); // warm left strip
    box(12, 3, 0xeaf2ff, 1.3, [-2, 3, 13]); // behind the camera: front-face highlights
    box(5, 2, 0x8fb6ff, 0.8, [6, 9, -9]); // cool kicker, high behind
    const envRT = pmrem.fromScene(env, 0.02);
    this.scene.environment = envRT.texture;
    this.scene.environmentIntensity = 0.9;
    pmrem.dispose();
    env.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
      }
    });

    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(60, 32, 16),
      new THREE.ShaderMaterial({ vertexShader: BACKDROP_VERT, fragmentShader: BACKDROP_FRAG, side: THREE.BackSide, depthWrite: false, fog: false }),
    );
    sky.renderOrder = -10;
    this.scene.add(sky);

    // key: soft overhead spot with soft shadows
    const key = new THREE.SpotLight(0xfff1e0, 30, 30, 0.25, 0.8, 1.6);
    key.position.set(0.8, 9.5, 1.6);
    key.target.position.set(0, 0.6, 0);
    key.castShadow = true;
    const sm = this.mobile ? 1024 : 2048;
    key.shadow.mapSize.set(sm, sm);
    key.shadow.camera.near = 4;
    key.shadow.camera.far = 16;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.015;
    key.shadow.radius = 5;
    this.scene.add(key, key.target);
    // warm rim from back-left, cool blue fill from the right
    const rim = new THREE.DirectionalLight(0xffb070, 1.8);
    rim.position.set(-6, 4, -5);
    rim.name = "norim";
    const fill = new THREE.DirectionalLight(0x6f9cff, 0.9);
    fill.position.set(6, 2.5, 4);
    fill.name = "nofill";
    // low cool front fill so the plinth's bevels and faces read
    const front = new THREE.DirectionalLight(0xb4c8ff, 1.5);
    front.position.set(-3, 1.6, 7);
    front.name = "nofront";
    this.scene.add(rim, fill, front, new THREE.HemisphereLight(0x2a3a58, 0x0a0806, 0.5));
    this.innerLight = new THREE.PointLight(0xfff0dd, 1.2, 1.7, 2);
    this.innerLight.position.set(0, HERO_Y + 0.2, 0);
    this.scene.add(this.innerLight);
  }

  private buildPlinth(): void {
    const tex = marble(this.mobile ? 512 : 1024);
    const map = new THREE.CanvasTexture(tex.color);
    map.colorSpace = THREE.SRGBColorSpace;
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    map.anisotropy = 4;
    const rough = new THREE.CanvasTexture(tex.rough);
    rough.wrapS = rough.wrapT = THREE.RepeatWrapping;
    const mat = (rep: number) => {
      const m = map.clone();
      m.repeat.set(rep, rep);
      m.needsUpdate = true;
      const r = rough.clone();
      r.repeat.set(rep, rep);
      r.needsUpdate = true;
      return new THREE.MeshPhysicalMaterial({ map: m, roughnessMap: r, roughness: 0.6, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.22, envMapIntensity: 0.5 });
    };
    const top = new THREE.Mesh(new RoundedBoxGeometry(3.3, 0.34, 3.3, 3, 0.03), mat(0.9));
    top.position.set(0, TIER1_Y - 0.17, 0);
    const low = new THREE.Mesh(new RoundedBoxGeometry(11, 0.4, 7.2, 3, 0.04), mat(2.2));
    (low.material as THREE.MeshPhysicalMaterial).clearcoat = 0.25;
    (low.material as THREE.MeshPhysicalMaterial).color.setScalar(0.7);
    low.position.set(-1.6, TIER2_Y - 0.2, 1.3);
    for (const m of [top, low]) {
      m.receiveShadow = true;
      this.scene.add(m);
    }
    // polished bevels catching the light along the top edges of both tiers
    const bevel = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.55, 0.6, 0.7), transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false });
    const edge = (len: number, x: number, y: number, z: number, alongX: boolean) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(alongX ? len : 0.008, 0.008, alongX ? 0.008 : len), bevel);
      m.position.set(x, y, z);
      this.scene.add(m);
    };
    const t1 = TIER1_Y - 0.012;
    edge(3.24, 0, t1, 1.645, true);
    edge(3.24, 1.645, t1, 0, false);
    edge(3.24, -1.645, t1, 0, false);
    edge(3.24, 0, t1, -1.645, true);
    const t2 = TIER2_Y - 0.014;
    edge(10.9, -1.6, t2, 1.3 + 3.595, true);
    // floor far below so the plinth doesn't float
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x040507, roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = TIER2_Y - 0.4;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // fake caustics: soft amber + teal light pools on the plinth that follow the beads
    this.poolTex = new THREE.CanvasTexture(pool());
    this.poolTex.colorSpace = THREE.SRGBColorSpace;
    for (const c of [cAmber, cTeal, cAmber, cTeal]) {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({ map: this.poolTex, color: c.clone(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.5 }),
      );
      m.rotation.x = -Math.PI / 2;
      m.renderOrder = 1;
      this.pools.push(m);
      this.scene.add(m);
    }
  }

  private glassGeometry(size: number): { glass: THREE.BufferGeometry; edges: THREE.BufferGeometry } {
    let glass = this.glassGeo.get(size);
    let edges = this.edgeGeo.get(size);
    if (!glass || !edges) {
      const bevel = size * 0.035;
      glass = new RoundedBoxGeometry(size, size, size, 3, bevel);
      // 12 thin bright bars along the bevels: polished edges catching the light
      const parts: THREE.BufferGeometry[] = [];
      const h = size / 2 - bevel * 0.3, t = Math.max(0.004, size * 0.0045);
      for (let axis = 0; axis < 3; axis++) {
        for (const a of [-1, 1]) {
          for (const b of [-1, 1]) {
            const dims = [t, t, t];
            dims[axis] = size - bevel * 1.6;
            const g = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
            const p = [0, 0, 0];
            p[(axis + 1) % 3] = a * h;
            p[(axis + 2) % 3] = b * h;
            g.translate(p[0], p[1], p[2]);
            // brighter on top edges (key light), dimmer underneath
            const yTop = p[1] > 0 || axis === 1 ? 1 : 0.45;
            const col = new Float32Array(g.attributes.position.count * 3).fill(yTop);
            g.setAttribute("color", new THREE.BufferAttribute(col, 3));
            parts.push(g);
          }
        }
      }
      edges = mergeGeometries(parts)!;
      parts.forEach((g) => g.dispose());
      this.glassGeo.set(size, glass);
      this.edgeGeo.set(size, edges);
    }
    return { glass, edges };
  }

  /** A thick-acrylic look without full transmission: faint body + env reflections + bright bevels. */
  private makeGlass(size: number, edgeBright: number): { group: THREE.Group; glass: THREE.Mesh; edges: THREE.Mesh } {
    const { glass, edges } = this.glassGeometry(size);
    const group = new THREE.Group();
    const back = new THREE.Mesh(glass, this.glassBackMat);
    back.renderOrder = 3;
    const front = new THREE.Mesh(glass, this.glassMat);
    front.renderOrder = 5;
    const em = new THREE.MeshBasicMaterial({ vertexColors: true, color: new THREE.Color(0.75, 0.86, 1.0).multiplyScalar(edgeBright), transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false });
    const e = new THREE.Mesh(edges, em);
    e.renderOrder = 6;
    group.add(back, front, e);
    return { group, glass: front, edges: e };
  }

  private buildHero(): void {
    this.glassMat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 0,
      roughness: 0.04,
      transparent: true,
      opacity: 0.13,
      clearcoat: 1,
      clearcoatRoughness: 0.03,
      envMapIntensity: 2.6,
      ior: 1.49,
      specularIntensity: 1,
      depthWrite: false,
      side: THREE.FrontSide,
    });
    this.glassBackMat = this.glassMat.clone();
    this.glassBackMat.side = THREE.BackSide;
    this.glassBackMat.opacity = 0.06;
    this.glassBackMat.envMapIntensity = 1.4;

    this.hero.position.set(0, HERO_Y, 0);
    this.scene.add(this.hero);
    const { group } = this.makeGlass(CUBE, 0.9);
    this.hero.add(group);
    // the cube's base: a thin darker slab like a display base
    const base = new THREE.Mesh(new RoundedBoxGeometry(CUBE * 1.02, 0.03, CUBE * 1.02, 2, 0.012), new THREE.MeshPhysicalMaterial({ color: 0x111317, roughness: 0.3, clearcoat: 1 }));
    base.position.y = -CUBE / 2 - 0.005;
    base.receiveShadow = true;
    this.hero.add(base);
    this.inner.scale.setScalar(IN);
    this.hero.add(this.inner);

    // beads (allocated in rebuild)
    // lattice
    const latMat = new THREE.ShaderMaterial({
      uniforms: this.latU,
      vertexShader: LATTICE_VERT,
      fragmentShader: LATTICE_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    this.lattice = new THREE.LineSegments(new THREE.BufferGeometry(), latMat);
    this.lattice.frustumCulled = false;
    this.lattice.renderOrder = 2;
    this.inner.add(this.lattice);

    // decision sheet: a convex polygon (plane ∩ cube), frosted glass + cyan rim
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(8 * 3), 3));
    sg.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(8 * 3), 3));
    sg.setIndex(new THREE.BufferAttribute(new Uint16Array(6 * 3), 1));
    this.sheetMat = new THREE.MeshPhysicalMaterial({
      color: 0xcff6ff,
      roughness: 0.42,
      metalness: 0,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
      clearcoat: 0.6,
      clearcoatRoughness: 0.35,
      emissive: cCyan.clone(),
      emissiveIntensity: 0.06,
      envMapIntensity: 1.6,
    });
    this.sheet = new THREE.Mesh(sg, this.sheetMat);
    this.sheet.renderOrder = 4;
    this.sheet.frustumCulled = false;
    this.inner.add(this.sheet);
    this.rimMat = new THREE.MeshBasicMaterial({ color: cCyan.clone().multiplyScalar(2.2), transparent: true, opacity: 0, toneMapped: false, depthWrite: false, blending: THREE.AdditiveBlending });
    const rimGeo = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
    rimGeo.translate(0, 0.5, 0);
    for (let i = 0; i < 6; i++) {
      const r = new THREE.Mesh(rimGeo, this.rimMat);
      r.renderOrder = 5;
      r.visible = false;
      this.rim.push(r);
      this.inner.add(r);
    }

    // trace thread + keyframe dots
    this.traceMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 1.9, 2.2), toneMapped: false, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
    this.traceDots = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), this.traceMat, 16);
    this.traceDots.count = 0;
    this.traceDots.frustumCulled = false;
    this.inner.add(this.traceDots);
  }

  private buildBeads(n: number): void {
    if (this.beads) {
      this.inner.remove(this.beads);
      this.beads.dispose();
    }
    const geo = new THREE.SphereGeometry(1, this.mobile ? 12 : 16, this.mobile ? 8 : 11);
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.1,
      metalness: 0,
      clearcoat: 1,
      clearcoatRoughness: 0.04,
      envMapIntensity: 1.5,
      ior: 1.5,
      specularIntensity: 1,
    });
    this.beadGlow = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    this.beadGlow.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aGlow", this.beadGlow);
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nattribute float aGlow;\nvarying float vGlow;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\nvGlow = aGlow;");
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying float vGlow;")
        .replace(
          "#include <emissivemap_fragment>",
          `#include <emissivemap_fragment>
          {
            float ndv = clamp(dot(normalize(vNormal), normalize(vViewPosition)), 0.0, 1.0);
            // glass bead: a lit core (light focused through the sphere) and a glowing rim
            float core = pow(ndv, 3.0);
            float rimF = pow(1.0 - ndv, 2.5);
            totalEmissiveRadiance += vColor.rgb * (0.05 + 0.24 * core + 0.24 * rimF);
            totalEmissiveRadiance += vColor.rgb * vGlow * (0.5 + 1.2 * rimF);
            totalEmissiveRadiance += vec3(0.9, 1.0, 1.1) * max(0.0, vGlow - 1.0) * 0.8;
          }`,
        );
    };
    this.beads = new THREE.InstancedMesh(geo, mat, n);
    this.beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.beads.castShadow = true;
    this.beads.frustumCulled = false;
    this.inner.add(this.beads);
  }

  private buildLattice(lab: Lab): void {
    const g = this.lattice.geometry;
    const G = lab.grid.count;
    this.latPos = new THREE.BufferAttribute(new Float32Array(G * 3), 3);
    this.latPos.setUsage(THREE.DynamicDrawUsage);
    this.latPred = new THREE.BufferAttribute(new Float32Array(G), 1);
    this.latPred.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("position", this.latPos);
    g.setAttribute("aPred", this.latPred);
    g.setIndex(new THREE.BufferAttribute(lab.grid.segs, 1));
  }

  private buildFilm(lab: Lab): void {
    for (const m of this.minis) {
      this.film.remove(m.group);
      m.beads.dispose();
      (m.edges.material as THREE.Material).dispose();
      (m.pad.material as THREE.Material).dispose();
    }
    this.minis = [];
    const count = lab.depth + 1;
    const span = this.layout.phone ? 3.4 : 4.4;
    const step = Math.min(1.05, span / Math.max(1, count - 1));
    const size = Math.min(MINI, step * 0.66);
    for (let v = 0; v < count; v++) {
      const { group, glass, edges } = this.makeGlass(size, 0.9);
      const beads = new THREE.InstancedMesh(this.miniBeadGeo, this.miniBeadMat, lab.ds.n);
      beads.frustumCulled = false;
      beads.castShadow = false;
      const inner = new THREE.Group();
      inner.scale.setScalar(size * 0.4);
      inner.add(beads);
      group.add(inner);
      const pad = new THREE.Mesh(
        new THREE.PlaneGeometry(size * 2.4, size * 2.4),
        new THREE.MeshBasicMaterial({ map: this.poolTex, color: cCyan.clone(), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
      );
      pad.rotation.x = -Math.PI / 2;
      pad.position.y = -size / 2 + 0.004;
      group.add(pad);
      glass.userData.view = v;
      this.minis.push({ group, beads, edges, pad, glass, view: v, lit: 0, size });
      this.film.add(group);
    }
    this.placeFilm(lab);
  }

  private placeFilm(lab: Lab): void {
    const count = lab.depth + 1;
    const phone = this.layout.phone;
    const span = phone ? 3.4 : 4.4;
    const step = Math.min(1.05, span / Math.max(1, count - 1));
    const size = Math.min(MINI, step * 0.66);
    this.minis.forEach((m, v) => {
      const u = count > 1 ? v / (count - 1) - 0.5 : 0;
      const w = step * (count - 1);
      if (phone) m.group.position.set(u * w, TIER2_Y + size / 2 + 0.002, 2.25);
      else m.group.position.set(-0.7 + u * w, TIER2_Y + size / 2 + 0.002, 2.25 + u * 0.3);
      m.group.rotation.y = phone ? 0 : -0.1;
    });
    this.film.visible = this.layout.film && !location.search.includes("nofilm");
  }

  private buildFx(): void {
    // dust motes drifting in the light
    const N = this.mobile ? 220 : 420;
    const p = new Float32Array(N * 3);
    let s = 1234567;
    const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < N; i++) {
      p[i * 3] = (r() - 0.5) * 12;
      p[i * 3 + 1] = -0.2 + r() * 5.5;
      p[i * 3 + 2] = (r() - 0.5) * 9 - 1;
    }
    this.dustBase = p.slice();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    const tex = new THREE.CanvasTexture(dot());
    this.dust = new THREE.Points(
      g,
      new THREE.PointsMaterial({ map: tex, size: 0.03, transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending, color: 0xcfe3ff, sizeAttenuation: true }),
    );
    this.dust.frustumCulled = false;
    this.scene.add(this.dust);
  }

  private rebuild(lab: Lab): void {
    this.version = lab.version;
    const n = lab.ds.n;
    if (!this.beads || this.beads.count !== n) this.buildBeads(n);
    for (let i = 0; i < n; i++) this.beads.setColorAt(i, lab.ds.y[i] ? cTeal : cAmber);
    this.beads.instanceColor!.needsUpdate = true;
    if (!this.latPos || this.latPos.count !== lab.grid.count) this.buildLattice(lab);
    this.pos = new Float32Array(n * 3);
    this.gridPos = this.latPos.array as Float32Array;
    const key = `${lab.depth}:${n}:${this.layout.phone}`;
    if (key !== this.filmKey) {
      this.filmKey = key;
      this.buildFilm(lab);
    }
    for (const m of this.minis) {
      for (let i = 0; i < n; i++) m.beads.setColorAt(i, lab.ds.y[i] ? cTeal : cAmber);
      m.beads.instanceColor!.needsUpdate = true;
    }
    this.traceKey = "";
  }

  // ───────────────────────────── view API ─────────────────────────────

  resize(w: number, h: number, layout: Layout): void {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    const phoneChanged = layout.phone !== this.layout.phone;
    if (!this.lay.size || phoneChanged) this.lay = { cx: layout.cx, cy: layout.cy, size: layout.size };
    this.layout = layout;
    this.renderer.setPixelRatio(this.dpr);
    this.renderer.setSize(this.w, this.h, false);
    this.composer.setPixelRatio(this.dpr);
    this.composer.setSize(this.w, this.h);
    const bloomScale = this.mobile ? 0.25 : 0.5;
    this.bloom.setSize(this.w * this.dpr * bloomScale, this.h * this.dpr * bloomScale);
    this.camera.aspect = this.w / this.h;
    if (phoneChanged) {
      this.filmKey = "";
      this.version = -1;
    }
    this.film.visible = layout.film && !location.search.includes("nofilm");
  }

  poke(): void {
    this.idle = 0;
  }

  setStepped(on: boolean): void {
    this.stepped = on;
    if (on) {
      this.dpr = 1;
      this.resize(this.w, this.h, this.layout);
    }
  }

  orbit(dx: number, dy: number): void {
    this.idle = 0;
    this.dragging = true;
    this.lastDrag = this.time;
    const ky = dx * 0.006, kp = dy * 0.004;
    this.yaw -= ky;
    this.pitch = THREE.MathUtils.clamp(this.pitch + kp, 0.06, 0.78);
    // remember a velocity for the flick
    this.yawV = THREE.MathUtils.lerp(this.yawV, -ky * 60, 0.5);
    this.pitchV = THREE.MathUtils.lerp(this.pitchV, kp * 60, 0.5);
  }

  release(): void {
    this.dragging = false;
    if (this.time - this.lastDrag > 0.12) {
      this.yawV = 0;
      this.pitchV = 0;
    }
  }

  zoom(f: number): void {
    this.idle = 0;
    this.zoomT = THREE.MathUtils.clamp(this.zoomT * f, 0.65, 1.5);
  }

  pick(clientX: number, clientY: number): PickResult {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    // beads: nearest projected centre within a finger-sized radius, front-most wins ties
    if (this.beads && this.pos.length) {
      this.inner.updateMatrixWorld();
      const mw = this.inner.matrixWorld;
      const v = this.v3;
      let best = -1, bestScore = Infinity;
      const rad = this.layout.phone ? 22 : 16;
      for (let i = 0; i < this.pos.length / 3; i++) {
        v.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]).applyMatrix4(mw).project(this.camera);
        const sx = (v.x * 0.5 + 0.5) * this.w, sy = (-v.y * 0.5 + 0.5) * this.h;
        const d = Math.hypot(sx - x, sy - y);
        if (d < rad) {
          const score = d + v.z * 400; // prefer nearer beads
          if (score < bestScore) {
            bestScore = score;
            best = i;
          }
        }
      }
      if (best >= 0) return { kind: "bead", index: best };
    }
    if (this.film.visible && this.minis.length) {
      this.ray.setFromCamera(new THREE.Vector2((x / this.w) * 2 - 1, -(y / this.h) * 2 + 1), this.camera);
      const hit = this.ray.intersectObjects(this.minis.map((m) => m.glass), false)[0];
      if (hit) return { kind: "film", view: hit.object.userData.view as number };
    }
    return { kind: "none" };
  }

  // ───────────────────────────── frame ─────────────────────────────

  private updateCamera(dt: number): void {
    this.idle += dt;
    if (!this.dragging) {
      // flick inertia, then a gentle idle drift (±10° yaw, slight push-in)
      this.yaw += this.yawV * dt;
      this.pitch = THREE.MathUtils.clamp(this.pitch + this.pitchV * dt, 0.06, 0.78);
      const damp = Math.exp(-dt * 2.4);
      this.yawV *= damp;
      this.pitchV *= damp;
    }
    const drift = THREE.MathUtils.smoothstep(this.idle, 4, 7);
    const t = this.time;
    const yawD = drift * 0.17 * Math.sin(t * 0.16);
    const pitchD = drift * 0.025 * Math.sin(t * 0.11 + 1);
    if (drift > 0) {
      // ease back toward the composed angle while idle
      this.yaw += (-0.5 - this.yaw) * (1 - Math.exp(-dt * 0.35)) * drift;
      this.pitch += (0.4 - this.pitch) * (1 - Math.exp(-dt * 0.35)) * drift;
    }
    this.zoomF += (this.zoomT - this.zoomF) * (1 - Math.exp(-dt * 6));
    const push = 1 - drift * 0.03 * (0.5 + 0.5 * Math.sin(t * 0.13));

    const k = 1 - Math.exp(-dt * 5);
    const L = this.lay;
    L.cx += (this.layout.cx - L.cx) * k;
    L.cy += (this.layout.cy - L.cy) * k;
    L.size += (this.layout.size - L.size) * k;
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    // distance so the cube (≈1.45 × side tall on screen at this pitch) is L.size px
    const dist = ((CUBE * 1.42) / 2 / Math.tan(fov / 2)) * (this.h / Math.max(80, L.size)) * this.zoomF * push;
    const yaw = this.yaw + yawD, pitch = this.pitch + pitchD;
    const target = new THREE.Vector3(0, HERO_Y, 0);
    this.camera.position.set(target.x + Math.sin(yaw) * Math.cos(pitch) * dist, target.y + Math.sin(pitch) * dist, target.z + Math.cos(yaw) * Math.cos(pitch) * dist);
    this.camera.lookAt(target);
    // shift the image so the cube centre lands at (cx, cy)
    this.camera.setViewOffset(this.w, this.h, this.w / 2 - L.cx, this.h / 2 - L.cy, this.w, this.h);
    this.camera.updateProjectionMatrix();
  }

  render(dt: number, lab: Lab, ui: UiState): void {
    this.time += dt;
    if (lab.version !== this.version) this.rebuild(lab);
    this.updateCamera(dt);
    const n = lab.ds.n;
    const D = lab.depth;
    const s = ui.s;

    // ── beads: interpolated layer view, scramble burst, glow for traced / misclassified
    lab.sample(s, this.pos, ui.showGrid ? this.gridPos : null);
    const scr = Math.max(0, 1 - (lab.time - lab.scrambleAt) / 0.9);
    const burst = scr * scr * 0.9;
    const outW = THREE.MathUtils.smoothstep(s, D - 0.6, D);
    const r = BEAD_R / IN;
    const glow = this.beadGlow.array as Float32Array;
    for (let i = 0; i < n; i++) {
      let x = this.pos[i * 3], y = this.pos[i * 3 + 1], z = this.pos[i * 3 + 2];
      if (burst > 0) {
        const a = i * 2.399, b = i * 0.7071;
        x += Math.sin(a + lab.time * 9) * burst * Math.cos(b);
        y += Math.cos(a * 1.3 + lab.time * 7) * burst;
        z += Math.sin(b * 3.1 + lab.time * 8) * burst;
        this.pos[i * 3] = x;
        this.pos[i * 3 + 1] = y;
        this.pos[i * 3 + 2] = z;
      }
      const wrong = (lab.pred[i] > 0.5 ? 1 : 0) !== lab.ds.y[i];
      let gl = wrong ? (0.25 + 0.3 * Math.sin(this.time * 7 + i)) * (0.4 + 0.6 * outW) : 0;
      let sc = r;
      if (i === ui.traced) {
        gl = 1.6 + 0.3 * Math.sin(this.time * 5);
        sc = r * 1.5;
      } else if (ui.traced >= 0) sc = r * 0.92;
      glow[i] = gl;
      this.m4.makeScale(sc, sc, sc).setPosition(x, y, z);
      this.beads.setMatrixAt(i, this.m4);
    }
    this.beads.instanceMatrix.needsUpdate = true;
    this.beadGlow.needsUpdate = true;

    // ── lattice
    this.lattice.visible = ui.showGrid;
    if (ui.showGrid) {
      this.latPos.needsUpdate = true;
      (this.latPred.array as Float32Array).set(lab.gridPred);
      this.latPred.needsUpdate = true;
      this.latU.uTint.value = outW;
      this.latU.uOpacity.value = 0.16 - 0.06 * outW;
    }

    // ── decision sheet (exact plane w·h + b = 0 at the output view)
    const accW = THREE.MathUtils.smoothstep(lab.acc, 0.72, 0.93);
    const sheetA = outW * accW * ui.sheetIn * (lab.plane.ok ? 1 : 0);
    this.updateSheet(lab, sheetA, ui);

    // ── trace thread
    this.updateTrace(lab, ui);

    // ── jelly wobble on layer change
    const wb = ui.wobble * 0.06;
    this.hero.scale.set(1 - wb * 0.5, 1 + wb, 1 - wb * 0.5);
    this.hero.position.y = HERO_Y + wb * CUBE * 0.5;

    // ── light pools follow each class's centroid
    let ax = 0, az = 0, bx = 0, bz = 0, na = 0, nb = 0;
    for (let i = 0; i < n; i++) {
      if (lab.ds.y[i]) {
        bx += this.pos[i * 3];
        bz += this.pos[i * 3 + 2];
        nb++;
      } else {
        ax += this.pos[i * 3];
        az += this.pos[i * 3 + 2];
        na++;
      }
    }
    ax /= Math.max(1, na);
    az /= Math.max(1, na);
    bx /= Math.max(1, nb);
    bz /= Math.max(1, nb);
    const apart = Math.min(1, Math.hypot(ax - bx, az - bz));
    const pl = [
      [ax * IN * 1.1 + 0.15, az * IN * 1.1 + 0.25, TIER1_Y + 0.003, 2.0, 0.42 + 0.25 * apart],
      [bx * IN * 1.1 + 0.15, bz * IN * 1.1 + 0.25, TIER1_Y + 0.003, 2.0, 0.42 + 0.25 * apart],
      [ax * 1.2 - 1.2, 2.2 + az * 0.4, TIER2_Y + 0.003, 3.0, 0.1],
      [bx * 1.2 + 0.8, 2.4 + bz * 0.4, TIER2_Y + 0.003, 3.0, 0.09],
    ];
    this.pools.forEach((m, k) => {
      const [px, pz, py, sz, op] = pl[k];
      m.position.set(px, py, pz);
      m.scale.setScalar(sz * (1 + 0.04 * Math.sin(this.time * 1.3 + k)));
      (m.material as THREE.MeshBasicMaterial).opacity = op * (1 + ui.flash * 0.8);
    });
    this.innerLight.intensity = 1.0 + ui.flash * 3;

    // ── filmstrip: each mini cube shows its layer's beads; the current one glows
    if (this.film.visible) this.updateFilm(lab, ui, dt);

    // ── dust
    const dp = (this.dust.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
    const t = this.time;
    for (let i = 0; i < dp.length; i += 3) {
      dp[i] = this.dustBase[i] + Math.sin(t * 0.07 + i) * 0.3;
      dp[i + 1] = this.dustBase[i + 1] + Math.sin(t * 0.05 + i * 0.3) * 0.25 + ((t * 0.03 + i * 0.001) % 1) * 0.2;
      dp[i + 2] = this.dustBase[i + 2] + Math.cos(t * 0.06 + i * 0.7) * 0.3;
    }
    this.dust.geometry.attributes.position.needsUpdate = true;

    this.bloom.strength = 0.38 + ui.flash * 0.8;
    this.finish.uniforms.uTime.value = this.time;
    this.composer.render(dt);
    this.adaptQuality(dt);
  }

  private updateSheet(lab: Lab, alpha: number, ui: UiState): void {
    const show = alpha > 0.004;
    this.sheet.visible = show;
    this.rim.forEach((r) => (r.visible = false));
    if (!show) return;
    const [nx, ny, nz] = lab.plane.n;
    // slide in from the teal side
    const d = lab.plane.d - (1 - ui.sheetIn) * 1.4;
    const lim = 1.0 / IN - 0.02; // the inside of the glass, in display units
    const pts = planeBox(nx, ny, nz, d, lim);
    if (pts.length < 3) {
      this.sheet.visible = false;
      return;
    }
    const geo = this.sheet.geometry;
    const P = geo.attributes.position as THREE.BufferAttribute;
    const N = geo.attributes.normal as THREE.BufferAttribute;
    const idx = geo.index!;
    // centre + fan
    let cx = 0, cy = 0, cz = 0;
    for (const p of pts) {
      cx += p[0];
      cy += p[1];
      cz += p[2];
    }
    cx /= pts.length;
    cy /= pts.length;
    cz /= pts.length;
    P.setXYZ(0, cx, cy, cz);
    N.setXYZ(0, nx, ny, nz);
    pts.forEach((p, k) => {
      P.setXYZ(k + 1, p[0], p[1], p[2]);
      N.setXYZ(k + 1, nx, ny, nz);
    });
    for (let k = 0; k < 6; k++) {
      if (k < pts.length) idx.setXYZ(k, 0, k + 1, ((k + 1) % pts.length) + 1);
      else idx.setXYZ(k, 0, 0, 0);
    }
    P.needsUpdate = true;
    N.needsUpdate = true;
    idx.needsUpdate = true;
    geo.setDrawRange(0, pts.length * 3);
    const shimmer = 0.5 + 0.5 * Math.sin(this.time * 2.2);
    this.sheetMat.opacity = 0.3 * alpha + ui.flash * 0.12;
    this.sheetMat.emissiveIntensity = 0.05 + 0.05 * shimmer * alpha + ui.flash * 0.5;
    // rim bars
    this.rimMat.opacity = alpha * (0.75 + 0.25 * shimmer);
    const up = new THREE.Vector3(0, 1, 0);
    pts.forEach((p, k) => {
      const q = pts[(k + 1) % pts.length];
      const r = this.rim[k];
      r.visible = true;
      const a = this.v3.set(p[0], p[1], p[2]);
      const dir = this.s3.set(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      const len = dir.length();
      r.position.copy(a);
      r.quaternion.setFromUnitVectors(up, dir.normalize());
      r.scale.set(0.012, len, 0.012);
    });
  }

  private updateTrace(lab: Lab, ui: UiState): void {
    if (ui.traced < 0 || ui.traced >= lab.ds.n) {
      if (this.trace) this.trace.visible = false;
      this.traceDots.count = 0;
      this.traceKey = "";
      return;
    }
    // rebuild a few times a second while learning (the path drifts), else on change
    const key = `${ui.traced}:${lab.version}:${lab.training ? Math.floor(this.time * 4) : lab.steps}`;
    if (key !== this.traceKey) {
      this.traceKey = key;
      const pts = lab.journey(ui.traced).map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      const tube = new THREE.TubeGeometry(curve, Math.max(24, pts.length * 14), 0.011, 6, false);
      if (!this.trace) {
        this.trace = new THREE.Mesh(tube, this.traceMat);
        this.trace.frustumCulled = false;
        this.trace.renderOrder = 6;
        this.inner.add(this.trace);
      } else {
        this.trace.geometry.dispose();
        this.trace.geometry = tube;
      }
      this.traceDots.count = Math.min(16, pts.length);
      pts.slice(0, 16).forEach((p, k) => {
        const post = lab.stages[k].kind === "post";
        const sc = post ? 0.03 : 0.018;
        this.m4.makeScale(sc, sc, sc).setPosition(p);
        this.traceDots.setMatrixAt(k, this.m4);
      });
      this.traceDots.instanceMatrix.needsUpdate = true;
    }
    this.trace!.visible = true;
    this.traceMat.opacity = 0.65 + 0.25 * Math.sin(this.time * 3);
  }

  private updateFilm(lab: Lab, ui: UiState, dt: number): void {
    const n = lab.ds.n;
    const active = Math.round(ui.s);
    for (const m of this.minis) {
      const st = lab.stages[lab.viewStage[m.view]];
      if (!st) continue;
      const on = m.view === active ? 1 : 0;
      m.lit += (on - m.lit) * (1 - Math.exp(-dt * 6));
      const rb = 0.042;
      for (let i = 0; i < n; i++) {
        const tr = i === ui.traced;
        const sc = tr ? rb * 2.4 : rb;
        this.m4.makeScale(sc, sc, sc).setPosition(st.data[i * 3], st.data[i * 3 + 1], st.data[i * 3 + 2]);
        m.beads.setMatrixAt(i, this.m4);
      }
      m.beads.instanceMatrix.needsUpdate = true;
      if (ui.traced >= 0 && ui.traced < n) {
        m.beads.setColorAt(ui.traced, this.col.setRGB(2.2, 2.4, 2.6));
        m.beads.instanceColor!.needsUpdate = true;
        m.beads.userData.traced = ui.traced;
      } else if (m.beads.userData.traced !== undefined) {
        const k = m.beads.userData.traced as number;
        if (k < n) m.beads.setColorAt(k, lab.ds.y[k] ? cTeal : cAmber);
        m.beads.instanceColor!.needsUpdate = true;
        delete m.beads.userData.traced;
      }
      (m.pad.material as THREE.MeshBasicMaterial).opacity = 0.18 + 0.55 * m.lit;
      (m.edges.material as THREE.MeshBasicMaterial).opacity = 0.4 + 0.6 * m.lit;
      m.group.position.y = TIER2_Y + m.size / 2 + 0.002 + m.lit * 0.04;
    }
  }

  /** Dynamic resolution: drop DPR if sustained fps is low (never switches renderer). */
  private adaptQuality(dt: number): void {
    if (this.hq || this.stepped) return;
    this.fpsAcc += dt;
    this.fpsFrames++;
    if (this.fpsAcc < 1) return;
    const fps = this.fpsFrames / this.fpsAcc;
    this.fpsAcc = 0;
    this.fpsFrames = 0;
    this.lowFor = fps < 40 ? this.lowFor + 1 : 0;
    if (this.lowFor >= 2 && this.dpr > 0.75) {
      this.dpr = Math.max(0.75, this.dpr * 0.8);
      this.lowFor = 0;
      this.resize(this.w, this.h, this.layout);
    }
  }

  dispose(): void {
    this.renderer.dispose();
    this.canvas.remove();
  }
}
