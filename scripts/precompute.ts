/**
 * Pre-compute three-body orbit positions for the WebGL viewer.
 *
 * Reads scripts/init_conditions.json, integrates each periodic orbit using
 * an adaptive Dormand-Prince 5(4) integrator (the same method scipy's RK45
 * uses), samples it on a uniform time grid via Hermite interpolation, and
 * writes:
 *
 *   static/public/orbits.json          — index + per-orbit metadata
 *   static/public/orbits/<slug>.bin    — one little-endian float32 file per orbit
 *
 * Splitting the binary per orbit lets the page only download the one orbit
 * it picks on load, instead of fetching all of them up front.
 *
 * Run from the repo root:  npm run precompute
 */

import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
  unlinkSync,
} from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

// --- Physics ----------------------------------------------------------------

const G = 1, m1 = 1, m2 = 1, m3 = 1;

// State layout (12 doubles):
//   [0..1]  r1 = body-1 position (x, y)
//   [2..3]  r2
//   [4..5]  r3
//   [6..7]  v1 = body-1 velocity
//   [8..9]  v2
//   [10..11] v3
type State = Float64Array;

// Compute dy/dt = [v1, v2, v3, a1, a2, a3] from the current state, into `out`.
// Hand-unrolled — runs many thousands of times per orbit, so allocations
// here would be measurable.
function deriv(y: State, out: State): void {
  const r1x = y[0],  r1y = y[1];
  const r2x = y[2],  r2y = y[3];
  const r3x = y[4],  r3y = y[5];

  const dx12 = r2x - r1x, dy12 = r2y - r1y;
  const dx23 = r3x - r2x, dy23 = r3y - r2y;
  const dx31 = r1x - r3x, dy31 = r1y - r3y;

  const d12sq = dx12 * dx12 + dy12 * dy12;
  const d23sq = dx23 * dx23 + dy23 * dy23;
  const d31sq = dx31 * dx31 + dy31 * dy31;

  const inv12 = 1 / (d12sq * Math.sqrt(d12sq));
  const inv23 = 1 / (d23sq * Math.sqrt(d23sq));
  const inv31 = 1 / (d31sq * Math.sqrt(d31sq));

  // a1 = G*m2*(r2-r1)/|r2-r1|^3 + G*m3*(r3-r1)/|r3-r1|^3
  // r3-r1 = -(r1-r3) = (-dx31, -dy31)
  const a1x = G * m2 * dx12 * inv12 - G * m3 * dx31 * inv31;
  const a1y = G * m2 * dy12 * inv12 - G * m3 * dy31 * inv31;
  const a2x = G * m3 * dx23 * inv23 - G * m1 * dx12 * inv12;
  const a2y = G * m3 * dy23 * inv23 - G * m1 * dy12 * inv12;
  const a3x = G * m1 * dx31 * inv31 - G * m2 * dx23 * inv23;
  const a3y = G * m1 * dy31 * inv31 - G * m2 * dy23 * inv23;

  out[0]  = y[6];   out[1]  = y[7];
  out[2]  = y[8];   out[3]  = y[9];
  out[4]  = y[10];  out[5]  = y[11];
  out[6]  = a1x;    out[7]  = a1y;
  out[8]  = a2x;    out[9]  = a2y;
  out[10] = a3x;    out[11] = a3y;
}

// --- Dormand-Prince 5(4) ----------------------------------------------------
// Butcher tableau coefficients (https://en.wikipedia.org/wiki/Dormand–Prince_method).
// 7 stages, FSAL: k7 of step n becomes k1 of step n+1 — saves one deriv eval.

const A21 = 1/5;
const A31 = 3/40,        A32 = 9/40;
const A41 = 44/45,       A42 = -56/15,      A43 = 32/9;
const A51 = 19372/6561,  A52 = -25360/2187, A53 = 64448/6561, A54 = -212/729;
const A61 = 9017/3168,   A62 = -355/33,     A63 = 46732/5247, A64 = 49/176, A65 = -5103/18656;

// 5th-order solution weights (B2 = B7 = 0).
const B1 = 35/384, B3 = 500/1113, B4 = 125/192, B5 = -2187/6784, B6 = 11/84;

// Error estimate weights = b5 - b4. Component-wise: yErr = h * sum(E_i * k_i).
const E1 = 71/57600, E3 = -71/16695, E4 = 71/1920, E5 = -17253/339200, E6 = 22/525, E7 = -1/40;

const N = 12;
const RTOL = 1e-10;
const ATOL = 1e-10;
const SAFETY = 0.9;
const MIN_FACTOR = 0.2;
const MAX_FACTOR = 5.0;

interface Workspace {
  k1: State; k2: State; k3: State; k4: State; k5: State; k6: State; k7: State;
  yNew: State; yErr: State; tmp: State;
}

function newWorkspace(): Workspace {
  return {
    k1:  new Float64Array(N),
    k2:  new Float64Array(N),
    k3:  new Float64Array(N),
    k4:  new Float64Array(N),
    k5:  new Float64Array(N),
    k6:  new Float64Array(N),
    k7:  new Float64Array(N),
    yNew: new Float64Array(N),
    yErr: new Float64Array(N),
    tmp: new Float64Array(N),
  };
}

// Take one DP54 step from y over h. Mutates yNew, yErr, k7. k1 must be filled
// already (either deriv(y) or, on FSAL chain, the previous step's k7).
function dp54Step(y: State, h: number, w: Workspace): void {
  const { k1, k2, k3, k4, k5, k6, k7, yNew, yErr, tmp } = w;

  for (let i = 0; i < N; i++) tmp[i] = y[i] + h * A21 * k1[i];
  deriv(tmp, k2);

  for (let i = 0; i < N; i++) tmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
  deriv(tmp, k3);

  for (let i = 0; i < N; i++) tmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
  deriv(tmp, k4);

  for (let i = 0; i < N; i++) tmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
  deriv(tmp, k5);

  for (let i = 0; i < N; i++) tmp[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
  deriv(tmp, k6);

  for (let i = 0; i < N; i++) {
    yNew[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
  }
  deriv(yNew, k7);  // FSAL — also the next step's k1.

  for (let i = 0; i < N; i++) {
    yErr[i] = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
  }
}

// Cubic Hermite interpolation between (t_n, y_n, k1=f(t_n)) and (t_{n+1},
// y_{n+1}, k7=f(t_{n+1})). Third-order accurate; sufficient for our 1e-6
// tolerance on the dense sample grid. Writes only the 6 position components
// since that's all the renderer reads.
function hermitePos(
  yn: State, ynNext: State, k1: State, k7: State, h: number, theta: number,
  out: Float32Array, outOffset: number,
): void {
  const t  = theta;
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 =  2 * t3 - 3 * t2 + 1;
  const h10 =      t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 =      t3 -     t2;
  for (let i = 0; i < 6; i++) {
    out[outOffset + i] = h00 * yn[i] + h10 * h * k1[i] + h01 * ynNext[i] + h11 * h * k7[i];
  }
}

// Adaptive integrator. Drives y from t=0 to t=T. At each accepted step we
// catch up any pending sample times (uniform grid t = T * s / nSamples)
// using cubic Hermite over the just-completed step. Returns the final state
// (for the seam check) and fills `samples`.
function integrate(
  y0: State, T: number, nSamples: number, samples: Float32Array,
): State {
  const w = newWorkspace();
  const y = new Float64Array(y0);
  const yPrev = new Float64Array(N);

  // Sample 0 = initial state.
  for (let k = 0; k < 6; k++) samples[k] = y[k];

  // Initial step size guess. The Hairer/Wanner heuristic is overkill here;
  // 1e-3 of T is a fine starting point — the controller adapts within a few
  // steps regardless.
  let h = T * 1e-3;
  let t = 0;

  deriv(y, w.k1);

  let nextIdx = 1;
  let nextT = (T * nextIdx) / nSamples;

  while (t < T) {
    if (t + h > T) h = T - t;

    dp54Step(y, h, w);

    // RMS-normalised error; standard PI-style tolerance check.
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      const sc = ATOL + RTOL * Math.max(Math.abs(y[i]), Math.abs(w.yNew[i]));
      const e  = w.yErr[i] / sc;
      sumSq += e * e;
    }
    const errNorm = Math.sqrt(sumSq / N);

    if (errNorm < 1) {
      // Accept. Cache the previous y for Hermite interpolation, then advance.
      for (let i = 0; i < N; i++) yPrev[i] = y[i];
      const tPrev = t;
      const hUsed = h;
      const k1Prev = w.k1;
      const k7Curr = w.k7;
      for (let i = 0; i < N; i++) y[i] = w.yNew[i];
      t += h;

      // Catch up sample times that fell inside this step.
      while (nextIdx < nSamples && nextT <= t + 1e-15) {
        const theta = (nextT - tPrev) / hUsed;
        hermitePos(yPrev, y, k1Prev, k7Curr, hUsed, theta, samples, nextIdx * 6);
        nextIdx++;
        nextT = (T * nextIdx) / nSamples;
      }

      // FSAL: k7 of this step is k1 of the next.
      const swap = w.k1;
      w.k1 = w.k7;
      w.k7 = swap;

      // Adapt step size for the next step (PI controller, simple form).
      const factor = errNorm === 0
        ? MAX_FACTOR
        : Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, SAFETY * Math.pow(errNorm, -0.2)));
      h *= factor;
    } else {
      // Reject — shrink and retry. Don't change y.
      const factor = Math.max(MIN_FACTOR, SAFETY * Math.pow(errNorm, -0.2));
      h *= factor;
    }
  }

  return y;
}

// --- Per-orbit precompute ---------------------------------------------------

interface InitCondition {
  period: number;
  positions: [number, number][];
  velocities: [number, number][];
}

interface OrbitMeta {
  name: string;
  period: number;
  sampleCount: number;
  center: [number, number];
  extent: [number, number];
  avgStartSpeed: number;
  file: string;
}

interface OrbitOutput extends OrbitMeta {
  samples: Float32Array;
}

function precomputeOne(name: string, init: InitCondition): OrbitOutput {
  const T = init.period;
  const nSamples = Math.max(2048, Math.ceil(T / 0.01));

  const y0 = new Float64Array(N);
  for (let b = 0; b < 3; b++) {
    y0[b * 2]         = init.positions[b][0];
    y0[b * 2 + 1]     = init.positions[b][1];
    y0[6 + b * 2]     = init.velocities[b][0];
    y0[6 + b * 2 + 1] = init.velocities[b][1];
  }

  const samples = new Float32Array(nSamples * 6);
  const yT = integrate(y0, T, nSamples, samples);

  // Centre of all body positions across the whole orbit, plus per-axis extent.
  let cx = 0, cy = 0;
  const totalPos = nSamples * 3;
  for (let s = 0; s < nSamples; s++) {
    for (let b = 0; b < 3; b++) {
      cx += samples[s * 6 + b * 2];
      cy += samples[s * 6 + b * 2 + 1];
    }
  }
  cx /= totalPos; cy /= totalPos;

  let ex = 0, ey = 0;
  for (let s = 0; s < nSamples; s++) {
    for (let b = 0; b < 3; b++) {
      const dx = Math.abs(samples[s * 6 + b * 2]     - cx);
      const dy = Math.abs(samples[s * 6 + b * 2 + 1] - cy);
      if (dx > ex) ex = dx;
      if (dy > ey) ey = dy;
    }
  }

  let speedSum = 0;
  for (let b = 0; b < 3; b++) {
    const vx = init.velocities[b][0], vy = init.velocities[b][1];
    speedSum += Math.sqrt(vx * vx + vy * vy);
  }
  const avgStartSpeed = speedSum / 3;

  // Periodicity seam: ||y(T) - y(0)||.
  let seamSq = 0;
  for (let i = 0; i < N; i++) {
    const d = yT[i] - y0[i];
    seamSq += d * d;
  }
  const seam = Math.sqrt(seamSq);
  const seamRel = seam / Math.max(Math.max(ex, ey), 1e-12);
  if (seamRel > 1e-6) {
    console.warn(`  warning: '${name}' periodicity seam = ${seamRel.toExponential(2)} (relative)`);
  }

  return {
    name,
    period: T,
    sampleCount: nSamples,
    center: [cx, cy],
    extent: [ex, ey],
    avgStartSpeed,
    file: '',
    samples,
  };
}

function slugify(name: string): string {
  const s = name.replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase();
  return s || 'orbit';
}

// --- Driver -----------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const REPO_ROOT  = join(__dirname, '..');
const INIT_PATH  = join(__dirname, 'init_conditions.json');
const OUT_DIR    = join(REPO_ROOT, 'static', 'public');
const BIN_DIR    = join(OUT_DIR, 'orbits');

function main(): void {
  mkdirSync(BIN_DIR, { recursive: true });
  for (const f of readdirSync(BIN_DIR)) {
    if (f.endsWith('.bin')) unlinkSync(join(BIN_DIR, f));
  }

  const initConditions: Record<string, InitCondition> =
    JSON.parse(readFileSync(INIT_PATH, 'utf-8'));

  const orbits: OrbitMeta[] = [];
  let totalBytes = 0;

  for (const [name, conditions] of Object.entries(initConditions)) {
    console.log(`integrating '${name}' (T=${conditions.period.toFixed(3)}) ...`);
    const orbit = precomputeOne(name, conditions);

    const slug = slugify(name);
    const binPath = join(BIN_DIR, `${slug}.bin`);
    const bytes = new Uint8Array(orbit.samples.buffer, orbit.samples.byteOffset, orbit.samples.byteLength);
    writeFileSync(binPath, bytes);
    totalBytes += bytes.length;

    const { samples: _drop, ...meta } = orbit;
    meta.file = `orbits/${slug}.bin`;
    orbits.push(meta);
    console.log(`  wrote ${relative(REPO_ROOT, binPath)} (${(bytes.length / 1024).toFixed(1)} KiB)`);
  }

  const jsonPath = join(OUT_DIR, 'orbits.json');
  writeFileSync(jsonPath, JSON.stringify(orbits, null, 2));

  console.log();
  console.log(`wrote ${relative(REPO_ROOT, jsonPath)} (${orbits.length} orbits, ${(totalBytes / 1024).toFixed(1)} KiB total)`);
}

main();
