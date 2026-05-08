// WebGL 2 three-body orbit background.
// Ping-pong framebuffer feedback for the trail; per-frame fade + 3 short
// capsule draws + present. Orbit positions are pre-baked by precompute.py.

// Each body should sweep across the screen at roughly the same speed across
// orbits regardless of period or extent. We achieve this by stretching the
// orbit's playback time so the *initial* body speed maps to this target
// on-screen speed. Bigger value = faster bodies.
const TARGET_BODY_SPEED = 0.65;  // worldHalfSize units per real second

// 1.0 = neutral (proper sRGB gamma); >1 crushes mid-tones for a more
// contrasty look; <1 lifts mid-tones for a softer/flatter look.
const CONTRAST = 1.8;
const FADE_TAU = 1.6;            // 1/s — per-second decay rate of the trail (lower = longer trails)
const FBO_SHORT = 720;           // shorter FBO axis — orbit is sized to this; longer axis scales with viewport aspect
const FBO_LONG_MAX = 1920;       // safety cap so ultra-wide viewports don't blow up GPU cost
const FRAMING_MARGIN = 0.05;     // 5% breathing room around the orbit's bounding box

const COLORS = new Float32Array([
  0.10, 0.35, 1.00,  // body C — saturated blue
  1.00, 0.30, 0.05,  // body A — amber / orange
  1.00, 0.85, 0.55,  // body B — warm cream
]);
const LUMS = new Float32Array(3);
for (let i = 0; i < 3; i++) {
  LUMS[i] = 0.299 * COLORS[3*i] + 0.587 * COLORS[3*i+1] + 0.114 * COLORS[3*i+2];
}

const VS_FULLSCREEN = `#version 300 es
out vec2 vUv;
const vec2 verts[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() {
  vec2 p = verts[gl_VertexID];
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const FS_FADE = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uPrev;
uniform float uFade;
uniform vec2 uTexel;
out vec4 fragColor;
void main() {
  // 5-tap cross blur + fade — diffusion makes the trail bloom softly as it fades
  // and hides single-pixel aliasing.
  vec4 c = texture(uPrev, vUv) * 0.5;
  c += texture(uPrev, vUv + vec2( uTexel.x, 0.0)) * 0.125;
  c += texture(uPrev, vUv + vec2(-uTexel.x, 0.0)) * 0.125;
  c += texture(uPrev, vUv + vec2(0.0,  uTexel.y)) * 0.125;
  c += texture(uPrev, vUv + vec2(0.0, -uTexel.y)) * 0.125;
  fragColor = c * uFade;
}`;

// Present: combine the linear trail FBO with analytic 1/d^2 sun-like halos
// for each current body position, then tonemap and gamma-encode the sum.
// All light combines in linear space — bright overlaps blow out to white.
const FS_PRESENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uFbo;
uniform vec2 uPos[3];
uniform vec3 uColor[3];
uniform float uLum[3];
uniform vec2 uCenter;
uniform vec2 uWorldExtent;       // half-extent of the FBO's view in world units (per-axis)
uniform float uNucleus;
uniform float uHalo;
uniform float uContrast;
out vec4 fragColor;
void main() {
  vec3 c = texture(uFbo, vUv).rgb;
  vec2 worldPos = uCenter + (vUv * 2.0 - 1.0) * uWorldExtent;
  for (int i = 0; i < 3; i++) {
    float d = max(distance(worldPos, uPos[i]), 1e-5);
    float core = pow(uNucleus / d, 2.0) / uLum[i];   // tight pinpoint
    float halo = pow(uHalo    / d, 2.0) / uLum[i];   // 1/r^2 sun glow
    c += uColor[i] * min(core + halo, 60.0);
  }
  // Crush the ambient 1/r^2 floor to true black using a SMOOTH ramp on the
  // max channel. A hard subtract here would create a visible boundary circle
  // around each body where the halo crosses the cutoff in linear space.
  float m = max(c.r, max(c.g, c.b));
  c *= smoothstep(0.0, 0.06, m);
  c = vec3(1.0) - exp(-c);
  // sRGB gamma + contrast: crushes mid-tones when uContrast > 1.
  c = pow(c, vec3(uContrast / 2.2));
  fragColor = vec4(c, 1.0);
}`;

const VS_SEGMENT = `#version 300 es
in vec2 aPos;            // unit quad [-0.5, 0.5]^2
uniform vec2 uPrev[3];
uniform vec2 uCurr[3];
uniform vec3 uColor[3];
uniform float uLum[3];
uniform float uRadius;
uniform vec4 uProj;      // (sx, sy, tx, ty): clip = world * sxy + txy
flat out vec2 vA;
flat out vec2 vB;
flat out vec3 vColor;
flat out float vLum;
out vec2 vWorld;
void main() {
  vec2 a = uPrev[gl_InstanceID];
  vec2 b = uCurr[gl_InstanceID];
  vec2 mid = 0.5 * (a + b);
  vec2 d = b - a;
  float len = length(d);
  vec2 ux = (len > 1e-7) ? d / len : vec2(1.0, 0.0);
  vec2 uy = vec2(-ux.y, ux.x);
  float halfLen = 0.5 * len + uRadius;
  vec2 offset = aPos.x * 2.0 * halfLen * ux + aPos.y * 2.0 * uRadius * uy;
  vec2 world = mid + offset;
  vA = a; vB = b;
  vColor = uColor[gl_InstanceID];
  vLum = uLum[gl_InstanceID];
  vWorld = world;
  gl_Position = vec4(world * uProj.xy + uProj.zw, 0.0, 1.0);
}`;

const FS_SEGMENT = `#version 300 es
precision highp float;
flat in vec2 vA;
flat in vec2 vB;
flat in vec3 vColor;
flat in float vLum;
in vec2 vWorld;
uniform float uStrength;
uniform float uExponent;
uniform float uRadius;
out vec4 fragColor;
float sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
  return length(pa - ba * h);
}
// Linear emission per body — no per-draw tonemap. The FBO accumulates raw
// HDR contributions; the present pass tonemaps the final sum so that trail
// crossings combine like real light rather than clipped LDR colors.
void main() {
  float d = max(sdSegment(vWorld, vA, vB), 1e-5);
  float intensity = pow(uStrength / d, uExponent) / vLum;
  intensity = min(intensity, 60.0);
  // Soft-mask as we approach the capsule boundary. Without this, emission
  // jumps from "tiny glow" inside the quad to absolute zero outside, and
  // the gamma curve amplifies that step into a visible perpendicular line
  // at the front/back of the body's halo.
  intensity *= 1.0 - smoothstep(uRadius * 0.65, uRadius, d);
  fragColor = vec4(vColor * intensity, 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh) + '\n--\n' + src);
  }
  return sh;
}

function program(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

function makeFbo(gl, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('framebuffer incomplete');
  }
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return { tex, fbo, w, h };
}

function disposeFbo(gl, fbo) {
  gl.deleteFramebuffer(fbo.fbo);
  gl.deleteTexture(fbo.tex);
}

// Choose FBO dimensions matching the canvas aspect exactly (so the present
// pass never stretches). Default: short axis = FBO_SHORT. If that pushes the
// long axis past FBO_LONG_MAX, shrink the short axis to keep aspect.
function fboDimsForAspect(aspect) {
  let w, h;
  if (aspect >= 1) {
    h = FBO_SHORT;
    w = Math.round(h * aspect);
    if (w > FBO_LONG_MAX) { w = FBO_LONG_MAX; h = Math.round(w / aspect); }
  } else {
    w = FBO_SHORT;
    h = Math.round(w / aspect);
    if (h > FBO_LONG_MAX) { h = FBO_LONG_MAX; w = Math.round(h * aspect); }
  }
  return { w: Math.max(1, w), h: Math.max(1, h) };
}

async function main() {
  const orbitsMeta = await fetch('orbits.json').then(r => r.json());

  // Each orbit goes into the pool twice: original orientation and 90° CCW
  // rotated. Rotating swaps the bounding-box axes so wide orbits become
  // tall and vice versa, doubling the chance of finding one that fits the
  // viewport. The actual sample rotation happens client-side after fetch.
  const pool = [];
  for (const o of orbitsMeta) {
    pool.push({ ...o, rotated: false });
    pool.push({
      ...o,
      name: o.name + ' (rotated)',
      center: [-o.center[1], o.center[0]],
      extent: [o.extent[1], o.extent[0]],
      rotated: true,
    });
  }

  // Bias the random pick toward orbits whose bounding-box aspect matches
  // the viewport. Distance is measured in log-aspect space (so 2:1 and
  // 1:2 are equivalently "off") with a Gaussian falloff. Bigger sigma =
  // softer bias; smaller = stricter aspect matching.
  const viewLogAspect = Math.log(window.innerWidth / window.innerHeight);
  const sigma = 0.5;
  const weights = pool.map(o => {
    const diff = Math.log(o.extent[0] / o.extent[1]) - viewLogAspect;
    return Math.exp(-(diff * diff) / (2 * sigma * sigma));
  });
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  let orbit = pool[pool.length - 1];
  for (let i = 0; i < pool.length; i++) {
    r -= weights[i];
    if (r <= 0) { orbit = pool[i]; break; }
  }
  console.log('Orbit:', orbit.name);

  const binBuf = await fetch(orbit.file).then(r => r.arrayBuffer());
  const samples = new Float32Array(binBuf);
  if (orbit.rotated) {
    // 90° CCW: (x, y) → (-y, x). Samples are tightly packed [r1x, r1y, ...].
    for (let i = 0; i < samples.length; i += 2) {
      const x = samples[i];
      samples[i]     = -samples[i + 1];
      samples[i + 1] =  x;
    }
  }

  const canvas = document.getElementById('c');
  const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false });
  if (!gl) {
    console.error('WebGL 2 unavailable');
    return;
  }
  if (!gl.getExtension('EXT_color_buffer_float')) {
    console.error('EXT_color_buffer_float unavailable; trail accumulator will band');
  }

  // ---- Resources

  const fadeProg = program(gl, VS_FULLSCREEN, FS_FADE);
  const presentProg = program(gl, VS_FULLSCREEN, FS_PRESENT);
  const segmentProg = program(gl, VS_SEGMENT, FS_SEGMENT);

  // Empty VAO for the fullscreen passes (vertices generated in the vertex shader).
  const fullscreenVao = gl.createVertexArray();

  // Unit quad for segment instances.
  const quadVerts = new Float32Array([
    -0.5,-0.5,  0.5,-0.5,  0.5, 0.5,
    -0.5,-0.5,  0.5, 0.5, -0.5, 0.5,
  ]);
  const quadVbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
  gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
  const segmentVao = gl.createVertexArray();
  gl.bindVertexArray(segmentVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
  const aPosLocSeg = gl.getAttribLocation(segmentProg, 'aPos');
  gl.enableVertexAttribArray(aPosLocSeg);
  gl.vertexAttribPointer(aPosLocSeg, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  let fboFront = null;
  let fboBack = null;

  // ---- Camera (world → clip)
  const worldHalfSize = Math.max(orbit.extent[0], orbit.extent[1]) * (1 + FRAMING_MARGIN);
  const proj = new Float32Array(4);
  const worldExtent = new Float32Array(2);

  // Per-orbit visual period: stretch the orbit's playback so the initial
  // body speed (in worldHalfSize units / orbit-time) lands at TARGET_BODY_SPEED
  // (in worldHalfSize units / real second). Faster-starting orbits play longer.
  const visualPeriodS = orbit.period * (orbit.avgStartSpeed / worldHalfSize) / TARGET_BODY_SPEED;

  // Trail thickness scales with the orbit so visual proportion stays constant.
  const TRAIL_RADIUS    = worldHalfSize * 0.15;   // bounding capsule — far enough out that emission has faded by the edge
  const TRAIL_STRENGTH  = worldHalfSize * 0.005;
  // Head sizing is in CANVAS PIXELS — converted to world units on resize so
  // the visual size stays constant across orbits AND viewport sizes.
  const HEAD_NUCLEUS_PX = 5;       // tight bright pinpoint
  const HEAD_HALO_PX    = 45;      // 1/r^2 sun-like glow

  // Recompute camera + world extent so the orbit fits the shorter FBO axis
  // and halos extend naturally into the longer axis.
  function updateCamera(fboW, fboH) {
    const aspect = fboW / fboH;
    let scaleX, scaleY, extX, extY;
    if (aspect >= 1) {
      scaleY = 1 / worldHalfSize;
      scaleX = scaleY / aspect;
      extX = worldHalfSize * aspect;
      extY = worldHalfSize;
    } else {
      scaleX = 1 / worldHalfSize;
      scaleY = scaleX * aspect;
      extX = worldHalfSize;
      extY = worldHalfSize / aspect;
    }
    proj[0] = scaleX;
    proj[1] = scaleY;
    proj[2] = -orbit.center[0] * scaleX;
    proj[3] = -orbit.center[1] * scaleY;
    worldExtent[0] = extX;
    worldExtent[1] = extY;
  }

  // ---- Per-frame state
  const prevPos = new Float32Array(6);
  const currPos = new Float32Array(6);

  // Initialize prevPos to the position at t=0 so the first segment is zero-length.
  for (let k = 0; k < 6; k++) prevPos[k] = samples[k];
  currPos.set(prevPos);

  const epoch = performance.now();
  let prevNow = epoch;
  let hiddenAt = null;
  let pausedDt = 0;

  function pickPosition(timeMs, out) {
    const phase = (((timeMs - epoch - pausedDt) / (visualPeriodS * 1000.0)) % 1.0 + 1.0) % 1.0;
    const f = phase * orbit.sampleCount;
    const i0 = Math.floor(f);
    const i1 = (i0 + 1) % orbit.sampleCount;
    const t = f - i0;
    const o0 = i0 * 6, o1 = i1 * 6;
    for (let k = 0; k < 6; k++) {
      out[k] = samples[o0 + k] + (samples[o1 + k] - samples[o0 + k]) * t;
    }
  }

  const segUniforms = {
    uPrev: gl.getUniformLocation(segmentProg, 'uPrev[0]'),
    uCurr: gl.getUniformLocation(segmentProg, 'uCurr[0]'),
    uColor: gl.getUniformLocation(segmentProg, 'uColor[0]'),
    uLum: gl.getUniformLocation(segmentProg, 'uLum[0]'),
    uProj: gl.getUniformLocation(segmentProg, 'uProj'),
    uRadius: gl.getUniformLocation(segmentProg, 'uRadius'),
    uStrength: gl.getUniformLocation(segmentProg, 'uStrength'),
    uExponent: gl.getUniformLocation(segmentProg, 'uExponent'),
  };
  const fadeUniforms = {
    uPrev: gl.getUniformLocation(fadeProg, 'uPrev'),
    uFade: gl.getUniformLocation(fadeProg, 'uFade'),
    uTexel: gl.getUniformLocation(fadeProg, 'uTexel'),
  };
  const presentUniforms = {
    uFbo: gl.getUniformLocation(presentProg, 'uFbo'),
    uPos: gl.getUniformLocation(presentProg, 'uPos[0]'),
    uColor: gl.getUniformLocation(presentProg, 'uColor[0]'),
    uLum: gl.getUniformLocation(presentProg, 'uLum[0]'),
    uCenter: gl.getUniformLocation(presentProg, 'uCenter'),
    uWorldExtent: gl.getUniformLocation(presentProg, 'uWorldExtent'),
    uNucleus: gl.getUniformLocation(presentProg, 'uNucleus'),
    uHalo: gl.getUniformLocation(presentProg, 'uHalo'),
    uContrast: gl.getUniformLocation(presentProg, 'uContrast'),
  };

  // Bind once: shared static uniforms (per-resize camera bindings handled below).
  gl.useProgram(segmentProg);
  gl.uniform3fv(segUniforms.uColor, COLORS);
  gl.uniform1fv(segUniforms.uLum, LUMS);

  gl.useProgram(presentProg);
  gl.uniform3fv(presentUniforms.uColor, COLORS);
  gl.uniform1fv(presentUniforms.uLum, LUMS);
  gl.uniform2f(presentUniforms.uCenter, orbit.center[0], orbit.center[1]);
  gl.uniform1f(presentUniforms.uContrast, CONTRAST);

  // ---- Resize: canvas backing-store + FBO dimensions + camera projection.
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(window.innerWidth * dpr));
    const ch = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;

    const aspect = cw / ch;
    const dims = fboDimsForAspect(aspect);
    if (!fboFront || fboFront.w !== dims.w || fboFront.h !== dims.h) {
      if (fboFront) disposeFbo(gl, fboFront);
      if (fboBack) disposeFbo(gl, fboBack);
      fboFront = makeFbo(gl, dims.w, dims.h);
      fboBack = makeFbo(gl, dims.w, dims.h);
    }
    updateCamera(dims.w, dims.h);

    // Convert pixel-defined head sizes to world units. Orbit fits the shorter
    // canvas axis, so 1 world unit spans shortPx / (2 * worldHalfSize) pixels.
    const shortPx = Math.min(cw, ch);
    const worldPerPx = (2 * worldHalfSize) / shortPx;

    gl.useProgram(segmentProg);
    gl.uniform4fv(segUniforms.uProj, proj);
    gl.useProgram(presentProg);
    gl.uniform2fv(presentUniforms.uWorldExtent, worldExtent);
    gl.uniform1f(presentUniforms.uNucleus, HEAD_NUCLEUS_PX * worldPerPx);
    gl.uniform1f(presentUniforms.uHalo, HEAD_HALO_PX * worldPerPx);
  }
  resize();
  window.addEventListener('resize', resize);

  // ---- Visibility: pause the simulation clock so the orbit doesn't jump.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      hiddenAt = performance.now();
    } else if (hiddenAt != null) {
      pausedDt += performance.now() - hiddenAt;
      hiddenAt = null;
      prevNow = performance.now();
    }
  });

  // ---- Render loop
  function frame(now) {
    if (document.hidden) {
      requestAnimationFrame(frame);
      return;
    }

    const dtSec = Math.min(0.1, (now - prevNow) / 1000.0);
    prevNow = now;

    prevPos.set(currPos);
    pickPosition(now, currPos);

    // 1. Fade + diffusion pass: fboFront × uFade (with cross blur) → fboBack
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBack.fbo);
    gl.viewport(0, 0, fboBack.w, fboBack.h);
    gl.disable(gl.BLEND);
    gl.useProgram(fadeProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fboFront.tex);
    gl.uniform1i(fadeUniforms.uPrev, 0);
    gl.uniform1f(fadeUniforms.uFade, Math.exp(-FADE_TAU * dtSec));
    gl.uniform2f(fadeUniforms.uTexel, 1 / fboBack.w, 1 / fboBack.h);
    gl.bindVertexArray(fullscreenVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 2. Trail segment pass: additive into fboBack (prev → curr capsule per body)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(segmentProg);
    gl.bindVertexArray(segmentVao);
    gl.uniform2fv(segUniforms.uPrev, prevPos);
    gl.uniform2fv(segUniforms.uCurr, currPos);
    gl.uniform1f(segUniforms.uRadius, TRAIL_RADIUS);
    gl.uniform1f(segUniforms.uStrength, TRAIL_STRENGTH);
    gl.uniform1f(segUniforms.uExponent, 2.0);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, 3);

    // 3. Present: trail FBO + analytic 1/r^2 head halos, summed in linear
    //    HDR and tonemapped once. Fills the entire canvas viewport.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(presentProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fboBack.tex);
    gl.uniform1i(presentUniforms.uFbo, 0);
    gl.uniform2fv(presentUniforms.uPos, currPos);
    gl.bindVertexArray(fullscreenVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 4. Swap FBOs.
    [fboFront, fboBack] = [fboBack, fboFront];

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
