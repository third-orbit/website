"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Periodic Solutions of the Three-Body Problem (2D Animation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script was inspired by this post:

https://www.reddit.com/r/physicsgifs/comments/14db21p/a_few_three_body_periodic_orbits/

The author claims he used Blender and Python, but I thought a lightweight animation with
real-time preview would be better, so I made this animation using vispy and custom glsl code.

I've included only a few periodic solutions here. More can be found at:

https://observablehq.com/@rreusser/periodic-planar-three-body-orbits

The only things you need to set are the period, the initial positions and velocities.

There are many variables you can tweak to control the visual effect. For example, `dt` and
`trail_length` both affect the length of the trails. When it comes to controlling the glowing
effect of the trails, several variables in the fragment shader are also involved. I haven't
had time to figure out a universal config of parameters that work for all solutions, so you might
need to experiment and adjust them manually.
"""

import json
import numpy as np
from vispy import app, gloo
from vispy.gloo import gl as _vispy_gl
from vispy.gloo.glir import GlirProgram as _GlirProgram
from vispy.io import imsave
from scipy.integrate import solve_ivp


# Monkey-patch: vispy's GlirProgram.set_uniform has a bug where the array
# element count is only computed on the first call (when the uniform's GL
# handle is being cached). On every subsequent call, count defaults to 1,
# so glUniform*v silently uploads only the first element of an array
# uniform — making per-frame array updates impossible via the bulk path.
# See vispy/gloo/glir.py:set_uniform. Without this patch we would have to
# upload each element individually (300 calls/frame for our trail data).
_orig_set_uniform = _GlirProgram.set_uniform


def _set_uniform_array_fix(self, name, type_, value):
    if not type_.startswith("mat") and name in self._handles:
        count = value.nbytes // (4 * self.ATYPEINFO[type_][0])
        if count > 1:
            self.activate()
            getattr(_vispy_gl, self.UTYPEMAP[type_])(self._handles[name], count, value)
            return
    _orig_set_uniform(self, name, type_, value)


_GlirProgram.set_uniform = _set_uniform_array_fix

resolution = (1000, 1000)
G = m1 = m2 = m3 = 1.0
dt = 0.02
trail_length = 100
name = "figure 8"


def calc_accel(t, y):
    r1, r2, r3, v1, v2, v3 = np.split(y, 6)
    d12 = d21 = np.linalg.norm(r2 - r1) ** 3
    d23 = d32 = np.linalg.norm(r3 - r2) ** 3
    d31 = d13 = np.linalg.norm(r1 - r3) ** 3

    a1 = G * m2 * (r2 - r1) / d12 + G * m3 * (r3 - r1) / d13
    a2 = G * m3 * (r3 - r2) / d23 + G * m1 * (r1 - r2) / d21
    a3 = G * m1 * (r1 - r3) / d31 + G * m2 * (r2 - r3) / d32
    return np.concatenate([v1, v2, v3, a1, a2, a3])


with open("init_conditions.json", encoding="utf-8") as f:
    init_conditions = json.load(f)

T = init_conditions[name]["period"]
y0 = np.concatenate(
    init_conditions[name]["positions"] + init_conditions[name]["velocities"]
)
num_steps = int(T / dt) + 1
t_eval = np.linspace(0, T, num_steps)

sol = solve_ivp(
    calc_accel, [0, T], y0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9
)
solution = sol.y.T
r1 = solution[:, 0:2]
r2 = solution[:, 2:4]
r3 = solution[:, 4:6]

all_points = np.vstack((r1, r2, r3))
center = np.mean(all_points, axis=0)
max_x_distance = np.max(np.abs(all_points[:, 0] - center[0]))
zoom = max_x_distance * 1.2


class ThreeBody(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(
            self, size=resolution, title="Three-Body Simulation", keys="interactive"
        )
        with open("./glsl/default.vert", "r") as f:
            vertex_shader = f.read()

        with open("./glsl/3body.frag", "r") as f:
            fragment_shader = f.read().replace("trail_length", str(trail_length))

        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.program["position"] = [
            [-1, -1],
            [-1, 1],
            [1, 1],
            [-1, -1],
            [1, 1],
            [1, -1],
        ]
        self.program["iResolution"] = self.physical_size
        self.program["iTime"] = 0.0
        self.program["zoom"] = zoom
        self.program["center"] = center
        self.timer = app.Timer("auto", connect=self.on_timer, start=False)
        self.frame_index = 0

    def on_draw(self, event):
        gloo.clear()
        indices = (
            np.arange(self.frame_index - trail_length, self.frame_index) % num_steps
        )
        # Requires the monkey patch above
        self.program["pointsA"] = r1[indices].astype(np.float32)
        self.program["pointsB"] = r2[indices].astype(np.float32)
        self.program["pointsC"] = r3[indices].astype(np.float32)
        self.program.draw("triangles")
        self.frame_index += 1

    def on_timer(self, event):
        self.program["iTime"] = event.elapsed
        self.update()

    def save_screenshot(self, fname):
        img = gloo.util._screenshot((0, 0, self.size[0], self.size[1]))
        imsave(fname, img)

    def on_key_press(self, event):
        if event.key == "Enter":
            self.save_screenshot("capture.png")

    def on_resize(self, event):
        self.program["iResolution"] = event.physical_size
        gloo.set_viewport(0, 0, *event.physical_size)

    def run(self):
        self.timer.start()
        self.show(run=True)


if __name__ == "__main__":
    anim = ThreeBody()
    anim.run()
