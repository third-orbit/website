"""
Pre-compute three-body orbit positions for the WebGL viewer.

Reads init_conditions.json, integrates each periodic orbit at high precision,
samples it on a uniform time grid, and writes:

  web/orbits.json          — index + per-orbit metadata + binary file path
  web/orbits/<slug>.bin    — one little-endian float32 file per orbit

Splitting the binary per orbit lets the page only download the one orbit it
picks on load, instead of fetching all 16.

Run from the repo root:  python precompute.py
"""

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


G = m1 = m2 = m3 = 1.0


def calc_accel(t, y):
    r1, r2, r3, v1, v2, v3 = np.split(y, 6)
    d12 = np.linalg.norm(r2 - r1) ** 3
    d23 = np.linalg.norm(r3 - r2) ** 3
    d31 = np.linalg.norm(r1 - r3) ** 3

    a1 = G * m2 * (r2 - r1) / d12 + G * m3 * (r3 - r1) / d31
    a2 = G * m3 * (r3 - r2) / d23 + G * m1 * (r1 - r2) / d12
    a3 = G * m1 * (r1 - r3) / d31 + G * m2 * (r2 - r3) / d23
    return np.concatenate([v1, v2, v3, a1, a2, a3])


def precompute_one(name, conditions):
    T = conditions["period"]
    y0 = np.concatenate(conditions["positions"] + conditions["velocities"])

    sol = solve_ivp(
        calc_accel,
        [0, T],
        y0,
        method="RK45",
        rtol=1e-10,
        atol=1e-10,
        dense_output=True,
    )

    n_samples = max(2048, math.ceil(T / 0.01))
    t_eval = np.linspace(0, T, n_samples, endpoint=False)
    y = sol.sol(t_eval).T  # (n_samples, 12)

    samples = y[:, :6].astype(np.float32)  # (n_samples, 6) — r1x r1y r2x r2y r3x r3y

    bodies = samples.reshape(-1, 2)  # all body positions, (n_samples*3, 2)
    center = bodies.mean(axis=0)
    extent = np.max(np.abs(bodies - center), axis=0)

    # Mean speed of the bodies at t=0 — used by the viewer to pick a per-orbit
    # playback duration that keeps on-screen speed roughly constant across
    # orbits, regardless of period or spatial extent.
    v_init = np.array(conditions["velocities"])  # (3, 2)
    avg_start_speed = float(np.linalg.norm(v_init, axis=1).mean())

    seam = np.linalg.norm(sol.sol(T) - sol.sol(0))
    seam_rel = seam / max(extent.max(), 1e-12)
    if seam_rel > 1e-6:
        print(f"  warning: {name!r} periodicity seam = {seam_rel:.2e} (relative)")

    return {
        "name": name,
        "period": T,
        "sampleCount": n_samples,
        "center": [float(center[0]), float(center[1])],
        "extent": [float(extent[0]), float(extent[1])],
        "avgStartSpeed": avg_start_speed,
        "_samples": samples,
    }


def slugify(name):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "orbit"


def main():
    repo_root = Path(__file__).parent
    init_path = repo_root / "init_conditions.json"
    out_dir = repo_root / "web"
    bin_dir = out_dir / "orbits"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Clean up the previous monolithic file and any stale per-orbit bins so
    # renames between runs don't leave orphans.
    old_monolith = out_dir / "orbits.bin"
    if old_monolith.exists():
        old_monolith.unlink()
    for f in bin_dir.glob("*.bin"):
        f.unlink()

    with open(init_path, encoding="utf-8") as f:
        init_conditions = json.load(f)

    orbits = []
    total_bytes = 0

    for name, conditions in init_conditions.items():
        print(f"integrating {name!r} (T={conditions['period']:.3f}) ...")
        orbit = precompute_one(name, conditions)
        samples = orbit.pop("_samples")

        slug = slugify(name)
        bin_path = bin_dir / f"{slug}.bin"
        with open(bin_path, "wb") as f:
            f.write(samples.tobytes(order="C"))
        size = bin_path.stat().st_size
        total_bytes += size

        orbit["file"] = f"orbits/{slug}.bin"
        orbits.append(orbit)
        print(f"  wrote {bin_path.relative_to(repo_root)} ({size / 1024:.1f} KiB)")

    json_path = out_dir / "orbits.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(orbits, f, indent=2)

    print()
    print(f"wrote {json_path.relative_to(repo_root)} ({len(orbits)} orbits, {total_bytes / 1024:.1f} KiB total)")


if __name__ == "__main__":
    main()
