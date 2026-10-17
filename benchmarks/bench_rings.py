"""flux_dev_from_tau for a ringed planet against spherical and oblate ones.

The fused fp32 kernels (metal_hybrid, metal_oblate, metal_rings), at the
production shape (512 chains x 5,000 phase-ordered points around a
transit, Kepler long cadence), per hybrid law:

  <law>              spherical planet
  <law>_oblate       f = 0.2, theta = 0.5 rad
  <law>_ring         spherical planet with a ring: ring_in = 1.4,
                     ring_out = 2.6, ring_f = 0.7, ring_tau = 0.6,
                     theta = 0.5 rad
  <law>_ring_oblate  the same ring around the oblate planet

Each configuration runs in its own subprocess, interleaved over rounds;
reported is the median over rounds of each round's median. value+grad
evaluates value AND gradients in every input (tau, period, a, b, r,
[f,] [theta, ring_in, ring_out, ring_f, ring_tau,] w). The oblate and
ringed times include the per-call contact or kink solve.

    python benchmarks/bench_rings.py [--chains 512] [--points 5000]
                                     [--integration contact|none]

Run on a quiet machine: GPU timings on a loaded one mean little.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

P, A, B, R = 3.4525, 8.84, 0.30, 0.1153
F, TH = 0.2, 0.5
RING = (1.4, 2.6, 0.7, 0.6)            # ring_in, ring_out, ring_f, ring_tau
EXP = 29.4 / 60.0 / 24.0
LAWS = ("hybrid2", "hybrid4", "hybrid5")
KINDS = ("", "_oblate", "_ring", "_ring_oblate")
ROUTES = tuple(law + k for law in LAWS for k in KINDS)
WEIGHTS = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
           "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}


def worker(route, grad, n, m, integration, reps):
    import numpy as np
    import mlx.core as mx
    from elome.metal import flux_dev_from_tau

    rng = np.random.default_rng(11)
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    tau = mx.array(np.sort(tau, axis=1).astype(np.float32))

    def full(v):
        return mx.array(np.full(n, v, np.float32))

    law = route.split("_")[0]
    oblate = route.endswith("_oblate")
    ringed = "_ring" in route
    kw = dict(exp_time=EXP, integration=integration, n_gl=5, limb_dark=law)
    w = mx.array(np.tile(WEIGHTS[law], (n, 1)).astype(np.float32))
    args = [tau] + [full(v) for v in (P, A, B, R)] + [w]
    names = []
    if oblate:
        names += ["f"]
    if oblate or ringed:
        names += ["theta"]
    if ringed:
        names += ["ring_in", "ring_out", "ring_f", "ring_tau"]
    vals = dict(zip(("f", "theta", "ring_in", "ring_out", "ring_f",
                     "ring_tau"), (F, TH) + RING))
    args += [full(vals[k]) for k in names]

    def model(t, p, a, b, r, u, *extra):
        return flux_dev_from_tau(t, p, a, b, r, u=u, **dict(zip(names, extra)),
                                 **kw)

    ct = mx.array(rng.normal(size=(n, m)).astype(np.float32))
    vg = mx.value_and_grad(lambda *v: mx.sum(ct * model(*v)),
                           argnums=tuple(range(len(args))))

    def once():
        if grad:
            val, gs = vg(*args)
            mx.eval(val, *gs)
        else:
            mx.eval(model(*args))

    for _ in range(3):
        once()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--integration", default="contact")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--laws", default=",".join(LAWS))
    ap.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        dt = worker(a._worker[0], a._worker[1] == "1", a.chains, a.points,
                    a.integration, a.reps)
        print(json.dumps({"s": dt}))
        return

    laws = a.laws.split(",")
    routes = [law + k for law in laws for k in KINDS]
    here = os.path.abspath(__file__)
    res = {(r, g): [] for r in routes for g in (False, True)}
    for _ in range(a.rounds):
        for g in (False, True):
            for r in routes:
                out = subprocess.run(
                    [sys.executable, here, "--chains", str(a.chains),
                     "--points", str(a.points), "--integration",
                     a.integration, "--reps", str(a.reps),
                     "--_worker", r, "1" if g else "0"],
                    check=True, capture_output=True, text=True).stdout
                res[(r, g)].append(json.loads(out.splitlines()[-1])["s"])

    print(f"\n{a.chains} chains x {a.points} points, integration="
          f"{a.integration!r}, n_gl=5 (median of {a.rounds} isolated rounds)\n")
    print("| route | forward | vs spherical | vs oblate | value+grad "
          "| vs spherical | vs oblate |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for law in laws:
        f0 = statistics.median(res[(law, False)])
        g0 = statistics.median(res[(law, True)])
        f1 = statistics.median(res[(law + "_oblate", False)])
        g1 = statistics.median(res[(law + "_oblate", True)])
        for k in KINDS:
            f = statistics.median(res[(law + k, False)])
            g = statistics.median(res[(law + k, True)])
            print(f"| {law + k} | {f * 1e3:.1f} ms | {f / f0:.1f}x | "
                  f"{f / f1:.1f}x | {g * 1e3:.1f} ms | {g / g0:.1f}x | "
                  f"{g / g1:.1f}x |")


if __name__ == "__main__":
    main()
