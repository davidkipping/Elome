"""Reproduce the measured batching and long-baseline numbers quoted in the docs.

    python benchmarks/verify_doc_claims.py           # measure; writes doc_claims.json
    python benchmarks/verify_doc_claims.py --check   # check the docs quote them

Every such number in the docs traces here or to vjp.json (written by
bench_vjp.py). --check measures nothing: it formats the stored numbers as
the docs print them and fails, listing each one, if a page does not
contain it. Run the measurements on an otherwise-idle GPU.
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")


def _latex(x):
    """'2.7e-07' -> '2.7\\times10^{-7}', as the docs write it."""
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def check():
    claims = json.load(open(os.path.join(HERE, "doc_claims.json")))
    v = json.load(open(os.path.join(HERE, "vjp.json")))["median_ms"]
    fa, fv, fm = v["autodiff_forward"], v["analytic_forward"], v["metal_forward"]
    ga, gv, gm = v["autodiff_value_grad"], v["analytic_value_grad"], v["metal_value_grad"]
    fe, ge = v["metal_ecc_forward"], v["metal_ecc_value_grad"]
    le, lc = v["chunked_eager"], v["chunked_compiled"]
    x_ad = f"{round(ga / gm, -1):.0f}"
    b = claims["batch_vs_loop"]
    per_loop_us = b["looped_ms"] / b["npv"] * 1e3
    per_batch_ns = b["batched_ms"] / b["npv"] * 1e6
    w = claims["fp32_baseline_flux_error"]
    expect = {
        "reference/performance.md": [
            f"| {fa:.0f} ms | {ga:,.0f} ms |", f"| {fv:.0f} ms | {gv:.0f} ms |",
            f"**{fm:.1f} ms** | **{gm:.1f} ms**", f"**{fe:.1f} ms** | **{ge:.1f} ms**",
            f"{round(fv / fm)} times faster", f"about {x_ad} times faster"],
        "reference/how-it-works.md": [f"{round(ga / gv, -1):.0f}–{x_ad} times faster"],
        "spherical/fitting.md": [
            f"{fm:.1f} ms at 1024 chains", f"pays {gm:.1f} ms", f"about {x_ad} times cheaper",
            f"about {le / lc:.1f} times as slow", f"({le:.1f} against {lc:.1f} ms"],
        "getting-started/installation.md": [
            f"about {round(fv / fm)} times slower", f"about\n  {round(gv / gm)} times slower"],
        "guides/batching.md": [
            f"about {round(b['looped_ms'], -2):,.0f} ms | about {round(per_loop_us, -1):.0f} µs",
            f"about {b['batched_ms']:.1f} ms | about {round(per_batch_ns, -1):.0f} ns",
            f"about {round(b['speedup'], -2):.0f} here"],
        "guides/gradients-and-precision.md": [
            f"${_latex(w['1'])}$ for a span of one orbit",
            f"${_latex(w['100'])}$ for 100", f"${_latex(w['1000'])}$ for 1,000"],
    }
    bad = []
    for page, needles in expect.items():
        text = open(os.path.join(DOCS, page)).read()
        bad += [f"{page}: missing {n!r}" for n in needles if n not in text]
    print("\n".join(bad) if bad else f"docs agree with the measurements "
          f"({sum(map(len, expect.values()))} numbers on {len(expect)} pages)")
    return not bad


if "--check" in sys.argv:
    sys.exit(0 if check() else 1)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

out = {}

# ---- claim 1: looped vs batched (the batching rule) ----------------------
from elome.metal import flux_dev_metal  # noqa: E402

rng = np.random.default_rng(0)
npv, npt = 10_000, 1_000
z_row = np.abs(rng.uniform(0, 1.3, npt)).astype(np.float32)
r_all = (0.1 * (1 + 0.1 * rng.standard_normal(npv))).astype(np.float32)
u1 = np.full(npv, 0.4, np.float32)
u2 = np.full(npv, 0.25, np.float32)

z1 = mx.array(z_row[None, :])
for _ in range(3):
    mx.eval(flux_dev_metal(z1, mx.array(r_all[:1]), mx.array(u1[:1]),
                           mx.array(u2[:1])))
t0 = time.perf_counter()
for i in range(200):
    mx.eval(flux_dev_metal(z1, mx.array(r_all[i:i + 1]), mx.array(u1[:1]),
                           mx.array(u2[:1])))
t_loop = (time.perf_counter() - t0) / 200 * npv

zb = mx.array(np.broadcast_to(z_row, (npv, npt)).copy())
rb, u1b, u2b = mx.array(r_all), mx.array(u1), mx.array(u2)
for _ in range(3):
    mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
ts = []
for _ in range(7):
    t0 = time.perf_counter()
    mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
    ts.append(time.perf_counter() - t0)
t_batch = float(np.median(ts))
out["batch_vs_loop"] = {
    "npv": npv, "npt": npt,
    "looped_ms": round(t_loop * 1e3, 1),
    "batched_ms": round(t_batch * 1e3, 2),
    "speedup": round(t_loop / t_batch),
}
print("batch_vs_loop:", out["batch_vs_loop"], flush=True)

# ---- claim 2: fp32 phase error vs the baseline the data span ------------
# The frontend re-centres the times on their midpoint in float64, so what
# float32 still sees is the half-span of the data. Here: transits at the
# start and at the end of a baseline of n_orbits orbits (t0 = 0), fp32
# against fp64 through the frontend, max in-transit flux error.
from elome import TransitModel, TransitParams  # noqa: E402

per = 10.0
wrap = {}
for n_orbits in (1, 100, 1000):
    p = TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, per, 0.1, 12.0, 88.0
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4, 0.25], "quadratic"
    win = np.linspace(-0.15, 0.15, 4001)
    t = np.concatenate([win, n_orbits * per + win])
    f32 = TransitModel(p, t, dtype=mx.float32).light_curve(p)
    f64 = TransitModel(p, t).light_curve(p)
    wrap[str(n_orbits)] = float(np.abs(f32 - f64).max())
out["fp32_baseline_flux_error"] = {k: f"{v:.1e}" for k, v in wrap.items()}
print("fp32_baseline_flux_error:", out["fp32_baseline_flux_error"], flush=True)

with open(os.path.join(HERE, "doc_claims.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote doc_claims.json")
