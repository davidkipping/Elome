"""Benchmark: analytic VJP vs reverse-mode autodiff for the gradient
of the chunked log-likelihood — the operation ChEES-HMC pays for at
every leapfrog step.

Each (path x measurement) runs in a FRESH SUBPROCESS: sustained GPU work
perturbs later measurements through thermal state and the MLX buffer
cache.

Setup mirrors production: 1024 chains x 65536-point chunk, fp32, GPU,
compiled. Reports medians of 7 in-process reps.

Also measures the docs' "compile the likelihood" claim: anvil's
ChunkedGaussianLogLike over the fused model, at the same size, eager
against mx.compile (forward).

Writes benchmarks/vjp.json, which benchmarks/make_report.py and the docs
check (benchmarks/verify_doc_claims.py --check) read. Run on a quiet
machine:  python benchmarks/bench_vjp.py
"""

import json
import os
import subprocess
import sys

CFGS = [
    ("chunked", "eager"),
    ("chunked", "compiled"),
    ("autodiff", "forward"),
    ("analytic", "forward"),
    ("metal", "forward"),
    ("autodiff", "value_grad"),
    ("analytic", "value_grad"),
    ("metal", "value_grad"),
    ("metal_ecc", "forward"),          # e = 0.3, w = 63 deg: the Kepler solve
    ("metal_ecc", "value_grad"),
]

WORKER = r"""
import json, os, time
import mlx.core as mx
import numpy as np
from elome.anvil import make_quad_transit_flux

mode = os.environ["BENCH_MODE"]
meas = os.environ["BENCH_MEAS"]
N_CHAINS, CHUNK, PERIOD_REF = 1024, 65536, 3.4565

rng = np.random.default_rng(0)
v0 = np.tile([0.009, -0.0005, 0.10, 0.30, 8.80, 0.4225, 0.3077, 0.0],
             (N_CHAINS, 1)).astype(np.float32)
v0 += 1e-3 * rng.standard_normal(v0.shape).astype(np.float32)
v = mx.array(v0)
dt = rng.uniform(-PERIOD_REF / 2, PERIOD_REF / 2, CHUNK)
k = rng.integers(0, 26, CHUNK).astype(np.float64)
x = mx.array(np.stack([dt, k]).astype(np.float32))
y = mx.array(rng.normal(0.0, 5e-4, CHUNK).astype(np.float32))
w = mx.array(np.full(CHUNK, 1.0 / 5e-4, np.float32))

if mode == "metal_ecc":
    from elome.anvil import make_ecc_transit_flux
    import math
    k_, h_ = (math.sqrt(0.3) * f_(math.radians(63.0)) for f_ in (math.cos, math.sin))
    v0 = np.insert(v0, 7, np.tile([k_, h_], (N_CHAINS, 1)).T, axis=1).astype(np.float32)
    v = mx.array(v0)                   # t0_off p_off r b a q1 q2 secosw sesinw df0
    model = make_ecc_transit_flux(PERIOD_REF, core="metal")
else:
    model = make_quad_transit_flux(
        PERIOD_REF, core={"autodiff": "autodiff", "analytic": "analytic",
                          "metal": "metal", "chunked": "metal"}[mode])

def loglike(v_):
    r_ = (y - model(v_, x)) * w
    return mx.sum(-0.5 * r_ * r_, axis=-1)

if mode == "chunked":
    from elome.anvil import import_engine
    _, ChunkedGaussianLogLike = import_engine()
    ll = ChunkedGaussianLogLike(model, np.stack([dt, k]),
                                np.asarray(y, np.float64), np.full(CHUNK, 5e-4))
    fn = ll if meas == "eager" else mx.compile(lambda v_: ll(v_))
elif meas == "forward":
    fn = mx.compile(loglike)
else:
    def value_grad(v_):
        val, vjps = mx.vjp(loglike, [v_], [mx.ones((N_CHAINS,))])
        return val[0], vjps[0]
    fn = mx.compile(value_grad)

for _ in range(3):
    mx.eval(fn(v))
ts = []
for _ in range(7):
    t0 = time.perf_counter()
    mx.eval(fn(v))
    ts.append(time.perf_counter() - t0)
print(json.dumps({"mode": mode, "meas": meas,
                  "median_ms": float(np.median(ts) * 1e3)}))
"""

results = {}
for mode, meas in CFGS:
    env = dict(os.environ, BENCH_MODE=mode, BENCH_MEAS=meas)
    out = subprocess.run([sys.executable, "-c", WORKER], env=env,
                         capture_output=True, text=True, check=True)
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    results[(mode, meas)] = rec["median_ms"]
    print(f"{mode:>9s} {meas:>10s}: {rec['median_ms']:9.2f} ms", flush=True)

fa = results[("autodiff", "forward")]
fv = results[("analytic", "forward")]
fm = results[("metal", "forward")]
ga = results[("autodiff", "value_grad")]
gv = results[("analytic", "value_grad")]
gm = results[("metal", "value_grad")]
le, lc = results[("chunked", "eager")], results[("chunked", "compiled")]
fe, ge = results[("metal_ecc", "forward")], results[("metal_ecc", "value_grad")]
print(f"\nforward: autodiff {fa:.1f} / analytic {fv:.1f} / metal {fm:.1f} ms "
      f"(metal speedup {fv / fm:.2f}x over graph)")
print(f"value+grad: autodiff {ga:.0f} / analytic {gv:.0f} / metal {gm:.1f} ms "
      f"(metal {gv / gm:.2f}x over analytic, {ga / gm:.1f}x over autodiff)")
print(f"eccentric (e = 0.3) fused kernel: forward {fe:.1f} / value+grad {ge:.1f} ms")
print(f"chunked log-likelihood: eager {le:.2f} / compiled {lc:.2f} ms "
      f"({le / lc:.2f}x)")

import platform, time  # noqa: E401,E402
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "vjp.json"), "w") as f:
    json.dump({"machine": platform.platform(), "date": time.strftime("%Y-%m-%d"),
               "n_chains": 1024, "n_points": 65536, "median_ms":
               {f"{m}_{q}": round(t, 2) for (m, q), t in results.items()}},
              f, indent=1)
print("wrote", os.path.join(here, "vjp.json"))
