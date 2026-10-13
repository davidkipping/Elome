# Performance and accuracy

All measurements are on an Apple M2 Max (12 CPU cores, 30-core GPU), each
configuration in its own process on a quiet machine. The scripts are in
the repository's `benchmarks/` directory, and each table names the one
that produced it.

## Against other transit codes (spherical planets)

The benchmark transit is quadratic limb-darkened, with $P = 3.456$ d,
$a/R_\star = 8.8$, $b = 0.45$, $R_p/R_\star = 0.1$ and $u = (0.40, 0.25)$
(`benchmarks/RESULTS.md`).

**Accuracy**, as the maximum error against a 30-digit direct integral
over 241 points:

| code | circular | eccentric ($e = 0.3$) |
|---|---:|---:|
| Elómë, float64 | $2.2\times10^{-16}$ | $4.4\times10^{-16}$ |
| exoplanet-core | $3.3\times10^{-16}$ | $5.6\times10^{-16}$ |
| jaxoplanet | $1.8\times10^{-12}$ (order 50) | $4.6\times10^{-9}$ (order 10) |
| batman | $6.0\times10^{-9}$ | $5.4\times10^{-8}$ |
| PyTransit (exact) | $4.0\times10^{-8}$ | $2.9\times10^{-7}$ |
| Elómë, float32 (GPU) | $1.8\times10^{-7}$ | $1.3\times10^{-7}$ |

**One light curve**, median wall time:

| code | $N = 10^3$ | $10^5$ | $10^6$ | $10^7$ |
|---|---:|---:|---:|---:|
| Elómë, float32 (GPU) | 0.34 ms | 0.42 ms | 1.13 ms | 9.76 ms |
| Elómë, float64 (CPU) | 0.42 ms | 14.3 ms | 129 ms | 1.29 s |
| batman (1 thread) | 0.02 ms | 1.92 ms | 18.9 ms | 194 ms |
| jaxoplanet | 0.13 ms | 2.10 ms | 20.2 ms | 183 ms |
| exoplanet-core | 0.04 ms | 2.82 ms | 29.8 ms | 318 ms |
| PyTransit | 0.04 ms | 3.55 ms | 35.7 ms | 360 ms |

Eccentric orbits, with each code's own Kepler solver inside the timed
region, take Elómë's GPU path to 15.3 ms at $10^7$ points, against 207 ms
for the fastest CPU code. Below about $10^4$ points the GPU's fixed
dispatch cost dominates and the CPU codes are faster.

**Many light curves**, 512 parameter sets × $10^5$ points each:

| code | wall time | light curves per second |
|---|---:|---:|
| Elómë, float32 (GPU), one call | 0.010 s | 53,590 |
| PyTransit (native arrays, numba) | 0.236 s | 2,174 |
| jaxoplanet (`jax.vmap`, CPU) | 0.903 s | 567 |
| exoplanet-core (loop) | 1.50 s | 342 |
| batman (loop) | 1.79 s | 287 |

Elómë's throughput stays at about 55,000 light curves per second from 256
to 4,096 sets, with no memory cliff.

## Gradients

Only Elómë and jaxoplanet are differentiable, and Elómë's gradients are
analytic, fused into Metal kernels. For the gradient of a chunked
log-likelihood at 1024 chains × 65,536 points, compiled, in float32
(`examples/bench_vjp.py`):

| | forward | value + gradient |
|---|---:|---:|
| reverse-mode automatic differentiation | 495 ms | 27,252 ms |
| analytic gradient (MLX graph) | 498 ms | 618 ms |
| **fused Metal kernel**, circular orbit | **21.8 ms** | **60.2 ms** |
| **fused Metal kernel**, eccentric ($e = 0.3$) | **29.4 ms** | **76.2 ms** |

The fused kernel keeps the whole model (orbit, Kepler solve and
photometry) in registers, with about 12 bytes of memory traffic per
point. That is 20 times faster than the same model as a compiled MLX graph,
and about 450 times faster for the gradient than reverse-mode
differentiation. End to end, a 1024-chain ChEES-HMC
injection-recovery fit runs at about 13 effective samples per second
with no divergences.

## Exposure integration, limb darkening, oblate planets

- Exposure rules: {doc}`../guides/exposures` and
  {doc}`../guides/sampler-api`.
- The hybrid laws against the quadratic law: {doc}`../guides/sampler-api`.
- Oblate planets, including against other oblate-planet codes:
  {doc}`../oblate/accuracy-and-cost`.
