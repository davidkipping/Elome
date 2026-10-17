# Accuracy, cost and fitting

## Accuracy

**float64** is exact to rounding. The test suite compares every
limb-darkening term against two unrelated brute-force integrations of the
occulted intensity, which agree with each other to $4\times10^{-14}$:
rays about the star's centre, and rays about the ring's centre with
closed-form radial integrals. Against them, Elómë is within
$5\times10^{-16}$ of each term's full-disc flux in every geometry tested:

- a planet inside the ring's gap, and overlapping one or both edges;
- on the stellar limb, off the star with the ring on it, and centred on
  it;
- a ring wider than its distance to the limb, crossing it four times;
- a ring larger than the star;
- face-on and nearly edge-on rings, down to $|\cos i| = 10^{-10}$;
- exact and near tangencies between the planet and a ring edge,
  including where they meet on the limb;
- spherical and oblate planets, opaque and translucent rings.

A transparent ring (`ring_tau = 0`) reproduces the spherical and oblate
light curves to $10^{-14}$, and the light curve is exactly linear in the
ring's blocked fraction.

**float32** (the GPU kernels) is within $10^{-7}$ of float64 in flux,
on every exposure rule and orbit. Its gradients in every input are
within $10^{-6}$ of float64 automatic differentiation, relative to their
scale, and typically within $10^{-7}$. The inputs are `tau`, the orbit,
`r`, `f`, `theta`, `ring_in`, `ring_out`, `ring_f`, `ring_tau` and every
weight.

As for an oblate planet, a flattening below $10^{-5}$ is evaluated as a
circle in float32 -- here also `ring_f` below $10^{-5}$, a face-on ring
-- and the derivative with respect to it is 0 in that band.

## Cost

Every outline -- the planet and both ring edges -- is integrated, and
each splits into arcs where it crosses the others, so a ringed planet
costs about 10–20 times an oblate one. Fused float32 kernels, 512 chains
× 5,000 points around a transit, a spherical planet with a ring
(`benchmarks/bench_rings.py`, M2 Max):

| | forward | vs oblate | value + gradient | vs oblate |
|---|---:|---:|---:|---:|
| hybrid2, one evaluation per point | 108 ms | 11× | 420 ms | 13× |
| hybrid4, one evaluation per point | 175 ms | 11× | 707 ms | 15× |
| hybrid5, one evaluation per point | 229 ms | 11× | 980 ms | 16× |
| hybrid2, contact rule (`n_gl = 5`) | 1.59 s | 15× | 5.57 s | 17× |
| hybrid4, contact rule (`n_gl = 5`) | 2.52 s | 16× | 9.05 s | 19× |
| hybrid5, contact rule (`n_gl = 5`) | 3.27 s | 16× | 12.2 s | 20× |

An oblate planet with the same ring costs 10–20% more. The gradients
are in every input, the ring's included; the contact-rule times include
finding each chain's kinks.

## Fitting a ringed planet

Everything in {doc}`../oblate/fitting` about oblate planets applies:
batch every chain into one call with `flux_dev_from_tau`, wrap `theta`
onto $[0, \pi)$, and fold it onto $[0, \pi/2]$ for a near-central
transit. What is specific to a ring:

- **Sample `ring_f` uniformly** on $[0, 1)$ for an isotropic prior on the
  ring's orientation.
- **Optical depth and tilt trade off.** On a nearly edge-on ring the
  blocked fraction $1 - e^{-\tau/(1-\mathtt{ring\_f})}$ saturates, and
  the ring's area shrinks; a thin opaque ring and a wider translucent one
  can look alike. A prior on `ring_tau` (or sampling the blocked fraction
  directly and deriving `ring_tau`) helps.
- **Keep chains in the domain.** `ring_in` must reach past the planet's
  equator, `ring_in` $\ge 1/\sqrt{1-f}$, and `ring_out` must exceed
  `ring_in`; outside, an MLX array gives NaN. Sampling, for example,
  `ring_in` and the ratio `ring_out / ring_in` with bounded priors keeps
  them there by construction.
- **Expect correlations with the planet.** A ring's ingress and egress
  overlap the planet's, so `rp`, `ring_in` and the impact parameter
  correlate, more strongly the closer the ring is to face-on.
