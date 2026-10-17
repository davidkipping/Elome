# Accuracy and fitting

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
