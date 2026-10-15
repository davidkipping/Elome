# The hybrid limb-darkening laws

The hybrid laws build the intensity profile from a few even powers of
$\mu$ and *double-pole* terms

$$
P_\epsilon(\mu) = \frac{1}{(\mu^2 + \epsilon)^2}, \qquad \epsilon > 0,
$$

which are infinite only at a projected radius $\sqrt{1+\epsilon}$, just
outside the star, and so rise steeply towards the limb as real profiles
do. A polynomial in $\mu^2$ cannot do that. The laws were designed so
that the light curve of a planet of any elliptical shape, and therefore
of a spherical one too, is *elementary*: every term has a closed form in
algebraic functions, arctangents and logarithms, with no elliptic
integrals.[^theory]

[^theory]: The hybrid-law paper (in preparation): the laws, and the proof
    that they give elementary light curves for spherical and oblate
    planets. See {doc}`../about/citing`.

## Definitions

Each law is written in a **shape basis**: every term $T_j(\mu)$ falls
from 1 at the limb ($\mu = 0$) to 0 at the centre ($\mu = 1$), and

$$
\frac{I(\mu)}{I(1)} = 1 - \sum_j w_j\, T_j(\mu),
\qquad
\frac{I(0)}{I(1)} = 1 - \sum_j w_j .
$$

The pole terms are normalised the same way:

$$
\Pi_\epsilon(\mu) = \frac{P_\epsilon(\mu) - (1+\epsilon)^{-2}}{\epsilon^{-2} - (1+\epsilon)^{-2}} .
$$

| law | weights | terms $T_j$ | poles $\epsilon$ |
|---|---:|---|---|
| `hybrid2` | 2 | $1-\mu^2$, $\Pi_{\epsilon}$ | 0.36 |
| `hybrid4` | 4 | $1-\mu^4$, $(1-\mu^2)^2$, $\Pi_{\epsilon_0}$, $\Pi_{\epsilon_1}$ | 0.0286, 0.386 |
| `hybrid5` | 5 | $1-\mu^4$, $(1-\mu^2)^2$, $\Pi_{\epsilon_0}$, $\Pi_{\epsilon_1}$, $\Pi_{\epsilon_2}$ | 0.0016, 0.0413, 0.387 |

The weights are ordered as the terms: even terms first, then poles.

### The poles

The poles of `hybrid4` and `hybrid5` lie on a ladder,

$$
\epsilon_k = \exp\!\left[\frac{2\pi}{\sqrt{\alpha}}\left(\sqrt{k + \tfrac12} - \sqrt{K}\right)\right],
\qquad k = 0, \ldots, K-1,
$$

with $K = 3$ and $\alpha = 1$ for `hybrid5`, and $K = 2$ and
$\alpha = 25/16$ for `hybrid4`. `elome.hybrid.ladder(K, alpha_eff)`
computes it. The `hybrid2` pole and the `hybrid4` ladder scale were tuned
on PHOENIX and MPS-ATLAS stellar models across 37 passbands.[^coeffs]
The exact values are `elome.HYBRID2_EPS`, `elome.HYBRID4_EPS` and
`elome.HYBRID5.eps`.

[^coeffs]: The hybrid-law limb-darkening coefficients paper (in
    preparation), which also tabulates the weights for stellar models
    and passbands. See {doc}`../about/citing`.

## The physical region

A profile is physical if its intensity is non-negative and does not
brighten towards the limb. For `hybrid4` and `hybrid5` that is the
simplex $w_j \ge 0$, $\sum_j w_j \le 1$; for `hybrid2` it is a triangle.
`simplex_from_q` and `hybrid2_from_q` map the unit cube onto these
regions uniformly, for priors; see {doc}`../spherical/limb-darkening`.

## Normalisation

The unocculted stellar flux, which normalises the light curve, is linear
in the weights. `elome.hybrid_norms(law)` returns the flux of each basis
column $[E_0, T_1, \ldots, T_n]$ (with $E_0$ the uniform disc), so that
the stellar flux is $N \cdot c$ with $c = (1, -w)$. The flux of
$1 - \mu^2$ is $\pi/2$, of $1 - \mu^4$ is $2\pi/3$, of $(1-\mu^2)^2$ is
$\pi/3$, and of $\Pi_\epsilon$ is $\pi\epsilon/(1 + 2\epsilon)$.

## Custom laws

A `HybridLaw` defines a law of the same family with other poles or even
terms. It is accepted wherever a law name is: in
`TransitParams.limb_dark`, `flux_dev_from_tau(limb_dark=...)` and
`flux_dev_metal`:

```python
from elome import HybridLaw

# shapes: even terms as coefficients over (mu^0, mu^2, mu^4);
# (1, -1, 0) is 1 - mu^2, (1, 0, -1) is 1 - mu^4, (1, -2, 1) is (1 - mu^2)^2
my_law = HybridLaw("my_hybrid3", eps=(0.05, 0.4), shapes=((1.0, -1.0, 0.0),))
params.limb_dark, params.u = my_law, [0.4, 0.1, 0.1]   # 1 + 2 weights
```

For a law with one even term and one pole, tie the prior to it with
`hybrid2_from_q(q1, q2, law=my_law)`.

The GPU kernels are compiled once per law *definition*, meaning its
poles and terms rather than its name. A law equal to a built-in one
reuses its kernels. Each new definition costs one compilation for the
life of the process, so scan over poles with a few laws rather than
thousands.

## Spherical planets: the closed forms

For a spherical planet every column has a short closed form in the
quantities the standard (Agol, Luger & Foreman-Mackey 2020) solution
already computes. The even powers $\mu^0$, $\mu^2$, $\mu^4$ come from its
even recursion. A pole term is an arc term plus an arctangent or
logarithm of rational functions of the separation $z$ and radius $r$,
switching to a short series where the two forms meet. Neither needs an
elliptic integral, which is why the hybrid laws are cheaper than the
quadratic law, whose $\mu^1$ term does. The closed forms are checked
against a 40-digit direct integral to $10^{-13}$.

For an oblate planet the boundary of the overlap is an elliptical arc,
and each pole term needs the roots of a quartic per point. Elómë's GPU
kernel seeds them from the quartic's closed-form solution and polishes
them (see {doc}`how-it-works`).
