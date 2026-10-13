# How it works

A short tour of the numerical methods, for readers who want to know why
the results can be trusted.

## The spherical light curve

The occulted flux of a limb-darkened star is computed with the
closed-form solution of Agol, Luger & Foreman-Mackey (2020): the
intensity is expanded in Green's basis, and each basis term's occulted
flux is a closed form in the planet's radius $r$ and its sky separation
$z$ from the star's centre. The quadratic law's linear term needs
complete elliptic integrals, which Elómë evaluates with Bulirsch's
algorithm at a fixed iteration count. The polynomial law of any order
uses the same solution's recursion.

- **Regimes.** No overlap, partial overlap, a planet fully inside the
  disc, and (for `rp > 1`) total occultation are separate formulas.
  Elómë computes them all and selects with masks, with every branch's
  arguments made safe on the inactive side. A clamped-but-tiny
  denominator would give a finite value and an infinite derivative,
  which turns into NaN in the backward pass.
- **Degenerate lines.** Two narrow Taylor switches, $|z - r| < 10\epsilon$
  and $|z + r - 1| < \sqrt{\epsilon}$, handle the only places where the
  closed forms lose precision. The analytic gradients need no switches.
- **Fixed iteration counts everywhere**, with no data-dependent control
  flow, so every function can be compiled and batched.

## Gradients

The photometric core has analytic derivatives with respect to $r$, $z$
and the limb-darkening coefficients, written as MLX custom functions.
They avoid reverse-mode differentiation's memory-bound backward pass,
and are 40–450 times faster depending on the path.

## Orbits

- **Circular orbits** use the phase measured from mid-transit, computed in
  float64 on the host as an orbit number and a residual
  (`epoch_center_times`), so float32 never sees a large number.
- **Eccentric orbits.** Kepler's equation is solved by Markley's
  non-iterative method with one fifth-order refinement: exact to
  rounding, branch-free, and differentiable through an implicit
  derivative. The separation comes from the orbital-plane Cartesian
  coordinates $X = a(\cos E - e)$, $Y = a\sqrt{1-e^2}\sin E$, never from
  the true anomaly, which avoids cancellations near transit and at
  apastron.
- **Transit-anchored.** The solve is made for the eccentric anomaly's
  offset from inferior conjunction rather than for the anomaly itself,
  and it uses only $e\cos\omega$ and $e\sin\omega$, taken from the sampled
  $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$. The textbook form divides
  by $e$ in its derivatives; in float32 its gradients are 5% wrong at
  $e = 10^{-5}$ and entirely wrong at $10^{-8}$. The anchored form is
  algebraically identical (to $2.5\times10^{-15}$ in flux) and accurate to
  about $10^{-7}$ throughout.

## The GPU kernels

The float32 path fuses the whole model, orbit and photometry, into one
hand-written Metal kernel per call, keeping every intermediate in
registers. The gradient kernel evaluates the analytic derivatives in the
same pass and reduces them per chain. A circular orbit is the $e = 0$
case of the eccentric kernel, with the Kepler solve skipped per chain.
Exposure integration runs inside the kernel, per output point.

The float64 path computes the same function as an MLX graph on the CPU.
It shares the kernels' choices, such as where exposures are split, so
that it serves as their exact reference.

## The hybrid laws and oblate planets

For a spherical planet, the hybrid laws' terms have short closed forms
in the same quantities as the quadratic law, without elliptic integrals;
see {doc}`hybrid-laws`.

For an oblate planet, the occulted flux is a line integral around the
boundary of the overlap, which consists of arcs of the ellipse and of
the stellar limb. Along the ellipse, the even terms integrate to finite
Fourier sums, and each pole term to a sum over the roots of a quartic,
one quartic per pole and point. Where the ellipse crosses the limb is
another quartic. The GPU kernel finds the roots with its own small
eigenvalue solver (Metal has no linear algebra), polishes them, and
checks them. The contact times for exposure integration are solved per
side with a monotone Newton iteration on convex distance functions, and
a separate kernel computes them.

## MLX caveats

- `mx.sin`, `mx.cos`, `mx.exp` and `mx.erf` are only float32-accurate on
  float64 arrays in MLX 0.32, so Elómë uses its own float64 `sincos`
  (Cody–Waite reduction, accurate to $10^{-16}$).
- MLX operations on Python numbers create float32 scalars, so where a
  parameter may be a Python number Elómë does the arithmetic in Python
  first.
