# Companions larger than their star

A planet transiting a white dwarf can be several times larger than its
star (WD 1856+534 b has $R_p/R_\star = 7.28$). Every entry point accepts
`rp > 1`, with every limb-darkening law, in both precisions, with
gradients.

A fourth regime then appears beside no overlap, partial overlap and a
planet fully inside the disc: **total occultation**, $z \le r_p - 1$
(with $z$ the sky separation in stellar radii), where the star is fully
covered, $F = 0$ exactly and every gradient is zero.

```python
import math
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a = 0.0, 1.4079, 7.28, 336.0     # WD 1856+534 b-like
p.inc = math.degrees(math.acos(7.79 / p.a))           # b = 7.79: grazing
p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
t = np.linspace(-0.01, 0.01, 2001)                    # a 5-minute, 16%-deep graze
flux = elome.TransitModel(p, t).light_curve(p)        # float64: exact
```

## Accuracy

Against a direct integral of the occulted intensity:

| `rp` | float64 (quadratic / polynomial / hybrid) | float32, partial overlap |
|---|---|---|
| 1.5 | $2\times10^{-16}$ | $8\times10^{-8}$ |
| 7.28 | $6\times10^{-15}$ / $7\times10^{-14}$ / $1\times10^{-14}$ | $3$–$9\times10^{-6}$ |
| 20 | $8\times10^{-14}$ / $2\times10^{-11}$ / $1\times10^{-13}$ | $10^{-4}$ |
| 50 | $3\times10^{-12}$ / $10^{-9}$ / $4\times10^{-12}$ | $2\times10^{-3}$ |

Total occultation is exact in both precisions. In partial overlap the
float32 error grows as $r_p^2$, because terms of size $\sim r_p$ cancel
to order 1. **Use float64 above $r_p \approx 10$**, or budget for the
error in the table.

## Things to know

- **Exposure integration.** The contact rule converges as it does for a
  planet, but its absolute error scales with the eclipse depth. For a
  *total* eclipse ($b < r_p - 1$, 100% deep; not the grazing example
  above), a 2-minute exposure of an 8-minute eclipse is accurate to about
  $10^{-5}$ at `n_gl = 5` and to below $10^{-6}$ at `n_gl = 9`.
- **Sampler bounds.** The ready-made anvil targets bound `r` to
  $[0.005, 0.5]$, `b` to $[0, 1.2]$ and `a` to $[1.5, 200]$ by default
  (`elome.anvil.DEFAULT_BOUNDS`). For a
  WD 1856+534 b-like system pass wider ones, for example
  `bounds={"r": (1.0, 12.0), "b": (0.0, 13.0), "a": (1.5, 1000.0)}` to
  `make_transit_target`.
- **`rp = 1` exactly** is a degenerate point, where the contacts meet at
  $z = 0$. The result there is good to $5\times10^{-9}$ in float64 and
  $10^{-4}$ in float32, and finite everywhere.
