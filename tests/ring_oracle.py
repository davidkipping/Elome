"""An independent reference for ringed planets, for the tests.

Shares no code with ``elome``: the occulted flux of a radially symmetric
intensity over any region built from concentric, coaxial ellipses (a
planet, a ring's edges) and clipped to the stellar disc is integrated
directly, in two unrelated ways:

* ``method="star"`` -- polar rays about the star's centre. Along a ray
  every conic cuts one chord; the ray is split at all chord ends, each
  piece's membership is read at its midpoint, and the radial integral is
  the closed-form primitive of ``oblate_oracle``. The ray angle is split
  at every tangent ray, every limb crossing and every conic-conic
  intersection, cos-mapped and integrated adaptively (Gauss-Legendre with
  bisection).
* ``method="centre"`` -- polar rays about the ring's centre. Every conic
  is then a single radius rho_k(psi) and the star a chord [t1, t2]; the
  radial integrals are closed forms in rho (polynomials for the even
  powers, a rational-plus-atanh primitive for a pole), the angle is split
  where the order of those radii changes (limb crossings, conic-conic
  intersections, the star's tangent rays) and integrated the same way.

The two share nothing past the problem statement, so their agreement
checks each.

Regions are given by a weight on the membership flags: ``weight(ins)``
takes a list of boolean arrays (inside conic k, k in order) and returns
the blocked fraction (1 opaque, 0 clear, anything between for a partly
transparent ring). Columns follow the package's generator order: mu^0,
mu^2, mu^4, then one double pole (mu^2 + eps)^-2 per eps; deviation form
(minus the occulted flux).

Geometry: all conics centred at (x0, y0), semi-axes (A, B) along x and y
(the ring's principal frame), the star is the unit disc at the origin.
"""

import math

import numpy as np

from oblate_oracle import _limb_angles, _tangent_angles, even_prim, pole_prim

__all__ = ["region_cols", "ring_cols", "conic_intersections"]


def _prims(eps):
    return [even_prim(n) for n in range(3)] + [pole_prim(e) for e in eps]


# ---------------------------------------------------------------------------
# geometry shared by both methods
# ---------------------------------------------------------------------------

def conic_intersections(c1, c2):
    """The (up to four) intersection points, relative to the common centre,
    of two concentric coaxial ellipses c = (A, B): linear in (x^2, y^2)."""
    (A1, B1), (A2, B2) = c1, c2
    M = np.array([[1 / A1 ** 2, 1 / B1 ** 2], [1 / A2 ** 2, 1 / B2 ** 2]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) <= 1e-14 * M[0, 0] * M[1, 1]:
        return []
    X, Y = np.linalg.solve(M, [1.0, 1.0])
    if X < 0.0 or Y < 0.0:
        return []
    x, y = math.sqrt(X), math.sqrt(Y)
    return [(sx * x, sy * y) for sx in (1, -1) for sy in (1, -1)]


def _limb_points(x0, y0, A, B):
    """Points where a conic meets the unit circle."""
    return [(math.cos(t), math.sin(t)) for t in _limb_angles(x0, y0, A, B)]


def _turn(angles, start):
    """Sorted breakpoints over one turn from ``start``, duplicates merged."""
    pts = [start, start + 2 * math.pi]
    pts += [start + (a - start) % (2 * math.pi) for a in angles]
    pts = np.unique(np.array(pts))
    return pts[np.concatenate([[True], np.diff(pts) > 1e-15])]


_GL = {n: np.polynomial.legendre.leggauss(n) for n in (10, 20)}


def _adaptive(a, b, f, ncol, rtol=1e-15, depth=45):
    """Integral of f(phi) -> (k, ncol) over [a, b], cos-mapped so square-
    root edges at either end become smooth; Gauss-Legendre 10 vs 20 with
    bisection, vectorised over panels."""
    def g(v):
        phi = a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * v))
        jac = (b - a) * 0.5 * np.pi * np.sin(np.pi * v)
        return jac[:, None] * f(phi)

    panels = np.array([[0.0, 1.0]])
    total = np.zeros(ncol)
    for _ in range(depth):
        lo_, hi_ = panels[:, :1], panels[:, 1:]
        mid, half = 0.5 * (lo_ + hi_), 0.5 * (hi_ - lo_)
        est = []
        for n in (10, 20):
            x, w = _GL[n]
            vals = g((mid + half * x).ravel()).reshape(len(panels), n, -1)
            est.append(half[:, :, None][:, 0] * np.einsum("pnc,n->pc", vals, w))
        err = np.abs(est[1] - est[0]).max(1)
        scale = np.maximum(np.abs(est[1]).max(1), 1.0)
        done = err <= rtol * scale * 50 + 1e-300
        total += est[1][done].sum(0)
        if done.all():
            return total
        rest = panels[~done]
        m = rest.mean(1, keepdims=True)
        panels = np.concatenate([np.hstack([rest[:, :1], m]), np.hstack([m, rest[:, 1:]])])
    raise RuntimeError("ring oracle: quadrature did not converge")


# ---------------------------------------------------------------------------
# method "star": rays about the star's centre
# ---------------------------------------------------------------------------

def _chord_raw(phi, x0, y0, A, B):
    """Unclipped chord [s1, s2] of the ray in the conic (s may be negative),
    and whether the ray's line meets it."""
    c, s = np.cos(phi), np.sin(phi)
    a = c * c / (A * A) + s * s / (B * B)
    bq = c * x0 / (A * A) + s * y0 / (B * B)
    cc = (x0 / A) ** 2 + (y0 / B) ** 2 - 1
    Q = ((1 - (y0 / B) ** 2) / (A * A) * c * c + 2 * x0 * y0 / (A * A * B * B) * c * s
         + (1 - (x0 / A) ** 2) / (B * B) * s * s)
    hit = Q >= 0
    sq = np.sqrt(np.where(hit, Q, 0.0))
    pos = bq >= 0
    big = np.where(pos, bq + sq, bq - sq) / a              # no cancellation
    other = cc / (a * np.where(big != 0, big, 1.0))
    s1, s2 = np.where(pos, other, big), np.where(pos, big, other)
    return np.where(hit, s1, 0.0), np.where(hit, s2, 0.0), hit


def _star_integrand(x0, y0, conics, weight, prims):
    def f(phi):
        chords = [_chord_raw(phi, x0, y0, A, B) for A, B in conics]
        ends = [np.zeros_like(phi), np.ones_like(phi)]
        for s1, s2, hit in chords:
            ends += [np.clip(np.where(hit, s1, 0.0), 0, 1), np.clip(np.where(hit, s2, 0.0), 0, 1)]
        ends = np.sort(np.stack(ends, -1), -1)
        out = np.zeros((phi.size, len(prims)))
        for k in range(ends.shape[1] - 1):
            lo, hi = ends[:, k], ends[:, k + 1]
            m = 0.5 * (lo + hi)
            ins = [hit & (s1 < m) & (m < s2) for s1, s2, hit in chords]
            w = np.where(hi > lo, weight(ins), 0.0)
            for j, p in enumerate(prims):
                out[:, j] += w * (p(hi) - p(lo))
        return out
    return f


def _star_breakpoints(x0, y0, conics):
    angles = []
    for A, B in conics:
        angles += _tangent_angles(x0, y0, A, B) + _limb_angles(x0, y0, A, B)
    for i in range(len(conics)):
        for j in range(i + 1, len(conics)):
            for X, Y in conic_intersections(conics[i], conics[j]):
                angles.append(math.atan2(y0 + Y, x0 + X))
    return _turn(angles, math.atan2(y0, x0) - math.pi)


# ---------------------------------------------------------------------------
# method "centre": rays about the conics' common centre
# ---------------------------------------------------------------------------

_GL_EVEN = np.polynomial.legendre.leggauss(4)      # exact for degree <= 7


def _radial_even(n, b, c0, lo, hi):
    """int_lo^hi (1 - s)^n rho d rho, s = rho^2 + 2 b rho + c0 (degree
    2n + 1 <= 5 in rho: 4-node Gauss-Legendre is exact)."""
    x, w = _GL_EVEN
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    acc = 0.0
    for xi, wi in zip(x, w):
        r = mid + half * xi
        s = r * r + 2 * b * r + c0
        acc = acc + wi * (1.0 - s) ** n * r
    return half * acc


def _radial_pole(e, b, c0, lo, hi):
    """int_lo^hi rho d rho / (p - s)^2, s as above, p = 1 + e: with
    u = rho + b and D = p - c0 + b^2 (> 0 on any ray that meets the
    star), g = D - u^2 = p - s > 0 on the star's chord and
        F(rho) = 1/(2 g) - b [u / (2 D g) + atanh(u / sqrt D) / (2 D^1.5)]."""
    p = 1.0 + e
    D = p - c0 + b * b
    D = np.where(D > 0, D, 1.0)
    sD = np.sqrt(D)

    def F(r):
        u = r + b
        g = D - u * u
        ok = g > 0                                 # always, on a star chord
        g = np.where(ok, g, 1.0)
        at = np.arctanh(np.where(ok, u / sD, 0.0))
        return 1.0 / (2 * g) - b * (u / (2 * D * g) + at / (2 * D * sD))
    return F(hi) - F(lo)


def _centre_integrand(x0, y0, conics, weight, eps):
    c0 = x0 * x0 + y0 * y0

    def f(psi):
        cs, sn = np.cos(psi), np.sin(psi)
        b = x0 * cs + y0 * sn
        disc = b * b - c0 + 1.0
        meets = disc > 0
        sq = np.sqrt(np.where(meets, disc, 0.0))
        # roots of t^2 + 2 b t + c0 - 1, the larger-magnitude one first
        big = np.where(b < 0, -b + sq, -b - sq)
        other = (c0 - 1.0) / np.where(big != 0, big, 1.0)
        t1 = np.where(meets, np.minimum(big, other), 0.0)
        t2 = np.where(meets, np.maximum(big, other), 0.0)
        rhos = [1.0 / np.sqrt(cs * cs / (A * A) + sn * sn / (B * B)) for A, B in conics]
        ends = np.sort(np.stack([np.zeros_like(psi), np.clip(t1, 0, None), np.clip(t2, 0, None)]
                                + rhos, -1), -1)
        out = np.zeros((psi.size, 3 + len(eps)))
        for k in range(ends.shape[1] - 1):
            lo, hi = ends[:, k], ends[:, k + 1]
            m = 0.5 * (lo + hi)
            star = meets & (t1 < m) & (m < t2)
            w = np.where((hi > lo) & star, weight([m < r for r in rhos]), 0.0)
            for n in range(3):
                out[:, n] += w * _radial_even(n, b, c0, lo, hi)
            for j, e in enumerate(eps):
                out[:, 3 + j] += np.where(w != 0, w * _radial_pole(e, b, c0, lo, hi), 0.0)
        return out
    return f


def _centre_breakpoints(x0, y0, conics):
    angles = []
    for A, B in conics:
        for X, Y in _limb_points(x0, y0, A, B):
            angles.append(math.atan2(Y - y0, X - x0))
    for i in range(len(conics)):
        for j in range(i + 1, len(conics)):
            for X, Y in conic_intersections(conics[i], conics[j]):
                angles.append(math.atan2(Y, X))
    d = math.hypot(x0, y0)
    if d > 1.0:                                     # the star's tangent rays
        back, g = math.atan2(-y0, -x0), math.asin(1.0 / d)
        angles += [back - g, back + g]
    return _turn(angles, 0.0)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def region_cols(x0, y0, conics, weight, eps, method="star", rtol=1e-15):
    """Deviation-form generator columns (3 + len(eps),) of the region with
    blocked fraction ``weight(ins)`` over the conics, clipped to the star."""
    conics = [(float(A), float(B)) for A, B in conics]
    ncol = 3 + len(eps)
    if method == "star":
        f = _star_integrand(x0, y0, conics, weight, _prims(eps))
        pts = _star_breakpoints(x0, y0, conics)
    elif method == "centre":
        f = _centre_integrand(x0, y0, conics, weight, tuple(eps))
        pts = _centre_breakpoints(x0, y0, conics)
    else:
        raise ValueError(method)
    return -sum(_adaptive(a, b, f, ncol, rtol) for a, b in zip(pts[:-1], pts[1:]) if b > a)


def ring_cols(x0, y0, planet, edges, T, eps, method="star", which="union"):
    """A planet (A_p, B_p) -- or None -- inside ring bands: ``edges`` the
    band edges (A, B), innermost first; T[k] the blocked fraction between
    edges k and k + 1 (len(edges) - 1 values; 1 is opaque). The planet is
    opaque, so the blocked fraction is 1 inside it and the band's T
    elsewhere.

    ``which``: "union" the whole occultor; "planet", "rings" (the bands
    alone), or "lens" (planet n first band's outer edge) for the pieces."""
    conics = ([planet] if planet is not None else []) + list(edges)
    off = 1 if planet is not None else 0

    def band(ins):
        w = np.zeros(ins[0].shape)
        for k, t in enumerate(T):
            w = np.where(ins[off + k + 1] & ~ins[off + k], t, w)
        return w

    def weight(ins):
        inP = ins[0] if planet is not None else np.zeros(ins[0].shape, bool)
        if which == "planet":
            return inP.astype(float)
        if which == "rings":
            return band(ins)
        if which == "lens":
            return (inP & ins[off + 1]).astype(float)
        return np.where(inP, 1.0, band(ins))
    return region_cols(x0, y0, conics, weight, eps, method)
