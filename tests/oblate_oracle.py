"""An independent reference for oblate (elliptical) planets, for the tests.

It shares no code with ``elome.oblate``: the occulted flux of a radially
symmetric intensity I(rho^2) over the part of an ellipse inside the unit
disc is integrated directly, by polar rays about the star's centre.

Geometry is the package's principal frame: the ellipse is centred at
(x0, y0) with semi-axes A along x and B along y, the star is the unit disc
at the origin. A ray at angle phi from the origin meets the ellipse on the
chord s in [s1, s2] (a quadratic); clipped to the disc, the ray contributes
P(hi) - P(lo), where P is the closed-form radial primitive of the
intensity, dP/ds = I(s^2) s. The outer integral over phi is split at every
point where the integrand is not smooth -- the tangent rays from the
origin (where the chord opens, a square-root edge) and the angles where
the ellipse crosses the limb (where the clipping switches on) -- and each
piece is mapped by phi = a + (b - a)(1 - cos(pi v))/2, which turns a
square-root edge into a smooth one, then integrated adaptively: Gauss-
Legendre with bisection in fp64 (every column at once, vectorised), or
mpmath's tanh-sinh at 30 digits.

Columns follow the package's generator order: mu^0, mu^2, mu^4, then one
double pole (mu^2 + eps)^-2 per eps; deviation form (minus the occulted
flux).

Also here: the stress-set generators (topology classes, limb and pole
tangencies) and a numpy Keplerian orbit in the package's sky frame, both
written independently of the package.
"""

import math

import numpy as np

__all__ = ["columns", "columns_mp", "law_flux_dev", "even_prim", "pole_prim",
           "make_configs", "tangent_config", "planets", "stress_sets",
           "sky_position", "lens_area"]


# ---------------------------------------------------------------------------
# radial primitives: P(s) with dP/ds = I(s^2) s, I a function of rho^2 = s^2
# ---------------------------------------------------------------------------

def even_prim(n):
    """mu^(2n) = (1 - rho^2)^n."""
    return lambda s: -(1.0 - s * s) ** (n + 1) / (2.0 * (n + 1))


def pole_prim(eps):
    """(mu^2 + eps)^-2 = (1 + eps - rho^2)^-2."""
    return lambda s: 1.0 / (2.0 * (1.0 + eps - s * s))


def _prims(eps):
    return [even_prim(n) for n in range(3)] + [pole_prim(e) for e in eps]


# ---------------------------------------------------------------------------
# breakpoints
# ---------------------------------------------------------------------------

def _tangent_angles(x0, y0, A, B):
    """Directions of the rays from the origin tangent to the ellipse (none
    when the origin is inside it). On a ray (cos phi, sin phi) the chord's
    discriminant is the quadratic form Q = al c^2 + 2 be c s + ga s^2."""
    cc = (x0 / A) ** 2 + (y0 / B) ** 2 - 1.0
    if cc <= 0.0:
        return []
    al = (1.0 - (y0 / B) ** 2) / (A * A)
    ga = (1.0 - (x0 / A) ** 2) / (B * B)
    be = x0 * y0 / (A * A * B * B)
    m, h = 0.5 * (al + ga), 0.5 * (al - ga)
    R = math.hypot(h, be)
    if R == 0.0 or abs(m) > R:
        return []
    d, g = math.atan2(be, h), math.acos(-m / R)
    out = []
    for phi in ((d + g) / 2, (d - g) / 2):
        for _ in range(3):                                  # Newton polish
            Q = m + h * math.cos(2 * phi) + be * math.sin(2 * phi)
            dQ = -2 * h * math.sin(2 * phi) + 2 * be * math.cos(2 * phi)
            if dQ == 0.0:
                break
            phi -= Q / dQ
        out += [phi, phi + math.pi]
    return out


def _limb_angles(x0, y0, A, B):
    """Angles at which the ellipse meets the unit circle: the real roots of
    g(psi) = B^2 (cos psi - x0)^2 + A^2 (sin psi - y0)^2 - A^2 B^2, as a
    quartic in t = tan((psi - psi0)/2) about the centre's direction psi0.
    Near-real roots are kept too: an extra breakpoint costs nothing."""
    P = np.polynomial.polynomial
    p0 = math.atan2(y0, x0)
    C0, S0 = math.cos(p0), math.sin(p0)
    def pad(q):                       # numpy trims trailing zeros; keep length 5
        return np.pad(q, (0, 5 - len(q)))

    pc = np.array([C0, -2 * S0, -C0])                       # (1 + t^2) cos psi
    ps = np.array([S0, 2 * C0, -S0])                        # (1 + t^2) sin psi
    w = np.array([1.0, 0.0, 1.0])                           # 1 + t^2
    G = (B * B * pad(P.polymul(pc, pc)) + A * A * pad(P.polymul(ps, ps))
         - 2 * B * B * x0 * pad(P.polymul(pc, w)) - 2 * A * A * y0 * pad(P.polymul(ps, w))
         + (B * B * x0 * x0 + A * A * y0 * y0 - A * A * B * B) * pad(P.polymul(w, w)))

    def g(psi):
        return B * B * (math.cos(psi) - x0) ** 2 + A * A * (math.sin(psi) - y0) ** 2 - A * A * B * B

    def dg(psi):
        return (-2 * B * B * (math.cos(psi) - x0) * math.sin(psi)
                + 2 * A * A * (math.sin(psi) - y0) * math.cos(psi))

    out = []
    for t in P.polyroots(G):
        if abs(t.imag) > 1e-6 * (1.0 + abs(t)):
            continue
        psi = p0 + 2.0 * math.atan(t.real)
        for _ in range(4):                                  # Newton polish
            d = dg(psi)
            if d == 0.0:
                break
            step = g(psi) / d
            if abs(step) > 1e-6:
                break
            psi -= step
        out.append(psi)
    if abs(g(p0 + math.pi)) < 1e-9 * A * A * B * B:        # the root at t = inf
        out.append(p0 + math.pi)
    return out


def _pieces(x0, y0, A, B):
    """Sorted breakpoints covering one turn, starting at a direction the
    integrand is smooth through (opposite the centre)."""
    start = math.atan2(y0, x0) - math.pi
    pts = [start, start + 2 * math.pi]
    for phi in _tangent_angles(x0, y0, A, B) + _limb_angles(x0, y0, A, B):
        pts.append(start + (phi - start) % (2 * math.pi))
    pts = np.unique(np.array(pts))
    return pts[np.concatenate([[True], np.diff(pts) > 0])]


# ---------------------------------------------------------------------------
# the chord on one ray
# ---------------------------------------------------------------------------

def _chord(phi, x0, y0, A, B, sqrt=math.sqrt, cos=math.cos, sin=math.sin):
    """(lo, hi) of the ray's chord inside the ellipse and the disc, or None."""
    c, s = cos(phi), sin(phi)
    a = c * c / (A * A) + s * s / (B * B)
    bq = c * x0 / (A * A) + s * y0 / (B * B)
    cc = (x0 / A) ** 2 + (y0 / B) ** 2 - 1
    Q = ((1 - (y0 / B) ** 2) / (A * A) * c * c + 2 * x0 * y0 / (A * A * B * B) * c * s
         + (1 - (x0 / A) ** 2) / (B * B) * s * s)
    if Q < 0:
        return None
    sq = sqrt(Q)
    if bq >= 0:                                # no cancellation in either root
        s2 = (bq + sq) / a
        s1 = cc / (a * s2) if s2 != 0 else s2
    else:
        s1 = (bq - sq) / a
        s2 = cc / (a * s1)
    lo, hi = min(max(s1, 0), 1), min(max(s2, 0), 1)
    return (lo, hi) if hi > lo else None


# ---------------------------------------------------------------------------
# the integrals
# ---------------------------------------------------------------------------

def _chord_vec(phi, x0, y0, A, B):
    """_chord over an array of ray angles: (lo, hi), with hi = lo on rays
    that miss."""
    c, s = np.cos(phi), np.sin(phi)
    a = c * c / (A * A) + s * s / (B * B)
    bq = c * x0 / (A * A) + s * y0 / (B * B)
    cc = (x0 / A) ** 2 + (y0 / B) ** 2 - 1
    Q = ((1 - (y0 / B) ** 2) / (A * A) * c * c + 2 * x0 * y0 / (A * A * B * B) * c * s
         + (1 - (x0 / A) ** 2) / (B * B) * s * s)
    hit = Q >= 0
    sq = np.sqrt(np.where(hit, Q, 0.0))
    pos = bq >= 0
    big = np.where(pos, bq + sq, bq - sq) / a              # the root without cancellation
    other = cc / (a * np.where(big != 0, big, 1.0))
    s1 = np.where(pos, other, big)
    s2 = np.where(pos, big, other)
    lo, hi = np.clip(s1, 0, 1), np.clip(s2, 0, 1)
    ok = hit & (hi > lo)
    return np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)


_GL = {n: np.polynomial.legendre.leggauss(n) for n in (10, 20)}


def _piece(a, b, x0, y0, A, B, prims, rtol=1e-15, depth=40):
    """Adaptive Gauss-Legendre (10 vs 20 nodes, bisection) of one piece in
    v, with phi = a + (b - a)(1 - cos(pi v))/2. Vectorised over panels."""
    def f(v):                                               # v: (k,) -> (k, ncol)
        phi = a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * v))
        lo, hi = _chord_vec(phi, x0, y0, A, B)
        jac = (b - a) * 0.5 * np.pi * np.sin(np.pi * v)
        return jac[:, None] * np.stack([p(hi) - p(lo) for p in prims], -1)

    panels = np.array([[0.0, 1.0]])
    total = np.zeros(len(prims))
    for _ in range(depth):
        lo_, hi_ = panels[:, :1], panels[:, 1:]
        mid, half = 0.5 * (lo_ + hi_), 0.5 * (hi_ - lo_)
        est = []
        for n in (10, 20):
            x, w = _GL[n]
            vals = f((mid + half * x).ravel()).reshape(len(panels), n, -1)
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
    raise RuntimeError("oblate oracle: quadrature did not converge")


def _occulted_np(x0, y0, A, B, prims):
    pts = _pieces(x0, y0, A, B)
    return sum(_piece(a, b, x0, y0, A, B, prims) for a, b in zip(pts[:-1], pts[1:]))


def columns(x0, y0, A, B, eps):
    """Deviation-form generator columns (n, 3 + len(eps)) for rows of
    (x0, y0, A, B), in fp64. Zero when the ellipse misses the disc."""
    x0, y0, A, B = (np.ravel(v) for v in np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (x0, y0, A, B))))
    prims = _prims(eps)
    out = np.zeros((x0.size, len(prims)))
    for i in range(x0.size):
        if math.hypot(x0[i], y0[i]) >= 1.0 + A[i]:          # cannot overlap
            continue
        out[i] = -_occulted_np(x0[i], y0[i], A[i], B[i], prims)
    return out


def columns_mp(x0, y0, A, B, eps, dps=30):
    """The same, one configuration, at ``dps`` digits (mpmath tanh-sinh)."""
    import mpmath as mp
    with mp.workdps(dps):
        X0, Y0, AA, BB = (mp.mpf(v) for v in (x0, y0, A, B))
        prims = ([lambda s, n=n: -(1 - s * s) ** (n + 1) / (2 * (n + 1)) for n in range(3)]
                 + [lambda s, e=mp.mpf(e): 1 / (2 * (1 + e - s * s)) for e in eps])
        pts = [mp.mpf(p) for p in _pieces(x0, y0, A, B)]
        out = []
        for p in prims:
            def f(phi, p=p):
                ch = _chord(phi, X0, Y0, AA, BB, mp.sqrt, mp.cos, mp.sin)
                return 0 if ch is None else p(ch[1]) - p(ch[0])
            out.append(-float(mp.quad(f, pts)))
        return np.array(out)


def law_flux_dev(law, w, x0, y0, A, B):
    """F - 1 of a hybrid law with weights w, from the columns: the law in
    expanded form I = c0 + c1 mu^2 + c2 mu^4 + sum d_k P_eps_k."""
    from elome.hybrid import get_law, w_to_c
    L = get_law(law)
    c, d = w_to_c(L, np.asarray(w, dtype=np.float64))
    coef = np.concatenate([np.pad(c, (0, 3 - len(c))), d])
    norm = math.pi * (coef[0] + coef[1] / 2 + coef[2] / 3
                      + sum(dk / (e * (1 + e)) for dk, e in zip(d, L.eps)))
    return columns(x0, y0, A, B, L.eps) @ coef / norm


def lens_area(d, r):
    """Area of the unit disc covered by a circle of radius r at distance d
    (the uniform column's closed form at f = 0)."""
    if d >= 1 + r:
        return 0.0
    if d <= 1 - r:
        return math.pi * r * r
    k0 = math.acos((d * d + r * r - 1) / (2 * d * r))
    k1 = math.acos((d * d + 1 - r * r) / (2 * d))
    return (r * r * k0 + k1
            - 0.5 * math.sqrt(max(4 * d * d - (1 + d * d - r * r) ** 2, 0.0)))


# ---------------------------------------------------------------------------
# stress sets, in the principal frame with A >= B
# ---------------------------------------------------------------------------

def _polar(rng, d):
    t = rng.uniform(0, 2 * np.pi)
    return d * math.cos(t), d * math.sin(t)


def _make_config(rng, cls):
    A = rng.uniform(0.01, 0.6)
    B = A * rng.uniform(0.5, 1.0)
    if cls == "disjoint":                     # clear of the star
        x0, y0 = _polar(rng, 1 + A + rng.uniform(1e-6, 1.0))
    elif cls == "two_int":                    # straddling the limb
        x0, y0 = _polar(rng, 1 + rng.uniform(-0.9, 0.9) * B)
    elif cls == "inside":                     # wholly on the disc
        x0, y0 = _polar(rng, rng.uniform(0, 1 - A))
    elif cls == "origin_in_partial":          # the star's centre inside, partial
        A, B = rng.uniform(0.6, 0.99), rng.uniform(0.3, 0.6)
        x0, y0 = _polar(rng, rng.uniform(0.1, 0.25))
        y0 += 0.4
    else:
        raise ValueError(cls)
    return float(x0), float(y0), float(A), float(B)


def make_configs(seed, cls, n):
    """n random (x0, y0, A, B) of one topology class."""
    rng = np.random.default_rng(seed)
    return [_make_config(rng, cls) for _ in range(n)]


def tangent_config(A, B, phit, kind, gap=0.0):
    """An ellipse touching the unit circle at its own parameter angle phit:
    from outside ("external") or inside ("internal"). gap > 0 moves it
    along the common normal so the curves separate; gap < 0 overlaps."""
    p = np.array([A * math.cos(phit), B * math.sin(phit)])
    n = np.array([math.cos(phit) / A, math.sin(phit) / B])
    n /= np.linalg.norm(n)                    # the ellipse's outward normal at p
    T = -n if kind == "external" else n
    if kind not in ("external", "internal"):
        raise ValueError(kind)
    c = T - p - gap * n
    return float(c[0]), float(c[1]), float(A), float(B)


def planets(C, f_min=1e-9):
    """Keep the planetary domain: A < 1, A^2 / B <= 1, f in (f_min, 0.5]."""
    x0, y0, A, B = C
    m = (A < 1.0) & (A * A / B <= 1.0) & (B / A >= 0.5) & (B / A < 1.0 - f_min)
    return x0[m], y0[m], A[m], B[m]


def stress_sets(eps, n=150, limb=None, pole_gaps=None):
    """Topology classes, limb tangencies and pole tangencies (the circle
    rho^2 = 1 + eps, where a pole column's arcs are tangent)."""
    sets = {cls: make_configs(700 + k, cls, n) for k, cls in
            enumerate(("two_int", "inside", "disjoint", "origin_in_partial"))}
    lt = []
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, -1e-9, -1e-6, -1e-3):
        for kind in ("external", "internal"):
            for A, B, ph in (limb or ((0.12, 0.08, 1.1), (0.05, 0.035, 2.0), (0.3, 0.2, 4.0))):
                lt.append(tangent_config(A, B, ph, kind, gap))
    sets["limb_tangent"] = lt
    pt = []
    for e in eps:
        rho = math.sqrt(1.0 + e)
        for gap in (pole_gaps or (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, -1e-12, -1e-9)):
            for A, B, ph in ((0.12, 0.08, 1.1), (0.3, 0.2, 2.5), (0.02, 0.014, 4.0)):
                x0, y0, a, b = tangent_config(A / rho, B / rho, ph, "internal", gap)
                pt.append((x0 * rho, y0 * rho, a * rho, b * rho))
    sets["pole_tangent"] = pt
    return sets


# ---------------------------------------------------------------------------
# a Keplerian orbit in the package's sky frame
# ---------------------------------------------------------------------------

def sky_position(tau, period, a, inc, e=0.0, omega=math.pi / 2):
    """(X, Y, front) at times tau since inferior conjunction: X along the
    direction of motion at transit, Y across it, in stellar radii; front is
    the planet's being nearer than the star. Plain numpy: Newton on Kepler's
    equation from the conjunction's mean anomaly."""
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    fc = math.pi / 2 - omega                                  # true anomaly at conjunction
    Ec = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(fc / 2))
    M = Ec - e * math.sin(Ec) + 2 * math.pi * tau / period
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(math.sqrt(1 + e) * np.sin(E / 2), math.sqrt(1 - e) * np.cos(E / 2))
    rad = a * (1 - e * np.cos(E))
    u, v = rad * np.cos(omega + nu), rad * np.sin(omega + nu)
    return -u, -v * math.cos(inc), v > 0
