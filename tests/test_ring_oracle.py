"""The ring oracle (tests/ring_oracle.py) checked before anything is checked
against it: its two methods against each other, against the oblate oracle
for a single ellipse, and against the package's spherical closed forms for
a planet alone."""

import math

import mlx.core as mx
import numpy as np
import pytest

import oblate_oracle as OO
import ring_oracle as RO
from elome.hybrid import HYBRID2, HYBRID4, HYBRID5, even_cols, pole_col

EPS = tuple(HYBRID5.eps) + tuple(HYBRID4.eps) + tuple(HYBRID2.eps)


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def union_configs():
    """Planet + one ring, from face-on to nearly edge-on, inside the star,
    on the limb (including an outer edge crossing it four times) and
    partly off it; opaque and partly transparent."""
    out = []
    for (d, th, c, Ro, tin, Rp, T) in [
            (0.30, 0.4, 0.9, 0.20, 0.6, 0.08, 1.0),     # planet inside the cavity
            (0.20, 2.0, 0.3, 0.25, 0.5, 0.10, 1.0),     # planet overlaps both edges
            (0.95, 0.3, 0.3, 0.25, 0.7, 0.08, 0.55),    # on the limb, partly transparent
            (0.97, 1.4, 0.05, 0.30, 0.6, 0.06, 1.0),    # Saturn-like: 4 limb crossings
            (1.10, -0.5, 0.6, 0.20, 0.5, 0.05, 1.0),    # planet off the star, ring on it
            (0.00, 0.0, 0.12, 0.30, 0.5, 0.10, 0.55)]:  # centred on the star
        out.append((d * math.cos(th), d * math.sin(th), Ro, Ro * c, tin, Rp, T))
    return out


@pytest.mark.parametrize("cfg", union_configs())
def test_two_methods_agree(cfg):
    x0, y0, Ao, Bo, tin, Rp, T = cfg
    edges = [(Ao * tin, Bo * tin), (Ao, Bo)]
    a = RO.ring_cols(x0, y0, (Rp, Rp), edges, [T], EPS, "star")
    b = RO.ring_cols(x0, y0, (Rp, Rp), edges, [T], EPS, "centre")
    assert rel(b, a) < 1e-12


@pytest.mark.parametrize("g", [(0.3, 0.2, 0.12, 0.05), (0.95, 0.1, 0.15, 0.06),
                               (0.0, 0.0, 0.3, 0.1), (1.05, 0.0, 0.1, 0.1)])
@pytest.mark.parametrize("method", ["star", "centre"])
def test_single_ellipse_is_the_oblate_oracle(g, method):
    x0, y0, A, B = g
    got = RO.region_cols(x0, y0, [(A, B)], lambda ins: ins[0].astype(float), EPS, method)
    assert rel(got, OO.columns(x0, y0, A, B, EPS)[0]) < 1e-12


@pytest.mark.parametrize("cfg", union_configs())
def test_planet_alone_is_the_spherical_closed_form(cfg):
    x0, y0, Ao, Bo, tin, Rp, T = cfg
    got = RO.ring_cols(x0, y0, (Rp, Rp), [(Ao * tin, Bo * tin), (Ao, Bo)], [T], EPS,
                       "centre", which="planet")
    with mx.stream(mx.cpu):
        z = mx.array([math.hypot(x0, y0)], dtype=mx.float64)
        r = mx.array(Rp, dtype=mx.float64)
        ref = np.array([c.item() for c in even_cols(z, r)]
                       + [pole_col(z, r, e).item() for e in EPS])
    assert rel(got, ref) < 1e-12


@pytest.mark.parametrize("cfg", union_configs())
def test_union_is_inclusion_exclusion(cfg):
    """P u T (O \\ I) = P + T [O - I - (P n O) + (P n I)], each from its own
    region: the weight bookkeeping, not the quadrature."""
    x0, y0, Ao, Bo, tin, Rp, T = cfg
    P, I, O = (Rp, Rp), (Ao * tin, Bo * tin), (Ao, Bo)

    def reg(conics, w):
        return RO.region_cols(x0, y0, conics, w, EPS)

    one = lambda ins: ins[0].astype(float)
    both = lambda ins: (ins[0] & ins[1]).astype(float)
    parts = reg([P], one) + T * (reg([O], one) - reg([I], one) - reg([P, O], both) + reg([P, I], both))
    whole = RO.ring_cols(x0, y0, P, [I, O], [T], EPS)
    assert np.max(np.abs(parts - whole)) < 1e-12 * np.max(np.abs(whole))
