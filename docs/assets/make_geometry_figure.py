"""Draw the oblate-planet geometry figure used in docs/oblate/geometry.md.

    python docs/assets/make_geometry_figure.py

Writes oblate_geometry_light.svg and oblate_geometry_dark.svg beside this
script (the docs show the one that matches the reader's theme).
"""
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# geometry: rp is the area-equivalent radius, A = rp / sqrt(1 - f), B = A (1 - f)
RP, F, THETA = 0.36, 0.42, 30.0           # THETA in degrees
B_IMPACT, X_PLANET = 0.38, 0.47
A = RP / math.sqrt(1.0 - F)
B = A * (1.0 - F)


def draw(fg, star_lo, star_hi, accent, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # limb-darkened star: intensity falls towards the limb
    n = 600
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    r2 = X * X + Y * Y
    mu = np.sqrt(np.clip(1 - r2, 0, 1))
    inten = 1 - 0.45 * (1 - mu) - 0.2 * (1 - mu) ** 2
    rgba = np.zeros((n, n, 4))
    for k in range(3):
        rgba[..., k] = star_lo[k] + (star_hi[k] - star_lo[k]) * inten
    rgba[..., 3] = (r2 <= 1).astype(float)
    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", zorder=0)
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=fg, lw=0.8, alpha=0.5))

    # the transit chord and the direction of motion
    ax.plot([-1.55, 1.55], [-B_IMPACT, -B_IMPACT], ls=(0, (4, 3)), color=fg,
            lw=0.9, alpha=0.7)
    ax.annotate("", xy=(1.55, -B_IMPACT - 0.32), xytext=(0.95, -B_IMPACT - 0.32),
                arrowprops=dict(arrowstyle="-|>", color=fg, lw=1.0))
    ax.text(1.25, -B_IMPACT - 0.45, "direction of motion", ha="center",
            va="top", color=fg, fontsize=9)
    ax.plot([0], [0], marker="o", ms=2.5, color=fg)
    ax.annotate("", xy=(0, -B_IMPACT), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=fg, lw=0.8))
    ax.text(-0.04, -B_IMPACT / 2, "$b$", ha="right", va="center", color=fg,
            fontsize=11)

    # the planet: an ellipse, long axis at theta from the motion
    cx, cy = X_PLANET, -B_IMPACT
    ax.add_patch(Ellipse((cx, cy), 2 * A, 2 * B, angle=THETA, fc="#111111",
                         ec=accent, lw=1.2, zorder=3))
    t = math.radians(THETA)
    ct, st = math.cos(t), math.sin(t)
    ax.plot([cx, cx + A * ct], [cy, cy + A * st], color=accent, lw=1.3, zorder=4)
    ax.plot([cx, cx - B * st], [cy, cy + B * ct], color=accent, lw=1.3, zorder=4)
    ax.text(cx + 0.5 * A * ct - 0.07 * st, cy + 0.5 * A * st + 0.07 * ct, "$A$",
            color=accent, fontsize=12, ha="center", va="center", zorder=5)
    ax.text(cx - 0.5 * B * st - 0.07 * ct, cy + 0.5 * B * ct - 0.07 * st, "$B$",
            color=accent, fontsize=12, ha="center", va="center", zorder=5)
    # theta: from the motion direction to the long axis, drawn outside the planet
    R = 1.55 * A
    ax.plot([cx + A, cx + 1.75 * A], [cy, cy], color=fg, lw=0.9, zorder=4)
    ax.plot([cx + A * ct, cx + 1.75 * A * ct], [cy + A * st, cy + 1.75 * A * st],
            color=accent, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.add_patch(Arc((cx, cy), 2 * R, 2 * R, theta1=0, theta2=THETA, color=fg,
                     lw=1.0, zorder=4))
    ax.text(cx + 1.12 * R * math.cos(t / 2), cy + 1.12 * R * math.sin(t / 2),
            r"$\theta$", color=fg, fontsize=13, ha="left", va="center", zorder=5)

    ax.text(1.30, 0.80, r"$f = 1 - B/A$" "\n" r"$r_p = \sqrt{AB}$",
            color=fg, fontsize=10, ha="left", va="top")
    ax.text(-0.95, 0.88, "star", color=fg, fontsize=9, alpha=0.8)

    ax.set_xlim(-1.15, 2.15)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)


draw("#1f2328", (0.93, 0.62, 0.25), (1.0, 0.93, 0.70), "#c2410c",
     HERE / "oblate_geometry_light.svg")
draw("#e6edf3", (0.80, 0.50, 0.18), (1.0, 0.90, 0.62), "#fb923c",
     HERE / "oblate_geometry_dark.svg")
print("wrote", HERE / "oblate_geometry_light.svg", "and the dark version")
