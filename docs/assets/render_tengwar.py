"""Render a Tengwar string to an SVG of outlines (no font needed to view):
HarfBuzz shapes it (mark-to-base placement of the tehtar), fontTools draws
each positioned glyph."""
import sys
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen

FONT = "AlcarinTengwar-Regular.ttf"   # from github.com/Tosche/Alcarin-Tengwar (SIL OFL 1.1)
TEXT = ""     # E-l-ó-m-ë, Quenya mode

def render(text, fill, out, height=96, pad=0.06):
    blob = hb.Blob.from_file_path(FONT)
    face = hb.Face(blob)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {})
    tt = TTFont(FONT)
    gs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    x = 0
    placed = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = order[info.codepoint]
        placed.append((name, x + pos.x_offset, pos.y_offset))
        x += pos.x_advance
    # bounds in font units (y up)
    bp = BoundsPen(gs)
    for name, dx, dy in placed:
        gs[name].draw(TransformPen(bp, (1, 0, 0, 1, dx, dy)))
    xmin, ymin, xmax, ymax = bp.bounds
    w, h = xmax - xmin, ymax - ymin
    p = pad * h
    paths = []
    for name, dx, dy in placed:
        sp = SVGPathPen(gs)
        # flip y: SVG y down
        gs[name].draw(TransformPen(sp, (1, 0, 0, -1, dx - xmin + p, ymax + p - dy)))
        d = sp.getCommands()
        if d:
            paths.append(f'<path d="{d}"/>')
    W, H = w + 2 * p, h + 2 * p
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'height="{height}" width="{height * W / H:.0f}" role="img" '
           f'aria-label="Elómë in Tengwar (Quenya mode)">'
           f'<title>Elómë in Tengwar (Quenya mode)</title>'
           f'<g fill="{fill}">' + "".join(paths) + '</g></svg>\n')
    open(out, "w").write(svg)
    print(out, [n for n, _, _ in placed], f"{W:.0f}x{H:.0f}")

render(TEXT, "#1f2328", "elome_tengwar_light.svg")   # for light backgrounds
render(TEXT, "#e6edf3", "elome_tengwar_dark.svg")    # for dark backgrounds
