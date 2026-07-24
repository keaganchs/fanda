"""Generate a clean SVG recreation of Figure 1 (supervision strategies) from
Liao & Poggio 2026, 'Simple Recursive Model'. CC BY 4.0 -> redraw is fine."""

N = 11                      # nodes z^0 .. z^10
X0, DX, R = 60, 60, 13
CY_OFF = 82                 # node centre offset within a row block
ROW_H = 150
TOP = 66
WIDTH = 60 + X0 + (N - 1) * DX + 40  # a bit of right margin
HEIGHT = TOP + 6 * ROW_H + 10

SUP = "#1a1aff"            # supervision blue
RED = "#e02b2b"           # gradient-flow red
REDF = "#f9d2d2"          # segment highlight fill
GREY = "#9aa0a6"          # no-gradient arrows
BLACK = "#222222"
NODE_FILL = "#cfe8f3"
NODE_STROKE = "#3a7ca5"
SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def xi(i):
    return X0 + i * DX


def sup_label(i):
    s = str(i)
    return "z" + "".join(SUPER[int(c)] for c in s)


def node(cx, cy):
    """A little clock/gauge 'refinement core' icon."""
    import math
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="{NODE_FILL}" '
             f'stroke="{NODE_STROKE}" stroke-width="1.5"/>']
    for k in range(12):                       # tick marks
        a = math.pi / 6 * k
        x1, y1 = cx + 9.5 * math.cos(a), cy + 9.5 * math.sin(a)
        x2, y2 = cx + 12 * math.cos(a), cy + 12 * math.sin(a)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{NODE_STROKE}" stroke-width="0.8"/>')
    # two hands
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+5.5:.1f}" y2="{cy-3.5:.1f}" '
                 f'stroke="#1f4e6b" stroke-width="1.4" stroke-linecap="round"/>')
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{cx-1.5:.1f}" y2="{cy-6.5:.1f}" '
                 f'stroke="#1f4e6b" stroke-width="1.4" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="1.6" fill="#1f4e6b"/>')
    return "".join(parts)


def arrow(i, cy, color, dashed=False, width=1.6):
    x1, x2 = xi(i) + R + 2, xi(i + 1) - R - 2
    dash = ' stroke-dasharray="4,3"' if dashed else ''
    mk = {SUP: 'sup', RED: 'red', GREY: 'grey', BLACK: 'black'}[color]
    return (f'<line x1="{x1}" y1="{cy}" x2="{x2-6}" y2="{cy}" stroke="{color}" '
            f'stroke-width="{width}"{dash} marker-end="url(#ah_{mk})"/>')


def seg_box(a, b, cy):
    x = xi(a) - R - 7
    w = xi(b) + R + 7 - x
    y = cy - R - 9
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{2*(R+9)}" rx="7" '
            f'fill="{REDF}" stroke="{RED}" stroke-width="1" opacity="0.9"/>')


def supervision(i, cy):
    x = xi(i)
    y_head, y_tail = cy + R + 8, cy + R + 30
    return (f'<line x1="{x}" y1="{y_tail}" x2="{x}" y2="{y_head}" stroke="{SUP}" '
            f'stroke-width="1.8" marker-end="url(#ah_sup)"/>'
            f'<text x="{x}" y="{y_tail+13}" font-size="9" fill="{SUP}" '
            f'text-anchor="middle" font-style="italic">supervision</text>')


def text(x, y, s, size=12, weight="normal", anchor="middle", style="normal",
         fill="#111"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def refinement_core(cy):
    # small italic label + tiny arrow onto node 0, above-left
    x = xi(0)
    return (text(x - 4, cy - R - 19, "refinement", 8.5, anchor="middle",
                 style="italic", fill="#555")
            + text(x - 4, cy - R - 11, "core", 8.5, anchor="middle",
                   style="italic", fill="#555"))


rows = []

# Row 1: refinement sequence (all solid black, per-node z^i labels)
def row1(cy):
    s = [refinement_core(cy)]
    for i in range(N - 1):
        s.append(arrow(i, cy, BLACK, width=1.8))
    for i in range(N):
        s.append(node(xi(i), cy))
        s.append(text(xi(i), cy + R + 15, sup_label(i), 9, fill="#444"))
    return "".join(s)


# Row 2: point supervision on a subset (grey solid arrows)
def row2(cy):
    s = [refinement_core(cy)]
    for i in range(N - 1):
        s.append(arrow(i, cy, GREY, width=1.6))
    for i in range(N):
        s.append(node(xi(i), cy))
    for i in (1, 4, 7, 10):
        s.append(supervision(i, cy))
    return "".join(s)


# Row 3: point supervision on all points (grey dashed arrows)
def row3(cy):
    s = [refinement_core(cy)]
    for i in range(N - 1):
        s.append(arrow(i, cy, GREY, dashed=True, width=1.4))
    for i in range(N):
        s.append(node(xi(i), cy))
    for i in range(N):
        s.append(supervision(i, cy))
    return "".join(s)


# Row 4: segment supervision (two 3-node segments, red gradient flow inside)
def row4(cy):
    segs = [(2, 4), (7, 9)]
    s = [refinement_core(cy)]
    for a, b in segs:
        s.append(seg_box(a, b, cy))
    seg_arrows = set()
    for a, b in segs:
        for i in range(a, b):
            seg_arrows.add(i)
    for i in range(N - 1):
        if i in seg_arrows:
            s.append(arrow(i, cy, RED, width=1.9))
        else:
            s.append(arrow(i, cy, GREY, dashed=True, width=1.4))
    for i in range(N):
        s.append(node(xi(i), cy))
    for a, b in segs:
        s.append(supervision(b, cy))
    return "".join(s)


# Row 5: end-to-end backprop (whole sequence one red segment)
def row5(cy):
    s = [refinement_core(cy), seg_box(0, N - 1, cy)]
    for i in range(N - 1):
        s.append(arrow(i, cy, RED, width=1.9))
    for i in range(N):
        s.append(node(xi(i), cy))
    s.append(supervision(N - 1, cy))
    return "".join(s)


# Row 6: truncated backprop, last T=3 (last segment red, rest grey dashed)
def row6(cy):
    a, b = 8, 10
    s = [refinement_core(cy), seg_box(a, b, cy)]
    for i in range(N - 1):
        if a <= i < b:
            s.append(arrow(i, cy, RED, width=1.9))
        else:
            s.append(arrow(i, cy, GREY, dashed=True, width=1.4))
    for i in range(N):
        s.append(node(xi(i), cy))
    s.append(supervision(b, cy))
    return "".join(s)


ROW_TITLES = [
    ["Refinement Sequence:  z⁰ → z¹ → … → z¹⁰"],
    ["Point Supervision:", "(No gradient flow between points)"],
    ["Point Supervision: All Points (Greedy Layerwise Learning)",
     "(No gradient flow between points)"],
    ["Segment Supervision:", "(Gradient flow within each segment)"],
    ["Segment Supervision: End-to-End Backpropagation",
     "(Gradient flow through entire sequence)"],
    ["Segment Supervision: Truncated Backpropagation",
     "(Gradient flow only in last T steps, T=3)"],
]
ROW_FN = [row1, row2, row3, row4, row5, row6]

body = []
body.append(text(WIDTH / 2, 30, "Refinement Learning: Supervision Strategies",
                 15, weight="bold"))
for r in range(6):
    top = TOP + r * ROW_H
    cy = top + CY_OFF
    lines = ROW_TITLES[r]
    ty = top + (14 if len(lines) == 2 else 22)
    for ln in lines:
        body.append(text(WIDTH / 2, ty, ln, 11.5, weight="bold"))
        ty += 15
    body.append(ROW_FN[r](cy))


def marker(name, color):
    return (f'<marker id="ah_{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker>')


defs = ('<defs>'
        + marker('black', BLACK) + marker('grey', GREY)
        + marker('red', RED) + marker('sup', SUP)
        + '</defs>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
       f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">'
       f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>'
       + defs + "".join(body) + '</svg>')

import sys
out = sys.argv[1] if len(sys.argv) > 1 else 'srm_supervision_strategies.svg'
with open(out, 'w') as f:
    f.write(svg)
print(f'wrote {out}  ({WIDTH}x{HEIGHT})')
