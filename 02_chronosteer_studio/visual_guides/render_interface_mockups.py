"""
Render 3 high-resolution ChronoSteer Studio UI visual guides matching the exact
minimal #000000 terminal + semicircular radial wheel aesthetic of the reference video.
Outputs (inside 02_chronosteer_studio/visual_guides/):
    1. 01_main_workspace_scrubber_and_attention_rail.png
    2. 02_radial_steering_wheel_and_topk_inspector.png
    3. 03_multiverse_3x3_branching_word_tree.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Arc, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent

BG = "#000000"
FG = "#E8EAED"
MUTED = "#6E7681"
DIM = "#2D333B"
CYAN = "#58A6FF"
ORANGE = "#F97316"
CORAL = "#EF4444"
GOLD = "#EAB308"
GREEN = "#22C55E"


def new_canvas(footer_text: str):
    fig = plt.figure(figsize=(14, 8.2), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    # Minimal bottom status bar like the video
    ax.plot([2.5, 97.5], [5.5, 5.5], color=DIM, lw=0.8)
    ax.text(50.0, 2.8, footer_text, color=MUTED, fontsize=8.3, ha="center", va="center", family="DejaVu Sans Mono")
    return fig, ax


def draw_pixel_rail(ax, lines, y_top=92.0, dy=4.2, steer_rows=(4, 5)):
    """Draw the exact video-style left attention rail with dashed bars & steering box."""
    for idx, (ln, txt, attn, hl_col) in enumerate(lines):
        y = y_top - idx * dy
        is_steer = (idx + 1) in steer_rows
        ax.text(3.2, y, f"{idx+1:02d}", color=MUTED if not is_steer else FG, fontsize=7.8, family="DejaVu Sans Mono", va="center")
        # Draw segmented pixel bars (like in video)
        n_segs = max(1, int(round(attn * 28)))
        for s in range(n_segs):
            sx = 6.2 + s * 0.65
            col = ORANGE if is_steer else ("#9CA3AF" if attn > 0.12 else "#374151")
            ax.plot([sx, sx + 0.42], [y, y], color=col, lw=3.0, solid_capstyle="butt")
        if is_steer:
            ax.text(14.0, y, f".{int(attn*100):02d}", color=FG, fontsize=7.6, family="DejaVu Sans Mono", va="center")

    # Dashed steering box around lines 04-05
    if steer_rows:
        y_hi = y_top - (steer_rows[0] - 1) * dy + 1.8
        y_lo = y_top - (steer_rows[-1] - 1) * dy - 1.8
        rect = FancyBboxPatch((2.6, y_lo), 14.0, y_hi - y_lo, boxstyle="square,pad=0", facecolor="none", edgecolor="#9CA3AF", lw=1.0, ls="--")
        ax.add_patch(rect)


def render_mockup_1():
    fig, ax = new_canvas("kv-cache: on (19.6 tok/s)   |   t = 142   |   c 50%   |   s 0%   |   [drag ▲/▼] scrub time   |   [click rail] steer attention")

    lines = [
        ("01", "There was an old legend about a forest that never slept,", 0.04, None),
        ("02", "where the trees whispered secrets to anyone brave enough", 0.06, None),
        ("03", "to listen. One night, under a sky brushed with clouds,", 0.05, None),
        ("04", "a young wanderer named Elara stepped through the tangled", 0.28, ORANGE),
        ("05", "undergrowth, guided only by the faint hum of an ancient", 0.24, ORANGE),
        ("06", "lullaby carried on the wind. As she pushed forward, the", 0.11, None),
        ("07", "forest grew quieter, its mossy floor pulsing with soft", 0.09, None),
        ("08", "bioluminescent glow. She paused beside a silver pool", 0.14, None),
        ("09", "where the water's surface rippled with a soft, rhythmic", 0.18, CORAL),
        ("10", "pulse, reflecting a ring of ancient runes beneath her feet.", 0.16, CORAL),
        ("11", "She knelt, and as her fingertips brushed the cool liquid,", 0.22, CYAN),
    ]

    draw_pixel_rail(ax, lines, y_top=91.5, dy=4.4, steer_rows=(4, 5))

    y_top, dy = 91.5, 4.4
    for idx, (ln, txt, attn, hl_col) in enumerate(lines):
        y = y_top - idx * dy
        ax.text(20.0, y, txt, color=FG, fontsize=9.1, family="DejaVu Sans Mono", va="center")
        if hl_col == ORANGE:
            ax.plot([20.0, 68.5], [y - 1.3, y - 1.3], color=ORANGE, lw=1.2, alpha=0.85)
        elif hl_col == CORAL:
            ax.plot([20.0, 68.5], [y - 1.3, y - 1.3], color=CORAL, lw=1.2, alpha=0.85)
        elif hl_col == CYAN:
            ax.plot([20.0, 68.5], [y - 1.3, y - 1.3], color=CYAN, lw=1.2, alpha=0.85)

    # Video-exact horizontal dashed scrubber line at cursor
    scrub_y = y_top - 10.7 * dy
    ax.plot([18.5, 73.0], [scrub_y, scrub_y], color=MUTED, lw=1.0, ls=(0, (3, 4)))
    ax.add_patch(FancyBboxPatch((47.2, scrub_y - 0.65), 1.3, 1.3, boxstyle="square,pad=0", facecolor=CORAL, edgecolor="none"))
    ax.text(74.2, scrub_y, "▲ rewind  |  ▼ generate", color=MUTED, fontsize=8.0, family="DejaVu Sans Mono", va="center")

    # Right-side minimal pills matching the video!
    pills = [
        (76.5, 62.0, "● tone: tender", "61%", CYAN),
        (76.5, 56.5, "● tension: very strained", "78%", CORAL),
        (76.5, 51.0, "● focus: Elara", "85%", ORANGE),
    ]
    for px, py, plbl, ppct, pcol in pills:
        ax.text(px, py, f"{plbl}  {ppct}", color=pcol, fontsize=8.6, fontweight="bold", family="DejaVu Sans Mono", va="center")

    # Semicircular dotted radial wheel right below the cursor (matching 00:01-00:03 of video!)
    cx, cy = 47.8, scrub_y
    ax.add_patch(Arc((cx, cy), 22.0, 22.0, angle=0, theta1=180, theta2=360, color=DIM, lw=1.0, ls="--"))
    spokes = [
        (200, "pace: somewhat compressed", MUTED),
        (232, "tension: very relaxed", CORAL),
        (268, "focus: Elara", ORANGE),
        (308, "surprise: very unexpected", MUTED),
        (340, "tone: tender", CYAN),
    ]
    for deg, slbl, scol in spokes:
        r = np.deg2rad(deg)
        x2, y2 = cx + 11.0 * np.cos(r), cy + 11.0 * np.sin(r)
        ax.plot([cx, x2], [cy, y2], color=scol if scol != MUTED else DIM, lw=1.0, ls=":")
        ax.add_patch(Circle((x2, y2), 0.45, facecolor=scol, edgecolor="none"))
        lx, ly = cx + 13.0 * np.cos(r), cy + 13.0 * np.sin(r)
        ha = "right" if np.cos(r) < -0.15 else ("left" if np.cos(r) > 0.15 else "center")
        ax.text(lx, ly, slbl, color=scol if scol != MUTED else MUTED, fontsize=7.8, ha=ha, va="center", family="DejaVu Sans Mono")

    out = OUT_DIR / "01_main_workspace_scrubber_and_attention_rail.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    return out


def render_mockup_2():
    fig, ax = new_canvas("wheel click: open spectrum slider + top-k inspector   |   click any word below to swap & fork timeline")

    # Top context lines in pure #000000 monochrome
    ctx = [
        "10  pulse, reflecting a ring of ancient runes beneath her feet.",
        "11  She knelt, and as her fingertips brushed the cool",
    ]
    ax.text(10.0, 91.0, ctx[0], color=MUTED, fontsize=9.2, family="DejaVu Sans Mono")
    ax.text(10.0, 86.5, ctx[1], color=FG, fontsize=9.2, family="DejaVu Sans Mono")
    ax.plot([9.0, 88.0], [83.5, 83.5], color=DIM, lw=1.0, ls=(0, (3, 4)))
    ax.add_patch(FancyBboxPatch((49.0, 82.85), 1.3, 1.3, boxstyle="square,pad=0", facecolor=CORAL, edgecolor="none"))

    # Video-exact horizontal spectrum bar expanded from the wheel!
    ax.text(50.0, 77.5, "VECTOR SPECTRUM:  tone  (dry  <--->  wondrous)", color=MUTED, fontsize=8.4, ha="center", family="DejaVu Sans Mono")
    bar_y = 71.5
    ax.plot([14.0, 86.0], [bar_y, bar_y], color=DIM, lw=2.2)
    stops = [
        (18.0, "dry", MUTED, False),
        (39.0, "tender", CYAN, True),
        (61.0, "somber", MUTED, False),
        (82.0, "wondrous", MUTED, False),
    ]
    for sx, slbl, scol, active in stops:
        ax.plot([sx, sx], [bar_y - 1.2, bar_y + 1.2], color=scol, lw=1.8 if active else 1.1)
        ax.text(sx, bar_y + 2.8, slbl, color=scol, fontsize=8.8, fontweight="bold" if active else "normal", ha="center", family="DejaVu Sans Mono")
    # Active orange/cyan slider knob at 61% tender
    ax.plot([39.0, 44.5], [bar_y, bar_y], color=CYAN, lw=3.2)
    ax.add_patch(Circle((44.5, bar_y), 1.1, facecolor=CORAL, edgecolor=FG, lw=1.0))
    ax.text(50.0, 64.5, "tone: tender   61%     |     x_L6 = x_L6 + 0.61 * v_tender", color=CYAN, fontsize=8.8, fontweight="bold", ha="center", family="DejaVu Sans Mono")

    # Minimal Top-K Logit & Entropy list directly below (zero heavy boxes!)
    ax.plot([12.0, 88.0], [59.0, 59.0], color=DIM, lw=0.8)
    ax.text(14.0, 55.2, "TOP-K CANDIDATES AT CURSOR (t = 142)", color=GOLD, fontsize=9.2, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(86.0, 55.2, "T = 0.80   |   H(p) = 1.42 bits", color=MUTED, fontsize=8.6, ha="right", family="DejaVu Sans Mono")

    cands = [
        ("1. liquid", 48.2, 11.40, True),
        ("2. surface", 21.6, 10.60, False),
        ("3. water", 12.1, 9.95, False),
        ("4. stone", 6.8, 9.38, False),
        ("5. glass", 4.1, 8.85, False),
        ("6. silver", 2.7, 8.42, False),
    ]
    for i, (w, p, z, chosen) in enumerate(cands):
        yc = 48.0 - i * 6.4
        col = GREEN if chosen else FG
        ax.text(14.0, yc, w, color=col, fontsize=9.5, fontweight="bold" if chosen else "normal", family="DejaVu Sans Mono", va="center")
        # Thin sleek bar
        bw = (p / 50.0) * 34.0
        ax.plot([34.0, 34.0 + bw], [yc, yc], color=GREEN if chosen else CYAN, lw=6.0, solid_capstyle="butt")
        ax.text(86.0, yc, f"{p:4.1f}%    logit z = {z:5.2f}", color=col if chosen else MUTED, fontsize=8.8, ha="right", family="DejaVu Sans Mono", va="center")

    out = OUT_DIR / "02_radial_steering_wheel_and_topk_inspector.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    return out


def render_mockup_3():
    fig, ax = new_canvas("multiverse word tree (3 x 3 x ...)   |   click any ancestor node to rewind time & fork future")

    ax.text(5.0, 92.5, "BRANCHING PROBABILITY TREE FROM CURSOR (t = 140)", color=FG, fontsize=11.0, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(5.0, 89.2, "ACTIVE PATH", color=GREEN, fontsize=8.6, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(20.0, 89.2, "ALTERNATIVE FORKS", color=CYAN, fontsize=8.6, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(44.0, 89.2, "CLICK ANY NODE TO REWIND past_kv[:t]", color=GOLD, fontsize=8.6, fontweight="bold", family="DejaVu Sans Mono")

    def node(x, y, w, h, title, sub, active=False, dim=False):
        ec = GREEN if active else (CYAN if not dim else DIM)
        fc = "#071A10" if active else BG
        tcol = "#FFFFFF" if active else (FG if not dim else MUTED)
        scol = GREEN if active else (CYAN if not dim else MUTED)
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="square,pad=0", facecolor=fc, edgecolor=ec, lw=1.6 if active else 1.0))
        ax.text(x, y + 1.0, title, color=tcol, fontsize=8.8, fontweight="bold", ha="center", va="center", family="DejaVu Sans Mono")
        ax.text(x, y - 1.4, sub, color=scol, fontsize=7.5, ha="center", va="center", family="DejaVu Sans Mono")

    def edge(x1, y1, x2, y2, active=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=GREEN if active else DIM, lw=2.0 if active else 1.0))

    rx, ry = 13.0, 46.0
    node(rx, ry, 18.0, 8.8, "...ancient mirror", "ROOT (t=140)", active=True)

    l1_x = 39.5
    for y1, w1, s1, act1 in [(71.0, "glowed", "48%  z=11.4", True), (46.0, "shattered", "27%  z=10.8", False), (21.0, "whispered", "18%  z=10.4", False)]:
        edge(rx + 9.1, ry, l1_x - 8.1, y1, active=act1)
        node(l1_x, y1, 16.0, 7.6, f'"{w1}"', s1, active=act1)

    l2_x = 67.5
    l2_groups = [
        (71.0, True, [(79.0, "softly", "52%", True), (71.0, "crimson", "29%", False), (63.0, "within", "14%", False)]),
        (46.0, False, [(54.0, "into", "44%", False), (46.0, "beneath", "31%", False), (38.0, "without", "19%", False)]),
        (21.0, False, [(29.0, "Elara's", "41%", False), (21.0, "ancient", "35%", False), (13.0, "warnings", "16%", False)]),
    ]
    for py, p_act, children in l2_groups:
        for cy, cw, cp, c_act in children:
            edge(l1_x + 8.1, py, l2_x - 7.3, cy, active=(p_act and c_act))
            node(l2_x, cy, 14.4, 5.8, f'"{cw}"', f"p={cp}", active=(p_act and c_act), dim=(not p_act))

    l3_x = 89.5
    for sy, stxt, ssub, s_act in [(79.0, "in blue...", "t+3", True), (71.0, "with fire...", "t+3", False), (46.0, "the floor...", "t+3", False), (21.0, "runes...", "t+3", False)]:
        edge(l2_x + 7.3, sy, l3_x - 6.1, sy, active=s_act)
        node(l3_x, sy, 12.0, 5.0, stxt, ssub, active=s_act, dim=(not s_act))

    out = OUT_DIR / "03_multiverse_3x3_branching_word_tree.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    return out


def render_mockup_4():
    fig, ax = new_canvas("wheel -> click multiple words: draw attention tethers   |   att[..., pinned_cols] += boost before Softmax")

    lines = [
        ("01", "There was an old legend about a forest that never slept,", 0.04, None),
        ("02", "where the trees whispered secrets to anyone brave enough", 0.06, None),
        ("03", "to listen. One night, under a sky brushed with clouds,", 0.05, None),
        ("04", "a young wanderer named Elara stepped through the tangled", 0.34, ORANGE),
        ("05", "undergrowth, guided only by the faint hum of an ancient", 0.31, ORANGE),
        ("06", "lullaby carried on the wind. As she pushed forward, the", 0.12, None),
        ("07", "forest grew quieter, its mossy floor pulsing with soft", 0.09, None),
        ("08", "bioluminescent glow. She paused beside a silver pool", 0.29, CYAN),
        ("09", "where the water's surface rippled with a soft, rhythmic", 0.14, None),
    ]
    draw_pixel_rail(ax, lines, y_top=90.5, dy=4.5, steer_rows=(4, 5))

    y_top, dy = 90.5, 4.5
    for idx, (ln, txt, attn, hl_col) in enumerate(lines):
        y = y_top - idx * dy
        ax.text(20.0, y, txt, color=FG, fontsize=9.1, family="DejaVu Sans Mono", va="center")

    # Cursor position below line 09
    cx, cy = 48.0, y_top - 9.0 * dy
    ax.plot([18.5, 76.0], [cy, cy], color=MUTED, lw=1.0, ls=(0, (3, 4)))
    ax.add_patch(FancyBboxPatch((cx - 0.7, cy - 0.7), 1.4, 1.4, boxstyle="square,pad=0", facecolor=CORAL, edgecolor="none"))

    # Highlighted pinned words + curved glowing tether lines straight to the cursor!
    # Monospace width at fontsize 9.1 is ~0.77 x-units per char from x=20.0
    pinned = [
        (20.0 + 25.5 * 0.77, y_top - 3 * dy, "Elara", ORANGE, "+3.0", 0.20),
        (20.0 + 50.0 * 0.77, y_top - 4 * dy, "ancient", GOLD, "+2.6", -0.22),
        (20.0 + 46.5 * 0.77, y_top - 7 * dy, "silver pool", CYAN, "+3.2", -0.18),
    ]
    for wx, wy, wtxt, wcol, boost, rad in pinned:
        ax.plot([wx - 2.5, wx + 2.5], [wy - 1.2, wy - 1.2], color=wcol, lw=1.8)
        ax.add_patch(Circle((wx, wy - 1.2), 0.55, facecolor=wcol, edgecolor="#FFFFFF", lw=0.7))
        ax.add_patch(FancyArrowPatch(
            (wx, wy - 1.4), (cx, cy + 0.8),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=10,
            color=wcol, lw=1.6, ls="--", alpha=0.9,
        ))
        ax.text(wx + 3.2, wy - 1.2, f"[{boost}]", color=wcol, fontsize=7.8, fontweight="bold", family="DejaVu Sans Mono", va="center")

    # Generated steered output below the cursor weaving ALL pinned words together!
    ax.text(20.0, cy - 5.2, "STEERED CONTINUATION (emphasizing pinned words):", color=MUTED, fontsize=8.0, family="DejaVu Sans Mono")
    out_lines = [
        '->  Elara leaned over the silver pool, hearing the ancient',
        '    melody echo back from beneath the dark water...',
    ]
    ax.text(20.0, cy - 9.5, out_lines[0], color=GREEN, fontsize=9.2, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(20.0, cy - 13.8, out_lines[1], color=GREEN, fontsize=9.2, fontweight="bold", family="DejaVu Sans Mono")

    # Right-side pinned word list
    ax.text(78.5, 68.0, "PINNED TETHERS (3)", color=GOLD, fontsize=8.8, fontweight="bold", family="DejaVu Sans Mono")
    for i, (_, _, wtxt, wcol, boost, _) in enumerate(pinned):
        ax.text(78.5, 63.0 - i * 4.5, f"● {wtxt:<12s} {boost}", color=wcol, fontsize=8.5, fontweight="bold", family="DejaVu Sans Mono")

    out = OUT_DIR / "04_multi_word_attention_tether.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = render_mockup_1()
    p2 = render_mockup_2()
    p3 = render_mockup_3()
    p4 = render_mockup_4()
    print("Rendered:", p1.name, p2.name, p3.name, p4.name)
