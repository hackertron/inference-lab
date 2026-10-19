#!/usr/bin/env python3
"""Plot a concurrency sweep as an SVG latency-vs-throughput chart (no deps).

    python3 scripts/plot_sweep.py                      # journal 04 defaults
    python3 scripts/plot_sweep.py --prefix 04-sweep-c --out docs/journal/img/04-sweep.svg

Reads results/<prefix><c>.json for every concurrency found. Two panels share
the x-axis (output throughput): TPOT on the left, mean TTFT on the right.
"""
import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Written down in journal 04 before the run (midpoints of the predicted ranges).
PREDICTED = {  # concurrency: (output tok/s, TPOT ms, TTFT ms)
    1: (95, 10, 45),
    2: (185, 10.5, 60),
    4: (350, 11, 100),
    8: (650, 12, 200),
    16: (1100, 14, 450),
}
ROOFLINE_TPOT_MS = 7.7  # 3.46 GB weights / 448 GB/s

W, H = 880, 420
PANEL_W, PANEL_H = 360, 250
TOP, LEFT, GAP = 104, 64, 80

STYLE = """
  .root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984;
          --grid:#e7e6e2; --s1:#2a78d6; }
  @media (prefers-color-scheme: dark) {
    .root { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#8f8e86;
            --grid:#383835; --s1:#3987e5; }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); font-size: 12px; }
  .title { fill: var(--ink); font-size: 16px; font-weight: 600; }
  .ptitle { fill: var(--ink); font-size: 13px; font-weight: 600; }
  .tick { fill: var(--muted); font-size: 11px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--muted); stroke-width: 1; }
  .meas { stroke: var(--s1); stroke-width: 2; fill: none; }
  .measpt { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
  .pred { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 4; fill: none; }
  .predpt { fill: var(--surface); stroke: var(--muted); stroke-width: 1.5; }
  .ref { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 2 3; }
  .ptlabel { fill: var(--ink); stroke: var(--surface); stroke-width: 4px; paint-order: stroke; stroke-linejoin: round; }
"""


def load(prefix):
    rows = {}
    for path in glob.glob(os.path.join(ROOT, "results", f"{prefix}*.json")):
        m = re.search(rf"{re.escape(prefix)}(\d+)\.json$", path)
        if m:
            d = json.load(open(path))
            rows[int(m.group(1))] = (d["output_throughput"], d["mean_tpot_ms"], d["mean_ttft_ms"])
    return dict(sorted(rows.items()))


def nice_max(v, step, floor):
    """Round v up to a multiple of step, never below floor."""
    return max(floor, -(-v // step) * step)


def panel(x0, title, ymax, yticks, measured, predicted, yidx, unit, xmax, ref=None):
    sx = lambda v: x0 + v / xmax * PANEL_W
    sy = lambda v: TOP + PANEL_H - v / ymax * PANEL_H
    out = [f'<text class="ptitle" x="{x0}" y="{TOP - 16}">{title}</text>']
    for t in yticks:
        out.append(f'<line class="grid" x1="{x0}" x2="{x0 + PANEL_W}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in range(0, xmax + 1, xmax // 4):
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{TOP + PANEL_H + 18}" text-anchor="middle">{t:,}</text>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x0 + PANEL_W}" y1="{sy(0)}" y2="{sy(0)}"/>')
    out.append(f'<text x="{x0 + PANEL_W / 2}" y="{TOP + PANEL_H + 40}" text-anchor="middle">output throughput (tok/s)</text>')
    if ref is not None:
        val, label = ref
        out.append(f'<line class="ref" x1="{x0}" x2="{x0 + PANEL_W}" y1="{sy(val):.1f}" y2="{sy(val):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 + PANEL_W}" y="{sy(val) + 14:.1f}" text-anchor="end">{label}</text>')

    def series(points, line_cls, pt_cls, r, label_pts):
        pts = [(sx(v[0]), sy(v[yidx])) for v in points.values()]
        out.append(f'<polyline class="{line_cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for (c, v), (x, y) in zip(points.items(), pts):
            tip = f"c={c}: {v[0]:,.0f} tok/s, {v[yidx]:.1f} {unit}"
            out.append(f'<circle class="{pt_cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"><title>{tip}</title></circle>')
            if label_pts:
                out.append(f'<text x="{x - 8:.1f}" y="{y - 10:.1f}" text-anchor="end" class="ptlabel">c={c}</text>')

    series(predicted, "pred", "predpt", 3.5, False)
    series(measured, "meas", "measpt", 5, True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="04-sweep-c")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "journal", "img", "04-sweep.svg"))
    args = ap.parse_args()

    measured = load(args.prefix)
    predicted = {c: PREDICTED[c] for c in measured if c in PREDICTED}

    body = [
        f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
        f'<text class="title" x="{LEFT - 40}" y="30">Qwen3-1.7B on an RTX 3060 Ti: batching more requests</text>',
        f'<circle class="measpt" cx="{LEFT - 32}" cy="50" r="5"/><text x="{LEFT - 22}" y="54">measured</text>',
        f'<line class="pred" x1="{LEFT + 50}" x2="{LEFT + 74}" y1="50" y2="50"/>'
        f'<circle class="predpt" cx="{LEFT + 62}" cy="50" r="3.5"/><text x="{LEFT + 82}" y="54">predicted</text>',
        f'<text x="{LEFT + 160}" y="54">c = concurrent requests</text>',
    ]
    allpts = list(measured.values()) + list(predicted.values())
    xmax = int(nice_max(max(p[0] for p in allpts), 400, 1200))
    tpot_max = nice_max(max(p[1] for p in allpts), 4, 16)
    ttft_max = nice_max(max(p[2] for p in allpts), 200, 600)
    ticks = lambda m: [m * i / 4 for i in range(5)]
    body += panel(LEFT, "Time per output token (ms)", tpot_max, ticks(tpot_max), measured, predicted, 1, "ms",
                  xmax, ref=(ROOFLINE_TPOT_MS, "bandwidth floor 7.7 ms"))
    body += panel(LEFT + PANEL_W + GAP, "Mean time to first token (ms)", ttft_max, ticks(ttft_max),
                  measured, predicted, 2, "ms", xmax)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="root" '
           f'role="img" aria-label="Latency versus throughput for concurrency 1 to 16">\n'
           f'<style>{STYLE}</style>\n' + "\n".join(body) + "\n</svg>\n")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(svg)
    print(f"wrote {os.path.relpath(args.out, ROOT)} ({len(measured)} points)")


if __name__ == "__main__":
    main()
