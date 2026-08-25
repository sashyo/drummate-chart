"""Aggregate a learn_songs batch: learn_report.py DATA_LEARN_DIR [--out report.md]

Per song: exact bars, kick/snare/hat/tom agreement, and alignment quality (the
best 8-bar block's kick+snare agreement - low means the grid, tempo or bar
line is wrong, not the notes). Across songs: the most common deviation
clusters per class, so fixes are chosen by how many songs they touch."""
import json, re, subprocess, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import compare_tab as ct  # noqa: E402

def block_quality(doc, tab_bars):
    chart = ct.chart_bars(doc); tab = [{c: frozenset(v) for c, v in b.items()} for b in tab_bars]
    best = 0.0
    for off in range(-12, 13):
        for b0 in range(0, len(tab), 8):
            n = ok = 0
            for i in range(b0, min(b0 + 8, len(tab))):
                j = i + off
                if 0 <= j < len(chart):
                    n += 1; ok += all(tab[i].get(c, frozenset()) == frozenset(chart[j].get(c, ())) for c in ("K", "S"))
            if n >= 6: best = max(best, ok / n)
    return best

def main():
    d = Path(sys.argv[1]); out = None
    if "--out" in sys.argv: out = Path(sys.argv[sys.argv.index("--out") + 1])
    rows = []; devs = collections.Counter(); dev_songs = collections.defaultdict(set)
    for song in sorted(p for p in d.iterdir() if (p / "score.json").exists() and (p / "tab.json").exists()):
        doc = json.loads((song / "score.json").read_text()); tab = json.loads((song / "tab.json").read_text())["bars"]
        cmp = subprocess.run([sys.executable, str(ROOT / "tools/compare_tab.py"), str(song / "score.json"), str(song / "tab.json"), "--show", "0"],
                             capture_output=True, text=True).stdout
        m = re.search(r"exactly matching the tab: (\d+)/(\d+)", cmp)
        ag = dict(re.findall(r"^\s+([KSxT]): (\d+)/(\d+)", cmp, re.M) and [(c, f"{a}/{b}") for c, a, b in re.findall(r"^\s+([KSxT]): (\d+)/(\d+)", cmp, re.M)])
        for c, kind, slots, k in re.findall(r"^\s+([KSxT]) (extra|missing) at \[([^\]]*)\]\s+x(\d+)", cmp, re.M):
            key = (c, kind, slots); devs[key] += int(k); dev_songs[key].add(song.name)
        q = block_quality(doc, tab)
        rows.append((song.name, int(m.group(1)) if m else 0, int(m.group(2)) if m else 0, ag, q, doc.get("tempo")))
    rows.sort(key=lambda r: (r[1] / max(1, r[2])))
    lines = ["# Learn batch report", "", f"{len(rows)} songs", "", "| song | exact | K | S | x | T | best 8-bar block (K+S) | tempo |", "|---|---|---|---|---|---|---|---|"]
    for name, ex, of, ag, q, tempo in rows:
        lines.append(f"| {name[:40]} | {ex}/{of} | {ag.get('K','-')} | {ag.get('S','-')} | {ag.get('x','-')} | {ag.get('T','-')} | {q:.2f} | {tempo:.0f} |")
    bad_grid = [r for r in rows if r[4] < 0.5]
    lines += ["", f"**Grid/meter suspects** (no 8-bar block reaches 50 % kick+snare agreement): {len(bad_grid)} songs — " + ", ".join(r[0][:30] for r in bad_grid), ""]
    lines += ["## Deviation clusters across songs (count, songs)", ""]
    for (c, kind, slots), k in devs.most_common(25):
        lines.append(f"- `{c} {kind} at [{slots}]` — {k} bars in {len(dev_songs[(c, kind, slots)])} songs")
    text = "\n".join(lines)
    print(text if not out else f"wrote {out}")
    if out: out.write_text(text)

if __name__ == "__main__":
    main()
