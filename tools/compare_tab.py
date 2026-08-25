"""Bar-by-bar comparison of a generated chart with an ASCII drum tab.

  compare_tab.py SCORE_JSON TAB_TXT [--offset N] [--show 30]

The tab is parsed into bars (each |...| cell; 16 chars = 16ths, 8 = 8ths,
32 = 32nds). Line labels map to classes: hats/ride/crash -> x, snare -> S,
bass -> K, toms -> T. Any character other than '-' or '|' in a cell is a
stroke. The chart is aligned to the tab by the bar offset that maximises
exact matches (searched over --offset if given, else -12..12)."""
import json, re, sys, collections
from pathlib import Path
CLS = {"HH": "x", "HH2": "x", "H": "x", "HH1": "x", "X": "x", "R": "x", "RD": "x", "RIDE": "x", "C": "x", "CC": "x", "CR": "x", "CRASH": "x", "C1": "x", "C2": "x", "SP": "x", "CH": "x",
       "S": "S", "SD": "S", "SN": "S", "B": "K", "BD": "K", "K": "K", "KD": "K", "B1": "K", "B2": "K",
       "T": "T", "T1": "T", "T2": "T", "T3": "T", "FT": "T", "F": "T", "HT": "T", "MT": "T", "LT": "T", "TT": "T"}
MAP = {'kick': 'K', 'snare': 'S', 'hihat': 'x', 'openhh': 'x', 'ride': 'x', 'crash': 'x',
       'tom_low': 'T', 'tom_mid': 'T', 'tom_hi': 'T'}

def parse_tab(text: str):
    """-> list of bars; bar = {cls: set(slot)} at 16th resolution."""
    lines = [l.rstrip("\n") for l in text.splitlines()]
    groups, cur = [], []
    for l in lines:
        m = re.match(r"^\s*([A-Za-z]{1,5}[0-9]?)\s*[:|]?\s*\|(.*)$", l)
        if m and m.group(1).upper() in CLS:
            cur.append((CLS[m.group(1).upper()], "|" + m.group(2)))
        else:
            if cur:
                groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    bars = []
    for g in groups:
        cells = {}
        present = {cls for cls, _ in g}                 # classes this block notates
        n = None
        for cls, row in g:
            parts = [p for p in row.split("|")[1:] if p.strip()]
            if n is None:
                n = len(parts)
            for i, p in enumerate(parts[:n]):
                w = len(p)
                if w < 4:
                    continue
                res = 16 / w
                for j, ch in enumerate(p):
                    if ch not in "- ":
                        cells.setdefault(i, {}).setdefault(cls, set()).add(int(round(j * res)) % 16)
        for i in range(n or 0):
            b = cells.get(i, {})
            b["_present"] = present
            bars.append(b)
    return bars

def chart_bars(doc):
    s16 = doc["ppq"] // 4
    out = []
    for b in doc["bars"]:
        d = {}
        for h in b["hits"]:
            c = MAP.get(h["inst"])
            if c and h["tick"] // s16 < 16:
                d.setdefault(c, set()).add(h["tick"] // s16)
        out.append(d)
    return out

def norm(b, classes):
    return {c: frozenset(b.get(c, ())) for c in classes}

def main():
    args = sys.argv[1:]
    show = 30; off_fixed = None
    if "--show" in args:
        i = args.index("--show"); show = int(args[i + 1]); del args[i:i + 2]
    if "--offset" in args:
        i = args.index("--offset"); off_fixed = int(args[i + 1]); del args[i:i + 2]
    doc = json.load(open(args[0])); tab = parse_tab(Path(args[1]).read_text(errors="ignore"))
    chart = chart_bars(doc)
    classes = sorted({c for b in tab for c in b if c != "_present"} & {"K", "S", "x", "T"}) or ["K", "S", "x"]
    tabN = [norm(b, classes) for b in tab]; chN = [norm(b, classes) for b in chart]
    pres = [b.get("_present", set(classes)) for b in tab]
    def same(i, j):
        return all(tabN[i][c] == chN[j][c] for c in classes if c in pres[i])
    def score(off):
        n = ok = 0
        for i in range(len(tabN)):
            j = i + off
            if 0 <= j < len(chN):
                n += 1; ok += same(i, j)
        return ok, n
    offs = [off_fixed] if off_fixed is not None else range(-12, 13)
    best = max(offs, key=lambda o: score(o)[0])
    ok, n = score(best)
    print(f"{doc['title'][:50]} | chart {len(chart)} bars, tab {len(tab)} bars, classes {classes}, offset {best:+d} (tab bar 0 = chart bar {best})")
    print(f"bars exactly matching the tab: {ok}/{n}")
    agree = collections.Counter(); dev = collections.Counter(); diffs = []
    for i, tb in enumerate(tabN):
        j = i + best
        if not (0 <= j < len(chN)):
            continue
        cb = chN[j]; bad = []
        for c in classes:
            if c not in pres[i]:
                continue                                   # the tab does not notate this class here
            if tb[c] == cb[c]:
                agree[c] += 1
            else:
                ex = sorted(cb[c] - tb[c]); mi = sorted(tb[c] - cb[c])
                if ex: dev[(c, "extra", tuple(ex))] += 1
                if mi: dev[(c, "missing", tuple(mi))] += 1
                bad.append(f"{c}: extra {ex} missing {mi}")
        if bad:
            diffs.append((i + 1, doc["bars"][j]["number"], round(doc["bars"][j]["startTime"], 1), "; ".join(bad)))
    for c in classes:
        nc = sum(1 for i in range(len(tabN)) if c in pres[i] and 0 <= i + best < len(chN))
        print(f"  {c}: {agree[c]}/{nc} bars agree (tab notates this class in {nc} bars)")
    print("most common deviations (16th slots from beat 1):")
    for (c, kind, sl), k in dev.most_common(10):
        print(f"   {c} {kind} at {list(sl)}  x{k}")
    print(f"first {min(show, len(diffs))} differing bars (tab bar, chart bar, time):")
    for t in diffs[:show]:
        print("   tab %3d / chart %3d @%6.1fs  %s" % t)

if __name__ == "__main__":
    main()
