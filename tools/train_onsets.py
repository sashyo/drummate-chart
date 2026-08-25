"""Learn kick / snare / tom decisions per detected onset from published charts.

  train_onsets.py DIR [DIR ...] [--min-agree 0.4] [--out model.pkl]

Each DIR holds a song: _analysis.pkl (raw detection), score.json (the
engine's chart: bar times), tab.json (published bars from songsterr_tab).
For every raw hit we build features from all six kit stems around the hit
and label it with what the published chart has at that bar/slot. Songs
whose chart disagrees with the tab on kick+snare in more than
(1 - min_agree) of bars are skipped: their bars are misaligned and the
labels would be noise. Leave-one-song-out cross-validation reports
precision / recall per class against the labels, next to what the current
rules produce, so a model only ships if it beats them.
"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import compare_tab as ct  # noqa: E402

SR = 44100
STEMS = ["kick", "snare", "toms", "hh", "ride", "crash"]
INSTS = ["kick", "snare", "tom_low", "tom_mid", "tom_hi", "hihat", "openhh", "ride", "crash", "perc"]
TARGETS = ["K", "S", "T"]


def stems_for(duration: float):
    n = int(round(duration * SR))
    c = []
    for f in glob.glob(str(ROOT / "data/research-cache/drumsep_*.npz")):
        try:
            if abs(np.load(f)["kick"].shape[0] - n) < SR:
                c.append(f)
        except Exception:  # noqa: BLE001 - a stem file the batch is still writing
            continue
    if not c:
        return None
    z = np.load(sorted(c, key=os.path.getmtime)[-1])
    return {k: z[k].astype(np.float32) for k in STEMS}


def env_peak(x: np.ndarray, t: float, pre=0.01, post=0.05) -> float:
    i = int(t * SR)
    seg = x[max(0, i - int(pre * SR)): i + int(post * SR)]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-7) if len(seg) else 1e-7


def align(doc: dict, tab_bars: list) -> tuple[int, float]:
    """Bar offset (tab bar i <-> chart bar i+off) by kick+snare agreement."""
    chart = ct.chart_bars(doc)
    tab = [{c: frozenset(v) for c, v in b.items()} for b in tab_bars]
    best, best_ok, best_blocks = 0, -1, 0.0
    for off in range(-12, 13):
        n = ok = 0; blocks = []
        for b0 in range(0, len(tab), 8):
            bn = bok = 0
            for i in range(b0, min(b0 + 8, len(tab))):
                j = i + off
                if 0 <= j < len(chart):
                    bn += 1
                    bok += all(tab[i].get(c, frozenset()) == frozenset(chart[j].get(c, ())) for c in ("K", "S"))
            n += bn; ok += bok
            if bn >= 6:
                blocks.append(bok / bn)
        if n and ok > best_ok:
            best, best_ok, best_n = off, ok, n
            # alignment is proven by SOME section matching nearly bar for bar,
            # even when other sections disagree (those are the bars to learn from)
            best_blocks = max(blocks) if blocks else 0.0
    return best, best_blocks


def song_rows(d: Path, min_agree: float):
    a = pickle.loads((d / "_analysis.pkl").read_bytes())
    doc = json.loads((d / "score.json").read_text())
    tab_bars = json.loads((d / "tab.json").read_text())["bars"]
    off, agree = align(doc, tab_bars)
    if agree < min_agree:                 # best 8-bar block must match this well
        return None, agree
    stems = stems_for(a["duration"])
    if stems is None:
        return None, agree
    bars = doc["bars"]
    starts = np.array([b["startTime"] for b in bars]); ends = np.array([b["endTime"] for b in bars])
    hits = sorted(a["hits"], key=lambda h: h.time)
    times = np.array([h.time for h in hits])
    # per-song normalisation: log energy relative to the stem's 95th percentile at kick/snare hits
    ref = {k: np.percentile([env_peak(stems[k], h.time) for h in hits[::max(1, len(hits) // 300)]], 95) + 1e-7 for k in STEMS}
    X, Y, R, meta = [], [], [], []
    chart_bars = ct.chart_bars(doc)
    for h in hits:
        j = int(np.searchsorted(starts, h.time, side="right") - 1)
        if j < 0 or j >= len(bars) or h.time >= ends[j]:
            continue
        tab_i = j - off
        if not (0 <= tab_i < len(tab_bars)):
            continue
        L = (ends[j] - starts[j]) / 16.0
        slot = int(round((h.time - starts[j]) / L))
        if slot > 15:
            continue
        tb = tab_bars[tab_i]
        y = [1 if slot in tb.get(c, []) else 0 for c in TARGETS]
        # the rules' answer for the same slot (what the chart currently says)
        cb = chart_bars[j]
        r = [1 if slot in cb.get(c, ()) else 0 for c in TARGETS]
        e = {k: env_peak(stems[k], h.time) for k in STEMS}
        f = [np.log(e[k] / ref[k]) for k in STEMS]
        f += [np.log(e["toms"] / e["kick"]), np.log(e["hh"] / e["snare"]), np.log(e["snare"] / e["kick"]),
              np.log(e["crash"] / e["hh"]), np.log(e["ride"] / e["hh"])]
        f += [1.0 if h.inst == i else 0.0 for i in INSTS]
        f += [h.velocity, getattr(h, "confidence", 1.0), 1.0 if h.ghost else 0.0, 1.0 if h.accent else 0.0]
        f += [1.0 if slot % 4 == 0 else 0.0, 1.0 if slot % 2 == 0 else 0.0, slot / 15.0]
        near = times[(times > h.time - 0.03) & (times < h.time + 0.03)]
        co = {hh.inst for hh in hits if abs(hh.time - h.time) < 0.03 and hh is not h}
        f += [len(near) - 1, 1.0 if "kick" in co else 0.0, 1.0 if "snare" in co else 0.0,
              1.0 if any(c.startswith("tom") for c in co) else 0.0, 1.0 if co & {"hihat", "openhh", "ride", "crash"} else 0.0]
        # context the rules already use: their own decision at this slot, and
        # how often the neighbouring bars carry each class at this slot - the
        # model learns when to overrule, not the groove from scratch
        f += r
        for c in TARGETS:
            f.append(sum(1 for jj in range(max(0, j - 4), min(len(chart_bars), j + 5)) if jj != j and slot in chart_bars[jj].get(c, ())) / 8.0)
        X.append(f); Y.append(y); R.append(r); meta.append((str(d.name), j, slot, h.inst))
    return (np.array(X, dtype=np.float32), np.array(Y), np.array(R), meta), agree


def main():
    args = sys.argv[1:]
    min_agree = 0.75; out = None
    if "--min-agree" in args:
        i = args.index("--min-agree"); min_agree = float(args[i + 1]); del args[i:i + 2]
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
    dirs = []
    for a in args:
        p = Path(a)
        dirs += [d for d in ([p] if (p / "tab.json").exists() else sorted(p.iterdir()))
                 if (d / "tab.json").exists() and (d / "_analysis.pkl").exists() and (d / "score.json").exists()]
    songs = []
    for d in dirs:
        try:
            rows, agree = song_rows(d, min_agree)
        except Exception as exc:  # noqa: BLE001
            print(f"  {d.name}: skipped ({type(exc).__name__}: {exc})"); continue
        if rows is None:
            print(f"  {d.name}: skipped (kick+snare agreement {agree:.2f} < {min_agree}, or no stems)"); continue
        print(f"  {d.name}: {len(rows[0])} hits, agreement {agree:.2f}")
        songs.append((d.name, rows))
    if len(songs) < 3:
        sys.exit("need at least 3 usable songs")
    from sklearn.ensemble import HistGradientBoostingClassifier
    names = [s for s, _ in songs]
    print(f"\n{len(songs)} songs, {sum(len(r[0]) for _, r in songs)} hits. Leave-one-song-out:")
    totals = {c: {"tp": 0, "fp": 0, "fn": 0, "rtp": 0, "rfp": 0, "rfn": 0} for c in TARGETS}
    for k, (held, (Xh, Yh, Rh, _)) in enumerate(songs):
        Xtr = np.concatenate([r[0] for s, r in songs if s != held]); Ytr = np.concatenate([r[1] for s, r in songs if s != held])
        for ci, c in enumerate(TARGETS):
            if Ytr[:, ci].sum() < 20 or Yh[:, ci].sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.5)
            clf.fit(Xtr, Ytr[:, ci])
            p = clf.predict(Xh)
            t = totals[c]
            t["tp"] += int(((p == 1) & (Yh[:, ci] == 1)).sum()); t["fp"] += int(((p == 1) & (Yh[:, ci] == 0)).sum()); t["fn"] += int(((p == 0) & (Yh[:, ci] == 1)).sum())
            t["rtp"] += int(((Rh[:, ci] == 1) & (Yh[:, ci] == 1)).sum()); t["rfp"] += int(((Rh[:, ci] == 1) & (Yh[:, ci] == 0)).sum()); t["rfn"] += int(((Rh[:, ci] == 0) & (Yh[:, ci] == 1)).sum())
    def prf(tp, fp, fn):
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); return p, r, (2 * p * r / max(1e-9, p + r))
    print(f"{'class':6s} {'model P/R/F1':>22s} {'rules P/R/F1':>22s}")
    for c in TARGETS:
        t = totals[c]
        mp, mr, mf = prf(t["tp"], t["fp"], t["fn"]); rp, rr, rf = prf(t["rtp"], t["rfp"], t["rfn"])
        print(f"{c:6s} {mp:6.2f} {mr:6.2f} {mf:6.2f}     {rp:6.2f} {rr:6.2f} {rf:6.2f}")
    if out:
        X = np.concatenate([r[0] for _, r in songs]); Y = np.concatenate([r[1] for _, r in songs])
        models = {}
        for ci, c in enumerate(TARGETS):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.5)
            clf.fit(X, Y[:, ci]); models[c] = clf
        pickle.dump({"models": models, "targets": TARGETS, "insts": INSTS, "stems": STEMS, "songs": names}, open(out, "wb"))
        print("saved", out)


if __name__ == "__main__":
    main()
