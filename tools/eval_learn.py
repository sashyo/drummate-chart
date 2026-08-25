"""Re-run grid + quantiser from saved detections for every learn song and total the
agreement with the published tabs: eval_learn.py DATA_LEARN_DIR [--jobs N]
Seconds per song, so engine changes can be measured across the whole set."""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
d = Path(sys.argv[1]); jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
songs = sorted(p for p in d.iterdir() if (p / "_analysis.pkl").exists() and (p / "tab.json").exists())
def one(p):
    subprocess.run([sys.executable, str(ROOT / "tools/requant.py"), str(p)], capture_output=True, text=True)
    cmp = subprocess.run([sys.executable, str(ROOT / "tools/compare_tab.py"), str(p / "score.json"), str(p / "tab.json"), "--show", "0"], capture_output=True, text=True).stdout
    m = re.search(r"exactly matching the tab: (\d+)/(\d+)", cmp)
    ag = {c: (int(a), int(b)) for c, a, b in re.findall(r"^\s+([KSxT]): (\d+)/(\d+)", cmp, re.M)}
    return p.name, (int(m.group(1)), int(m.group(2))) if m else (0, 0), ag
with ThreadPoolExecutor(jobs) as ex:
    res = list(ex.map(one, songs))
tot = {"exact": [0, 0], "K": [0, 0], "S": [0, 0], "x": [0, 0], "T": [0, 0]}
for name, ex, ag in res:
    tot["exact"][0] += ex[0]; tot["exact"][1] += ex[1]
    for c in "KSxT":
        if c in ag: tot[c][0] += ag[c][0]; tot[c][1] += ag[c][1]
print(f"{len(res)} songs | " + " | ".join(f"{k} {v[0]}/{v[1]} ({100*v[0]/max(1,v[1]):.1f}%)" for k, v in tot.items()))
if "--per-song" in sys.argv:
    for name, ex, ag in sorted(res, key=lambda r: r[1][0] / max(1, r[1][1])):
        print(f"  {name[:36]:36s} exact {ex[0]:3d}/{ex[1]:3d}  K {ag.get('K',(0,0))[0]:3d} S {ag.get('S',(0,0))[0]:3d}")
