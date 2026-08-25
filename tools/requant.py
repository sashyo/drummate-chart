"""Re-run grid + quantiser without separating again.

  requant.py OUT_DIR                         # uses OUT_DIR/_analysis.pkl
  requant.py OUT_DIR --npy drums.npy --npz drumsep.npz [--bpb 4]
                                             # re-detect from cached stems
Writes OUT_DIR/score.json; pass --pattern "..." to run check_groove too."""
import argparse, json, pickle, subprocess, sys, time
T0=time.time()
def lap(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.pipeline import onsets, quantize, rhythm, score, separate

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--npy"); ap.add_argument("--npz")
ap.add_argument("--bpb", type=int, default=4); ap.add_argument("--pattern")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

if a.npy:
    mono = np.load(a.npy).astype(np.float32)
    stems = {k: v for k, v in np.load(a.npz).items()}
    lap("stems loaded"); tracked = rhythm.analyse(mono, mix=mono, beats_per_bar=a.bpb); lap("tracked grid")
    det = onsets.detect_from_stems(stems, sensitivity=1.0, detect_toms=True,
                                   cymbal_detail=1.0, mono=mono); lap("onsets")
    bpb, fixed = a.bpb, None
else:
    d = pickle.loads((out / "_analysis.pkl").read_bytes())
    det = onsets.Detection(d["hits"], d["duration"], d["debug"])
    tracked, bpb, fixed = d["tracked"], d["beats_per_bar"], d["fixed_tempo"]

lap(f"duration {det.duration:.1f}s, {len(det.hits)} hits")
dg = rhythm.grid_from_drums(det.hits, det.duration, bpb, tempo_hint=tracked.tempo, fixed_tempo=fixed)
grid = rhythm.refine_with_hits(dg or tracked, det.hits); lap("drum grid")
q = quantize.quantize(det.hits, grid); lap("quantised")
doc = score.build(q, {"title": out.name, "source": "", "videoId": "", "separation": "replay",
                      "detector": det.debug.get("detector", "spectral"), "engine": 0, "offset": 0.0,
                      "audio": {}, "stats": {"hits": len(det.hits), "bars": len(q.bars),
                                             "duration": round(det.duration, 2)}})
(out / "score.json").write_text(json.dumps(doc, indent=1))
print(f"{grid.tempo:.1f} BPM, {len(q.bars)} bars -> {out/'score.json'}")
if a.pattern:
    print(subprocess.run([sys.executable, str(ROOT / "tools/check_groove.py"), str(out / "score.json"), a.pattern],
                         capture_output=True, text=True).stdout)
