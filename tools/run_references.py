"""Score the engine against every reference groove: run_references.py [name-filter]
Transcribes each song (all caches reused) and prints the check_groove summary."""
import json, sys, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from backend.pipeline.run import Options, transcribe
refs=json.load(open(ROOT/"tools/references.json"))
flt=sys.argv[1].lower() if len(sys.argv)>1 else ""
out=Path("/tmp/drummate-refs"); out.mkdir(exist_ok=True)
for r in refs:
    if flt and flt not in r["name"].lower(): continue
    d=out/r["name"].replace(" ","_").replace("/","-")
    doc=transcribe(r["url"], d, ROOT/"data/cache", Options(render_audio=False))
    res=subprocess.run([sys.executable, str(ROOT/"tools/check_groove.py"), str(d/"score.json"), r["pattern"]],
                       capture_output=True, text=True).stdout
    print("="*70); print(f"{r['name']}  (published: {r['bpm']} BPM)"); print(res)
