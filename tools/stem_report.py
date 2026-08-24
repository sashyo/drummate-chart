"""Per-stem diagnosis for a job: stem_report.py <drums.wav|mp3> [bars]
Shows each DrumSep stem's level, how many onsets the detector keeps vs what
a plain normalised picker would find, and the unexplained residual."""
import sys, glob, numpy as np, librosa
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pipeline import onsets as O
y,_=librosa.load(sys.argv[1], sr=44100, mono=True)
bars=float(sys.argv[2]) if len(sys.argv)>2 else None
z=None
for f in sorted(glob.glob("data/cache/drumsep_*.npz"), key=lambda p:-Path(p).stat().st_mtime):
    zz=np.load(f)
    if abs(len(zz['kick'])-len(y))<44100: z=zz; break
if z is None: sys.exit("no cached DrumSep stems match this audio")
n=min(len(y),*(len(z[k]) for k in z.files))
ref=max(float(np.percentile(np.abs(z[k][:n]),99.9)) for k in z.files)
rms=lambda a: 20*np.log10(np.sqrt(np.mean(a**2))+1e-9)
print(f"{'stem':6s} {'rms dB':>7s} {'kept':>5s} {'raw':>5s}  {'per bar' if bars else ''}")
gates={'kick':(0.3,0.045,0.08,0.04),'snare':(0.3,0.04,0.08,0.04),'hh':(0.24,0.03,0.12,0.045),
       'ride':(0.42,0.08,0.25,0.10),'crash':(0.42,0.08,0.25,0.10),'toms':(0.36,0.05,0.2,0.10)}
for k in ('kick','snare','hh','ride','crash','toms'):
    st=z[k][:n]; d,w,fl,g=gates[k]
    kept,_=O._stem_onsets(st,d,w,floor=fl,abs_ref=ref,abs_gate=g)
    raw,_=O._stem_onsets(st,d,w,floor=0.0)
    per=f"{len(kept)/bars:.1f}" if bars else ""
    print(f"{k:6s} {rms(st):7.1f} {len(kept):5d} {len(raw):5d}  {per}")
resid=y[:n].copy()
for k in z.files: resid-=z[k][:n]
print(f"residual {rms(resid):.1f} dB (input {rms(y[:n]):.1f} dB)")
