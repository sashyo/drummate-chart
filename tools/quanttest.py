"""Quantiser-level fixture: mixed 8ths/16ths, jitter, pickups, triplets.

Feeds controlled hit lists straight into the quantiser (no audio, no
detector) so the subdivision chooser is tested in isolation.
"""
import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from backend.pipeline.onsets import Hit
from backend.pipeline.rhythm import BeatGrid
from backend.pipeline import quantize as Q

BPM=100.0; T=60/BPM; BPB=4
rng=np.random.default_rng(3)

def grid(n_bars):
    bt=np.arange(0,(n_bars*BPB+2))*T
    return BeatGrid(tempo=BPM, beat_times=bt, beats_per_bar=BPB,
                    downbeat_index=0, tempo_curve=np.full(len(bt),BPM))

def bar_hits(bar, spec, jitter=0.0):
    """spec: list of (beat_pos, inst). jitter in seconds (std)."""
    out=[]
    for pos,inst in spec:
        t=(bar*BPB+pos)*T + (rng.standard_normal()*jitter)
        out.append(Hit(time=max(0,t), inst=inst, velocity=.8))
    return out

EIGHTH=[(i*0.5,'hihat') for i in range(8)]+[(0,'kick'),(2,'kick'),(1,'snare'),(3,'snare')]
SIXT  =[(i*0.25,'hihat') for i in range(16)]+[(0,'kick'),(2,'kick'),(1,'snare'),(3,'snare')]
PICKUP=EIGHTH+[(2.75,'kick')]                       # one genuine 16th
TRIP  =[(b+k/3,'hihat') for b in range(4) for k in range(3)]+[(0,'kick'),(1,'snare'),(3,'snare')]

CASES=[  # (label, spec, jitter_s, expected subdivision)
    ("clean 8ths",       EIGHTH, 0.000, 2),
    ("jitter 8ths 20ms", EIGHTH, 0.020, 2),
    ("jitter 8ths 35ms", EIGHTH, 0.035, 2),
    ("clean 16ths",      SIXT,   0.000, 4),
    ("jitter 16ths 20ms",SIXT,   0.020, 4),
    ("8ths+16th pickup", PICKUP, 0.012, 4),
    ("8th triplets",     TRIP,   0.012, 3),
    ("clean 8ths again", EIGHTH, 0.000, 2),
]

hits=[]
for bar,(label,spec,j,_e) in enumerate(CASES):
    hits+=bar_hits(bar,spec,j)
hits.sort(key=lambda h:h.time)

q=Q.quantize(hits, grid(len(CASES)), detect_swing=False)
by={b.index:b for b in q.bars}
ok=0
print(f"{'bar':3s} {'case':20s} {'want':>5s} {'got':>5s}  verdict")
for i,(label,spec,j,exp) in enumerate(CASES):
    b=by.get(i)
    got=b.subdivision if b else None
    # a 16th-demanding bar written as 8ths loses notes; 8ths written as 16ths
    # invents offbeat positions - both are wrong notation
    good = got==exp
    # extra check for the pickup bar: the 16th must land on an off-16th tick
    note=""
    if label.startswith("8ths+16th") and b is not None:
        picks=[n.tick for n in b.notes if n.inst=='kick' and n.tick%24]
        note=f" pickup tick={picks}"
        good = good and len(picks)==1 and picks[0]%12==0 and picks[0]%24!=0
    ok+=good
    print(f"{i:3d} {label:20s} {exp:5d} {str(got):>5s}  {'OK' if good else 'WRONG'}{note}")
print(f"\n{ok}/{len(CASES)} bars correctly spelled")
