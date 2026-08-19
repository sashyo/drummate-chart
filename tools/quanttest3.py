"""Sequence test: 60 consecutive jittered 8th bars - count flicker."""
import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from backend.pipeline.onsets import Hit
from backend.pipeline.rhythm import BeatGrid
from backend.pipeline import quantize as Q
BPM=100.0; T=60/BPM; BPB=4
EIGHTH=[(i*0.5,'hihat') for i in range(8)]+[(0,'kick'),(2,'kick'),(1,'snare'),(3,'snare')]
for sigma in (0.06,0.08):
    rng=np.random.default_rng(5)
    hits=[]
    N=60
    for bar in range(N):
        for pos,inst in EIGHTH:
            t=(bar*BPB+pos)*T+rng.standard_normal()*sigma*T
            hits.append(Hit(time=max(0,t),inst=inst,velocity=.8))
    hits.sort(key=lambda h:h.time)
    bt=np.arange(0,N*BPB+2)*T
    g=BeatGrid(tempo=BPM,beat_times=bt,beats_per_bar=BPB,downbeat_index=0,
               tempo_curve=np.full(len(bt),BPM))
    q=Q.quantize(hits,g,detect_swing=False)
    subs=[b.subdivision for b in q.bars]
    wrong=sum(1 for x in subs if x!=2)
    print(f"sigma={sigma}: {N-wrong}/{N} bars stay 8ths | flips: {[(i,x) for i,x in enumerate(subs) if x!=2][:8]}")
