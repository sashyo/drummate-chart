"""Stress the subdivision chooser where aggregate error is known to lie."""
import sys, collections
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from backend.pipeline.onsets import Hit
from backend.pipeline.rhythm import BeatGrid
from backend.pipeline import quantize as Q

BPM=100.0; T=60/BPM; BPB=4
def grid(n_bars):
    bt=np.arange(0,(n_bars*BPB+2))*T
    return BeatGrid(tempo=BPM, beat_times=bt, beats_per_bar=BPB,
                    downbeat_index=0, tempo_curve=np.full(len(bt),BPM))
EIGHTH=[(i*0.5,'hihat') for i in range(8)]+[(0,'kick'),(2,'kick'),(1,'snare'),(3,'snare')]
SIXT  =[(i*0.25,'hihat') for i in range(16)]+[(0,'kick'),(2,'kick'),(1,'snare'),(3,'snare')]
SPARSE=[(0,'kick'),(1,'snare'),(2.5,'kick'),(3,'snare')]           # 4 hits, one off-8th

def run_case(label,spec,jitter_beats,lag_beats,expect,reps=40,seed=0):
    rng=np.random.default_rng(seed)
    wrong=collections.Counter(); pos_err=0
    for r in range(reps):
        hits=[]
        for pos,inst in spec:
            p=pos+lag_beats+rng.standard_normal()*jitter_beats
            hits.append(Hit(time=max(0,p*T), inst=inst, velocity=.8))
        hits.sort(key=lambda h:h.time)
        q=Q.quantize(hits, grid(1), detect_swing=False)
        b=next((x for x in q.bars if x.index==0), None)
        got=b.subdivision if b else None
        if got!=expect: wrong[got]+=1
        elif expect==2 and b is not None:
            # even with the right grid, did any hit land on a phantom off-16th?
            if any(n.tick%24 for n in b.notes): pos_err+=1
    flips=sum(wrong.values())
    print(f"{label:34s} want {expect}: {reps-flips}/{reps} right"
          f"{'  wrong->'+str(dict(wrong)) if wrong else ''}")
    return flips

total=0
total+=run_case("8ths, jitter 0.06 beat (~35ms@170)",EIGHTH,0.06,0.0,2)
total+=run_case("8ths, jitter 0.08 beat",           EIGHTH,0.08,0.0,2)
total+=run_case("8ths, laid back +0.10 beat",       EIGHTH,0.03,0.10,2)
total+=run_case("8ths, laid back +0.15 beat",       EIGHTH,0.03,0.15,2)
total+=run_case("16ths, jitter 0.05 beat",          SIXT,  0.05,0.0,4)
total+=run_case("16ths, jitter 0.07 beat",          SIXT,  0.07,0.0,4)
total+=run_case("sparse bar w/ off-8th kick",       SPARSE,0.04,0.0,2)
print("\ntotal wrong:",total)
