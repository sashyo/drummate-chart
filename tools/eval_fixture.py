"""Score the transcriber against a fixture: eval_fixture.py <wav> <truth.json> <outdir> [detector]"""
import json,sys,collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pipeline.run import Options, transcribe
wav,truthp,outdir=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])
detector=sys.argv[4] if len(sys.argv)>4 else "auto"
opts=Options(separation='none', render_audio=False, detector=detector)
doc=transcribe(None,outdir,outdir/'cache',opts,local_file=wav,title=wav.stem)
truth=[tuple(x) for x in json.load(open(truthp))]
pred=[(h['time'],h['inst']) for b in doc['bars'] for h in b['hits']]
TOL=0.055
tp=collections.Counter(); fn=collections.Counter(); fp=collections.Counter(); used=set()
for t,inst in truth:
    m=[i for i,(pt,pi) in enumerate(pred) if pi==inst and abs(pt-t)<=TOL and i not in used]
    if m: used.add(m[0]); tp[inst]+=1
    else: fn[inst]+=1
for i,(pt,pi) in enumerate(pred):
    if i not in used: fp[pi]+=1
print(f"detector={doc.get('detector')}  tempo={doc['tempo']:.1f}")
print(f"{'inst':8s}{'truth':>6s}{'tp':>5s}{'fn':>5s}{'fp':>5s}    P     R    F1")
TP=FN=FP=0
for i in sorted(set(list(tp)+list(fn)+list(fp))):
    t_,f_,p_=tp[i],fn[i],fp[i]; TP+=t_;FN+=f_;FP+=p_
    P=t_/max(1,t_+p_); R=t_/max(1,t_+f_); F=2*P*R/max(1e-9,P+R)
    print(f"{i:8s}{t_+f_:6d}{t_:5d}{f_:5d}{p_:5d}  {P:.2f}  {R:.2f}  {F:.2f}")
P=TP/max(1,TP+FP);R=TP/max(1,TP+FN);F=2*P*R/max(1e-9,P+R)
print(f"{'TOTAL':8s}{TP+FN:6d}{TP:5d}{FN:5d}{FP:5d}  {P:.2f}  {R:.2f}  {F:.2f}")
