"""Notation lint: every bar, every voice, engraving invariants."""
import json, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from backend.pipeline import score as SC
from backend.pipeline.rebuild import from_json

BEATS={'w':4.0,'h':2.0,'q':1.0,'8':.5,'16':.25,'32':.125}
def dur_beats(e):
    d=BEATS[e['dur']]
    if e.get('dots'): d*=1.5**e['dots']
    t=e.get('tuplet')
    if t: d*=t['den']/t['num']
    return d

def lint(doc,label):
    problems=[]
    q=from_json(doc,[])
    rebuilt=SC.build(q,{'title':'lint'})
    for b in rebuilt['bars']:
        bpb=rebuilt['beatsPerBar']
        for vn,elems in b['voices'].items():
            if not elems: continue
            if len(elems)==1 and elems[0].get('barRest'): continue
            tot=sum(dur_beats(e) for e in elems)
            if abs(tot-bpb)>1e-6:
                problems.append(f"bar {b['number']} voice {vn}: sums to {tot} beats (want {bpb})")
            for e in elems:
                if e['type']=='rest' and e.get('dots'):
                    problems.append(f"bar {b['number']} {vn}: dotted rest")
                if e['dur'] not in BEATS:
                    problems.append(f"bar {b['number']} {vn}: bad duration {e['dur']}")
            # tuplet groups must fill whole beats
            i=0
            while i<len(elems):
                t=elems[i].get('tuplet')
                if not t: i+=1; continue
                j=i; s=0
                while j<len(elems) and elems[j].get('tuplet')==t and elems[j]['beat']==elems[i]['beat']:
                    s+=dur_beats(elems[j]); j+=1
                if abs(s-1.0)>1e-6:
                    problems.append(f"bar {b['number']} {vn}: tuplet group covers {s} beats")
                i=j
    empties=sum(1 for b in rebuilt['bars'] if b['empty'])
    runs=[]; run=0
    for b in rebuilt['bars']:
        run=run+1 if b['empty'] else 0
        if run: runs.append(run)
    print(f"{label}: {len(rebuilt['bars'])} bars, {empties} empty (longest run {max(runs) if runs else 0}), {len(problems)} problems")
    for p in problems[:12]: print("   !", p)
    return problems

allp=[]
import glob
paths=sys.argv[1:] or sorted(glob.glob("data/jobs/*/score.json"))
if not paths:
    print("usage: lint_notation.py [score.json ...]  (or run from the repo root with jobs present)")
for path in paths:
    allp+=lint(json.load(open(path)),path)
print("\nTOTAL PROBLEMS:",len(allp))
sys.exit(1 if allp else 0)
