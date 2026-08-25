"""Compare a chart against a known groove: check_groove.py <score.json> "<pattern>"

Pattern = 16 slots per bar (4/4), letters per slot, e.g. for a straight rock beat
  "Kx.x Sx.x Kx.x Sx.x"   (K kick, S snare, x hat, . nothing; a slot may hold several: "KS" or "Kx")
Reports, per instrument, how many bars agree with the pattern and the most
common deviations - the objective version of 'it should look like this'.
"""
import json, sys, collections
doc=json.load(open(sys.argv[1]))
# pattern: 16 whitespace-separated slot tokens (16ths from beat 1); a token is
# '.' for nothing or letters for every drum sounding on that 16th, e.g.
#   "Kx . x . Sx . x . Kx . x . Sx . x ."   = straight 8th hats, kick 1&3, snare 2&4
toks=sys.argv[2].split()
if len(toks)!=16:
    sys.exit(f"pattern must have 16 slot tokens separated by spaces, got {len(toks)}")
slots=[set() if t=='.' else set(t) for t in toks]
MAP={'kick':'K','snare':'S','hihat':'x','openhh':'x','ride':'x','crash':'x'}
bars=[b for b in doc["bars"] if not b["empty"] and b.get("beats",4)==4]
odd=[b for b in doc["bars"] if b.get("beats",4)!=4]
if odd: print(f"({len(odd)} bars of other meters skipped: {[b['number'] for b in odd]})")
print(f"{doc['title'][:50]} | {doc['tempo']:.1f} BPM | {len(bars)} bars | detector {doc.get('detector')} engine {doc.get('engine')}")
agree=collections.Counter(); total=collections.Counter(); dev=collections.Counter()
exact=0
for b in bars:
    got=[set() for _ in range(16)]
    for h in b["hits"]:
        s=h["tick"]//12
        if s<16 and h["inst"] in MAP: got[s].add(MAP[h["inst"]])
    ok=True
    for L in ('K','S','x'):
        want=[L in sl for sl in slots]; have=[L in g for g in got]
        total[L]+=1
        if want==have: agree[L]+=1
        else:
            ok=False
            extra=[i for i in range(16) if have[i] and not want[i]]
            missing=[i for i in range(16) if want[i] and not have[i]]
            dev[(L,'extra',tuple(extra))]+=1 if extra else 0
            dev[(L,'missing',tuple(missing))]+=1 if missing else 0
    exact+=ok
print(f"bars matching the whole pattern exactly: {exact}/{len(bars)}")
for L,name in (('x','hi-hat'),('K','kick'),('S','snare')):
    print(f"  {name:7s}: {agree[L]}/{total[L]} bars correct")
print("most common deviations (slot numbers 0-15 = 16ths from beat 1):")
for (L,kind,sl),n in sorted(dev.items(), key=lambda kv:-kv[1])[:6]:
    if n: print(f"   {L} {kind} at {list(sl)}  x{n}")
