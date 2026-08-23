"""Hard synthetic fixture: realistic broadband hi-hats on 16ths, ghost notes,
a 16th pickup, open hats, a tom fill. Writes wav + ground-truth json."""
import numpy as np, soundfile as sf, sys, json, collections
from scipy.signal import butter, lfilter
SR=44100; rng=np.random.default_rng(11)
def env(n,d): return np.exp(-np.arange(n)/(SR*d))
def bp(x,lo,hi,o=3):
    b,a=butter(o,[lo/(SR/2),hi/(SR/2)],btype='band'); return lfilter(b,a,x)
def hp(x,f,o=3):
    b,a=butter(o,f/(SR/2),btype='high'); return lfilter(b,a,x)
def kick():
    n=int(.28*SR); t=np.arange(n)/SR; f=100*np.exp(-t*30)+45
    click=hp(rng.standard_normal(n),3000)*env(n,.004)*.4
    return (np.sin(2*np.pi*np.cumsum(f)/SR)*env(n,.10)+click).astype(np.float32)
def snare(g=1.0):
    n=int(.25*SR); t=np.arange(n)/SR
    body=(np.sin(2*np.pi*195*t)+.7*np.sin(2*np.pi*330*t))*env(n,.05)*.8
    wires=bp(rng.standard_normal(n),1200,9000)*env(n,.09)*.9
    rattle=bp(rng.standard_normal(n),300,900)*env(n,.05)*.25
    return ((body+wires+rattle)*g).astype(np.float32)
def hat(open_=False):
    n=int((.6 if open_ else .12)*SR); d=.28 if open_ else .03
    x=bp(rng.standard_normal(n),2500,15000)*env(n,d)
    x+=bp(rng.standard_normal(n),1500,6000)*env(n,d*.8)*.5
    return (x*.5).astype(np.float32)
def crash():
    n=int(1.8*SR); return (hp(rng.standard_normal(n),2000)*env(n,.6)*.6).astype(np.float32)
def tom(f0):
    n=int(.4*SR); t=np.arange(n)/SR; f=f0*np.exp(-t*5)+f0*.8
    return (np.sin(2*np.pi*np.cumsum(f)/SR)*env(n,.15)*.8).astype(np.float32)
BPM=100.0; SIX=60/BPM/4; BARS=8
S={'kick':kick(),'snare':snare(),'ghost':snare(.28),'hihat':hat(),'openhh':hat(True),
   'crash':crash(),'tom_hi':tom(220),'tom_mid':tom(155),'tom_low':tom(100)}
out=np.zeros(int((BARS*16*SIX+3)*SR),dtype=np.float32); truth=[]
def place(inst,pos,gain=1.0,label=None):
    s=S[inst]; i=int(round((1.0+pos*SIX)*SR)); out[i:i+len(s)]+=s*gain
    truth.append((round(1.0+pos*SIX,4),label or inst))
for bar in range(BARS):
    b=bar*16
    if bar==0: place('crash',b+0)
    fill=(bar==7)
    for i in range(16):
        if fill and i>=12: break
        if bar in (3,7) and i==14: place('openhh',b+i,.9)
        else: place('hihat',b+i,.95 if i%4==0 else .6)
    place('snare',b+4); place('snare',b+12)
    if bar in (2,4,6): place('ghost',b+7,label='snare'); place('ghost',b+10,label='snare')
    place('kick',b+0); place('kick',b+6)
    if bar%2: place('kick',b+11,.85)
    if fill:
        for i,inst in enumerate(['tom_hi','tom_mid','tom_low','snare']): place(inst,b+12+i,.95)
out/=max(1.0,np.abs(out).max()/0.85)
sf.write(sys.argv[1],np.stack([out,out]).T,SR); json.dump(truth,open(sys.argv[2],'w'))
print("wrote",len(truth),"hits:",dict(collections.Counter(i for _,i in truth)))
