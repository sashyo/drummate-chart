/* Drum Notation — browser app.
 *
 * The server sends, per bar, the raw list of hits (tick + instrument). The
 * browser re-runs the same beat-by-beat layout the server uses for export,
 * so edits re-engrave instantly without a round trip.
 */
'use strict';
const APP_BUILD='2026-08-26i';
const ENGINE_CURRENT = 4;

const VF = Vex.Flow;
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

/* ── the kit ──────────────────────────────────────────────────────────── */
const PPQ = 48;
const DRUMS = {
  crash:   {label:'Crash',       key:'a/5/x2', midi:49, voice:'up',   order:0, short:'Cr'},
  hihat:   {label:'Hi-hat',      key:'g/5/x2', midi:42, voice:'up',   order:1, short:'HH'},
  openhh:  {label:'Open hi-hat', key:'g/5/cx', midi:46, voice:'up',   order:1, short:'oHH'},
  ride:    {label:'Ride',        key:'f/5/x2', midi:51, voice:'up',   order:2, short:'Rd'},
  tom_hi:  {label:'High tom',    key:'e/5',    midi:48, voice:'up',   order:3, short:'T1'},
  tom_mid: {label:'Mid tom',     key:'d/5',    midi:45, voice:'up',   order:4, short:'T2'},
  snare:   {label:'Snare',       key:'c/5',    midi:38, voice:'up',   order:5, short:'Sn'},
  tom_low: {label:'Floor tom',   key:'a/4',    midi:41, voice:'up',   order:6, short:'FT'},
  kick:    {label:'Bass drum',   key:'f/4',    midi:36, voice:'down', order:7, short:'BD'},
  hhfoot:  {label:'Hi-hat foot', key:'d/4/x2', midi:44, voice:'down', order:8, short:'HHf'},
  perc:    {label:'Perc / rim',  key:'b/5/d',  midi:37, voice:'up',   order:9, short:'Pc'},
};
const INSTS = Object.keys(DRUMS).sort((a,b)=>DRUMS[a].order-DRUMS[b].order);

/* slots-in-a-beat -> [duration, dots], per subdivision (mirrors score.py) */
const TABLES = {
  1:{1:['q',0]},
  2:{1:['8',0],2:['q',0]},
  4:{1:['16',0],2:['8',0],3:['8',1],4:['q',0]},
  8:{1:['32',0],2:['16',0],3:['16',1],4:['8',0],6:['8',1],8:['q',0]},
  3:{1:['8',0],2:['q',0],3:['q',1]},
  6:{1:['16',0],2:['8',0],3:['8',1],4:['q',0],6:['q',1]},
};
const TUPLET_OCCUPIED = {3:2, 6:4};

/* ── state ────────────────────────────────────────────────────────────── */
const S = {
  jobId:null, score:null, poll:null,
  source:'drums', playing:false, speed:1,
  loopFrom:null, loopTo:null, loopPick:0,
  editing:false, brush:'snare', detail:100, simple:false, teach:false,
  hatHand:(typeof localStorage!=='undefined' && localStorage.getItem('dm-hats-hand'))||'right',
  sayIt:(typeof localStorage!=='undefined' && localStorage.getItem('dm-say-hits'))==='1',
  cursorBar:-1, systems:[], yt:null, ytReady:false,
  ctx:null, scheduled:[], schedTimer:null, playFrom:0, playStarted:0,
};

/* ── helpers ──────────────────────────────────────────────────────────── */
const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
function fmtTime(s){
  if(!isFinite(s)||s<0) s=0;
  const m=Math.floor(s/60), r=Math.floor(s%60);
  return `${m}:${String(r).padStart(2,'0')}`;
}
function parseTime(v){
  if(!v) return null;
  v=String(v).trim(); if(!v) return null;
  if(v.includes(':')){
    const p=v.split(':').map(Number);
    if(p.some(isNaN)) return null;
    return p.length===3 ? p[0]*3600+p[1]*60+p[2] : p[0]*60+p[1];
  }
  const n=Number(v); return isNaN(n)?null:n;
}

/* ── layout: hits -> notation elements (same algorithm as the exporter) ── */
function splitCount(count, table, rest=false){
  /* rests avoid dotted values — an 8th + 16th rest reads at a glance */
  const sizes=Object.keys(table).map(Number)
    .filter(k=>!(rest && table[k][1])).sort((a,b)=>b-a);
  const out=[]; let rem=count, guard=0;
  while(rem>0 && guard++<16){
    const pick=sizes.find(s=>s<=rem);
    if(pick===undefined) break;
    out.push([...table[pick], pick]);
    rem-=pick;
  }
  return out;
}

function layoutVoice(hits, subdiv, beatsPerBar){
  /* a voice with nothing to play gets one whole-bar rest, not a stack of quarters */
  if(!hits.length)
    return [{type:'rest', dur:'w', dots:0, beat:0, keys:[], insts:[], barRest:true}];
  const table=TABLES[subdiv]||TABLES[4];
  const slotTicks=PPQ/subdiv;
  const elems=[];
  for(let beat=0; beat<beatsPerBar; beat++){
    const lo=beat*PPQ, hi=(beat+1)*PPQ;
    const slots=new Map();
    for(const h of hits){
      if(h.tick<lo||h.tick>=hi) continue;
      const s=Math.min(subdiv-1, Math.floor((h.tick-lo)/slotTicks));
      if(!slots.has(s)) slots.set(s,[]);
      slots.get(s).push(h);
    }
    const be=[];
    if(slots.size===0){
      be.push({type:'rest', dur:'q', dots:0, beat, keys:[], insts:[]});
    }else{
      const occ=[...slots.keys()].sort((a,b)=>a-b);
      if(occ[0]>0)
        for(const [d,dt] of splitCount(occ[0], table, true))
          be.push({type:'rest', dur:d, dots:dt, beat, keys:[], insts:[]});
      occ.forEach((s,i)=>{
        const next = i+1<occ.length ? occ[i+1] : subdiv;
        const pieces=splitCount(next-s, table);
        const [d,dt] = pieces.length?pieces[0]:['16',0];
        const group=slots.get(s).slice().sort((a,b)=>DRUMS[a.inst].order-DRUMS[b.inst].order);
        be.push({
          type:'note', dur:d, dots:dt, beat,
          keys:group.map(h=>DRUMS[h.inst].key),
          insts:group.map(h=>h.inst),
          accent:group.some(h=>h.accent),
          ghost:group.every(h=>h.ghost)&&group.some(h=>h.inst==='snare'),
          flam:group.some(h=>h.flam),
          open:group.some(h=>h.inst==='openhh'),
          time:Math.min(...group.map(h=>h.time??0)),
          tick:lo+s*slotTicks,
        });
        for(const [d2,dt2] of pieces.slice(1))
          be.push({type:'rest', dur:d2, dots:dt2, beat, keys:[], insts:[]});
      });
    }
    if(TUPLET_OCCUPIED[subdiv]){
      if(be.length>1) be.forEach(e=>e.tuplet={num:subdiv, den:TUPLET_OCCUPIED[subdiv]});
      else { be[0].dur='q'; be[0].dots=0; }   // a whole triplet beat = one quarter
    }
    elems.push(...be);
  }
  return elems;
}

function barBeats(bar){ return (bar && bar.beats) || S.score.beatsPerBar; }
function barTicks(bar){ return (bar && bar.ticksPerBar) || S.score.ticksPerBar; }
function layoutBar(bar, beatsPerBar){
  const up=bar.hits.filter(h=>DRUMS[h.inst]?.voice==='up');
  const down=bar.hits.filter(h=>DRUMS[h.inst]?.voice==='down');
  const laid={
    up: layoutVoice(up, bar.subdivision, beatsPerBar),
    down: layoutVoice(down, bar.subdivision, beatsPerBar),
  };
  /* Published charts don't stack rests under the hands: the kick voice only
   * shows a rest when it locates a mid-beat note (e.g. the 8th rest before a
   * kick on the "&"). Whole-beat rests in the feet stay for the voice math
   * but are not drawn. */
  for(const e of laid.down)
    if(e.type==='rest' && e.dur==='q' && !e.dots) e.hidden=true;
  return laid;
}

/* ── API ──────────────────────────────────────────────────────────────── */
async function api(path, opts){
  const r=await fetch(path, opts);
  if(!r.ok){
    let msg=`${r.status} ${r.statusText}`;
    try{ const j=await r.json(); if(j.detail) msg=j.detail; }catch(_){}
    throw new Error(msg);
  }
  return r.json();
}

function collectOptions(){
  return {
    start: parseTime($('#opt-start').value),
    end: parseTime($('#opt-end').value),
    beatsPerBar: Number($('#opt-meter').value),
    tempo: $('#opt-tempo').value ? Number($('#opt-tempo').value) : null,
    sensitivity: Number($('#opt-sens').value),
    maxSubdiv: Number($('#opt-subdiv').value),
    allowTriplets: $('#opt-triplets').checked,
    detectToms: $('#opt-toms').checked,
    cymbalDetail: $('#opt-cym').checked,
    detectSwing: $('#opt-swing').checked,
    separation: $('#opt-sep').value,
    lockGrid: $('#opt-grid') ? $('#opt-grid').value==='lock' : false,
    detector: $('#opt-detector') ? $('#opt-detector').value : 'auto',
    renderAudio: true,
  };
}

/* ── the wait-time ad: real DrumMate feature copy, rotating ─────────── */
const AD_SLIDES=[
  ['You drum. The band follows.', 'No backing track. Bass, keys and synth bend to <em>your</em> groove in real time \u2014 play a fill, push the tempo, lay back, the band stays glued to you.'],
  ['Follow tempo.', 'BPM, phase, swing and energy update as you play. Slow it to woodshed, crank it to double-time \u2014 any tempo, any time.'],
  ['Shape songs. You call the shots.', 'Pick style, key and progression; fills can move sections. Mute keys, add synth, strip to just bass \u2014 your kit, your rules.'],
  ['A whole band, no bandmates.', 'Analog synth, tine e-piano, FM, 808 built in \u2014 or bring your own SoundFonts and samples. Plug in USB MIDI and drum.'],
];
function startAd(){
  const t=$('#wait-ad-title'), x=$('#wait-ad-text'), dots=$('#wait-ad-dots'), v=$('#wait-ad-video');
  if(!t) return;
  if(v && !v.src){
    v.src='drummate-demo.mp4?v=2'; v.volume=0.8;
    const mute=$('#wait-ad-mute');
    const paint=()=>{ if(mute){ mute.innerHTML=v.muted?'&#128263; sound off':'&#128266; sound on'; mute.setAttribute('aria-pressed', v.muted?'true':'false'); } };
    // the user has already clicked "Chart it", so most browsers allow sound;
    // Safari/iOS may still refuse - then start silent and let the button unmute
    v.muted=false;
    v.play().then(paint).catch(()=>{ v.muted=true; v.play().catch(()=>{}); paint(); });
    if(mute) mute.onclick=()=>{ v.muted=!v.muted; if(!v.muted && v.paused) v.play().catch(()=>{}); paint(); };
  } else if(v){ v.play().catch(()=>{}); }
  let i=0;
  dots.innerHTML=AD_SLIDES.map((_,k)=>`<i class="${k===0?'on':''}"></i>`).join('');
  clearInterval(S.adTimer);
  S.adTimer=setInterval(()=>{
    i=(i+1)%AD_SLIDES.length;
    t.textContent=AD_SLIDES[i][0]; x.innerHTML=AD_SLIDES[i][1];
    [...dots.children].forEach((d,k)=>d.classList.toggle('on',k===i));
  }, 7000);
}
function stopAd(){ clearInterval(S.adTimer); const v=$('#wait-ad-video'); if(v) v.pause(); }

function showView(name){
  for(const v of ['setup','progress','score'])
    $('#view-'+v).classList.toggle('hidden', v!==name);
  $('#btn-new').classList.toggle('hidden', name==='setup');
  if(name==='progress') startAd(); else stopAd();
}

async function startJob(){
  const url=$('#url').value.trim();
  if(!url){ showError('Paste a YouTube link, or drop an audio file.'); return; }
  showError('');
  try{
    const {jobId}=await api('/api/transcribe',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url, rights: !!($('#rights')&&$('#rights').checked), ...collectOptions()}),
    });
    watch(jobId);
  }catch(e){ showError(e.message); }
}

async function startUpload(file){
  showError('');
  const fd=new FormData();
  fd.append('file', file);
  fd.append('options', JSON.stringify(collectOptions()));
  try{
    const {jobId}=await api('/api/upload',{method:'POST', body:fd});
    watch(jobId);
  }catch(e){ showError(e.message); }
}

function showError(msg){
  const el=$('#setup-error');
  el.textContent=msg; el.classList.toggle('hidden', !msg);
}

function watch(jobId){
  S.jobId=jobId;
  showView('progress');
  $('#prog-fill').style.width='2%';
  $('#prog-msg').textContent='Queued…';
  clearInterval(S.poll); clearInterval(S.creep);
  S.progT0=Date.now(); S.progSrv=0.02; S.progAt=Date.now();
  // Between server updates the bar creeps a little (never past the next
  // stage) and the clock ticks - a frozen bar reads as a hung app.
  S.stripe=0;
  S.creep=setInterval(()=>{
    const idle=(Date.now()-S.progAt)/1000;
    const creep=Math.min(0.035, 0.035*(1-Math.exp(-idle/45)));
    const shown=Math.min(0.99, S.progSrv+creep);
    const fill=$('#prog-fill');
    fill.style.width=(shown*100).toFixed(1)+'%';
    // move the stripes from JS as well - never rely on a CSS animation that
    // a cached stylesheet or reduced-motion setting can switch off
    S.stripe=(S.stripe+7)%28;
    fill.style.backgroundPosition=S.stripe+'px 0';
    if($('#prog-pct')) $('#prog-pct').textContent=Math.round(shown*100)+'%';
    if($('#prog-elapsed')) $('#prog-elapsed').textContent=fmtTime((Date.now()-S.progT0)/1000)+' elapsed';
  }, 250);
  S.poll=setInterval(async()=>{
    let j;
    try{ j=await api(`/api/jobs/${jobId}`); }
    catch(e){ return; }
    S.jobInfo=j;
    if(j.progress!==S.progSrv){ S.progSrv=j.progress; S.progAt=Date.now(); }
    let msg=j.message||'';
    // an exhausted estimate must not read as a stall
    if(/about 0:0\d left/.test(msg)) msg=msg.replace(/ \u2014 about 0:0\d left/,'')+' \u2014 taking longer than estimated (server is busy), still working';
    $('#prog-msg').textContent=msg;
    $$('#prog-steps li').forEach(li=>{
      const at=Number(li.dataset.at);
      li.classList.toggle('done', j.progress>=at);
      li.classList.toggle('active', j.progress<at && j.progress>at-0.35);
    });
    if(j.status==='done'){
      clearInterval(S.poll); clearInterval(S.creep);
      const score=await api(`/api/jobs/${jobId}/score`);
      openScore(score);
    }else if(j.status==='error'){
      clearInterval(S.poll); clearInterval(S.creep);
      showView('setup');
      showError(j.error||'Transcription failed.');
    }
  }, 700);
}


/* ── teach mode: find the groove, layer it, explain it ────────────────── */
const CLASS_MAP={openhh:'cym', ride:'cym', hihat:'cym', crash:'cym', perc:'perc',
                 tom_hi:'tom', tom_mid:'tom', tom_low:'tom',
                 kick:'kick', snare:'snare', hhfoot:'kick'};

function grooveOf(bar){
  /* fingerprint a bar on the 8th grid by drum class */
  const slot=PPQ/2, seen=new Map();
  for(const h of bar.hits){
    if(h.ghost) continue;
    const cls=CLASS_MAP[h.inst]||'cym';
    const tick=Math.min(S.score.ticksPerBar-slot, Math.round(h.tick/slot)*slot);
    const key=cls+':'+tick;
    if(!seen.has(key)) seen.set(key,{tick, cls, inst:h.inst, velocity:h.velocity||0.8});
  }
  return [...seen.values()].sort((a,b)=>a.tick-b.tick);
}

function analyzeGroove(){
  /* Per-slot voting: a slot is part of THE groove when most bars hit it.
   * Exact-fingerprint matching fails on real transcriptions - residual
   * detection noise makes nearly every bar unique. */
  const grooves=[];
  for(const b of S.score.bars){
    if(b.empty) continue;
    const g=grooveOf(b);
    if(!g.length || g.some(x=>x.cls==='tom')) continue;   // fills aren't the groove
    grooves.push({g, number:b.number});
  }
  if(grooves.length<2) return null;
  const freq=new Map();
  const classBars={};
  for(const {g} of grooves){
    const present=new Set();
    for(const h of g){
      const k=h.cls+':'+h.tick;
      const e=freq.get(k)||{n:0, tick:h.tick, cls:h.cls, insts:new Map(), vsum:0};
      e.n++; e.vsum+=h.velocity||0.8;
      e.insts.set(h.inst,(e.insts.get(h.inst)||0)+1);
      freq.set(k, e);
      present.add(h.cls);
    }
    for(const c of present) classBars[c]=(classBars[c]||0)+1;
  }
  const N=grooves.length;
  const byClass={};
  for(const e of freq.values()) (byClass[e.cls]=byClass[e.cls]||[]).push(e);
  const core=[];
  for(const cls in byClass){
    let sel=byClass[cls].filter(e=>e.n>=0.55*N);
    // A drum that moves around never wins one slot outright, but a groove
    // without its snare is simply the wrong groove. If the class plays in
    // most bars, take its strongest slots anyway.
    if(!sel.length && (classBars[cls]||0)>=0.5*N)
      sel=byClass[cls].filter(e=>e.n>=0.3*N)
        .sort((a,b)=>b.n-a.n).slice(0, cls==='cym'?8:3);
    core.push(...sel);
  }
  const pick=e=>{
    let inst=null, n=-1;
    for(const [i2,c2] of e.insts) if(c2>n){ n=c2; inst=i2; }
    return {tick:e.tick, cls:e.cls, inst, velocity:Math.min(1, e.vsum/e.n+0.15)};
  };
  const coreHits=core.map(pick)
    .sort((a,b)=>a.tick-b.tick||DRUMS[a.inst].order-DRUMS[b.inst].order);
  if(coreHits.length<3) return null;
  const coreKeys=new Set(coreHits.map(h=>h.cls+':'+h.tick));
  let match=0; const matchBars=[];
  for(const {g, number} of grooves){
    const keys=new Set(g.map(h=>h.cls+':'+h.tick));
    let inter=0;
    for(const k of keys) if(coreKeys.has(k)) inter++;
    if(inter/Math.max(1, keys.size+coreKeys.size-inter) >= 0.7){
      match++; matchBars.push(number);
    }
  }
  return {hits:coreHits, share:match/N, bars:matchBars};
}

function countLabel(t){
  if(t%PPQ===0) return String(Math.floor(t/PPQ)+1);
  if(t%(PPQ/2)===0) return '&';
  return t%PPQ===PPQ/4 ? 'e' : 'a';
}

function renderBarInto(el, hits, width, withCounts, subdivision=2){
  el.classList.add('step-bar');
  const r=new VF.Renderer(el, VF.Renderer.Backends.SVG);
  r.resize(width, 128);
  const ctx=r.getContext();
  const stave=new VF.Stave(4, 22, width-10, {num_lines:5});
  stave.setContext(ctx).draw();
  const bar={hits:hits.map(h=>({...h})), subdivision};
  const laid=layoutBar(bar, barBeats(bar));
  const voices=[], beams=[], drawn=[];
  for(const [elems,dir] of [[laid.up,'up'],[laid.down,'down']]){
    if(dir==='down' && !elems.some(e=>e.type==='note')) continue;
    const notes=buildNotes(elems, dir);
    if(!notes.length) continue;
    drawn.push(...notes);
    if(withCounts && dir==='up')
      for(const n of notes){
        if(n.el.type!=='note') continue;
        const txt = countLabel(n.el.tick??0);
        try{
          n.vf.addModifier(new VF.Annotation(txt).setFont('JetBrains Mono',9)
            .setVerticalJustify(VF.AnnotationVerticalJustify.BOTTOM), 0);
        }catch(_){}
      }
    const v=new VF.Voice({num_beats:barBeats(bar), beat_value:4});
    v.setStrict(false); v.addTickables(notes.map(n=>n.vf));
    voices.push(v);
    VF.Beam.generateBeams(notes.filter(n=>n.el.type==='note').map(n=>n.vf), {
      groups:[new VF.Fraction(1,4)],
      stem_direction: dir==='up'?VF.Stem.UP:VF.Stem.DOWN,
    }).forEach(b=>beams.push(b));
  }
  if(voices.length){
    const fmt=new VF.Formatter();
    fmt.joinVoices(voices); fmt.formatToStave(voices, stave);
    voices.forEach(v=>v.draw(ctx, stave));
    beams.forEach(b=>b.setContext(ctx).draw());
  }
  const ph=document.createElement('div');
  ph.className='playhead';
  el.appendChild(ph);
  // where each tick actually sits on the engraved stave - the formatter does
  // NOT space time linearly, so the playhead must follow the real noteheads
  const mm=new Map(), noteEls=new Map();
  for(const n of drawn){
    if(n.el.type!=='note' || n.el.tick==null) continue;
    try{
      const x=n.vf.getAbsoluteX();
      if(!mm.has(n.el.tick) || x<mm.get(n.el.tick)) mm.set(n.el.tick, x);
      const g=n.vf.getSVGElement && n.vf.getSVGElement();
      if(g){
        if(!noteEls.has(n.el.tick)) noteEls.set(n.el.tick, []);
        noteEls.get(n.el.tick).push(g);
      }
    }catch(_){}
  }
  const marks=[...mm.entries()].map(([tick,x])=>({tick,x})).sort((a,b)=>a.tick-b.tick);
  return {x0:stave.getNoteStartX(), x1:width-12, marks, noteEls};
}

/* ── lesson playback: loop any card through the synth kit ─────────────── */
function limbLabel(insts){
  const H=handNames();
  const hat=H.hat.split(' ')[0].toUpperCase(), sn=H.snare.split(' ')[0].toUpperCase();
  const parts=[];
  const has=i=>insts.includes(i);
  if(has('crash')) parts.push(hat+' \u00b7 crash');
  else if(has('ride')) parts.push(hat+' \u00b7 ride');
  else if(has('openhh')) parts.push(hat+' \u00b7 open hat');
  else if(has('hihat')) parts.push(hat+' \u00b7 hat');
  if(has('snare')) parts.push(sn+' \u00b7 snare');
  if(insts.some(i=>i.startsWith('tom_'))) parts.push('HANDS \u00b7 tom');
  if(has('kick')||has('hhfoot')) parts.push('FOOT \u00b7 kick');
  return parts.join('   +   ');
}

function barEvents(hits){
  const by=new Map();
  for(const h of hits){
    if(!by.has(h.tick)) by.set(h.tick, []);
    by.get(h.tick).push(h.inst);
  }
  const SPOKEN=[['crash','crash'],['tom_','tom'],['openhh','open hat'],
                ['snare','snare'],['kick','kick'],['ride','ride'],['hihat','hat']];
  const wordFor=insts=>{
    for(const [k,w] of SPOKEN)
      if(insts.some(i=>i===k||i.startsWith(k))) return w;
    return '';
  };
  return [...by.entries()].sort((a,b)=>a[0]-b[0])
    .map(([tick,insts])=>({tick, label:limbLabel(insts), word:wordFor(insts)}));
}

/* Spoken callouts are pre-rendered files scheduled through WebAudio -
 * speechSynthesis has 50-300 ms of jitter, which is musically unusable. */
const VOICE_WORDS=['kick','snare','hat','openhat','tom','crash','ride'];
function loadVoices(){
  if(S.voiceBuf || S.voiceLoading) return;
  S.voiceLoading=true;
  ensureCtx();
  const buf={};
  const load=(w,ext)=>fetch('voice/'+w+'.'+ext).then(r=>r.arrayBuffer())
      .then(a=>S.ctx.decodeAudioData(a)).then(b=>{ buf[w]=b; });
  Promise.all(VOICE_WORDS.map(w=>
    load(w,'ogg').catch(()=>load(w,'mp3')).catch(()=>{})   // Safari/iOS: no Vorbis
  )).then(()=>{ S.voiceBuf=buf; S.voiceLoading=false;
                if(S.sayIt) sayWord('kick'); });
}
function sayWord(word, when){
  const buf=S.voiceBuf && S.voiceBuf[word.replace(' ','')];
  if(!buf) return 0;
  ensureCtx();
  const src=S.ctx.createBufferSource(); src.buffer=buf;
  const g=S.ctx.createGain(); g.gain.value=1.7;
  src.connect(g).connect(S.ctx.destination);
  src.start(Math.max(S.ctx.currentTime, when||0));
  if(S.lessonRun) S.lessonRun.voiceSrcs.push(src);
  return buf.duration;
}

function stopLesson(){
  if(S.lessonTimer){ clearInterval(S.lessonTimer); S.lessonTimer=null; }
  if(S.lessonRun) for(const src of S.lessonRun.voiceSrcs||[])
    try{ src.stop(); }catch(_){}
  S.lessonRun=null;
  $$('.lesson-step.playing').forEach(e=>e.classList.remove('playing'));
  $$('.lesson-step .playhead').forEach(e=>e.style.opacity=0);
  $$('.step-play').forEach(b=>b.textContent='\u25b6');
  $$('.coach').forEach(e=>{
    e.querySelector('.coach-count').textContent='\u2014';
    e.querySelector('.coach-now').textContent='';
    e.querySelector('.coach-next').textContent='';
  });
  $$('.hit-now').forEach(e=>e.classList.remove('hit-now'));
}

function playLesson(card, seq, pct){
  const same = S.lessonRun && S.lessonRun.card===card;
  stopLesson();
  if(same) return;                                   // toggle off
  pause();
  ensureCtx(); S.ctx.resume();
  const bpm=Math.max(30, S.score.tempo*pct);
  const spb=60/bpm, bpb=S.score.beatsPerBar, barLen=bpb*spb;
  let t0=S.ctx.currentTime+0.12;
  for(let i=0;i<bpb;i++) hit(i?'click':'click1', t0+i*spb, 1);   // count-in
  t0+=bpb*spb;
  card.classList.add('playing');
  card.querySelector('.step-play').textContent='\u25a0';
  if(S.sayIt) loadVoices();
  seq.forEach(sq=>{ sq.events=barEvents(sq.hits); });
  const coach=card.querySelector('.coach');
  if(coach){
    coach.querySelector('.coach-count').textContent='\u2026';
    coach.querySelector('.coach-now').textContent='count-in\u2026';
    coach.querySelector('.coach-next').textContent='';
  }
  const run={card, seq, t0, barLen, spb, done:0, coach, flashKey:null,
             voiceSrcs:[], voiceUntil:0};
  S.lessonRun=run;
  S.lessonTimer=setInterval(()=>{
    const ahead=S.ctx.currentTime+0.5;
    while(run.t0+run.done*barLen < ahead){
      const bar=seq[run.done%seq.length];
      const start=run.t0+run.done*barLen;
      for(let b2=0;b2<bpb;b2++) hit(b2?'click':'click1', start+b2*spb, 0.45);
      for(const h of bar.hits)
        hit(h.inst, start+(h.tick/PPQ)*spb,
            h.accent ? 1.0 : Math.max(0.75, h.velocity??0.8));
      if(S.sayIt && S.voiceBuf && bar.events){
        for(const e of bar.events){
          const buf=S.voiceBuf[(e.word||'').replace(' ','')];
          if(!buf) continue;
          const tHit=start+(e.tick/PPQ)*spb;
          const t=tHit-Math.min(buf.duration+0.05, 0.8*spb);
          if(t < (run.voiceUntil||0)+0.05) continue;  // drop, never drag
          sayWord(e.word, t);
          run.voiceUntil=t+buf.duration;
        }
      }
      run.done++;
    }
  }, 110);
}

function _coach(run, sq, barIdx, frac, box){
  if(!run.coach || !sq.events) return;
  const tpb=S.score.ticksPerBar;
  const tickNow=frac*tpb;
  const ev=sq.events;
  // the hit that just fired (or is firing), and the one to prepare for
  let now=null, next=null;
  for(const e of ev){
    if(e.tick<=tickNow+2) now=e;
    else if(!next) next=e;
  }
  if(!next){
    const nsq=run.seq[(barIdx+1)%run.seq.length];
    if(nsq && nsq.events && nsq.events.length) next=nsq.events[0];
  }
  run.coach.querySelector('.coach-count').textContent=
    countLabel(Math.round(tickNow/(PPQ/2))*(PPQ/2)%tpb);
  run.coach.querySelector('.coach-now').textContent= now ? now.label : '';
  run.coach.querySelector('.coach-next').textContent=
    next ? `next \u2014 ${next.label.toLowerCase()}  (on ${beatName(next.tick)})` : '';
  // flash the engraved notes as they sound
  const key=barIdx+':'+(now?now.tick:'-');
  if(key!==run.flashKey){
    run.flashKey=key;
    $$('.hit-now').forEach(e=>e.classList.remove('hit-now'));
    if(now && sq.geo && sq.geo.noteEls){
      (sq.geo.noteEls.get(now.tick)||[]).forEach(g=>g.classList.add('hit-now'));
    }
  }
}

function lessonPlayhead(){
  const run=S.lessonRun;
  if(!run || !S.ctx) return;
  const el=run.card;
  const boxes=[...el.querySelectorAll('.step-bar')];
  if(!boxes.length) return;
  const elapsed=S.ctx.currentTime-run.t0;
  boxes.forEach(b2=>{ const p=b2.querySelector('.playhead'); if(p) p.style.opacity=0; });
  if(elapsed<0) return;                              // still counting in
  const barIdx=Math.floor(elapsed/run.barLen)%run.seq.length;
  const frac=(elapsed%run.barLen)/run.barLen;
  const sq=run.seq[Math.min(barIdx, run.seq.length-1)];
  const box=boxes[Math.min(sq.box??barIdx, boxes.length-1)];
  _coach(run, sq, barIdx, (elapsed%run.barLen)/run.barLen, box);
  const geo=sq.geo||{x0:30,x1:300};
  const ph=box.querySelector('.playhead');
  if(!ph) return;
  ph.style.opacity=1;
  const tpb=S.score.ticksPerBar;
  const tick=frac*tpb;
  const pts=[...(geo.marks||[])];
  if(!pts.length || pts[0].tick>0) pts.unshift({tick:0, x:geo.x0});
  pts.push({tick:tpb, x:geo.x1});
  let x=pts[pts.length-1].x;
  for(let i2=0;i2<pts.length-1;i2++){
    if(tick>=pts[i2].tick && tick<pts[i2+1].tick){
      const f=(tick-pts[i2].tick)/Math.max(1e-6, pts[i2+1].tick-pts[i2].tick);
      x=pts[i2].x + f*(pts[i2+1].x-pts[i2].x);
      break;
    }
  }
  ph.style.left=x+'px';
}

function beatName(tick){
  const beat=Math.floor(tick/PPQ)+1;
  return tick%PPQ===0 ? String(beat) : 'the & of '+beat;
}
function listBeats(ticks){
  const parts=ticks.map(beatName);
  return parts.length<=1 ? parts.join('') :
    parts.slice(0,-1).join(', ')+' and '+parts[parts.length-1];
}

function handNames(){
  const hat = S.hatHand==='right' ? 'Right' : 'Left';
  const snare = S.hatHand==='right' ? 'Left' : 'Right';
  return {hat:hat+' hand', snare:snare+' hand'};
}

function grooveTips(core){
  const H=handNames();
  const tips=[];
  const bpm=Math.round(S.score.tempo);
  tips.push(`Full speed is <b>\u2669 = ${bpm}</b>. Start around 65% with the speed slider and only speed up when a loop feels easy.`);
  const kicks=core.hits.filter(h=>h.cls==='kick').map(h=>h.tick);
  const snares=core.hits.filter(h=>h.cls==='snare').map(h=>h.tick);
  const cyms=core.hits.filter(h=>h.cls==='cym').map(h=>h.tick);
  if(cyms.length){
    const off=cyms.every(t=>t%PPQ!==0);
    if(off) tips.push(`${H.hat} (hi-hat): play the <b>offbeats only</b> (every "&") \u2014 count "1 <b>&</b> 2 <b>&</b>" and play on the &s.`);
    else if(cyms.length>=S.score.beatsPerBar*2-1) tips.push(`${H.hat} (hi-hat): <b>steady 8th notes</b> \u2014 "1 & 2 & 3 & 4 &", no gaps.`);
    else tips.push(`${H.hat} (hi-hat): <b>${listBeats(cyms)}</b>.`);
  }
  if(snares.length){
    const backbeat = snares.length===2 && snares.every(t=>t===PPQ||t===3*PPQ);
    tips.push(backbeat
      ? `${H.snare} (snare): <b>2 and 4</b> \u2014 the classic backbeat. Land it with confidence; it drives the song.`
      : `${H.snare} (snare): <b>${listBeats(snares)}</b>.`);
  }
  if(kicks.length) tips.push(`Kick foot: <b>${listBeats(kicks)}</b>. Say it out loud while you play the hats \u2014 the mouth teaches the foot.`);
  const opens=S.score.bars.reduce((a,b)=>a+b.hits.filter(h=>h.inst==='openhh').length,0);
  if(opens>3) tips.push(`Notes with a <b>\u2297</b> (circled x) are open hi-hats \u2014 loosen your hi-hat foot there, close it on the next hit.`);
  const ghosts=S.score.bars.reduce((a,b)=>a+b.hits.filter(h=>h.ghost).length,0);
  if(ghosts>3) tips.push(`Notes in <b>(parentheses)</b> are ghost notes \u2014 tiny taps, felt more than heard. Ignore them until the groove is solid.`);
  const fills=S.score.bars.filter(b=>b.hits.some(h=>(CLASS_MAP[h.inst]||'')==='tom')).map(b=>b.number);
  if(fills.length) tips.push(`Fills live in bars <b>${fills.slice(0,6).join(', ')}${fills.length>6?'\u2026':''}</b>. Learn the groove first; add fills last.`);
  tips.push(`Loop one bar (click it; shift-click extends), turn on <b>Click</b>, and stay with a loop until you can hold it for a minute without thinking.`);
  tips.push(`Solid? Go drum with a band: <a href="https://drummate.app/?utm_source=chart&utm_medium=app&utm_campaign=teach-tip" target="_blank" rel="noopener"><b>DrumMate</b></a> is a live band for your e-kit that follows you.`);
  return tips;
}

function findFill(){
  let best=null;
  for(const b of S.score.bars){
    if(b.empty) continue;
    const toms=b.hits.filter(h=>(CLASS_MAP[h.inst]||'')==='tom').length;
    if(!toms) continue;
    const score=toms*4+b.hits.length;
    if(!best || score>best.score) best={score, bar:b};
  }
  return best && best.bar;
}

function addStepCard(container, name, hint, displayBars, seq, opts={}){
  const step=document.createElement('div');
  step.className='lesson-step';
  const nm=document.createElement('div'); nm.className='step-name'; nm.textContent=name;
  step.appendChild(nm);
  const row=document.createElement('div'); row.className='step-bars';
  step.appendChild(row);
  container.appendChild(step);      // must be in the DOM before VexFlow draws:
  try{                              // getSVGElement resolves via getElementById
    displayBars.forEach((d,i)=>{
      const box=document.createElement('div');
      if(d.tag){
        const tag=document.createElement('div'); tag.className='bar-tag';
        tag.textContent=d.tag; box.appendChild(tag);
      }
      row.appendChild(box);
      const geo=renderBarInto(box, d.hits, d.width||330, d.counts!==false, d.subdivision||2);
      // remember geometry on every seq entry that displays in this box
      seq.forEach((sq,si)=>{ if((sq.box??si)===i) sq.geo=geo; });
    });
  }catch(_){ step.remove(); return; }
  const ctl=document.createElement('div'); ctl.className='step-controls';
  const play=document.createElement('button'); play.className='step-play'; play.textContent='\u25b6';
  const sel=document.createElement('select');
  for(const [v,l] of [[0.5,'50%'],[0.7,'70%'],[0.85,'85%'],[1,'100%']]){
    const o=document.createElement('option'); o.value=v; o.textContent=l;
    if(v===0.7) o.selected=true;
    sel.appendChild(o);
  }
  ctl.append(play, sel);
  const ht=document.createElement('div'); ht.className='step-hint'; ht.textContent=hint;
  const coach=document.createElement('div');
  coach.className='coach';
  coach.innerHTML='<div class="coach-count">\u2014</div>'+
    '<div class="coach-now"></div><div class="coach-next"></div>';
  step.append(ctl, ht, coach);
  const wsum=displayBars.reduce((a,d)=>a+(d.width||330),0)+8*(displayBars.length-1);
  step.style.width=(wsum+24)+'px';
  play.onclick=()=>playLesson(step, seq, Number(sel.value));
  sel.onchange=()=>{ if(S.lessonRun && S.lessonRun.card===step) playLesson(step, seq, Number(sel.value)); };
}

function renderLesson(){
  const host=$('#lesson');
  if(!host) return;
  host.classList.toggle('hidden', !S.teach);
  stopLesson();
  if(!S.teach || !S.score) return;
  host.innerHTML='';
  const core=analyzeGroove();
  if(!core){ host.innerHTML='<h3>Teach</h3><p>No repeating groove found \u2014 use Simple mode and loop short sections.</p>'; return; }

  const h3=document.createElement('h3');
  h3.textContent='The groove to learn';
  const cov=document.createElement('div');
  cov.className='coverage';
  cov.textContent=`this one pattern is ${(100*core.share).toFixed(0)}% of the song \u2014 e.g. bars ${core.bars.slice(0,4).join(', ')} \u00b7 press \u25b6 on a card to hear and loop it (with count-in + click)`;
  const handRow=document.createElement('div');
  handRow.className='hand-pick';
  handRow.innerHTML=`hi-hats with: <select id="hat-hand">
    <option value="right">right hand (crossed \u2014 standard)</option>
    <option value="left">left hand (open-handed)</option></select>
    <label class="check say-check"><input id="say-hits" type="checkbox">
      \ud83d\udd0a call the hits out loud</label>`;
  host.append(h3, cov, handRow);
  const sh=handRow.querySelector('#say-hits');
  sh.checked=S.sayIt;
  sh.onchange=()=>{ S.sayIt=sh.checked;
    try{ localStorage.setItem('dm-say-hits', sh.checked?'1':'0'); }catch(_){}
    if(sh.checked){ ensureCtx(); S.ctx.resume(); loadVoices(); sayWord('kick'); } };
  const hh=handRow.querySelector('#hat-hand');
  hh.value=S.hatHand;
  hh.onchange=()=>{ S.hatHand=hh.value;
    try{ localStorage.setItem('dm-hats-hand', hh.value); }catch(_){}
    renderLesson(); };

  const steps=document.createElement('div');
  steps.className='lesson-steps';
  host.appendChild(steps);

  const hats=core.hits.filter(h=>h.cls==='cym');
  const hands=core.hits.filter(h=>h.cls!=='kick');
  const H=handNames();
  const layers=[
    [`Step 1 \u00b7 ${H.hat.toLowerCase()} \u2014 hi-hats`, hats, 'count out loud with the click'],
    ['Step 2 \u00b7 add the snare', hands, 'hands only \u2014 no feet yet'],
    ['Step 3 \u00b7 the full groove', core.hits, 'loop until it plays itself'],
  ];
  let prevLen=-1;
  for(const [name,hits,hint] of layers){
    if(!hits.length || hits.length===prevLen) continue;
    prevLen=hits.length;
    addStepCard(steps, name, hint, [{hits}], [{hits}]);
  }

  const fill=findFill();
  if(fill){
    const fhits=fill.hits.map(h=>({...h}));
    addStepCard(steps, `The fill (bar ${fill.number})`,
      'slower is fine \u2014 accuracy first',
      [{hits:fhits, subdivision:fill.subdivision, counts:true}],
      [{hits:fhits}]);
    addStepCard(steps, 'Groove \u00d73 + fill',
      'how it sits in the song \u2014 fill lands every 4th bar',
      [{hits:core.hits, tag:'groove \u00d73', width:300},
       {hits:fhits, subdivision:fill.subdivision, tag:'then the fill', width:300, counts:false}],
      [{hits:core.hits, box:0},{hits:core.hits, box:0},{hits:core.hits, box:0},{hits:fhits, box:1}]);
  }

  const ul=document.createElement('ul');
  ul.className='lesson-tips';
  for(const t of grooveTips(core)){
    const li=document.createElement('li'); li.innerHTML=t; ul.appendChild(li);
  }
  host.appendChild(ul);
}

/* ── rendering ────────────────────────────────────────────────────────── */

function openScore(score){
  S.score=score;
  S.cursorBar=-1; S.loopFrom=S.loopTo=null; updateLoopLabel();
  $('#s-title').textContent=score.title;
  $('#s-tempo').textContent=`♩ = ${Math.round(score.tempo)}`;
  if($('#tempo-in')) $('#tempo-in').value=score.tempo.toFixed(1);
  $('#s-meter').textContent=score.timeSignature;
  $('#s-stale')?.classList.toggle('hidden', (score.engine||1) >= ENGINE_CURRENT);
  $('#s-swing').textContent = score.swing ? 'Swing 8ths' : 'Straight';
  $('#s-kit').textContent = (score.kit||[]).map(i=>DRUMS[i]?.label||i).join(' \u00b7 ');
  const src=$('#s-source');
  if(score.source){ src.href=score.source; src.classList.remove('hidden'); }
  else src.classList.add('hidden');

  $('#dl-ch').href=`/api/jobs/${S.jobId}/clonehero`;
  $('#dl-midi').href=`/api/jobs/${S.jobId}/files/drums.mid`;
  $('#dl-xml').href=`/api/jobs/${S.jobId}/files/drums.musicxml`;
  // audio downloads only for charts made from the user's own upload (the
  // server refuses them otherwise); never the original mix
  const safe=(S.score.title||'song').replace(/[^\w \-()]+/g,'').trim().slice(0,60)||'song';
  const own=!!(S.jobInfo && S.jobInfo.userAudio); const audio=S.score.audio||{};
  for(const [id,file,label] of [['#dl-drums','drums.mp3','drums only'],['#dl-drumless','backing.mp3','drumless']]){
    const a=$(id); if(!a) continue;
    const have=own && Object.values(audio).includes(file);
    a.classList.toggle('hidden', !have);
    a.href=`/api/jobs/${S.jobId}/files/${file}?dl=${encodeURIComponent(safe+' - '+label+'.mp3')}`;
    a.setAttribute('download', safe+' - '+label+'.mp3');
  }

  const have=score.audio||{};
  setSegEnabled('drums', !!have.drums);
  setSegEnabled('backing', !!have.backing);
  setSegEnabled('youtube', !!score.videoId);
  selectSource(have.drums ? 'drums' : (score.videoId ? 'youtube' : 'click'));

  buildPalette();
  showView('score');
  renderScore();
  requestAnimationFrame(tick);
}

/* Bars are priced by how many symbols they must hold, then packed into
 * systems that fit the page. A dense fill therefore gets the room it needs
 * instead of being crushed into a quarter of the width. */
const CLEF_W = 62, SYS_TOP = 54, SYS_H = 196;

function detailThresholds(){
  if(S._thrFor===S.score && S._thrDetail===S.detail) return S._thr;
  const by={};
  for(const b of S.score.bars)
    for(const h of b.hits) (by[h.inst]=by[h.inst]||[]).push(h.velocity||0);
  const thr={}, q=(100-S.detail)/100;
  for(const k in by){
    const a=by[k].sort((x,y)=>x-y);
    thr[k]= q<=0 ? -1 : a[Math.min(a.length-1, Math.floor(q*a.length))];
  }
  S._thr=thr; S._thrFor=S.score; S._thrDetail=S.detail;
  return thr;
}

/* The bars actually drawn: at full detail the originals, otherwise shallow
 * copies with hits hidden or simplified. `_orig` keeps edits pointing home. */
const SIMPLE_MAP={openhh:'hihat', ride:'hihat', tom_hi:'tom_mid', tom_low:'tom_mid'};
const SIMPLE_DROP=new Set(['perc']);

function visibleBars(){
  const bars=S.score.bars;
  if(S.detail>=100 && !S.simple){ bars.forEach(b=>b._orig=b); return bars; }
  const thr = S.detail<100 ? detailThresholds() : null;
  return bars.map(b=>{
    let hits=thr ? b.hits.filter(h=>(h.velocity||0)>=(thr[h.inst]??-1)) : b.hits.slice();
    const c={...b, _orig:b};
    if(S.simple){
      /* Kick / snare / hats on an 8th-note grid, no ornaments: the groove a
       * beginner learns first. A crash survives only on a bar's downbeat. */
      const slot=PPQ/2, merged=new Map();
      for(const h of hits){
        if(h.ghost || SIMPLE_DROP.has(h.inst)) continue;
        let inst = h.inst==='crash' ? (h.tick<slot ? 'crash' : 'hihat')
                                    : (SIMPLE_MAP[h.inst]||h.inst);
        const tick=Math.min(S.score.ticksPerBar-slot, Math.round(h.tick/slot)*slot);
        const key=inst+':'+tick, prev=merged.get(key);
        if(!prev || (h.velocity||0)>(prev.velocity||0))
          merged.set(key, {...h, inst, tick, accent:false, ghost:false, flam:false});
      }
      hits=[...merged.values()].sort((a,b2)=>a.tick-b2.tick||DRUMS[a.inst].order-DRUMS[b2.inst].order);
      c.subdivision=2; c.grid='8th';
    }
    c.hits=hits; c.empty=hits.length===0;
    return c;
  });
}

/* ── legend: what each symbol means ─────────────────────────────────── */
function legendCell(container, label, build){
  const cell=document.createElement('div'); cell.className='key-cell';
  const box=document.createElement('div');
  const lab=document.createElement('div'); lab.className='key-label'; lab.textContent=label;
  cell.append(box, lab);
  container.appendChild(cell);
  try{
    const r=new VF.Renderer(box, VF.Renderer.Backends.SVG);
    r.resize(112, 78);
    const ctx=r.getContext();
    const stave=new VF.Stave(2, 0, 106, {num_lines:5});
    stave.setContext(ctx).draw();
    const res=build();
    const notes=Array.isArray(res)?res:res.notes;
    VF.Formatter.FormatAndDraw(ctx, stave, notes);
    if(res.post) res.post.forEach(d=>d.setContext(ctx).draw());
  }catch(_){ cell.remove(); }
}

function mkKeyNote(inst, mods){
  const d=DRUMS[inst];
  const n=new VF.StaveNote({keys:[d.key], duration:'q', clef:'percussion',
    stem_direction: d.voice==='up'?VF.Stem.UP:VF.Stem.DOWN});
  (mods||[]).forEach(m=>m(n));
  return [n];
}

function renderLegend(host){
  const div=document.createElement('div');
  div.className='legend';
  host.appendChild(div);
  const kit=new Set([...(S.score.kit||[]), 'kick','snare','hihat']);
  for(const inst of INSTS){
    if(!kit.has(inst)) continue;
    legendCell(div, DRUMS[inst].label, ()=>mkKeyNote(inst, []));
  }
  legendCell(div, 'Accent', ()=>mkKeyNote('snare',
    [n=>n.addModifier(new VF.Articulation('a>').setPosition(VF.Modifier.Position.ABOVE),0)]));
  legendCell(div, 'Ghost note', ()=>mkKeyNote('snare',
    [n=>VF.Parenthesis.buildAndAttach([n])]));
  legendCell(div, 'Flam', ()=>mkKeyNote('snare', [n=>{
    const gn=new VF.GraceNote({keys:['c/5'], duration:'8', slash:true, clef:'percussion'});
    n.addModifier(new VF.GraceNoteGroup([gn], false), 0);
  }]));

  /* rhythm symbols */
  const hat=(dur)=>new VF.StaveNote({keys:[DRUMS.hihat.key], duration:dur,
    clef:'percussion', stem_direction:VF.Stem.UP});
  legendCell(div, 'Hit together', ()=>[new VF.StaveNote({
    keys:[DRUMS.snare.key, DRUMS.hihat.key], duration:'q',
    clef:'percussion', stem_direction:VF.Stem.UP})]);
  legendCell(div, '8ths - 1 beam', ()=>{
    const notes=[hat('8'),hat('8')];
    return {notes, post:[new VF.Beam(notes)]};
  });
  legendCell(div, '16ths - 2 beams', ()=>{
    const notes=[hat('16'),hat('16'),hat('16'),hat('16')];
    return {notes, post:[new VF.Beam(notes)]};
  });
  legendCell(div, 'Triplet', ()=>{
    const notes=[hat('8'),hat('8'),hat('8')];
    return {notes, post:[new VF.Beam(notes),
      new VF.Tuplet(notes,{num_notes:3, notes_occupied:2, bracketed:true, ratioed:false})]};
  });
  legendCell(div, 'Rests: \u00bc + \u215b beat', ()=>[
    new VF.StaveNote({keys:['b/4'], duration:'qr', clef:'percussion'}),
    new VF.StaveNote({keys:['b/4'], duration:'8r', clef:'percussion'})]);
  legendCell(div, 'Silent bar', ()=>[
    new VF.StaveNote({keys:['b/4'], duration:'wr', clef:'percussion'})]);
}

function renderScore(){
  const host=$('#score');
  host.innerHTML=''; S.systems=[]; S._lastDrawnBar=null;
  renderLesson();
  if($('#opt-key') ? $('#opt-key').checked : true) renderLegend(host);
  const bpb=S.score.beatsPerBar;
  const width=Math.max(340, host.clientWidth-16);

  const priced=visibleBars().map(bar=>{
    const laid=layoutBar(bar, barBeats(bar));
    const n=Math.max(laid.up.length, laid.down.length);
    return {bar, laid, need:Math.max(148, 56+24*n)};   // dense bars get real air
  });

  const rows=[]; let cur=[], curW=0;
  for(const p of priced){
    const lead=(rows.length===0 && cur.length===0) ? CLEF_W : 0;
    if(cur.length && curW+p.need+lead > width-14){ rows.push(cur); cur=[]; curW=0; }
    cur.push(p); curW+=p.need;
  }
  if(cur.length) rows.push(cur);

  rows.forEach((row,i)=>{
    const div=document.createElement('div');
    div.className='system';
    host.appendChild(div);
    drawSystem(div, row, width, i===0);
  });
  applyCursor(false);
  drawLoopRange();
  $('#bar-count').textContent=`${priced.length} bars`;
}

function drawSystem(div, row, width, isFirst){
  const renderer=new VF.Renderer(div, VF.Renderer.Backends.SVG);
  renderer.resize(width, SYS_H);
  const ctx=renderer.getContext();
  ctx.setFont('sans-serif', 10);

  const lead = isFirst ? CLEF_W : 0;
  const total = row.reduce((a,p)=>a+p.need, 0);
  const usable = width - 14 - lead;
  const scale = usable / total;             // fill the line, keep proportions
  const geo=[];
  let x=8;

  row.forEach((p,bi)=>{
    const w = p.need*scale + (bi===0 ? lead : 0);
    const stave=new VF.Stave(x, SYS_TOP, w, {num_lines:5});
    if(isFirst && bi===0){
      stave.addClef('percussion');
      stave.addTimeSignature(p.bar.timeSignature||S.score.timeSignature);
    } else {
      // a bar of a different length (a 2/4 turnaround) announces itself
      const prevBar=S._lastDrawnBar;
      if(prevBar && barBeats(p.bar)!==barBeats(prevBar)) stave.addTimeSignature(p.bar.timeSignature||`${barBeats(p.bar)}/4`);
    }
    S._lastDrawnBar=p.bar;
    stave.setMeasure(p.bar.number);
    stave.setContext(ctx).draw();

    const voices=[];
    for(const [elems,dir] of [[p.laid.up,'up'],[p.laid.down,'down']]){
      if(dir==='down' && !elems.some(e=>e.type==='note')) continue;
      const notes=buildNotes(elems, dir);
      if(!notes.length) continue;
      const v=new VF.Voice({num_beats:barBeats(p.bar), beat_value:4});
      v.setStrict(false);
      v.addTickables(notes.map(n=>n.vf));
      voices.push({voice:v, notes, dir});
    }
    if(voices.length){
      const fmt=new VF.Formatter();
      fmt.joinVoices(voices.map(v=>v.voice));
      fmt.formatToStave(voices.map(v=>v.voice), stave);
    }
    const tuplets=[], beams=[];
    for(const {notes,dir} of voices){
      collectTuplets(notes).forEach(t=>tuplets.push(t));
      VF.Beam.generateBeams(notes.filter(n=>n.el.type==='note').map(n=>n.vf), {
        groups:[new VF.Fraction(1,4)],
        stem_direction: dir==='up'?VF.Stem.UP:VF.Stem.DOWN,
      }).forEach(b=>beams.push(b));
    }
    for(const {voice} of voices) voice.draw(ctx, stave);
    beams.forEach(b=>b.setContext(ctx).draw());
    tuplets.forEach(t=>t.setContext(ctx).draw());

    geo.push({index:p.bar.index, bar:p.bar, x, y:SYS_TOP-30, w, h:104,
              noteX0:stave.getNoteStartX(), noteX1:x+w-6});
    x+=w;
  });

  const svg=div.querySelector('svg');
  svg.style.cursor='pointer';
  div.addEventListener('click', (ev)=>onSystemClick(ev, div, geo));
  S.systems.push({el:div, svg, bars:geo});
}

function buildNotes(elems, dir){
  const out=[];
  for(const e of elems){
    let vf;
    if(e.type==='rest'){
      const key = dir==='up' ? 'b/4' : 'd/4';
      vf=new VF.StaveNote({keys:[key], duration:e.dur+'r', clef:'percussion'});
      if(e.hidden) vf.setStyle({fillStyle:'none', strokeStyle:'none'});
    }else{
      vf=new VF.StaveNote({
        keys:e.keys, duration:e.dur, clef:'percussion',
        stem_direction: dir==='up'?VF.Stem.UP:VF.Stem.DOWN,
      });
      if(e.accent) vf.addModifier(new VF.Articulation('a>').setPosition(
        dir==='up'?VF.Modifier.Position.ABOVE:VF.Modifier.Position.BELOW), 0);
      if(e.ghost){
        try{ VF.Parenthesis.buildAndAttach([vf]); }catch(_){}
      }
      if(e.flam){
        try{
          const gn=new VF.GraceNote({keys:[e.keys[0]], duration:'8', slash:true,
                                     clef:'percussion'});
          vf.addModifier(new VF.GraceNoteGroup([gn], false).beamNotes(), 0);
        }catch(_){}
      }
    }
    if(e.dots) { try{ VF.Dot.buildAndAttach([vf], {all:true}); }catch(_){} }
    out.push({vf, el:e});
  }
  return out;
}

function collectTuplets(notes){
  const out=[]; let i=0;
  while(i<notes.length){
    const t=notes[i].el.tuplet;
    if(!t){ i++; continue; }
    let j=i;
    while(j+1<notes.length && notes[j+1].el.tuplet &&
          notes[j+1].el.tuplet.num===t.num &&
          notes[j+1].el.beat===notes[i].el.beat) j++;
    if(j>i){
      try{
        out.push(new VF.Tuplet(notes.slice(i,j+1).map(n=>n.vf),
          {num_notes:t.num, notes_occupied:t.den, bracketed:true, ratioed:false}));
      }catch(_){}
    }
    i=j+1;
  }
  return out;
}

/* ── playback ─────────────────────────────────────────────────────────── */
function duration(){
  const b=S.score?.bars; return b&&b.length ? b[b.length-1].endTime : 0;
}
function setSegEnabled(src, on){
  const b=$(`.seg[data-src="${src}"]`); if(b) b.disabled=!on;
}
function selectSource(src){
  if($(`.seg[data-src="${src}"]`)?.disabled) src='click';
  const at=position();
  pause();
  S.source=src;
  $$('.seg').forEach(b=>b.classList.toggle('active', b.dataset.src===src));
  const wrap=$('#yt-wrap'), audio=$('#audio');
  wrap.classList.toggle('hidden', src!=='youtube');
  if(src==='youtube'){ ensureYT(); }
  else if(src==='drums'||src==='backing'){
    const want=`/api/jobs/${S.jobId}/files/${src}.mp3`;
    if(!audio.src.endsWith(`${src}.mp3`)) audio.src=want;
    audio.playbackRate=S.speed;
  }
  seek(at||0);
}

function position(){
  if(!S.score) return 0;
  if(S.source==='youtube')
    return S.yt&&S.ytReady ? Math.max(0,S.yt.getCurrentTime()-S.score.offset) : 0;
  if(S.source==='click')
    return S.playing ? S.playFrom+(S.ctx.currentTime-S.playStarted)*S.speed : S.playFrom;
  const a=$('#audio');
  const ct=a.currentTime;
  /* Safari/iPad only refresh currentTime every ~250 ms - interpolate between
     updates or the cursor trails the sound by up to half a bar. */
  const now=performance.now();
  if(!S.playing || a.paused){ S._ctBase=ct; S._ctPerf=now; return ct; }
  if(ct!==S._ctBase){ S._ctBase=ct; S._ctPerf=now; return ct; }
  return ct + Math.min(0.35, (now-(S._ctPerf||now))/1000) * (a.playbackRate||1);
}
function seek(t){
  t=clamp(t,0,Math.max(0,duration()));
  if(S.source==='youtube'){ if(S.yt&&S.ytReady) S.yt.seekTo(t+S.score.offset,true); }
  else if(S.source==='click'){
    S.playFrom=t;
    if(S.playing){ resetSched(t); S.playStarted=S.ctx.currentTime; }
  }else{ $('#audio').currentTime=t; }
}
function play(){
  stopLesson();
  if(S.source!=='click' && $('#opt-hear')&&$('#opt-hear').checked){ ensureCtx(); S.ctx.resume(); resetSched(position()); }
  if(S.source==='youtube'){ if(S.yt&&S.ytReady) S.yt.playVideo(); }
  else if(S.source==='click'){
    ensureCtx(); S.ctx.resume();
    resetSched(S.playFrom); S.playStarted=S.ctx.currentTime;
  }else{
    const a=$('#audio');
    if(a.error){ const src=a.src, t=S.playFrom||0; a.src=''; a.src=src; a.currentTime=t; }
    a.play().catch(()=>{});
  }
  S.playing=true; $('#btn-play').textContent='❚❚';
}
function pause(){
  if(S.source==='youtube'){ if(S.yt&&S.ytReady) S.yt.pauseVideo(); }
  else if(S.source==='click'){ S.playFrom=position(); }
  else $('#audio').pause();
  S.playing=false; $('#btn-play').textContent='▶';
}
function togglePlay(){ S.playing?pause():play(); }

function setSpeed(v){
  S.speed=v;
  $('#speed-out').textContent=v.toFixed(2)+'×';
  $('#audio').playbackRate=v;
  if(S.yt&&S.ytReady){ try{ S.yt.setPlaybackRate(v); }catch(_){} }
}

/* YouTube iframe player, loaded lazily */
function ensureYT(){
  if(S.yt||!S.score?.videoId) return;
  if(!window.YT){
    const tag=document.createElement('script');
    tag.src='https://www.youtube.com/iframe_api';
    document.head.appendChild(tag);
    window.onYouTubeIframeAPIReady=()=>makeYT();
  }else makeYT();
}
function makeYT(){
  if(S.yt) return;
  S.yt=new YT.Player('yt-player',{
    videoId:S.score.videoId, playerVars:{controls:1, rel:0, modestbranding:1},
    events:{
      onReady:()=>{ S.ytReady=true; try{S.yt.setPlaybackRate(S.speed);}catch(_){} },
      onStateChange:(e)=>{
        if(e.data===YT.PlayerState.PLAYING){ S.playing=true; $('#btn-play').textContent='❚❚'; }
        if(e.data===YT.PlayerState.PAUSED||e.data===YT.PlayerState.ENDED){
          S.playing=false; $('#btn-play').textContent='▶';
        }
      },
    },
  });
}

/* ── the synth (notation playback + metronome) ────────────────────────── */
function ensureCtx(){
  if(!S.ctx) S.ctx=new (window.AudioContext||window.webkitAudioContext)();
  if(!S.noise){
    const n=S.ctx.sampleRate*2, buf=S.ctx.createBuffer(1,n,S.ctx.sampleRate);
    const d=buf.getChannelData(0);
    for(let i=0;i<n;i++) d[i]=Math.random()*2-1;
    S.noise=buf;
  }
  return S.ctx;
}
function noiseSrc(){ const s=S.ctx.createBufferSource(); s.buffer=S.noise; s.loop=true; return s; }
function envGain(t, peak, decay){
  const g=S.ctx.createGain();
  g.gain.setValueAtTime(0,t);
  g.gain.linearRampToValueAtTime(peak,t+0.002);
  g.gain.exponentialRampToValueAtTime(0.0005,t+decay);
  return g;
}
function hit(inst, t, vel=0.8){
  const c=ensureCtx(), out=c.destination, v=clamp(vel,0.1,1);
  const tone=(f0,f1,dec,gain,type='sine')=>{
    const o=c.createOscillator(); o.type=type;
    o.frequency.setValueAtTime(f0,t);
    o.frequency.exponentialRampToValueAtTime(Math.max(20,f1),t+dec*0.8);
    const g=envGain(t,gain*v,dec); o.connect(g).connect(out);
    o.start(t); o.stop(t+dec+0.02);
  };
  const noise=(hp,dec,gain,q=0.7)=>{
    const s=noiseSrc(), f=c.createBiquadFilter();
    f.type='highpass'; f.frequency.value=hp; f.Q.value=q;
    const g=envGain(t,gain*v,dec);
    s.connect(f).connect(g).connect(out);
    s.start(t); s.stop(t+dec+0.02);
  };
  switch(inst){
    case 'kick':    tone(125,42,0.22,0.9); noise(90,0.02,0.10); break;
    case 'snare':   noise(1500,0.16,0.42,0.6); tone(196,150,0.10,0.22,'triangle'); break;
    case 'hihat':   noise(6000,0.05,0.42,1.1); break;
    case 'openhh':  noise(5500,0.32,0.34); break;
    case 'ride':    noise(6200,0.55,0.10); tone(560,540,0.28,0.06,'triangle'); break;
    case 'crash':   noise(3200,1.30,0.20); break;
    case 'tom_hi':  tone(280,190,0.30,0.55); break;
    case 'tom_mid': tone(200,140,0.34,0.55); break;
    case 'tom_low': tone(140,95,0.42,0.60); break;
    case 'hhfoot':  noise(6000,0.05,0.14); break;
    case 'perc':    tone(1800,1200,0.06,0.35,'square'); noise(3000,0.03,0.12); break;
    case 'click':   tone(1400,1400,0.03,0.25,'square'); break;
    case 'click1':  tone(2100,2100,0.04,0.34,'square'); break;
  }
}
function flatHits(){
  if(S._flat && S._flatFor===S.score) return S._flat;
  const list=[];
  for(const b of S.score.bars)
    for(const h of b.hits)
      list.push({t:h.time, inst:h.inst, v:h.ghost?0.3:(h.accent?1:0.75)});
  list.sort((a,b)=>a.t-b.t);
  S._flat=list; S._flatFor=S.score;
  return list;
}
function beatTimes(){
  if(S._beats && S._beatsFor===S.score) return S._beats;
  const out=[];
  for(const b of S.score.bars){
    const n=barBeats(b);
    const span=(b.endTime-b.startTime)/n;
    for(let k=0;k<n;k++) out.push({t:b.startTime+k*span, down:k===0});
  }
  S._beats=out; S._beatsFor=S.score;
  return out;
}
function resetSched(from){
  S.scheduled=[]; S.synthIdx=0; S.metIdx=0;
  const hits=flatHits(); while(S.synthIdx<hits.length && hits[S.synthIdx].t<from) S.synthIdx++;
  const bts=beatTimes(); while(S.metIdx<bts.length && bts[S.metIdx].t<from) S.metIdx++;
  S.playFrom=from;
}
function pump(pos){
  if(!S.playing) return;
  const c=S.ctx; if(!c) return;
  const AHEAD=0.35;
  const toCtx=(t)=>c.currentTime+(t-pos)/S.speed;
  if(S.source==='click' || ($('#opt-hear')&&$('#opt-hear').checked)){
    const hits=flatHits();
    while(S.synthIdx<hits.length && hits[S.synthIdx].t < pos+AHEAD*S.speed){
      const h=hits[S.synthIdx++];
      if(h.t>=pos-0.05) hit(h.inst, Math.max(c.currentTime, toCtx(h.t)), h.v);
    }
  }
  if($('#opt-met').checked){
    const bts=beatTimes();
    while(S.metIdx<bts.length && bts[S.metIdx].t < pos+AHEAD*S.speed){
      const b=bts[S.metIdx++];
      if(b.t>=pos-0.05) hit(b.down?'click1':'click', Math.max(c.currentTime, toCtx(b.t)), 1);
    }
  }
}

/* ── cursor / loop ────────────────────────────────────────────────────── */
function barAt(t){
  const bars=S.score.bars;
  for(let i=0;i<bars.length;i++)
    if(t>=bars[i].startTime && t<bars[i].endTime) return i;
  return t<bars[0]?.startTime ? -1 : bars.length-1;
}
function applyCursor(force){
  const idx=S.cursorBar;
  const num=S.score.bars[idx]?.number;
  for(const sys of S.systems){
    let box=sys.el.querySelector('.cursor');
    const g=sys.bars.find(b=>b.bar.number===num);
    if(!g){ if(box) box.remove(); continue; }
    if(!box){
      box=document.createElement('div');
      box.className='cursor';
      Object.assign(box.style,{position:'absolute', pointerEvents:'none',
        background:'rgba(255,46,136,.14)', border:'2px solid rgba(255,46,136,.6)',
        borderRadius:'0', transition:'left .12s linear, width .12s linear'});
      sys.el.appendChild(box);
    }
    Object.assign(box.style,{left:g.x+'px', top:(g.y)+'px',
      width:g.w+'px', height:(g.h+14)+'px'});
    if(force && $('#opt-follow').checked && sys.el.scrollIntoView){
      const r=sys.el.getBoundingClientRect();
      if(r.top<70||r.bottom>window.innerHeight-40)
        sys.el.scrollIntoView({block:'center', behavior:'smooth'});
    }
  }
}
function drawLoopRange(){
  $$('.loop-box').forEach(e=>e.remove());
  if(S.loopFrom==null) return;
  const a=S.score.bars[S.loopFrom]?.number, b=S.score.bars[S.loopTo??S.loopFrom]?.number;
  for(const sys of S.systems){
    for(const g of sys.bars){
      if(g.bar.number<a || g.bar.number>b) continue;
      const box=document.createElement('div');
      box.className='loop-box';
      Object.assign(box.style,{position:'absolute', pointerEvents:'none',
        left:g.x+'px', top:(g.y-2)+'px', width:g.w+'px', height:(g.h+18)+'px'});
      sys.el.appendChild(box);
    }
  }
}

function updateLoopLabel(){
  const l=$('#loop-label'), btn=$('#btn-loop');
  if(btn) btn.classList.toggle('armed', S.loopPick>0);
  if(S.loopPick===1){ l.textContent='tap the first bar…'; return; }
  if(S.loopPick===2){ l.textContent='now tap the last bar…'; return; }
  if(S.loopFrom==null){ l.textContent='off'; return; }
  const a=S.score.bars[S.loopFrom]?.number, b=S.score.bars[S.loopTo??S.loopFrom]?.number;
  l.textContent=`bars ${a}–${b}`;
}
function tick(){
  requestAnimationFrame(tick);
  if(!S.score) return;
  lessonPlayhead();
  const pos=position();
  $('#clock').textContent=`${fmtTime(pos)} / ${fmtTime(duration())}`;
  pump(pos);
  if(S.loopFrom!=null && S.playing){
    const end=S.score.bars[S.loopTo??S.loopFrom]?.endTime ?? Infinity;
    const start=S.score.bars[S.loopFrom]?.startTime ?? 0;
    if(pos>=end-0.01 || pos<start-0.25) seek(start);
  }
  const idx=barAt(pos);
  if(idx!==S.cursorBar){ S.cursorBar=idx; applyCursor(true); }
  chartPlayhead(pos, idx);
}

function chartPlayhead(pos, idx){
  const bar=S.score && S.score.bars[idx];
  for(const sys of S.systems){
    let line=sys.el.querySelector('.chart-ph');
    const g=bar && sys.bars.find(b2=>b2.bar.number===bar.number);
    if(!g){ if(line) line.style.opacity=0; continue; }
    if(!line){
      line=document.createElement('div');
      line.className='chart-ph';
      sys.el.appendChild(line);
    }
    const frac=Math.max(0, Math.min(1, (pos-bar.startTime)/Math.max(1e-6, bar.endTime-bar.startTime)));
    line.style.opacity=S.playing?1:0;
    line.style.left=(g.noteX0 + frac*(g.noteX1-g.noteX0))+'px';
    line.style.top=(g.y+4)+'px';
    line.style.height=(g.h+8)+'px';
  }
}

/* ── editing ──────────────────────────────────────────────────────────── */
function buildPalette(){
  const p=$('#palette'); p.innerHTML='';
  for(const inst of INSTS){
    const b=document.createElement('button');
    b.textContent=DRUMS[inst].label;
    b.dataset.inst=inst;
    b.className = inst===S.brush ? 'active':'';
    b.onclick=()=>{ S.brush=inst; buildPalette(); };
    p.appendChild(b);
  }
}
function onSystemClick(ev, div, geo){
  const rect=div.querySelector('svg').getBoundingClientRect();
  const x=ev.clientX-rect.left;
  const g=geo.find(b=>x>=b.x && x<b.x+b.w) || geo[geo.length-1];
  if(!g) return;
  // the drawn bar may be a Detail/Simple copy - loop and edit on the original
  const barIdx=S.score.bars.indexOf(g.bar._orig || g.bar);
  if(barIdx<0) return;

  if(!S.editing && S.loopPick){
    // touch-friendly two-tap loop: first bar, then last bar
    if(S.loopPick===1){ S.loopFrom=barIdx; S.loopTo=barIdx; S.loopPick=2; seek(g.bar.startTime); }
    else {
      if(barIdx < S.loopFrom){ S.loopTo=S.loopFrom; S.loopFrom=barIdx; } else S.loopTo=barIdx;
      S.loopPick=0;
    }
    updateLoopLabel(); drawLoopRange(); return;
  }
  if(!S.editing){
    if((ev.shiftKey || ev.ctrlKey || ev.metaKey) && S.loopFrom!=null){
      ev.preventDefault();
      if(barIdx < S.loopFrom){ S.loopTo=S.loopFrom; S.loopFrom=barIdx; }   // extend backwards too
      else S.loopTo=barIdx;
    } else {
      S.loopFrom=barIdx; S.loopTo=barIdx;
      seek(g.bar.startTime);
    }
    updateLoopLabel();
    drawLoopRange();
    return;
  }

  const ticksPerBar=barTicks(g.bar);
  const slot=PPQ/(g.bar.subdivision||4);
  const frac=clamp((x-g.noteX0)/Math.max(1,(g.noteX1-g.noteX0)),0,0.999);
  const tick=clamp(Math.round(frac*ticksPerBar/slot)*slot, 0, ticksPerBar-slot);

  const target=g.bar._orig || g.bar;      // edits belong to the real bar
  const hits=target.hits;
  const at=hits.findIndex(h=>h.inst===S.brush && Math.abs(h.tick-tick)<slot/2);
  if(at>=0) hits.splice(at,1);
  else{
    const span=g.bar.endTime-g.bar.startTime;
    hits.push({tick, inst:S.brush, velocity:0.8, ghost:false, accent:false,
               flam:false, time:+(g.bar.startTime+span*tick/ticksPerBar).toFixed(4)});
    hits.sort((a,b)=>a.tick-b.tick||DRUMS[a.inst].order-DRUMS[b.inst].order);
  }
  target.empty=hits.length===0;
  S._flat=null; S._flatFor=null; S._thrFor=null;
  markDirty();
  renderScore();
}
let dirtyTimer=null;
function markDirty(){
  $('#edit-hint').textContent='Edited — saving exports…';
  clearTimeout(dirtyTimer);
  dirtyTimer=setTimeout(saveEdits, 900);
}
async function saveEdits(){
  try{
    await api(`/api/jobs/${S.jobId}/reexport`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({bars:S.score.bars.map(b=>({index:b.index, hits:b.hits}))}),
    });
    $('#edit-hint').textContent='Edits saved — MIDI and MusicXML updated.';
  }catch(e){ $('#edit-hint').textContent='Could not save exports: '+e.message; }
}

/* ── wiring ───────────────────────────────────────────────────────────── */
function on(sel, ev, fn){
  const el=$(sel);
  if(el) el.addEventListener(ev, fn);
  return el;
}

async function loadNotice(){
  // a site-wide banner driven by frontend/notice.json (served no-cache, so
  // editing the file is enough); empty text hides it
  try{
    const r=await fetch('notice.json?t='+Date.now(), {cache:'no-store'}); if(!r.ok) return;
    const n=await r.json(); const el=$('#notice'); if(!el) return;
    if(n && n.text){
      $('#notice-text').textContent=n.text;
      const a=$('#notice-link');
      if(n.link){ a.href=n.link; a.textContent=n.linkText||n.link; a.classList.remove('hidden'); } else a.classList.add('hidden');
      el.classList.remove('hidden');
    } else el.classList.add('hidden');
  }catch(_){}
}
function init(){
  loadNotice(); setInterval(loadNotice, 120000);
  // session-only audio: tell the server when this chart is left so its
  // drum/backing tracks are deleted right away (idle timeout covers the rest)
  const release=()=>{ if(S.jobId && S.score){ try{ navigator.sendBeacon(`/api/jobs/${S.jobId}/release`); }catch(_){} } };
  window.addEventListener('pagehide', release);
  api('/api/health').then(h=>{
    if(!h || h.links===false) return;                 // upload-only: the link section stays hidden
    for(const sel of ['#link-or','#link-row','#link-note']) $(sel)?.classList.remove('hidden');
    if(h.youtubeWithConsent && !h.youtube) $('#rights-row')?.classList.remove('hidden');
    if(h.youtube){ $('#url').placeholder='https://www.youtube.com/watch?v=… or a direct audio link'; $('#link-note').textContent='A YouTube link or a direct link to an audio file.'; $('#rights-row')?.classList.add('hidden'); }
    else if(!h.youtubeWithConsent){ $('#rights-row')?.classList.add('hidden'); $('#link-note').textContent='A link straight to an audio file (.mp3 / .wav / .m4a / .ogg / .flac). YouTube and other streaming sites can\'t be used.'; }
  }).catch(()=>{});
  $('#btn-go').onclick=startJob;
  $('#url').addEventListener('keydown', e=>{ if(e.key==='Enter') startJob(); });
  $('#file').addEventListener('change', e=>{ if(e.target.files[0]) startUpload(e.target.files[0]); });
  const drop=$('#filedrop');
  // the drop zone is a <label> around the input: the browser already opens
  // the picker on click; a second .click() here opened it twice every time
  ['dragenter','dragover'].forEach(t=>drop.addEventListener(t, e=>{
    e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave','drop'].forEach(t=>drop.addEventListener(t, e=>{
    e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', e=>{
    const f=e.dataTransfer.files[0]; if(f) startUpload(f);
  });

  on('#opt-sens','input', e=>
    $('#opt-sens-out').textContent=Number(e.target.value).toFixed(1));
  $('#btn-cancel').onclick=async()=>{
    clearInterval(S.poll); clearInterval(S.creep);
    if(S.jobId){ try{ await fetch(`/api/jobs/${S.jobId}`,{method:'DELETE'}); }catch(_){} }
    showView('setup');
  };
  $('#btn-new').onclick=()=>{ pause(); clearInterval(S.poll); if(S.jobId && S.score){ fetch(`/api/jobs/${S.jobId}/release`,{method:'POST'}).catch(()=>{}); } showView('setup'); };
  $('#btn-play').onclick=togglePlay;
  on('#audio','error', ()=>{
    $('#clock').textContent='audio failed to load \u2014 press play to retry';
  });
  $('#speed').addEventListener('input', e=>setSpeed(Number(e.target.value)));
  $('#btn-loop-clear').onclick=()=>{ S.loopFrom=S.loopTo=null; S.loopPick=0; updateLoopLabel(); drawLoopRange(); };
  on('#btn-loop','click', ()=>{ S.loopPick = S.loopPick ? 0 : 1; updateLoopLabel(); });
  $('#btn-print').onclick=()=>window.print();
  const regrid=async(t)=>{
    if(!S.score||!S.jobId) return;
    const lbl=$('#s-tempo'); const old=lbl.textContent; lbl.textContent='re-spelling…';
    try{
      const doc=await api(`/api/jobs/${S.jobId}/regrid`,{method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({tempo:t})});
      const at=position(); pause();
      openScore(doc); seek(at);
    }catch(e){ lbl.textContent=old; alert('Could not re-spell: '+e.message); }
  };
  on('#btn-half','click', ()=>regrid(S.score.tempo/2));
  on('#btn-double','click', ()=>regrid(S.score.tempo*2));
  on('#btn-regrid','click', ()=>regrid(Number($('#tempo-in').value)));
  on('#tempo-in','keydown', e=>{ if(e.key==='Enter') regrid(Number(e.target.value)); });
  $$('.seg').forEach(b=>b.onclick=()=>selectSource(b.dataset.src));
  on('#opt-edit','change', e=>{
    S.editing=e.target.checked;
    $('#palette').classList.toggle('hidden', !S.editing);
    $('#edit-hint').textContent = S.editing
      ? 'Pick a drum, then click the staff to add it — click it again to remove it.'
      : '';
  });
  on('#opt-simple','change', e=>{
    S.simple=e.target.checked;
    if(S.score) renderScore();
  });
  on('#opt-teach','change', e=>{
    S.teach=e.target.checked;
    if(S.teach && !S.simple){                 // teach implies the simple chart
      S.simple=true;
      const cb=$('#opt-simple'); if(cb) cb.checked=true;
    }
    renderLesson();
    if(S.score) renderScore();
  });
  on('#detail','input', e=>{
    S.detail=Number(e.target.value);
    $('#detail-out').textContent=S.detail+'%';
    if(S.score) renderScore();
  });
  on('#opt-key','change', ()=>{ if(S.score) renderScore(); });
  on('#opt-met','change', ()=>{ ensureCtx(); S.metIdx=0; resetSched(position()); });
  on('#opt-hear','change', ()=>{ ensureCtx(); S.ctx.resume(); resetSched(position()); });

  document.addEventListener('keydown', e=>{
    if(e.target.matches('input,select,textarea')) return;
    if(e.code==='Space'){ e.preventDefault(); togglePlay(); }
    if(e.key==='ArrowLeft') seek(position()-2);
    if(e.key==='ArrowRight') seek(position()+2);
  });

  let rt=null;
  window.addEventListener('resize', ()=>{
    clearTimeout(rt); rt=setTimeout(()=>{ if(S.score) renderScore(); }, 200);
  });

  if($('#build')) $('#build').textContent='build '+APP_BUILD;
  api('/api/health').then(h=>{
    const p=$('#engine');
    p.textContent = h.demucs ? (h.drumsep ? 'Demucs + kit split ready' : 'Demucs separation ready')
                             : 'Fast separation (no Demucs)';
    p.classList.toggle('pill-muted', !h.demucs);
  }).catch(()=>{ $('#engine').textContent='server offline'; });
}
document.addEventListener('DOMContentLoaded', init);
