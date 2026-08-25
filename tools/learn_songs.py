"""Widen the reference set: for each song, find its Songsterr drum track and a
YouTube upload, transcribe it, and compare bar by bar.

  learn_songs.py SONGLIST.json OUT_DIR      # [{"q": "Artist Title", "yt": "optional url"}, ...]

Writes OUT_DIR/<slug>/score.json, OUT_DIR/<slug>/tab.json, and OUT_DIR/results.json
(one row per song: bars, exact, per-class agreement, top deviations)."""
import json, re, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import songsterr_tab as st
from backend.pipeline.run import Options, transcribe

def yt_url(q: str) -> str:
    out = subprocess.run([str(ROOT / ".venv/bin/yt-dlp"), "--no-warnings", "--js-runtimes", "node",
                          "--print", "%(id)s|%(title)s|%(duration)s", f"ytsearch5:{q} official audio"],
                         capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
    best = None
    for line in out:
        vid, title, dur = (line.split("|") + ["", ""])[:3]
        try: dur = float(dur)
        except ValueError: dur = 0
        bad = re.search(r"live|cover|drum cover|karaoke|lesson|tutorial|remix|reaction|slowed|sped", title, re.I)
        if 60 < dur < 600 and not bad:
            best = vid; break
        best = best or vid
    return f"https://www.youtube.com/watch?v={best}"

songs = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
results = json.load(open(out / "results.json")) if (out / "results.json").exists() else {}
for s in songs:
    q = s["q"]; slug = re.sub(r"[^A-Za-z0-9]+", "_", q).strip("_")
    if slug in results and results[slug].get("exact") is not None:
        continue
    d = out / slug; d.mkdir(exist_ok=True)
    # users first: wait while the live server has a job queued or running
    while True:
        try:
            q = subprocess.run(["curl", "-s", "localhost:8000/api/queue"], capture_output=True, text=True, timeout=10).stdout
            # run alongside at most one live job (the GPU interleaves two
            # processes fine; three is where users start to feel it)
            if not q.strip() or len(json.loads(q)) < 2:
                break
        except Exception:  # noqa: BLE001
            break
        time.sleep(20)
    t0 = time.time()
    try:
        sid, idx, artist, title = st.find_song(q)
        bars = st.to_bars(st.fetch_track(sid, idx))
        json.dump({"artist": artist, "title": title, "songId": sid, "track": idx, "bars": bars}, open(d / "tab.json", "w"))
        url = s.get("yt") or yt_url(q)
        doc = transcribe(url, d, ROOT / "data/cache", Options(render_audio=False))
        cmp = subprocess.run([sys.executable, str(ROOT / "tools/compare_tab.py"), str(d / "score.json"), str(d / "tab.json"), "--show", "0"],
                             capture_output=True, text=True).stdout
        m = re.search(r"exactly matching the tab: (\d+)/(\d+)", cmp)
        agree = dict(re.findall(r"^\s+([KSxT]): (\d+/\d+)", cmp, re.M))
        devs = re.findall(r"^\s+([KSxT] (?:extra|missing) at \[[^\]]*\]\s+x\d+)", cmp, re.M)[:6]
        results[slug] = {"q": q, "url": url, "songsterr": f"{sid}/track{idx}", "title": doc["title"], "tempo": doc["tempo"],
                         "bars": len(doc["bars"]), "tab_bars": len(bars), "exact": int(m.group(1)) if m else None,
                         "of": int(m.group(2)) if m else None, "agree": agree, "devs": devs, "seconds": round(time.time() - t0)}
        print(f"{q}: {results[slug]['exact']}/{results[slug]['of']} exact, {agree}, {results[slug]['seconds']}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        results[slug] = {"q": q, "error": f"{type(exc).__name__}: {exc}"}
        print(f"{q}: FAILED {type(exc).__name__}: {exc}", flush=True); traceback.print_exc()
    json.dump(results, open(out / "results.json", "w"), indent=1)
print("ALL DONE", flush=True)
