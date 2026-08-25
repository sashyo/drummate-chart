"""Fetch a Songsterr drum track and write it as bars for compare_tab.py.

  songsterr_tab.py "artist title" OUT.json     # searches, picks the Drums track
  songsterr_tab.py --song SONGID OUT.json

Output: {"bars": [{"K": [slots], "S": [...], "x": [...], "T": [...]}, ...]}
at 16th resolution (slot = 16th from beat 1, 4/4 assumed per measure)."""
import gzip, json, re, sys, urllib.request
UA = {"User-Agent": "Mozilla/5.0"}
GM = {35: "K", 36: "K", 38: "S", 40: "S", 37: "S", 39: "S",       # kick, snare, rim, clap
      42: "x", 44: "x", 46: "x", 49: "x", 51: "x", 52: "x", 53: "x", 55: "x", 57: "x", 59: "x",
      41: "T", 43: "T", 45: "T", 47: "T", 48: "T", 50: "T",
      # Guitar Pro's extended drum map, which Songsterr keeps: 91 snare (hit),
      # 92/93 hat half-open / ride edge, 97/98 snare (side stick / rim)
      91: "S", 92: "x", 93: "x", 97: "S", 98: "S"}

def get(url, binary=False):
    # curl: songsterr answers with HTTP 103 Early Hints, which urllib rejects
    import subprocess
    data = subprocess.run(["curl", "-sL", "-A", UA["User-Agent"], url], capture_output=True, timeout=60).stdout
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data if binary else data.decode("utf-8", "ignore")

def find_song(query):
    songs = json.loads(get("https://www.songsterr.com/api/songs?pattern=" + urllib.parse.quote(query)))
    for s in songs:
        # the original recording's tab is listed first; take its LAST drum
        # track (a second 'Drums' track is usually percussion/overdubs
        # placed after the kit... but the kit is the first one)
        drums = [i for i, t in enumerate(s.get("tracks", []))
                 if t.get("instrument") == "Drums" or str(t.get("hash", "")).startswith("drums")]
        if drums:
            return s["songId"], drums[0], s["artist"], s["title"]
    raise SystemExit("no drum track found for " + query)

def fetch_track(song_id, track_index):
    html = get(f"https://www.songsterr.com/a/wsa/x-drum-tab-s{song_id}")
    rev = re.search(r'"revisionId":(\d{5,})', html).group(1)
    img = re.search(r'"image":"([^"]+)"', html).group(1)
    return json.loads(get(f"https://dqsljvtekg760.cloudfront.net/{song_id}/{rev}/{img}/{track_index}.json"))

def to_bars(track):
    bars = []
    for m in track["measures"]:
        sig = m.get("signature") or [4, 4]
        beats_per_bar = sig[0] * 4 / sig[1]
        cells = {}
        for v in m["voices"]:
            pos = 0.0                                   # in quarter notes
            for b in v["beats"]:
                num, den = b.get("duration", [1, 4])
                q = 4.0 * num / den
                if b.get("dots"):
                    q *= 1.5 if b["dots"] == 1 else 1.75
                tp = b.get("tuplet")
                if tp:
                    q *= 2.0 / 3.0 if tp == 3 else (tp.get("ratio", 1) if isinstance(tp, dict) else 1)
                if not b.get("rest"):
                    slot = int(round(pos * 4))
                    for n in b.get("notes", []):
                        c = GM.get(n.get("fret"))
                        if c and 0 <= slot < 16 * beats_per_bar / 4 + 1:
                            cells.setdefault(c, set()).add(min(slot, 15))
                pos += q
        bars.append({c: sorted(s) for c, s in cells.items()})
    return bars

if __name__ == "__main__":
    import urllib.parse
    args = sys.argv[1:]
    if args[0] == "--song":
        sid, idx = int(args[1]), None; out = args[2]
        songs = json.loads(get(f"https://www.songsterr.com/api/songs?pattern={sid}"))
        html = get(f"https://www.songsterr.com/a/wsa/x-drum-tab-s{sid}")
        m = re.search(r'"tracks":(\[.*?\])\s*,"', html)
        raise SystemExit("use the search form")
    sid, idx, artist, title = find_song(args[0]); out = args[1]
    track = fetch_track(sid, idx)
    bars = to_bars(track)
    json.dump({"artist": artist, "title": title, "songId": sid, "track": idx, "bars": bars}, open(out, "w"))
    print(f"{artist} - {title} (song {sid}, drum track {idx}): {len(bars)} bars -> {out}")
