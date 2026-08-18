"""Export a Clone Hero song package.

Clone Hero reads Rock Band-style MIDI: a tempo track plus a "PART DRUMS"
track. Expert lives at pitches 96-100 (kick, red, yellow, blue, green);
yellow/blue/green are CYMBALS unless a tom marker (110/111/112) spans the
note - that's what "Pro Drums" means. Lower difficulties sit at 84/72/60
and are generated here as progressively thinned reductions.

The package is a zip the player drops into their Songs folder:
    notes.mid   the chart
    song.ini    metadata (pro_drums = True)
    song.mp3    the backing (everything but the kit)
    drums.mp3   the kit - Clone Hero mutes it when you miss
"""
from __future__ import annotations

import zipfile
from pathlib import Path

RES = 480                              # ticks per quarter, the CH convention
SCALE = RES // 48                      # our PPQ is 48

# inst -> (lane offset from the difficulty base, tom marker pitch or None)
LANES = {
    "kick":    (0, None),
    "snare":   (1, None),
    "hihat":   (2, None),              # yellow cymbal
    "openhh":  (2, None),
    "tom_hi":  (2, 110),               # yellow + tom marker
    "ride":    (3, None),              # blue cymbal
    "tom_mid": (3, 111),
    "crash":   (4, None),              # green cymbal
    "tom_low": (4, 112),
}
DIFF_BASE = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
NOTE_LEN = 60                          # 1/32 - drums have no sustains


def _keep(diff: str, inst: str, tick: int, ghost: bool) -> bool:
    """Progressive reductions, same philosophy as the app's Simple mode."""
    if inst == "hhfoot":
        return False
    if diff == "expert":
        return True
    if ghost:
        return False
    if diff == "hard":                 # 8th-note grid
        return tick % 24 == 0
    if diff == "medium":               # 8ths, kick on quarters, no toms
        if inst.startswith("tom_"):
            return False
        if inst == "kick":
            return tick % 48 == 0
        return tick % 24 == 0
    # easy: quarters only, kick/snare/hats (crash allowed on the downbeat)
    if inst == "crash":
        return tick == 0
    if inst.startswith("tom_") or inst in ("ride", "openhh"):
        return False
    return tick % 48 == 0


def _lead_in(gap: float, bpb: int):
    """Whole bars of count-in so bar lines stay aligned with the music."""
    if gap <= 0.05:
        return 0, None
    beats = max(bpb, round(gap / 0.5 / bpb) * bpb)   # aim near 120 BPM
    bpm = 60.0 * beats / gap
    while bpm > 260:
        beats += bpb
        bpm = 60.0 * beats / gap
    while bpm < 30 and beats > bpb:
        beats -= bpb
        bpm = 60.0 * beats / gap
    return beats * RES, bpm


def write_notes_mid(doc: dict, qscore, path: Path) -> Path:
    import mido

    bpb = int(doc.get("beatsPerBar", 4))
    mid = mido.MidiFile(ticks_per_beat=RES, type=1)

    tempo_tr = mido.MidiTrack()
    drum_tr = mido.MidiTrack()
    mid.tracks += [tempo_tr, drum_tr]
    tempo_tr.append(mido.MetaMessage("track_name", name=doc.get("title", "song")[:100], time=0))
    tempo_tr.append(mido.MetaMessage("time_signature", numerator=bpb, denominator=4, time=0))
    drum_tr.append(mido.MetaMessage("track_name", name="PART DRUMS", time=0))

    bars = qscore.bars
    if not bars:
        raise ValueError("empty score")

    offset_ticks, leadin_bpm = _lead_in(float(bars[0].start_time), bpb)
    tempos = []                        # (tick, microseconds per beat)
    if leadin_bpm:
        tempos.append((0, int(round(60_000_000 / leadin_bpm))))

    events = []                        # (tick, order, pitch, velocity, on)
    markers = set()                    # (tick, marker_pitch)

    for i, bar in enumerate(bars):
        bar_tick = offset_ticks + i * bpb * RES
        dur = max(float(bar.end_time - bar.start_time), 1e-3)
        bpm = min(max(60.0 * bpb / dur, 20.0), 400.0)
        tempos.append((bar_tick, int(round(60_000_000 / bpm))))

        for n in bar.notes:
            lane = LANES.get(n.inst)
            if lane is None:
                continue
            off, marker = lane
            t = bar_tick + n.tick * SCALE
            vel = 127 if n.accent else (1 if n.ghost else 100)
            for diff, base in DIFF_BASE.items():
                if not _keep(diff, n.inst, n.tick, n.ghost):
                    continue
                events.append((t, 1, base + off, vel, True))
                events.append((t + NOTE_LEN, 0, base + off, 0, False))
            if marker is not None:
                markers.add((t, marker))

    for t, m in markers:
        events.append((t, 1, m, 100, True))
        events.append((t + NOTE_LEN, 0, m, 0, False))

    # dedupe identical note-ons (several difficulties can agree on a note)
    events = sorted(set(events), key=lambda e: (e[0], e[1]))

    prev = 0
    for tick, us in sorted(set(tempos)):
        tempo_tr.append(mido.MetaMessage("set_tempo", tempo=us, time=tick - prev))
        prev = tick
    tempo_tr.append(mido.MetaMessage("end_of_track", time=0))

    prev = 0
    for tick, _o, pitch, vel, on in events:
        delta = tick - prev
        prev = tick
        drum_tr.append(mido.Message("note_on" if on else "note_off",
                                    channel=0, note=pitch, velocity=vel, time=delta))
    drum_tr.append(mido.MetaMessage("end_of_track", time=0))

    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


def _song_ini(doc: dict) -> str:
    title = doc.get("title", "Unknown")
    artist, name = "Unknown", title
    if " - " in title:
        artist, name = title.split(" - ", 1)
    dur = float(doc.get("stats", {}).get("duration", 0.0))
    hits = int(doc.get("stats", {}).get("hits", 0))
    rate = hits / dur if dur else 0
    diff = 2 if rate < 4 else 3 if rate < 6 else 4 if rate < 8 else 5 if rate < 10 else 6
    return "\n".join([
        "[song]",
        f"name = {name}",
        f"artist = {artist}",
        "charter = DrumMate Chart",
        f"diff_drums = {diff}",
        "pro_drums = True",
        "delay = 0",
        f"song_length = {int(dur * 1000)}",
        f"preview_start_time = {int(dur * 300)}",
        "loading_phrase = Transcribed automatically by DrumMate Chart - chart.drummate.app",
        "",
    ])


def write_package(doc: dict, qscore, job_dir: Path) -> Path:
    """Assemble <job>/clonehero.zip; contents unzip into a CH Songs folder."""
    notes = job_dir / "notes.mid"
    write_notes_mid(doc, qscore, notes)

    title = doc.get("title", "song")
    safe = "".join(c for c in title if c.isalnum() or c in " -_()[]").strip() or "song"
    zpath = job_dir / "clonehero.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(notes, f"{safe}/notes.mid")
        z.writestr(f"{safe}/song.ini", _song_ini(doc))
        backing = job_dir / "backing.mp3"
        drums = job_dir / "drums.mp3"
        if backing.exists():
            z.write(backing, f"{safe}/song.mp3")
        if drums.exists():
            z.write(drums, f"{safe}/drums.mp3" if backing.exists() else f"{safe}/song.mp3")
    notes.unlink(missing_ok=True)
    return zpath
