"""Stage 7 - export the chart as MIDI and MusicXML.

MIDI carries per-bar tempo changes so it lines up with the original
recording. MusicXML opens in MuseScore / Sibelius / Dorico as a real
drumset part, which is how you get a printable PDF.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from . import drums as K

MIDI_PPQ = 480

_XML_DUR = {"w": 96, "h": 48, "q": 24, "8": 12, "16": 6, "32": 3}
_XML_TYPE = {"w": "whole", "h": "half", "q": "quarter", "8": "eighth",
             "16": "16th", "32": "32nd"}
DIVISIONS = 24


# --------------------------------------------------------------------------
# MIDI
# --------------------------------------------------------------------------

def write_midi(score: dict, qscore, path: Path) -> Path:
    import mido

    mid = mido.MidiFile(ticks_per_beat=MIDI_PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=score["title"][:120], time=0))

    bpb = score["beatsPerBar"]
    scale = MIDI_PPQ // qscore.ppq
    ticks_per_bar = qscore.ticks_per_bar * scale

    events: list[tuple[int, int, str, int, int]] = []   # (tick, order, kind, note, vel)
    base = qscore.bars[0].index if qscore.bars else 0

    for bar in qscore.bars:
        seq = bar.index - base
        bar_start = seq * ticks_per_bar
        dur = max(bar.end_time - bar.start_time, 1e-3)
        bpm = 60.0 * bpb / dur
        bpm = min(max(bpm, 30.0), 300.0)
        events.append((bar_start, 0, "tempo", int(mido.bpm2tempo(bpm)), 0))
        for n in bar.notes:
            vel = int(round(45 + 82 * min(max(n.velocity, 0.0), 1.0)))
            if n.ghost:
                vel = 34
            if n.accent:
                vel = min(127, vel + 18)
            t = bar_start + n.tick * scale
            if n.flam:
                events.append((max(0, t - MIDI_PPQ // 12), 1, "on", K.midi_of(n.inst), 40))
                events.append((max(0, t - MIDI_PPQ // 12) + 30, 2, "off", K.midi_of(n.inst), 0))
            events.append((t, 1, "on", K.midi_of(n.inst), vel))
            events.append((t + 30, 2, "off", K.midi_of(n.inst), 0))

    events.sort(key=lambda e: (e[0], e[1]))
    prev = 0
    for tick, _o, kind, a, b in events:
        delta = max(0, tick - prev)
        prev = tick
        if kind == "tempo":
            track.append(mido.MetaMessage("set_tempo", tempo=a, time=delta))
        elif kind == "on":
            track.append(mido.Message("note_on", channel=9, note=a, velocity=b, time=delta))
        else:
            track.append(mido.Message("note_off", channel=9, note=a, velocity=0, time=delta))

    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


# --------------------------------------------------------------------------
# MusicXML
# --------------------------------------------------------------------------

def _elem_duration(e: dict) -> int:
    d = _XML_DUR.get(e["dur"], 24)
    if e.get("dots"):
        d = int(d * (3 ** e["dots"]) / (2 ** e["dots"]))
    t = e.get("tuplet")
    if t:
        d = int(round(d * t["den"] / t["num"]))
    return max(1, d)


def write_musicxml(score: dict, path: Path) -> Path:
    root = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = score["title"]
    ident = ET.SubElement(root, "identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "DrumMate Chart"

    part_list = ET.SubElement(root, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Drumset"
    ET.SubElement(sp, "part-abbreviation").text = "Dr."

    used = score.get("kit") or ["kick", "snare", "hihat"]
    for inst in used:
        info = K.DRUMS.get(inst)
        if not info:
            continue
        iid = f"P1-I{info['midi']}"
        si = ET.SubElement(sp, "score-instrument", id=iid)
        ET.SubElement(si, "instrument-name").text = info["label"]
        mi = ET.SubElement(sp, "midi-instrument", id=iid)
        ET.SubElement(mi, "midi-channel").text = "10"
        ET.SubElement(mi, "midi-unpitched").text = str(info["midi"] + 1)

    part = ET.SubElement(root, "part", id="P1")

    for i, bar in enumerate(score["bars"]):
        m = ET.SubElement(part, "measure", number=str(bar["number"]))
        if i == 0:
            attrs = ET.SubElement(m, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = str(score["beatsPerBar"])
            ET.SubElement(time, "beat-type").text = "4"
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "percussion"
            ET.SubElement(clef, "line").text = "2"
            sd = ET.SubElement(attrs, "staff-details")
            ET.SubElement(sd, "staff-lines").text = "5"

            direction = ET.SubElement(m, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(int(round(score["tempo"])))
            ET.SubElement(direction, "sound", tempo=str(int(round(score["tempo"]))))
            if score.get("swing"):
                d2 = ET.SubElement(m, "direction", placement="above")
                dt2 = ET.SubElement(d2, "direction-type")
                ET.SubElement(dt2, "words").text = "Swing 8ths"

        total_up = _write_voice(m, bar["voices"].get(K.UP, []), voice=1, stem="up",
                                _bar_beats=score["beatsPerBar"])
        down = bar["voices"].get(K.DOWN, [])
        if down:
            if total_up:
                bk = ET.SubElement(m, "backup")
                ET.SubElement(bk, "duration").text = str(total_up)
            _write_voice(m, down, voice=2, stem="down", _bar_beats=score["beatsPerBar"])

    path.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    xml = ET.tostring(root, encoding="unicode")
    doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n' + xml)
    path.write_text(doc, encoding="utf-8")
    return path


def _write_voice(measure, elems: list[dict], voice: int, stem: str,
                 _bar_beats: int = 4) -> int:
    total = 0
    groups = _tuplet_groups(elems)
    for idx, e in enumerate(elems):
        dur = _elem_duration(e)
        total += dur
        if e["type"] == "rest":
            n = ET.SubElement(measure, "note")
            if e.get("hidden"):
                n.set("print-object", "no")
            if e.get("barRest"):
                # whole-measure rest: duration = the actual bar, no note type
                bar_div = DIVISIONS * _bar_beats
                total += bar_div - dur
                ET.SubElement(n, "rest", measure="yes")
                ET.SubElement(n, "duration").text = str(bar_div)
                ET.SubElement(n, "voice").text = str(voice)
                continue
            ET.SubElement(n, "rest")
            ET.SubElement(n, "duration").text = str(dur)
            ET.SubElement(n, "voice").text = str(voice)
            ET.SubElement(n, "type").text = _XML_TYPE.get(e["dur"], "quarter")
            for _ in range(e.get("dots", 0)):
                ET.SubElement(n, "dot")
            _tuplet_xml(n, e, idx, groups, rest=True)
            continue

        for j, inst in enumerate(e["insts"]):
            info = K.DRUMS.get(inst)
            if not info:
                continue
            n = ET.SubElement(measure, "note")
            if j > 0:
                ET.SubElement(n, "chord")
            up = ET.SubElement(n, "unpitched")
            ET.SubElement(up, "display-step").text = info["step"]
            ET.SubElement(up, "display-octave").text = str(info["octave"])
            ET.SubElement(n, "duration").text = str(dur)
            ET.SubElement(n, "instrument", id=f"P1-I{info['midi']}")
            ET.SubElement(n, "voice").text = str(voice)
            ET.SubElement(n, "type").text = _XML_TYPE.get(e["dur"], "quarter")
            for _ in range(e.get("dots", 0)):
                ET.SubElement(n, "dot")
            if info["head"] == "x":
                ET.SubElement(n, "notehead").text = "x"
            ET.SubElement(n, "stem").text = stem
            _tuplet_xml(n, e, idx, groups, rest=False, mark=(j == 0))

            notations_needed = (j == 0 and (e.get("accent") or e.get("open")
                                            or e.get("ghost") or e.get("flam")))
            if notations_needed:
                nt = n.find("notations")
                if nt is None:
                    nt = ET.SubElement(n, "notations")
                if e.get("accent"):
                    art = ET.SubElement(nt, "articulations")
                    ET.SubElement(art, "accent")
                if e.get("open"):
                    tech = ET.SubElement(nt, "technical")
                    ET.SubElement(tech, "open-string")
    return total


def _tuplet_groups(elems: list[dict]) -> dict[int, str]:
    """Map element index -> 'start' / 'stop' for tuplet notations."""
    marks: dict[int, str] = {}
    i = 0
    while i < len(elems):
        t = elems[i].get("tuplet")
        if not t:
            i += 1
            continue
        j = i
        while j + 1 < len(elems) and elems[j + 1].get("tuplet") == t \
                and elems[j + 1]["beat"] == elems[i]["beat"]:
            j += 1
        marks[i] = "start"
        marks[j] = "stop"
        i = j + 1
    return marks


def _tuplet_xml(note_el, e, idx, groups, rest: bool, mark: bool = True) -> None:
    t = e.get("tuplet")
    if not t:
        return
    # Every note in the chord is time-modified, but the bracket is drawn once.
    tm = ET.SubElement(note_el, "time-modification")
    ET.SubElement(tm, "actual-notes").text = str(t["num"])
    ET.SubElement(tm, "normal-notes").text = str(t["den"])
    mark = groups.get(idx) if mark else None
    if mark:
        nt = note_el.find("notations")
        if nt is None:
            nt = ET.SubElement(note_el, "notations")
        ET.SubElement(nt, "tuplet", type=mark, number="1")


def _indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem.tail or "").strip():
            elem.tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad
