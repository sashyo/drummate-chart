"""The kit: one place that defines how each drum is named, drawn and played.

Staff positions follow the standard drumset notation convention (Weinberg /
Percussive Arts Society), written on a 5-line staff with a percussion clef:

      A5  --- crash        (ledger space above)
      G5  --- hi-hat       (space above top line)
      F5  === ride         (top line, 5)
      E5  --- high tom     (space 4)
      D5  === mid tom      (line 4)
      C5  --- snare        (space 3)
      A4  --- floor tom    (space 2)
      F4  --- bass drum    (space 1)
      D4  --- hi-hat foot  (below the staff)
"""
from __future__ import annotations

UP, DOWN = "up", "down"

DRUMS: dict[str, dict] = {
    "crash":   dict(label="Crash",       key="a/5/x2", head="x",      midi=49, step="A", octave=5, voice=UP,   order=0),
    "hihat":   dict(label="Hi-hat",      key="g/5/x2", head="x",      midi=42, step="G", octave=5, voice=UP,   order=1),
    "openhh":  dict(label="Open hi-hat", key="g/5/x2", head="x",      midi=46, step="G", octave=5, voice=UP,   order=1),
    "ride":    dict(label="Ride",        key="f/5/x2", head="x",      midi=51, step="F", octave=5, voice=UP,   order=2),
    "tom_hi":  dict(label="High tom",    key="e/5",    head="normal", midi=48, step="E", octave=5, voice=UP,   order=3),
    "tom_mid": dict(label="Mid tom",     key="d/5",    head="normal", midi=45, step="D", octave=5, voice=UP,   order=4),
    "snare":   dict(label="Snare",       key="c/5",    head="normal", midi=38, step="C", octave=5, voice=UP,   order=5),
    "tom_low": dict(label="Floor tom",   key="a/4",    head="normal", midi=41, step="A", octave=4, voice=UP,   order=6),
    "kick":    dict(label="Bass drum",   key="f/4",    head="normal", midi=36, step="F", octave=4, voice=DOWN, order=7),
    "hhfoot":  dict(label="Hi-hat foot", key="d/4/x2", head="x",      midi=44, step="D", octave=4, voice=DOWN, order=8),
}

ORDER = [k for k, _ in sorted(DRUMS.items(), key=lambda kv: kv[1]["order"])]


def voice_of(inst: str) -> str:
    return DRUMS.get(inst, {}).get("voice", UP)


def key_of(inst: str) -> str:
    return DRUMS.get(inst, {}).get("key", "c/5")


def midi_of(inst: str) -> int:
    return DRUMS.get(inst, {}).get("midi", 38)


def label_of(inst: str) -> str:
    return DRUMS.get(inst, {}).get("label", inst)
