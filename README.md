# DrumMate Chart

**Drop in a song, get drum notation you can practise with** — plus the isolated drum track and a
drums-removed backing track to play along to.

Built by the people behind **[DrumMate](https://drummate.app/?utm_source=github&utm_medium=readme&utm_campaign=drummate-chart)**, the live band for your electronic drum
kit: you drum, the band follows. This is the sibling for learning the song first.

![DrumMate Chart](docs/screenshot.png)

Live: **[chart.drummate.app](https://chart.drummate.app/?utm_source=github&utm_medium=readme&utm_campaign=drummate-chart)** (free, upload your own audio). Open source under the [MIT License](LICENSE).

## What it does

1. **Separates the kit** from the mix — Demucs, then the MDX23C DrumSep model splits the kit into
   kick / snare / toms / hi-hat / ride / crash.
2. **Finds the pulse from the drums themselves** — period from the kick/snare autocorrelation, an
   octave/4:3 judge that scores candidates on whether the bars look like bars, a phase follower that
   only real backbeat strokes may move, and a linear-drift fold for songs recorded without a click.
3. **Detects every hit per stem**, with bleed rules learned by comparing against published charts:
   a floor-velocity kick under a snare is bleed unless the song doubles its backbeat; a stray ghost
   that repeats nowhere nearby is noise; and so on.
4. **Quantises** per bar (8ths / 16ths / 32nds / triplets, swing detection), then snaps weak
   deviations to the section's groove while leaving fills exactly as played — the way a
   transcriber writes a chart.
5. **Engraves** proper two-voice drumset notation, with a legend, a simple mode, a loop-practice
   tool and a teach mode that breaks the groove down and calls the hits out loud.
6. **Exports** MIDI, MusicXML and a Clone Hero package (Pro Drums).

Measured against full published transcriptions (Songsterr, tTabs) it matches **90–98 % of bars on
kick and snare** for clean studio recordings (Billie Jean 140/143 kick, 136/143 snare; Every
Breath You Take 110/112 kick). Hi-hats are where published charts disagree with each other; live
bands without a click are where it still struggles. Details and the tools to measure it yourself
are under [Accuracy](#accuracy).

## Quick start (your own machine)

**Easiest — one file to run.** Download the ZIP from the
[latest release](https://github.com/sashyo/drummate-chart/releases/latest), unzip it, then:

| | double-click | needs |
|---|---|---|
| Windows | `start.bat` | nothing — it installs Python (via winget) and ffmpeg itself if missing |
| macOS | `start.command` | Python 3.10+ (python.org or Homebrew); ffmpeg via Homebrew if missing |
| Linux | `start.sh` | Python 3.10+, ffmpeg (`apt install ffmpeg`) |

The first run installs the Python packages into the app folder (5–10 minutes, ~1.5 GB) and the
first chart downloads the separation models (~500 MB). After that it starts in seconds and opens
<http://127.0.0.1:8000> in your browser. Close the window to stop.

**Or with git**, if you'd rather:

```bash
git clone https://github.com/sashyo/drummate-chart.git
cd drummate-chart
./run.sh                      # same as start.sh, without opening the browser
```

Node.js is only needed for YouTube links (yt-dlp uses it as a JavaScript runtime).

**Links, on your own machine.** Running it locally is personal use, so you can turn link fetching on:

```bash
DRUMS_LINKS=1 DRUMS_ALLOW_YOUTUBE=1 ./run.sh
```

The public site is deliberately upload-only and deletes your audio the moment the chart is done
(see [Terms](frontend/terms.html)). Please don't run a public instance that fetches from YouTube;
that's a breach of their terms and, served to strangers, a distribution of other people's recordings.

**GPU.** On CPU a 4-minute song takes 10–20 minutes. With an NVIDIA card it takes about 4, and the
fine-tuned Demucs model becomes the default. Create a CUDA environment (`.venv-cuda`, torch build
matching your driver — see [DEPLOY.md](DEPLOY.md#2-code-and-python-environments)); `run.sh` uses it
when present. A 3 GB card is enough.

## Configuration

| variable | default | meaning |
|---|---|---|
| `DRUMS_LINKS` | `1` | accept links (direct audio files). `0` = upload only |
| `DRUMS_ALLOW_YOUTUBE` | `0` | fetch YouTube/streaming links (personal use on your own machine) |
| `DRUMS_YOUTUBE_WITH_CONSENT` | `1` | when YouTube is off, allow it behind a rights checkbox recorded on the job |
| `DRUMS_WORKERS` | `2` | transcriptions in parallel |
| `DRUMS_DEVICE` | auto | `cpu` to ignore the GPU |
| `DRUMS_SEPARATION` | auto | `htdemucs` to skip the fine-tuned model on a GPU |
| `DRUMS_MAX_SECONDS` | `600` | cap on analysed audio per job |
| `DRUMS_DATA` | `./data` | where jobs, caches and the usage counter live |
| `DRUMS_KEEP_SOURCES` | `0` | `1` keeps source audio and separation caches (default: deleted when the job ends) |
| `DRUMS_AUDIO_TTL_HOURS` | `6` | hard cap on how long a chart's drum/backing tracks exist |
| `DRUMS_SESSION_IDLE_MIN` | `20` | a chart's audio is released when its page is closed, or after this idle time |

## How the pieces fit

```
backend/pipeline/
  fetch.py      audio in (upload / link / yt-dlp) -> 44.1 kHz wav
  separate.py   Demucs drums stem + backing            gpu.py: one inference at a time, CPU fallback
  drumsep.py    MDX23C kit split (6 stems, cached by audio content)
  onsets.py     per-stem onset picking, velocities, open-hat test, toms by pitch
  rhythm.py     tempo from the drums: autocorrelation, octave judge, phase follower, drift fold
  quantize.py   per-bar grid choice, de-lag, bleed rules, section consolidation
  score.py      notation model (two voices, rests, tuplets)  ->  score.json
  exports.py    MIDI, MusicXML          clonehero.py   Clone Hero package
backend/server.py   FastAPI: jobs, queue, persistence, usage counter, janitor
frontend/           vanilla JS + VexFlow: engraving, playback, loop, teach mode, editing
tools/              accuracy tooling (below)
```

## Accuracy

Everything in the engine is measured against published charts, bar by bar, and changes ship only
when they win.

```bash
.venv/bin/python tools/songsterr_tab.py "Michael Jackson Billie Jean" bj.json   # a published drum track as bars
.venv/bin/python tools/compare_tab.py data/jobs/<id>/score.json bj.json         # per-bar agreement + deviation clusters
.venv/bin/python tools/run_references.py                                        # the reference suite (tools/references.json)
.venv/bin/python tools/requant.py data/jobs/<id>                                # re-run grid + quantiser from a saved detection (seconds)
.venv/bin/python tools/learn_songs.py songs.json data/learn                     # widen the set: fetch tabs, transcribe, compare
.venv/bin/python tools/train_onsets.py data/learn                               # per-onset classifier, leave-one-song-out vs the rules
```

If you find a bar that's wrong, that's the most useful thing you can send: the song, the bar number,
and what should be there. Every rule in `quantize.py` came from exactly that.

## Deploying

[DEPLOY.md](DEPLOY.md) is the full runbook for a public instance behind a Cloudflare Tunnel:
environments, models, data, verification, cutover and autostart. `Dockerfile` / `docker-compose.yml`
build a CPU-only container.

## License

[MIT](LICENSE) — use it, change it, ship it; keep the notice. The separation models are third-party
(see Credits) and carry their own terms.

## Credits

- Separation: [Demucs](https://github.com/facebookresearch/demucs) (MIT) and the MDX23C DrumSep
  model by aufr33 & jarredou via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator).
- Engraving: [VexFlow](https://github.com/0xfe/vexflow).
- Made by [DrumMate](https://drummate.app/?utm_source=github&utm_medium=readme&utm_campaign=drummate-chart). If the chart got you through the song, go play it with a band that follows you.
