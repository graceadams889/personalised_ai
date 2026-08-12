# Grace Adams — Thesis — Personalised Parametric Mastering Chain

A personalised, measurement-driven mastering tool. The chain learns an engineer's
mastering style from a small set of paired *pre-master* / *master* recordings by
measuring a few interpretable features per genre — integrated loudness, dynamics
(crest factor), stereo image (L–R correlation) and tonal balance (per-octave-band
energy) — and then reproduces those targets on a new mix through a fixed
signal-processing chain:

**parametric EQ → broadband compression → optional mid/side width → loudness-targeted peak limiting**

Every decision the chain makes is traceable to a measured quantity, which keeps the
model interpretable at the small data scale used in this project. The parametric
chain in `src/chain/` is the system described in the thesis; the `DeepAFx-ST/`
directory is retained only for exploratory / future work and is not part of the
submitted chain.

## Requirements

Python 3, with the packages listed in `requirements.txt`. The parametric chain
itself relies on `numpy`, `scipy`, `soundfile`, `pyloudnorm`, `pandas`, `matchering`,
`matplotlib` and `gradio`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data layout

Paired stereo WAVs (44.1 kHz) go in `data/raw/`. A master and its corresponding mix
**share the same filename**, and **each filename must contain its genre keyword** so
the chain can group tracks by genre. Recognised keywords (case-insensitive) are
`acoustic`, `altrock`, `classical`, and `electr`/`electronic` (or `rbpop`); a filename
matching none of these is treated as `Unknown` and excluded from the profiles.

```
data/raw/
├─ master/                         # your finished masters (24-bit)
│   ├─ Acoustic2_UnmasteredWAV.wav
│   ├─ AltRock2_UnmasteredWAV.wav
│   ├─ Classical2_UnMastered.wav
│   └─ ElectrPop2_UnmasteredWAV.wav
└─ pre/                            # unmastered mixes, produced by normalise.py
    ├─ Acoustic2_UnmasteredWAV.wav
    └─ ...
```

Keep your original mixes in a separate folder; `scripts/normalise.py` reads that
folder, normalises each mix to −18 LUFS, and writes it into `data/raw/pre/` under the
same filename. Place the matching finished master in `data/raw/master/`. Mixes should
be WAV, since the profiling step only reads `*.wav` from `pre/`.

## Usage

Run from the project root:

```bash
# 1. Normalise the unmastered mixes to −18 LUFS.
#    First edit INPUT_FOLDER / OUTPUT_FOLDER at the top of the script,
#    then run (writes into data/raw/pre/):
python scripts/normalise.py

# 2. Extract per-track features and aggregate them into per-genre profiles
#    (writes outputs/profile_per_track.csv):
python scripts/profile_pairs.py

# 3. Master a single file using a chosen genre profile
#    (genre: Acoustic, AltRock, Classical, or ElectronicPop):
python scripts/apply_chain.py --input <mix.wav> --output <master.wav> --genre <genre>
#    add --with-width to enable the optional mid/side width stage

# 4a. In-sample validation: master every training pair with its genre profile and
#     compare LUFS / crest / L-R correlation against the real master
#     (writes outputs/chain_validation_nowidth/ and a CSV):
python scripts/validate_chain.py            # add --with-width for the width variant

# 4b. Leave-one-out cross-validation (generalisation test): each track is mastered
#     from a profile refitted on the OTHER tracks in its genre. Run profile_pairs.py
#     first, as this reads outputs/profile_per_track.csv:
python scripts/validate_chain_loocv.py

# 5. Launch the drag-and-drop web interface (Gradio) in your browser:
python scripts/launch_chain_ui.py
```

## Configuration

Project-wide defaults live in `configs/default.yaml`.

## Note on the use of generative AI

Parts of the implementation were scaffolded with the assistance
of an AI coding tool. All design decisions, parameter choices, dataset creation and
listening-driven refinement are the researcher's own work.
