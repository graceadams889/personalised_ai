# Personalised AI Mastering

Fine-tune DeepAFx-ST on your own paired pre-master/master recordings to learn your mastering style.

## Setup

```bash

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
git clone https://github.com/adobe-research/DeepAFx-ST.git
```

## Data Layout

Place your paired WAVs in `data/raw/` with matching filenames:

```
data/raw/
  pre/
    track_001.wav
    track_002.wav
  master/
    track_001.wav
    track_002.wav
```

All files must be the same sample rate (44.1kHz) and bit depth (24-bit).

## Usage

Run the numbered scripts in order from the project root:

```bash
# 1. Prepare data (align, normalise, chunk, split)
python scripts/01_prepare_data.py

# 2. Train the model
python scripts/02_train.py

# 3. Evaluate against ground truth + Matchering baseline
python scripts/03_evaluate.py --predictions outputs/model --reference data/raw/master/track_001.wav

# 4. Run inference on new audio
python scripts/04_infer.py --input /path/to/premaster.wav --output /path/to/output.wav

# 5. Launch the web UI
python scripts/05_launch_ui.py
```

## Configuration

Edit `configs/default.yaml` to adjust sample rate, segment length, training hyperparameters, and effect chain.

## DeepAFx-ST Integration

The training and inference modules contain placeholder methods that need to be connected to the DeepAFx-ST model. Look for `_setup_model()` in `src/training/trainer.py` and `_load_model()` in `src/inference/infer.py`.
