import torch
import torchaudio
import soundfile as sf
import numpy as np
import yaml
from pathlib import Path
from typing import Optional


class MasteringInference:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device: Optional[str] = None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sr = self.config["data"]["sample_rate"]
        self.segment_length = int(self.config["data"]["segment_length_sec"] * self.sr)

        self._load_model(checkpoint_path)
        print(f"Model loaded on {self.device}")

    def _load_model(self, checkpoint_path: str):
        """
        Placeholder: load DeepAFx-ST model from checkpoint.
        Replace with actual model loading once integrated.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = None
        print("NOTE: Replace _load_model with DeepAFx-ST model loading")

    @torch.no_grad()
    def process_file(
        self,
        input_path: str,
        output_path: str,
        overlap: float = 0.5,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Integrate DeepAFx-ST first.")

        audio, sr = torchaudio.load(input_path)
        if sr != self.sr:
            audio = torchaudio.transforms.Resample(sr, self.sr)(audio)

        hop = int(self.segment_length * (1 - overlap))
        n_samples = audio.shape[-1]

        output = torch.zeros_like(audio)
        weight = torch.zeros(1, n_samples)

        window = torch.hann_window(self.segment_length)

        for start in range(0, n_samples - self.segment_length + 1, hop):
            end = start + self.segment_length
            chunk = audio[:, start:end].unsqueeze(0).to(self.device)

            processed = self.model(chunk).squeeze(0).cpu()

            output[:, start:end] += processed * window
            weight[:, start:end] += window

        mask = weight > 1e-8
        output[:, mask.squeeze()] /= weight[:, mask.squeeze()]

        remaining = n_samples % hop
        if remaining > 0 and n_samples > self.segment_length:
            chunk = audio[:, -self.segment_length:].unsqueeze(0).to(self.device)
            processed = self.model(chunk).squeeze(0).cpu()
            unprocessed = mask.squeeze() == False
            if unprocessed.any():
                output[:, unprocessed] = processed[:, -(~mask).sum():]

        inf_cfg = self.config["inference"]
        if inf_cfg.get("dither"):
            dither = torch.randn_like(output) * (1.0 / (2 ** 23))
            output = output + dither

        true_peak_limit = 10 ** (inf_cfg.get("true_peak_dbtp", -0.1) / 20)
        peak = output.abs().max()
        if peak > true_peak_limit:
            output = output * (true_peak_limit / peak)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, output, self.sr)

        print(f"Saved: {output_path}")
        return output_path

    def process_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(input_path.glob("*.wav"))
        print(f"Processing {len(wav_files)} files...")

        for wav_file in wav_files:
            out_file = output_path / wav_file.name
            self.process_file(str(wav_file), str(out_file))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run mastering inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input WAV file or directory")
    parser.add_argument("--output", required=True, help="Output WAV file or directory")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    engine = MasteringInference(args.checkpoint, args.config)

    if Path(args.input).is_dir():
        engine.process_directory(args.input, args.output)
    else:
        engine.process_file(args.input, args.output)
