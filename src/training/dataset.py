import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class MasteringDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        split: str = "train",
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
    ):
        self.split_dir = Path(split_dir) / split
        self.pre_dir = self.split_dir / "pre"
        self.master_dir = self.split_dir / "master"
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        self.files = sorted(self.pre_dir.glob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No WAV files found in {self.pre_dir}")

        print(f"  {split}: {len(self.files)} chunks")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        pre_path = self.files[idx]
        master_path = self.master_dir / pre_path.name

        pre, sr = torchaudio.load(str(pre_path))
        master, _ = torchaudio.load(str(master_path))

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            pre = resampler(pre)
            master = resampler(master)

        if self.segment_length and pre.shape[-1] > self.segment_length:
            max_start = pre.shape[-1] - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            pre = pre[:, start : start + self.segment_length]
            master = master[:, start : start + self.segment_length]

        min_len = min(pre.shape[-1], master.shape[-1])
        pre = pre[:, :min_len]
        master = master[:, :min_len]

        return {
            "pre": pre,
            "master": master,
            "name": pre_path.stem,
        }
