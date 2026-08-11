import torch
import torch.nn as nn
from typing import List


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: List[int] = None):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        self.fft_sizes = fft_sizes

    def _stft_loss(self, predicted: torch.Tensor, target: torch.Tensor, fft_size: int):
        hop_length = fft_size // 4
        window = torch.hann_window(fft_size, device=predicted.device)

        pred_stft = torch.stft(
            predicted.reshape(-1, predicted.shape[-1]),
            fft_size, hop_length=hop_length, window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target.reshape(-1, target.shape[-1]),
            fft_size, hop_length=hop_length, window=window,
            return_complex=True,
        )

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        spectral_convergence = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        log_mag_loss = nn.functional.l1_loss(
            torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8)
        )

        return spectral_convergence + log_mag_loss

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=predicted.device)
        for fft_size in self.fft_sizes:
            loss = loss + self._stft_loss(predicted, target, fft_size)
        return loss / len(self.fft_sizes)


class MasteringLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        spectral_weight: float = 1.0,
        stft_sizes: List[int] = None,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.l1_loss = nn.L1Loss()
        self.stft_loss = MultiResolutionSTFTLoss(stft_sizes)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> dict:
        l1 = self.l1_loss(predicted, target)
        spectral = self.stft_loss(predicted, target)

        total = self.l1_weight * l1 + self.spectral_weight * spectral

        return {
            "total": total,
            "l1": l1,
            "spectral": spectral,
        }
