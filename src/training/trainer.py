import torch
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import MasteringDataset
from .losses import MasteringLoss


class Trainer:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._setup_data()
        self._setup_model()
        self._setup_training()

    def _setup_data(self):
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        sr = data_cfg["sample_rate"]
        segment_length = int(data_cfg["segment_length_sec"] * sr)

        print("Loading datasets...")
        self.train_dataset = MasteringDataset(
            data_cfg["splits_dir"], "train", sr, segment_length
        )
        self.val_dataset = MasteringDataset(
            data_cfg["splits_dir"], "val", sr, segment_length
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def _setup_model(self):
        """
        Placeholder for DeepAFx-ST model loading.
        Replace this with the actual DeepAFx-ST model once the repo is cloned
        and dependencies are resolved.
        """
        self.model = None
        print("NOTE: Model not yet loaded — integrate DeepAFx-ST here")
        print("See: https://github.com/adobe-research/DeepAFx-ST")

    def _setup_training(self):
        train_cfg = self.config["training"]
        loss_cfg = train_cfg["loss"]

        self.criterion = MasteringLoss(
            l1_weight=loss_cfg["l1_weight"],
            spectral_weight=loss_cfg["spectral_weight"],
            stft_sizes=loss_cfg["stft_sizes"],
        )

        self.max_epochs = train_cfg["max_epochs"]
        self.patience = train_cfg["early_stopping_patience"]

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = models_dir
        self.writer = SummaryWriter(log_dir=str(models_dir / "tensorboard"))

    def _train_epoch(self, epoch: int) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Integrate DeepAFx-ST first.")

        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            pre = batch["pre"].to(self.device)
            master = batch["master"].to(self.device)

            self.optimizer.zero_grad()
            predicted = self.model(pre)
            losses = self.criterion(predicted, master)
            losses["total"].backward()
            self.optimizer.step()

            total_loss += losses["total"].item()
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "l1": f"{losses['l1'].item():.4f}",
                "spec": f"{losses['spectral'].item():.4f}",
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Integrate DeepAFx-ST first.")

        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            pre = batch["pre"].to(self.device)
            master = batch["master"].to(self.device)

            predicted = self.model(pre)
            losses = self.criterion(predicted, master)
            total_loss += losses["total"].item()

        return total_loss / len(self.val_loader)

    def train(self):
        if self.model is None:
            print("\n" + "=" * 60)
            print("Cannot train: DeepAFx-ST model not yet integrated.")
            print("Steps to integrate:")
            print("  1. Clone: git clone https://github.com/adobe-research/DeepAFx-ST.git")
            print("  2. Install its dependencies")
            print("  3. Update _setup_model() in this file to load the model")
            print("  4. Configure the effect chain in configs/default.yaml")
            print("=" * 60)
            return

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nStarting training for up to {self.max_epochs} epochs")
        print(f"Early stopping patience: {self.patience}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("lr", lr, epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": self.config,
                }
                torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        self.writer.close()
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train mastering model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()
