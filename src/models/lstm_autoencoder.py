"""LSTM Autoencoder for temporal anomaly detection (PyTorch).

Encodes sequences of UBFS vectors into a latent representation,
then reconstructs them. High reconstruction error indicates
anomalous temporal patterns.

Adapted from MSc thesis (TensorFlow) to PyTorch for better
Apple Silicon MPS support.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import TemporalDetector


def _get_device(preference: str = "auto") -> torch.device:
    """Select best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class _LSTMAutoencoderModule(nn.Module):
    """PyTorch LSTM autoencoder module.

    Architecture mirrors the MSc thesis TF/Keras implementation:
        Encoder: LSTM layers → Dense latent
        Decoder: RepeatVector → LSTM layers → Dense output
    """

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        encoder_units: List[int],
        decoder_units: List[int],
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        # Encoder LSTMs
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        for units in encoder_units:
            self.encoder_layers.append(
                nn.LSTM(
                    prev_dim, units,
                    batch_first=True, dropout=dropout,
                )
            )
            prev_dim = units

        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(encoder_units[-1], latent_dim),
            nn.ReLU(),
        )

        # Decoder LSTMs
        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim
        for units in decoder_units:
            self.decoder_layers.append(
                nn.LSTM(
                    prev_dim, units,
                    batch_first=True, dropout=dropout,
                )
            )
            prev_dim = units

        # Output projection
        self.output_proj = nn.Linear(decoder_units[-1], input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent vector."""
        h = x
        for lstm in self.encoder_layers:
            h, _ = lstm(h)
        # Take final hidden state
        h_final = h[:, -1, :]
        return self.latent_proj(h_final)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed sequence."""
        # Repeat latent for each timestep
        h = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        for lstm in self.decoder_layers:
            h, _ = lstm(h)
        return self.output_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class LSTMAutoencoderDetector(TemporalDetector):
    """LSTM Autoencoder anomaly detector on UBFS sequences.

    Anomaly score = mean squared reconstruction error.
    """

    def __init__(
        self,
        encoder_units: Optional[List[int]] = None,
        decoder_units: Optional[List[int]] = None,
        latent_dim: int = 16,
        dropout: float = 0.2,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__(name="LSTMAutoencoder", seed=seed)
        self.encoder_units = encoder_units or [64, 32]
        self.decoder_units = decoder_units or [32, 64]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.device = _get_device(device)
        self.verbose = verbose
        self.model_: Optional[_LSTMAutoencoderModule] = None
        self.history_: Dict[str, List[float]] = {
            "loss": [], "val_loss": [],
        }

    def _set_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "LSTMAutoencoderDetector":
        X = self.validate_input(X)
        self._set_seed()

        n_samples, seq_length, n_features = X.shape

        self.model_ = _LSTMAutoencoderModule(
            input_dim=n_features,
            seq_length=seq_length,
            encoder_units=self.encoder_units,
            decoder_units=self.decoder_units,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)

        # Train/val split (90/10)
        n_val = max(1, int(n_samples * 0.1))
        idx = np.random.permutation(n_samples)
        X_train = X[idx[n_val:]]
        X_val = X[idx[:n_val]]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32)
        )
        val_tensor = torch.tensor(
            X_val, dtype=torch.float32
        ).to(self.device)

        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= max(n_batches, 1)

            # Validate
            self.model_.eval()
            with torch.no_grad():
                val_recon = self.model_(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()

            self.history_["loss"].append(epoch_loss)
            self.history_["val_loss"].append(val_loss)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} "
                    f"loss={epoch_loss:.6f} "
                    f"val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch+1}"
                        )
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_.to(self.device)

        self.is_fitted = True

        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 95))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self.validate_input(X)

        self.model_.eval()
        tensor = torch.tensor(
            X, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            recon = self.model_(tensor)
            errors = torch.mean(
                (tensor - recon) ** 2, dim=(1, 2)
            )
        return errors.cpu().numpy()

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Get latent representations of sequences."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding")
        X = self.validate_input(X)

        self.model_.eval()
        tensor = torch.tensor(
            X, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            latent = self.model_.encode(tensor)
        return latent.cpu().numpy()

    def score_per_timestep(self, X: np.ndarray) -> np.ndarray:
        """Per-timestep reconstruction error (n_samples, seq_len)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self.validate_input(X)

        self.model_.eval()
        tensor = torch.tensor(
            X, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            recon = self.model_(tensor)
            errors = torch.mean((tensor - recon) ** 2, dim=2)
        return errors.cpu().numpy()

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "encoder_units": self.encoder_units,
            "decoder_units": self.decoder_units,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }
