"""Deep Clustering for behavioural profiling (PyTorch).

Combines a dense autoencoder with K-Means clustering in the
latent space. Entities far from any cluster centroid are flagged
as anomalous. This captures group-level deviations that point
anomaly detectors may miss.

The MSc thesis used a Dense Autoencoder (TF/Keras); this
implementation extends it with an explicit clustering step
in PyTorch.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from .base import StaticDetector


def _get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class _AutoencoderModule(nn.Module):
    """Dense autoencoder for learning latent representations."""

    def __init__(
        self,
        input_dim: int,
        encoder_layers: List[int],
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()

        # Encoder
        enc_modules = []
        prev = input_dim
        for units in encoder_layers:
            enc_modules.extend([
                nn.Linear(prev, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = units
        enc_modules.append(nn.Linear(prev, latent_dim))
        enc_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_modules)

        # Decoder (mirror)
        dec_modules = []
        prev = latent_dim
        for units in reversed(encoder_layers):
            dec_modules.extend([
                nn.Linear(prev, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = units
        dec_modules.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class DeepClusteringDetector(StaticDetector):
    """Deep Clustering anomaly detector on UBFS vectors.

    Training:
        1. Pretrain autoencoder (reconstruction loss)
        2. Extract latent representations
        3. Fit K-Means on latent space
        4. Score = distance to nearest centroid

    The dual scoring combines reconstruction error with
    cluster distance for robust anomaly detection.
    """

    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        latent_dim: int = 32,
        n_clusters: int = 5,
        dropout: float = 0.2,
        batch_size: int = 64,
        pretrain_epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        recon_weight: float = 0.5,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__(name="DeepClustering", seed=seed)
        self.encoder_layers = encoder_layers or [128, 64]
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.recon_weight = recon_weight
        self.device = _get_device(device)
        self.verbose = verbose

        self.model_: Optional[_AutoencoderModule] = None
        self.kmeans_: Optional[KMeans] = None
        self.recon_threshold_: Optional[float] = None
        self.cluster_threshold_: Optional[float] = None

    def _set_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "DeepClusteringDetector":
        X = self.validate_input(X)
        self._set_seed()

        input_dim = X.shape[1]

        # Build autoencoder
        self.model_ = _AutoencoderModule(
            input_dim=input_dim,
            encoder_layers=self.encoder_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)

        # Pretrain autoencoder
        self._pretrain(X)

        # Extract latent representations and cluster
        latent = self._encode_numpy(X)
        effective_k = min(self.n_clusters, len(latent))
        self.kmeans_ = KMeans(
            n_clusters=effective_k,
            random_state=self.seed,
            n_init=10,
        )
        self.kmeans_.fit(latent)

        self.is_fitted = True

        # Set thresholds
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 95))
        return self

    def _pretrain(self, X: np.ndarray) -> None:
        """Pretrain autoencoder on reconstruction loss."""
        n_val = max(1, int(len(X) * 0.1))
        idx = np.random.permutation(len(X))
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

        for epoch in range(self.pretrain_epochs):
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

            self.model_.eval()
            with torch.no_grad():
                val_recon = self.model_(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Pretrain {epoch+1}/{self.pretrain_epochs} "
                    f"loss={epoch_loss:.6f} val={val_loss:.6f}"
                )

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
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_.to(self.device)

    def _encode_numpy(self, X: np.ndarray) -> np.ndarray:
        """Encode to latent space, returning numpy array."""
        self.model_.eval()
        tensor = torch.tensor(
            X, dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            latent = self.model_.encode(tensor)
        return latent.cpu().numpy()

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error per sample."""
        self.model_.eval()
        tensor = torch.tensor(
            X, dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            recon = self.model_(tensor)
            errors = torch.mean((tensor - recon) ** 2, dim=1)
        return errors.cpu().numpy()

    def _cluster_distance(self, X: np.ndarray) -> np.ndarray:
        """Distance to nearest cluster centroid in latent space."""
        latent = self._encode_numpy(X)
        centroids = self.kmeans_.cluster_centers_
        dists = np.linalg.norm(
            latent[:, np.newaxis] - centroids[np.newaxis, :],
            axis=2,
        )
        return np.min(dists, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Combined anomaly score: weighted reconstruction + cluster distance."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self.validate_input(X)

        recon = self._recon_error(X)
        cluster = self._cluster_distance(X)

        # Normalise each component to [0, 1] range for combination
        recon_norm = (recon - recon.min()) / max(
            recon.max() - recon.min(), 1e-8
        )
        cluster_norm = (cluster - cluster.min()) / max(
            cluster.max() - cluster.min(), 1e-8
        )

        return (
            self.recon_weight * recon_norm
            + (1 - self.recon_weight) * cluster_norm
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Get latent representations."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding")
        X = self.validate_input(X)
        return self._encode_numpy(X)

    def get_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        """Get cluster assignments for samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        X = self.validate_input(X)
        latent = self._encode_numpy(X)
        return self.kmeans_.predict(latent)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "encoder_layers": self.encoder_layers,
            "latent_dim": self.latent_dim,
            "n_clusters": self.n_clusters,
            "dropout": self.dropout,
            "pretrain_epochs": self.pretrain_epochs,
            "learning_rate": self.learning_rate,
            "recon_weight": self.recon_weight,
        }
