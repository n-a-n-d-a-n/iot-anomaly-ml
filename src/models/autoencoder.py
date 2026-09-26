"""PyTorch autoencoder for anomaly detection via reconstruction error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Autoencoder(nn.Module):
    """Symmetric feedforward autoencoder.

    Args:
        input_dim: Number of input features.
        hidden_dims: Sequence of hidden layer sizes for the encoder
            (decoder mirrors in reverse).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers: list[nn.Module] = []
        for dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Reconstructed tensor of the same shape.
        """
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to the bottleneck representation.

        Args:
            x: Input tensor.

        Returns:
            Latent representation.
        """
        return self.encoder(x)


class AutoencoderDetector:
    """Anomaly detector based on autoencoder reconstruction error.

    Args:
        hidden_dims: Hidden layer sizes for the autoencoder.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimiser.
        batch_size: Mini-batch size.
        device: ``"auto"`` to prefer CUDA when available, else ``"cpu"``.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model: Autoencoder | None = None
        self.threshold: float = 0.0
        self.feature_names: list[str] = []
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
        X_val: np.ndarray | None = None,
    ) -> AutoencoderDetector:
        """Train the autoencoder and set the anomaly threshold.

        The threshold is set at the 95th percentile of training
        reconstruction errors.

        Args:
            X: Training feature array.
            feature_names: Column names.
            X_val: Optional validation set (unused but kept for API symmetry).

        Returns:
            ``self`` for method chaining.
        """
        self.feature_names = feature_names
        input_dim = X.shape[1]

        self.model = Autoencoder(input_dim, self.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, total_loss / len(loader)
                )

        # Set threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            self.threshold = float(np.percentile(errors, 95))
            self._score_min = float(errors.min())
            self._score_max = float(errors.max())

        logger.info("Autoencoder trained, threshold=%.6f", self.threshold)
        return self

    def _compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample mean squared reconstruction error.

        Args:
            X: Feature array.

        Returns:
            Error array of shape ``(n_samples,)``.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return errors.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1 = anomaly, 0 = normal.

        Args:
            X: Feature array.

        Returns:
            Binary prediction array.
        """
        errors = self._compute_reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute normalised anomaly scores in [0, 1].
        Uses training parameters to prevent data leakage.

        Args:
            X: Feature array.

        Returns:
            Score array.
        """
        errors = self._compute_reconstruction_error(X)
        error_range = (self._score_max - self._score_min) + 1e-10
        return np.clip((errors - self._score_min) / error_range, 0.0, 1.0)

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-feature reconstruction errors for root-cause analysis.

        Args:
            X: Feature array.

        Returns:
            Dict mapping feature name to normalised error array.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            per_feature_errors = (X_tensor - reconstructed) ** 2
            per_feature_errors = per_feature_errors.cpu().numpy()

        contributions: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.feature_names):
            errors = per_feature_errors[:, i]
            error_range = errors.max() - errors.min() + 1e-10
            contributions[name] = (errors - errors.min()) / error_range
        return contributions

    def save(self, path: str) -> None:
        """Persist model weights and configuration.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": {
                    "input_dim": self.model.input_dim,
                    "hidden_dims": self.hidden_dims,
                    "threshold": self.threshold,
                },
                "feature_names": self.feature_names,
                "score_min": self._score_min,
                "score_max": self._score_max,
            },
            path,
        )
        logger.info("Saved autoencoder to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> AutoencoderDetector:
        """Load model from disk.

        Args:
            path: File path saved by :meth:`save`.
            device: Torch device specifier.

        Returns:
            Restored detector instance.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(hidden_dims=data["config"]["hidden_dims"], device=device)
        detector.model = Autoencoder(
            data["config"]["input_dim"],
            data["config"]["hidden_dims"],
        ).to(detector.device)
        detector.model.load_state_dict(data["model_state"])
        detector.threshold = data["config"]["threshold"]
        detector.feature_names = data["feature_names"]
        detector._score_min = data.get("score_min", 0.0)
        detector._score_max = data.get("score_max", 1.0)
        return detector
