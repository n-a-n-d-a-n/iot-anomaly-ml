"""Configuration management using YAML files with dot-notation access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Singleton configuration loader with dot-notation key access.

    Loads a YAML config file once and provides nested key lookups via
    dotted strings (e.g. ``"database.path"``).

    Example::

        cfg = Config.load("configs/config.yaml")
        db_path = cfg.get("database.path", "data/default.db")
    """

    _instance: Config | None = None
    _config: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str = "configs/config.yaml") -> Config:
        """Load configuration from *path* (cached after first call).

        Args:
            path: Filesystem path to a YAML configuration file.

        Returns:
            The singleton ``Config`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if cls._instance is None:
            cls._instance = cls()
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(config_path) as fh:
                cls._instance._config = yaml.safe_load(fh)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton so a fresh config can be loaded (useful in tests)."""
        cls._instance = None
        cls._config = None

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value using dot-separated *key*.

        Args:
            key: Dot-separated path into the config dict (e.g. ``"models.autoencoder.lr"``).
            default: Fallback returned when the key is missing.

        Returns:
            The resolved value, or *default* if any segment is absent.
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
