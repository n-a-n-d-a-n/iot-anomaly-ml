"""Tests for configuration management."""

import pytest
import yaml

from src.utils.config import Config


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset singleton between tests."""
    Config.reset()
    yield
    Config.reset()


@pytest.fixture()
def config_file(tmp_path):
    """Create a temporary config YAML file."""
    data = {
        "database": {"path": "data/test.db"},
        "models": {
            "isolation_forest": {"contamination": 0.05, "n_estimators": 100},
            "autoencoder": {"hidden_dims": [32, 16, 8], "epochs": 50, "lr": 0.001},
        },
        "sensors": {"temperature": {"min": 60, "max": 80}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def test_load_valid_config(config_file: str) -> None:
    cfg = Config.load(config_file)
    assert cfg is not None
    assert cfg.get("database.path") == "data/test.db"


def test_load_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Config.load("/nonexistent/path/config.yaml")


def test_singleton_returns_same_instance(config_file: str) -> None:
    cfg1 = Config.load(config_file)
    cfg2 = Config.load(config_file)
    assert cfg1 is cfg2


def test_get_nested_key(config_file: str) -> None:
    cfg = Config.load(config_file)
    assert cfg.get("models.isolation_forest.contamination") == 0.05
    assert cfg.get("models.autoencoder.hidden_dims") == [32, 16, 8]


def test_get_with_default(config_file: str) -> None:
    cfg = Config.load(config_file)
    assert cfg.get("nonexistent.key", "fallback") == "fallback"
    assert cfg.get("database.missing", 42) == 42


def test_get_top_level_key(config_file: str) -> None:
    cfg = Config.load(config_file)
    db = cfg.get("database")
    assert isinstance(db, dict)
    assert db["path"] == "data/test.db"


def test_reset_allows_reload(config_file: str) -> None:
    cfg1 = Config.load(config_file)
    Config.reset()
    cfg2 = Config.load(config_file)
    assert cfg1 is not cfg2
