"""Testes de src.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import DataConfig, load_config

VALID = {
    "url": "https://example.com/reviews.csv",
    "sha256": "a" * 64,
    "raw_path": "data/raw/reviews.csv",
    "processed_dir": "data/processed",
    "text_column": "review_text",
    "rating_column": "overall_rating",
    "seed": 42,
    "val_size": 0.1,
    "test_size": 0.1,
    "min_chars": 3,
}


def test_repo_data_config_is_valid() -> None:
    cfg = load_config(Path("configs/data.yaml"), DataConfig)
    assert cfg.seed == 42
    assert cfg.val_size + cfg.test_size == pytest.approx(0.2)


@pytest.mark.parametrize(
    "override",
    [{"val_size": 0.5, "test_size": 0.5}, {"min_chars": 0}, {"sha256": "xyz"}, {"extra": 1}],
)
def test_invalid_data_config(override: dict) -> None:
    with pytest.raises(ValidationError):
        DataConfig.model_validate(VALID | override)
