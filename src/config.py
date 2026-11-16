"""Schemas pydantic das configs de experimento (configs/*.yaml)."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DataConfig(BaseModel):
    """Fonte, caminhos e parâmetros do pipeline de dados."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_path: Path
    processed_dir: Path
    text_column: str
    rating_column: str
    seed: int
    val_size: float = Field(gt=0, lt=1)
    test_size: float = Field(gt=0, lt=1)
    min_chars: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_split_sizes(self) -> "DataConfig":
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size precisa ser < 1")
        return self


def load_config(path: str | Path, schema: type[SchemaT]) -> SchemaT:
    """Lê um YAML e valida contra o schema pydantic informado."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return schema.model_validate(raw)
