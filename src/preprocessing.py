"""Limpeza, deduplicação, mapeamento de rótulos e split estratificado."""

import argparse
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataConfig, load_config
from src.data_loader import SPLITS, download, read_reviews

logger = logging.getLogger(__name__)

LABELS = ("negativo", "neutro", "positivo")
_RATING_TO_LABEL = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupStats:
    """Contagens da deduplicação."""

    duplicates_removed: int
    conflict_texts: int
    conflict_rows_removed: int


def clean_text(text: str) -> str:
    """Normaliza unicode (NFC) e espaços. Preserva caixa, acentos, pontuação e emojis."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalize_for_dedup(text: str) -> str:
    """Chave de deduplicação: texto limpo em minúsculas."""
    return clean_text(text).lower()


def map_rating_to_label(rating: int) -> str:
    """Mapeia nota 1–5 para negativo/neutro/positivo. Levanta ValueError se inválida."""
    try:
        return _RATING_TO_LABEL[rating]
    except (KeyError, TypeError):
        raise ValueError(f"Nota inválida: {rating!r} (esperado inteiro de 1 a 5)") from None


def clean_and_filter(df: pd.DataFrame, min_chars: int) -> tuple[pd.DataFrame, int, int]:
    """Limpa `text` e descarta nulos e textos com menos de `min_chars` caracteres.

    Retorna (df, n_nulos, n_curtos).
    """
    null_mask = df["text"].isna()
    cleaned = df.loc[~null_mask].assign(text=lambda d: d["text"].map(clean_text))
    short_mask = cleaned["text"].str.len() < min_chars
    kept = cleaned.loc[~short_mask].reset_index(drop=True)
    return kept, int(null_mask.sum()), int(short_mask.sum())


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupStats]:
    """Remove duplicatas por texto normalizado, mantendo a primeira ocorrência.

    Textos que aparecem com rótulos diferentes são conflitos: todas as cópias saem.
    """
    df = df.reset_index(drop=True)
    key = df["text"].map(normalize_for_dedup)
    conflict = df.groupby(key)["label"].transform("nunique") > 1
    duplicated = key.duplicated(keep="first") & ~conflict
    stats = DedupStats(
        duplicates_removed=int(duplicated.sum()),
        conflict_texts=int(key[conflict].nunique()),
        conflict_rows_removed=int(conflict.sum()),
    )
    return df.loc[~conflict & ~duplicated].reset_index(drop=True), stats


def stratified_split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado por `label` em treino/validação/teste."""
    holdout_size = val_size + test_size
    train, holdout = train_test_split(
        df, test_size=holdout_size, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        holdout,
        test_size=test_size / holdout_size,
        stratify=holdout["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def class_counts(labels: pd.Series) -> dict[str, int]:
    """Contagem por classe, sempre na ordem de LABELS."""
    counts = labels.value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def prepare(
    raw: pd.DataFrame, min_chars: int, val_size: float, test_size: float, seed: int
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Pipeline puro: limpeza -> rótulo -> deduplicação -> split. Retorna (splits, stats)."""
    df, n_null, n_short = clean_and_filter(raw, min_chars)
    df = df.assign(label=df["rating"].map(map_rating_to_label))
    df, dedup = deduplicate(df)
    train, val, test = stratified_split(df, val_size, test_size, seed)
    splits = dict(zip(SPLITS, (train, val, test), strict=True))

    def split_stats(part: pd.DataFrame) -> dict[str, Any]:
        counts = class_counts(part["label"])
        return {
            "n": len(part),
            "counts": counts,
            "proportions": {k: round(v / len(part), 4) for k, v in counts.items()},
        }

    stats = {
        "rows_raw": len(raw),
        "dropped_null_text": n_null,
        "dropped_min_chars": n_short,
        **asdict(dedup),
        "rows_final": len(df),
        "total": split_stats(df),
        "splits": {name: split_stats(part) for name, part in splits.items()},
    }
    return splits, stats


def write_outputs(splits: dict[str, pd.DataFrame], stats: dict[str, Any], out_dir: Path) -> None:
    """Grava os splits em parquet e as estatísticas em stats.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False, engine="pyarrow")
    payload = json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "stats.json").write_text(payload, encoding="utf-8")


# --- Funções de apoio à EDA (usadas pelo notebook) ---------------------------------------


def token_lengths(
    texts: Sequence[str],
    tokenize: Callable[[list[str]], list[list[int]]],
    batch_size: int = 1024,
) -> np.ndarray:
    """Tamanho em tokens de cada texto. `tokenize` recebe um lote e devolve os input_ids."""
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        lengths.extend(len(ids) for ids in tokenize(batch))
    return np.asarray(lengths, dtype=np.int64)


def length_percentiles(
    lengths: np.ndarray, percentiles: Sequence[int] = (50, 90, 95, 99)
) -> dict[int, int]:
    """Percentis de tamanho, arredondados para cima (servem de max_length)."""
    return {p: int(np.ceil(np.percentile(lengths, p))) for p in percentiles}


def sample_per_class(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra determinística de `n` exemplos por classe."""
    return df.groupby("label", group_keys=False).sample(n=n, random_state=seed)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint: baixa o dataset, processa e grava os splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config(args.config, DataConfig)
    raw_path = download(str(cfg.url), cfg.raw_path, cfg.sha256)
    raw = read_reviews(raw_path, cfg.text_column, cfg.rating_column)
    splits, stats = prepare(raw, cfg.min_chars, cfg.val_size, cfg.test_size, cfg.seed)
    stats = {"source": {"url": str(cfg.url), "sha256": cfg.sha256}, **stats}
    write_outputs(splits, stats, cfg.processed_dir)
    logger.info(
        "Splits gravados em %s: %s",
        cfg.processed_dir,
        {name: len(part) for name, part in splits.items()},
    )


if __name__ == "__main__":
    main()
