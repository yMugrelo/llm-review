"""Testes de src.preprocessing com um DataFrame sintético (sem dados reais)."""

from itertools import combinations

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    deduplicate,
    map_rating_to_label,
    normalize_for_dedup,
    prepare,
    stratified_split,
)

SEED = 42
N_POS, N_NEG, N_NEU = 30, 20, 10  # múltiplos de 10: splits 80/10/10 exatos


@pytest.fixture
def synthetic_reviews() -> pd.DataFrame:
    """60 textos únicos + ruído que o pipeline deve remover (14 linhas)."""
    rows: list[tuple[str | None, int]] = []
    rows += [(f"Avaliação {i}: produto excelente, chegou rápido!", 4 + i % 2) for i in range(N_POS)]
    rows += [(f"Avaliação {i}: péssimo, veio quebrado.", 1 + i % 2) for i in range(N_NEG)]
    rows += [(f"Avaliação {i}: é ok, nada demais.", 3) for i in range(N_NEU)]
    unique = list(rows)

    # 8 duplicatas com o mesmo rótulo (caixa/espaços diferentes; nota 4 vs 5 = mesmo rótulo).
    for text, rating in unique[:6] + unique[N_POS : N_POS + 2]:
        variant = "  " + text.upper().replace(" ", "   ") + " "
        other = {4: 5, 5: 4}.get(rating, rating)
        rows.append((variant, other))
    # 2 textos em conflito (rótulos diferentes) x 2 cópias cada.
    rows += [("Produto mediano mesmo", 1), ("produto  MEDIANO mesmo", 5)]
    rows += [("Nada a declarar", 3), ("nada a declarar", 4)]
    # 1 texto curto (< min_chars) e 1 nulo.
    rows += [("ok", 5), (None, 4)]

    return pd.DataFrame(rows, columns=["text", "rating"]).astype(
        {"text": "string", "rating": "Int64"}
    )


def run(df: pd.DataFrame, seed: int = SEED) -> tuple[dict[str, pd.DataFrame], dict]:
    return prepare(df, min_chars=3, val_size=0.1, test_size=0.1, seed=seed)


@pytest.mark.parametrize(
    ("rating", "label"),
    [(1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo")],
)
def test_map_rating_to_label(rating: int, label: str) -> None:
    assert map_rating_to_label(rating) == label


@pytest.mark.parametrize("rating", [0, 6, -1, None, "5"])
def test_map_rating_to_label_invalid(rating: object) -> None:
    with pytest.raises(ValueError, match="Nota inválida"):
        map_rating_to_label(rating)  # type: ignore[arg-type]


def test_clean_text_preserves_signal() -> None:
    nfd = "Péssimo"  # "é" decomposto (e + acento combinante)
    assert clean_text(f"  {nfd}!!\n\tNão  recomendo 😡 ") == "Péssimo!! Não recomendo 😡"
    assert clean_text(nfd) == "Péssimo"  # "é" composto (1 code point)
    assert clean_text("Ótimo 👍🏽, recomendo!") == "Ótimo 👍🏽, recomendo!"


def test_prepare_drop_counts(synthetic_reviews: pd.DataFrame) -> None:
    _, stats = run(synthetic_reviews)
    assert stats["dropped_null_text"] == 1
    assert stats["dropped_min_chars"] == 1
    assert stats["duplicates_removed"] == 8
    assert stats["conflict_texts"] == 2
    assert stats["conflict_rows_removed"] == 4
    assert stats["rows_final"] == N_POS + N_NEG + N_NEU
    assert sum(s["n"] for s in stats["splits"].values()) == stats["rows_final"]


def test_deduplicate_keeps_first_occurrence() -> None:
    df = pd.DataFrame({"text": ["Bom produto", "BOM  produto"], "label": ["positivo"] * 2})
    out, stats = deduplicate(df)
    assert out["text"].tolist() == ["Bom produto"]
    assert stats.duplicates_removed == 1
    assert stats.conflict_rows_removed == 0


def test_no_text_in_two_splits(synthetic_reviews: pd.DataFrame) -> None:
    splits, _ = run(synthetic_reviews)
    keys = {name: set(df["text"].map(normalize_for_dedup)) for name, df in splits.items()}
    for a, b in combinations(keys, 2):
        assert keys[a].isdisjoint(keys[b]), f"vazamento entre {a} e {b}"


def test_split_sizes_and_class_proportions(synthetic_reviews: pd.DataFrame) -> None:
    splits, stats = run(synthetic_reviews)
    assert {name: len(df) for name, df in splits.items()} == {"train": 48, "val": 6, "test": 6}
    total = stats["total"]["proportions"]
    for name, split in stats["splits"].items():
        for label, prop in split["proportions"].items():
            assert abs(prop - total[label]) <= 0.01, f"{name}/{label}: {prop} vs {total[label]}"


def test_prepare_is_deterministic(synthetic_reviews: pd.DataFrame) -> None:
    first, stats_a = run(synthetic_reviews)
    second, stats_b = run(synthetic_reviews.copy())
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert stats_a == stats_b


def test_seed_changes_split(synthetic_reviews: pd.DataFrame) -> None:
    """Garante que a seed é de fato usada (senão o teste de determinismo seria trivial)."""
    a, _ = run(synthetic_reviews, seed=SEED)
    b, _ = run(synthetic_reviews, seed=SEED + 1)
    assert set(a["test"]["text"]) != set(b["test"]["text"])


def test_stratified_split_rejects_tiny_class() -> None:
    df = pd.DataFrame(
        {"text": [f"t{i}" for i in range(10)], "label": ["positivo"] * 9 + ["neutro"]}
    )
    with pytest.raises(ValueError):
        stratified_split(df, val_size=0.1, test_size=0.1, seed=SEED)
