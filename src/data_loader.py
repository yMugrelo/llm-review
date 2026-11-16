"""Download e leitura do dataset B2W-Reviews01."""

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcula o sha256 de um arquivo lendo em blocos (não carrega tudo na memória)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str, timeout: float = 60.0) -> Path:
    """Baixa `url` para `dest` só se o arquivo não existe ou o checksum não bate.

    O download vai para `<dest>.part` e só é renomeado após validar o checksum,
    então `dest` nunca fica corrompido ou pela metade.
    """
    if dest.exists():
        if sha256_file(dest) == sha256:
            logger.info("Arquivo já existe com checksum válido, sem download: %s", dest)
            return dest
        logger.warning("Checksum de %s não bate; baixando novamente", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    logger.info("Baixando %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)

    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ValueError(f"Checksum inválido para {url}: esperado {sha256}, obtido {actual}")
    tmp.replace(dest)
    return dest


def read_reviews(path: Path, text_column: str, rating_column: str) -> pd.DataFrame:
    """Lê o CSV bruto mantendo só texto e nota, renomeados para `text` e `rating`."""
    df = pd.read_csv(
        path,
        usecols=[text_column, rating_column],
        dtype={text_column: "string", rating_column: "Int64"},
    )
    return df.rename(columns={text_column: "text", rating_column: "rating"})[["text", "rating"]]


def read_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Lê os splits train/val/test gerados por `src.preprocessing`."""
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in SPLITS}
