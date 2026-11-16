"""Testes de src.data_loader. O download é mockado: nenhum acesso à rede."""

import hashlib
import io
from pathlib import Path

import pytest

from src import data_loader
from src.data_loader import download, read_reviews

PAYLOAD = b"review_text,overall_rating,extra\nMuito bom,5,x\nRuim,1,y\n"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.com/reviews.csv"


@pytest.fixture
def fake_urlopen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(data_loader.urllib.request, "urlopen", _urlopen)
    return calls


def test_download_when_missing(tmp_path: Path, fake_urlopen: list[str]) -> None:
    dest = tmp_path / "raw" / "reviews.csv"
    assert download(URL, dest, PAYLOAD_SHA) == dest
    assert dest.read_bytes() == PAYLOAD
    assert fake_urlopen == [URL]


def test_download_skips_when_checksum_matches(tmp_path: Path, fake_urlopen: list[str]) -> None:
    dest = tmp_path / "reviews.csv"
    dest.write_bytes(PAYLOAD)
    download(URL, dest, PAYLOAD_SHA)
    assert fake_urlopen == []


def test_download_replaces_corrupted_file(tmp_path: Path, fake_urlopen: list[str]) -> None:
    dest = tmp_path / "reviews.csv"
    dest.write_bytes(b"corrompido")
    download(URL, dest, PAYLOAD_SHA)
    assert dest.read_bytes() == PAYLOAD
    assert fake_urlopen == [URL]


def test_download_rejects_bad_checksum(tmp_path: Path, fake_urlopen: list[str]) -> None:
    dest = tmp_path / "reviews.csv"
    with pytest.raises(ValueError, match="Checksum inválido"):
        download(URL, dest, "0" * 64)
    assert list(tmp_path.iterdir()) == []  # nem o arquivo final nem o .part ficam no disco


def test_read_reviews_keeps_only_needed_columns(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    path.write_bytes(PAYLOAD)
    df = read_reviews(path, "review_text", "overall_rating")
    assert df.columns.tolist() == ["text", "rating"]
    assert df["rating"].tolist() == [5, 1]
