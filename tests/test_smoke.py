"""Smoke test: o pacote src importa sem dependências pesadas (sem torch)."""

import src


def test_src_imports() -> None:
    assert src.__version__ == "0.1.0"
