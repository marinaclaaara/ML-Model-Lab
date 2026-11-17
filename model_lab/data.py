"""Download verificável do B2W-Reviews01 (reviews reais de e-commerce em PT-BR).

Licença do dataset: CC BY-NC-SA 4.0 (crédito a B2W Digital, uso NÃO comercial, derivados sob a mesma licença).
O arquivo NÃO é versionado neste repositório (49 MB + licença): o script baixa e confere o hash.
Referência: Real et al., "B2W-Reviews01: an open product reviews corpus", STIL 2019.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/master/B2W-Reviews01.csv"
DESTINO = Path("data/B2W-Reviews01.csv")
HASH_ESPERADO = Path("data/B2W-Reviews01.sha256")  # gravado no 1º download; verificado nos seguintes


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(destino: Path = DESTINO) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        print(f"baixando {URL} ...")
        urllib.request.urlretrieve(URL, destino)  # noqa: S310 - URL fixa e confiável (https)
    atual = sha256(destino)
    if HASH_ESPERADO.exists():
        if HASH_ESPERADO.read_text().strip() != atual:
            raise RuntimeError("Hash do dataset difere do esperado: arquivo alterado ou corrompido.")
    else:
        HASH_ESPERADO.write_text(atual + "\n")
    return destino


def carregar(caminho: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(caminho or baixar(), low_memory=False)
