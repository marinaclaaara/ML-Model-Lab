import json

import pytest

from model_lab.make_card_transformer import gerar
from model_lab.publish_hf import arquivos_a_enviar, main

BASELINE = "results/metrics.json"  # resultados REAIS dos baselines, versionados no repositório


def _resultados_ficticios(caminho, f1, lo, hi):
    """Resultados FICTÍCIOS, só para exercitar o gerador de card (nunca usados fora dos testes)."""
    caminho.write_text(json.dumps({
        "modelo_base": "base/x", "epocas": 1, "max_len": 128, "n_treino": 100,
        "ambiente": {"gpu": "cpu", "transformers": "0", "torch": "0"},
        "teste": {"n": 50, "acuracia": 0.9, "roc_auc": 0.9, "negativo": {"recall": 0.9}},
        "bootstrap_teste_f1_macro": {"transformer": {"f1_macro": f1, "ic95": [lo, hi]}}}), encoding="utf-8")


def test_card_recusa_sem_resultados_reais(tmp_path):
    with pytest.raises(SystemExit, match="rode o fine-tuning"):
        gerar(tmp_path / "nao_existe.json", __import__("pathlib").Path(BASELINE), "u/m")


@pytest.mark.parametrize("f1,lo,hi,trecho", [
    (0.99, 0.98, 0.995, "estatisticamente distinguível"),   # IC todo acima do baseline
    (0.50, 0.45, 0.55, "foi pior"),                         # IC todo abaixo
    (0.943, 0.93, 0.955, "não há evidência de diferença"),  # ICs se sobrepõem
])
def test_veredito_do_card_e_calculado_a_partir_dos_intervalos(tmp_path, f1, lo, hi, trecho):
    from pathlib import Path
    r = tmp_path / "r.json"
    _resultados_ficticios(r, f1, lo, hi)
    card = gerar(r, Path(BASELINE), "usuario/modelo")
    assert trecho in card and "cc-by-nc-sa-4.0" in card and "NÃO COMERCIAL" in card and "usuario/modelo" in card


def test_publicar_recusa_sem_modelo_sem_card_e_com_card_do_baseline(tmp_path):
    from pathlib import Path
    with pytest.raises(SystemExit, match="não tem um modelo"):
        arquivos_a_enviar(tmp_path, tmp_path / "card.md")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(SystemExit, match="não existe"):
        arquivos_a_enviar(tmp_path, tmp_path / "card.md")
    with pytest.raises(SystemExit, match="baseline"):
        arquivos_a_enviar(tmp_path, Path("MODEL_CARD.md"))  # o card do baseline está marcado NÃO PUBLICADO


def test_publicar_exige_autorizacao_e_token(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    card = tmp_path / "card.md"
    card.write_text("card do transformer")
    with pytest.raises(SystemExit, match="recusada"):
        main("u/m", tmp_path, card, autorizado=False, simular=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        main("u/m", tmp_path, card, autorizado=True, simular=False)
    assert "SIMULAÇÃO" in main("u/m", tmp_path, card, autorizado=False, simular=True)  # simular nunca envia
