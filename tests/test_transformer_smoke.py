"""Teste de ponta a ponta do código de fine-tuning com um modelo MINÚSCULO criado localmente (sem rede, sem GPU).

Valida a mecânica (Dataset, Trainer, métricas, salvar, id2label, card, publicação simulada). NÃO valida a qualidade do
BERTimbau: isso só se mede no Colab.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


def _modelo_minusculo(pasta: Path) -> str:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    letras = list("abcdefghijklmnopqrstuvwxyz0123456789.,!?")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + letras + ["##" + c for c in letras]
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(pasta / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    BertForSequenceClassification(cfg).save_pretrained(pasta)
    tok.save_pretrained(pasta)
    return str(pasta)


def _df(n):
    pos = [f"produto otimo, adorei {i}" for i in range(n)]
    neg = [f"produto pessimo, detestei {i}" for i in range(n)]
    return pd.DataFrame({"texto": pos + neg, "rotulo": [1] * n + [0] * n})


def test_fine_tuning_de_ponta_a_ponta_com_modelo_minusculo(tmp_path):
    from model_lab.make_card_transformer import gerar
    from model_lab.publish_hf import main as publicar
    from model_lab.transformer_finetune import treinar_e_avaliar

    base = _modelo_minusculo(tmp_path / "tiny")
    saida = tmp_path / "out"
    rel = treinar_e_avaliar(_df(40), _df(10), _df(10), base, saida, epocas=1, max_len=32, batch=8)

    assert {"validacao", "teste", "bootstrap_teste_f1_macro", "ambiente"} <= rel.keys()
    assert 0.0 <= rel["teste"]["f1_macro"] <= 1.0 and rel["teste"]["n"] == 20
    cfg = json.loads((saida / "transformer_model" / "config.json").read_text())
    assert cfg["id2label"] == {"0": "negativo", "1": "positivo"}           # o Hub mostrará nomes, não LABEL_0/1
    assert (saida / "metrics_transformer.json").exists()

    card_txt = gerar(saida / "metrics_transformer.json", Path("results/metrics.json"), "u/bertimbau-teste")
    (saida / "card.md").write_text(card_txt, encoding="utf-8")
    msg = publicar("u/bertimbau-teste", saida / "transformer_model", saida / "card.md", False, True)
    assert "SIMULAÇÃO" in msg and "config.json" in msg and "README.md" in msg
