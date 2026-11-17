"""Fine-tuning de um Transformer pré-treinado em português (BERTimbau) no MESMO split dos baselines.

Roda no Colab/Kaggle com GPU:
    pip install torch transformers datasets accelerate
    python -m model_lab.transformer_finetune --max-train 20000      # ensaio rápido (confira se tudo funciona)
    python -m model_lab.transformer_finetune                        # treino completo

STATUS: a lógica de treino e avaliação é testada com um modelo MINÚSCULO criado localmente
(tests/test_transformer_smoke.py), o que valida o código. O BERTimbau de verdade ainda NÃO foi executado
(sem GPU e sem acesso ao Hugging Face no ambiente de desenvolvimento).

Diferença para os baselines (declare ao comparar): o Transformer usa perda comum e limiar 0.5, sem
`class_weight=balanced`. O F1 macro continua sendo a métrica principal.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import baixar, carregar
from .prep import SEED, dividir, preparar
from .train import bootstrap_f1, metricas

MODELO = "neuralmind/bert-base-portuguese-cased"  # BERTimbau base. CONFIRME a licença na página do modelo no Hub.
ROTULOS = {0: "negativo", 1: "positivo"}


def prever(modelo, tok, textos: list[str], max_len: int, lote: int = 64) -> np.ndarray:
    """Probabilidade da classe 'positivo' para cada texto (inferência em lotes, sem gradiente)."""
    import torch
    modelo.eval()
    saida = []
    for i in range(0, len(textos), lote):
        enc = tok(textos[i:i + lote], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(modelo.device) for k, v in enc.items()}
        with torch.no_grad():
            saida.append(torch.softmax(modelo(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(saida)


def treinar_e_avaliar(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, modelo_id: str, saida: Path,
                      epocas: int = 2, max_len: int = 128, batch: int = 32, lr: float = 2e-5) -> dict:
    """Treina em `tr`, avalia em `va` e `te`. Separado de `main` para ser testável sem baixar o dataset."""
    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(modelo_id)
    # id2label/label2id: o Hub mostra "negativo"/"positivo" em vez de LABEL_0/LABEL_1
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_id, num_labels=2, id2label=ROTULOS, label2id={v: k for k, v in ROTULOS.items()})

    def tokenizar(lote):
        return tok(lote["texto"], truncation=True, max_length=max_len)

    base = tr[["texto", "rotulo"]].rename(columns={"rotulo": "labels"})
    ds_tr = Dataset.from_pandas(base, preserve_index=False).map(tokenizar, batched=True)
    # Aquecimento (warmup): a taxa de aprendizado sobe devagar nos primeiros ~6% dos passos, o que estabiliza o início.
    # Usamos passos (inteiro) e não `warmup_ratio`, que as versões novas do transformers removeram.
    aquecimento = int(0.06 * math.ceil(len(ds_tr) / batch) * epocas)
    obrigatorios = dict(output_dir=str(saida / "ckpt"), num_train_epochs=epocas, per_device_train_batch_size=batch,
                        learning_rate=lr, warmup_steps=aquecimento, seed=SEED, save_strategy="no")
    # Opcionais: se uma versão futura do transformers renomear/remover algum, avisamos e seguimos sem ele,
    # em vez de abortar o treino. (Os obrigatórios acima definem o experimento; esses só ajustam conforto.)
    opcionais = dict(weight_decay=0.01, report_to="none", fp16=torch.cuda.is_available(), logging_steps=50)
    aceitos = set(inspect.signature(TrainingArguments).parameters)
    ignorados = sorted(k for k in opcionais if k not in aceitos)
    if ignorados:
        print(f"AVISO: esta versão do transformers não aceita {ignorados}; seguindo sem eles.")
    args = TrainingArguments(**obrigatorios, **{k: v for k, v in opcionais.items() if k in aceitos})
    try:  # transformers >= 4.46 usa processing_class; versões antigas usam tokenizer
        trainer = Trainer(model=modelo, args=args, train_dataset=ds_tr, processing_class=tok)
    except TypeError:
        trainer = Trainer(model=modelo, args=args, train_dataset=ds_tr, tokenizer=tok)
    trainer.train()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rel: dict = {"modelo_base": modelo_id, "epocas": epocas, "max_len": max_len, "n_treino": int(len(tr)),
                 "ambiente": {"transformers": transformers.__version__, "torch": torch.__version__, "gpu": gpu,
                              "python": platform.python_version()}}
    scores = {}
    for nome, d in (("validacao", va), ("teste", te)):
        s = prever(modelo, tok, list(d["texto"]), max_len)
        scores[nome] = s
        rel[nome] = metricas(d["rotulo"].to_numpy(), s, 0.5)
    y_te = te["rotulo"].to_numpy()
    rel["bootstrap_teste_f1_macro"] = bootstrap_f1(y_te, {"transformer": (scores["teste"] >= 0.5).astype(int)})
    rel["tempo_total_s"] = round(time.time() - t0)

    saida.mkdir(parents=True, exist_ok=True)
    (saida / "metrics_transformer.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    modelo.save_pretrained(saida / "transformer_model")
    tok.save_pretrained(saida / "transformer_model")
    return rel


def main(max_train: int | None, epocas: int, saida: Path, modelo_id: str = MODELO) -> dict:
    d, _ = preparar(carregar(baixar()))
    d = dividir(d)  # mesma semente => EXATAMENTE o mesmo split dos baselines
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    if max_train:
        tr = tr.sample(min(max_train, len(tr)), random_state=SEED)
    return treinar_e_avaliar(tr, va, te, modelo_id, saida, epocas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=None, help="usa só N exemplos de treino (ensaio rápido)")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--saida", type=Path, default=Path("results"))
    ap.add_argument("--modelo", default=MODELO)
    a = ap.parse_args()
    print(json.dumps(main(a.max_train, a.epocas, a.saida, a.modelo), indent=2, ensure_ascii=False))
