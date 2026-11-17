"""Gera o model card do Transformer a partir dos resultados REAIS (results/metrics_transformer.json).

Nenhum número é digitado à mão: se o arquivo não existir, o script recusa. Isso impede publicar um card
com métricas inventadas ou do modelo errado.

Uso:  python -m model_lab.make_card_transformer --usuario SEU_USUARIO_HF
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CARD = """---
language: pt
license: cc-by-nc-sa-4.0
base_model: {base}
datasets:
- B2W-Reviews01
tags:
- sentiment-analysis
- text-classification
- portuguese
metrics:
- f1
- accuracy
pipeline_tag: text-classification
---
# {nome}: sentimento em português (BERTimbau ajustado)

Classifica reviews de e-commerce em português como **positivo** ou **negativo**. Projeto de estudo/portfólio.

> **Licença: uso NÃO COMERCIAL.** Treinado no B2W-Reviews01 (B2W Digital, CC BY-NC-SA 4.0). Crédito à B2W Digital.
> Redistribuição de derivados sob a mesma licença. Não use em produto comercial.

## Uso

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo}")
clf("Produto excelente, chegou antes do prazo!")
```

## Dados e treino
* B2W-Reviews01, 110.859 reviews após limpeza (duplicatas e nota 3 removidas); rótulo pela nota (1-2 negativo, 4-5 positivo).
* Divisão 80/10/10 **por autor** (nenhum autor em dois conjuntos). Só o texto é usado como entrada.
* Base: `{base}`. {epocas} épocas, `max_len`={max_len}, {n_treino} exemplos de treino, ambiente: {gpu}, transformers {tf}, torch {torch}.

## Resultados (teste, n={n_teste}), medidos pelo autor
| Modelo | Acurácia | F1 macro (IC 95%) | ROC-AUC | Recall negativo |
|---|---|---|---|---|
| **Este (Transformer)** | {acc_t} | {f1_t} ({lo_t} a {hi_t}) | {auc_t} | {rn_t} |
| TF-IDF + Linear SVM (baseline) | {acc_b} | {f1_b} ({lo_b} a {hi_b}) | {auc_b} | {rn_b} |
| Classe majoritária (piso) | {acc_m} | {f1_m} | 0,500 | 0,000 |

{veredito}

## Limitações
* Rótulo por nota é ruidoso (elogios com nota baixa e vice-versa); a proporção não foi quantificada.
* Só e-commerce brasileiro; não avaliado em outros domínios, gírias novas ou sarcasmo. Sem classe neutra.
* Produtos compartilhados entre treino e teste (~70% do teste) podem inflar levemente o resultado.
* O Transformer usa perda sem `class_weight` e limiar 0,5; os baselines usam `class_weight=balanced`.
* Textos acima de {max_len} tokens são truncados.
"""


def pt(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def gerar(resultados: Path, baseline: Path, repo: str) -> str:
    if not resultados.exists():
        raise SystemExit(f"{resultados} não existe: rode o fine-tuning antes. Não geramos card sem resultados reais.")
    t = json.loads(resultados.read_text(encoding="utf-8"))
    b = json.loads(baseline.read_text(encoding="utf-8"))
    bs_b, bs_t = b["bootstrap_teste_f1_macro"]["linearsvm"], t["bootstrap_teste_f1_macro"]["transformer"]
    svm, maj, tt = b["modelos"]["linearsvm"]["teste"], b["modelos"]["majoritaria"]["teste"], t["teste"]
    # o veredito é CALCULADO: só diz "melhor" se os intervalos de confiança não se sobrepõem
    if bs_t["ic95"][0] > bs_b["ic95"][1]:
        veredito = "O IC 95% do Transformer fica acima do IC do baseline: a melhora é estatisticamente distinguível."
    elif bs_t["ic95"][1] < bs_b["ic95"][0]:
        veredito = "O IC 95% do Transformer fica abaixo do baseline: neste experimento ele foi pior."
    else:
        veredito = "Os intervalos de confiança se sobrepõem: **não há evidência de diferença** entre o Transformer e o baseline."
    return CARD.format(
        base=t["modelo_base"], nome=repo.split("/")[-1], repo=repo, epocas=t["epocas"], max_len=t["max_len"],
        n_treino=t["n_treino"], gpu=t["ambiente"]["gpu"], tf=t["ambiente"]["transformers"], torch=t["ambiente"]["torch"],
        n_teste=tt["n"], acc_t=pt(tt["acuracia"]), f1_t=pt(bs_t["f1_macro"]), lo_t=pt(bs_t["ic95"][0]),
        hi_t=pt(bs_t["ic95"][1]), auc_t=pt(tt["roc_auc"]), rn_t=pt(tt["negativo"]["recall"]),
        acc_b=pt(svm["acuracia"]), f1_b=pt(bs_b["f1_macro"]), lo_b=pt(bs_b["ic95"][0]), hi_b=pt(bs_b["ic95"][1]),
        auc_b=pt(svm["roc_auc"]), rn_b=pt(svm["negativo"]["recall"]), acc_m=pt(maj["acuracia"]), f1_m=pt(maj["f1_macro"]),
        veredito=veredito)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usuario", required=True, help="seu usuário no Hugging Face")
    ap.add_argument("--nome", default="bertimbau-sentimento-b2w")
    ap.add_argument("--resultados", type=Path, default=Path("results/metrics_transformer.json"))
    ap.add_argument("--baseline", type=Path, default=Path("results/metrics.json"))
    ap.add_argument("--saida", type=Path, default=Path("results/MODEL_CARD_transformer.md"))
    a = ap.parse_args()
    texto = gerar(a.resultados, a.baseline, f"{a.usuario}/{a.nome}")
    a.saida.write_text(texto, encoding="utf-8")
    print(f"card gerado em {a.saida}. REVISE antes de publicar.")
