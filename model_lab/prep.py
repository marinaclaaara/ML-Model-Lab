"""Preparação: rótulo binário, limpeza, remoção de duplicatas/conflitos e divisão SEM vazamento.

Decisões (todas documentadas porque mudam o resultado):
  * rótulo vem da NOTA: 1-2 = negativo (0), 4-5 = positivo (1); nota 3 é descartada (sentimento ambíguo).
    Limitação: a nota é um rótulo "fraco" (o cliente pode dar 4 e reclamar no texto).
  * só o TEXTO da review é usado como entrada. Título, marca, idade, gênero e estado NÃO entram:
    o título "vaza" o sentimento e os dados demográficos geram risco de viés.
  * textos idênticos são removidos; se o mesmo texto aparece com rótulos DIFERENTES, todos são descartados.
  * a divisão é por AUTOR (reviewer_id): um autor nunca aparece em dois conjuntos.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"linhas_originais": len(df)}
    d = df[["reviewer_id", "product_id", "site_category_lv1", "overall_rating", "review_text"]].copy()
    d = d.dropna(subset=["review_text"])
    log["apos_remover_sem_texto"] = len(d)
    d["texto"] = d["review_text"].astype(str).str.strip()
    d = d[d["texto"].str.len() >= 3]
    log["apos_remover_textos_muito_curtos"] = len(d)
    d = d[d["overall_rating"] != 3]
    log["apos_remover_nota_3"] = len(d)
    d["rotulo"] = (d["overall_rating"] >= 4).astype(int)

    d["_chave"] = d["texto"].str.lower().str.replace(r"\s+", " ", regex=True)
    rotulos_por_texto = d.groupby("_chave")["rotulo"].nunique()
    conflitantes = set(rotulos_por_texto[rotulos_por_texto > 1].index)
    d = d[~d["_chave"].isin(conflitantes)]
    log["textos_com_rotulos_conflitantes_removidos"] = len(conflitantes)
    antes = len(d)
    d = d.drop_duplicates("_chave")
    log["duplicatas_removidas"] = antes - len(d)
    log["linhas_finais"] = len(d)
    return d.drop(columns=["_chave", "review_text"]).reset_index(drop=True), log


def dividir(d: pd.DataFrame, seed: int = SEED, fracoes=(0.8, 0.1, 0.1)) -> pd.DataFrame:
    """80/10/10 agrupando por autor. Devolve o df com a coluna `split`."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=fracoes[1] + fracoes[2], random_state=seed)
    i_tr, i_resto = next(g1.split(d, groups=d["reviewer_id"]))
    resto = d.iloc[i_resto]
    g2 = GroupShuffleSplit(n_splits=1, test_size=fracoes[2] / (fracoes[1] + fracoes[2]), random_state=seed)
    i_val, i_te = next(g2.split(resto, groups=resto["reviewer_id"]))
    d = d.copy()
    d["split"] = "train"
    d.loc[resto.index[i_val], "split"] = "val"
    d.loc[resto.index[i_te], "split"] = "test"
    return d


def diagnostico_split(d: pd.DataFrame) -> dict:
    """Verifica vazamento (autores) e mede o que NÃO controlamos (produtos compartilhados)."""
    por = {s: d[d["split"] == s] for s in ("train", "val", "test")}
    autores = {s: set(x["reviewer_id"]) for s, x in por.items()}
    prod_treino = set(por["train"]["product_id"])
    return {
        "tamanho": {s: len(x) for s, x in por.items()},
        "proporcao_positivos": {s: round(float(x["rotulo"].mean()), 3) for s, x in por.items()},
        "autores_compartilhados": {"train_val": len(autores["train"] & autores["val"]),
                                   "train_test": len(autores["train"] & autores["test"]),
                                   "val_test": len(autores["val"] & autores["test"])},
        "teste_com_produto_visto_no_treino": round(float(por["test"]["product_id"].isin(prod_treino).mean()), 3),
        "nota": "Produtos NÃO são controlados: parte do teste fala de produtos que aparecem no treino.",
    }
