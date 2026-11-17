"""Treino e avaliação dos baselines clássicos. Uso:  python -m model_lab.train

Modelos: (0) classe majoritária (piso de referência), (1) TF-IDF + Regressão Logística, (2) TF-IDF + Linear SVM.
O hiperparâmetro C de cada modelo é escolhido pelo F1 macro na VALIDAÇÃO; o teste é usado UMA vez, no final.

Por que F1 macro e não acurácia? ~71% das reviews são positivas: um modelo que sempre diz "positivo" já acerta 71%.
O F1 macro dá o mesmo peso à classe negativa (a minoria, e a que a empresa mais quer enxergar).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import baixar, carregar
from .prep import SEED, diagnostico_split, dividir, preparar

SAIDA = Path("results")
GRADES = {"logreg": [1.0, 4.0, 16.0], "linearsvm": [0.1, 0.3, 1.0]}


def novo_vetorizador() -> TfidfVectorizer:
    # strip_accents="unicode": reviews informais escrevem "nao" e "não": tratamos como a mesma palavra.
    # ngram (1,2): bigramas capturam negação ("não gostei", "nao recomendo").
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True,
                           strip_accents="unicode", lowercase=True, dtype=np.float32)


def pontuar(clf, X) -> tuple[np.ndarray, float]:
    """Devolve (score, limiar). Probabilidade usa 0.5; margem do SVM usa 0.0."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1], 0.5
    return clf.decision_function(X), 0.0


def metricas(y: np.ndarray, score: np.ndarray, limiar: float) -> dict:
    pred = (score >= limiar).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "acuracia": round(accuracy_score(y, pred), 4),
            "f1_macro": round(f1_score(y, pred, average="macro"), 4),
            "negativo": {"precision": round(p[0], 4), "recall": round(r[0], 4), "f1": round(f[0], 4)},
            "positivo": {"precision": round(p[1], 4), "recall": round(r[1], 4), "f1": round(f[1], 4)},
            "roc_auc": round(roc_auc_score(y, score), 4),
            "matriz_confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def bootstrap_f1(y: np.ndarray, preds: dict[str, np.ndarray], b: int = 1000, seed: int = SEED) -> dict:
    """IC 95% do F1 macro por reamostragem (com reposição) do conjunto de teste, e da DIFERENÇA entre modelos."""
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s = {k: [] for k in preds}
    for _ in range(b):
        i = rng.integers(0, n, n)
        for k, p in preds.items():
            f1s[k].append(f1_score(y[i], p[i], average="macro"))
    out = {k: {"f1_macro": round(float(np.mean(v)), 4),
               "ic95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
           for k, v in f1s.items()}
    if {"logreg", "linearsvm"} <= preds.keys():
        d = np.array(f1s["logreg"]) - np.array(f1s["linearsvm"])
        out["diferenca_logreg_menos_svm"] = {"media": round(float(d.mean()), 4),
                                             "ic95": [round(float(np.percentile(d, 2.5)), 4),
                                                      round(float(np.percentile(d, 97.5)), 4)]}
    return out


def faixa_tamanho(n_palavras: int) -> str:
    if n_palavras <= 5:
        return "ate 5 palavras"
    if n_palavras <= 20:
        return "6 a 20"
    return "21 a 60" if n_palavras <= 60 else "mais de 60"


def analise_de_erros(pipe: Pipeline, teste: pd.DataFrame, score: np.ndarray, limiar: float) -> dict:
    y, pred = teste["rotulo"].to_numpy(), (score >= limiar).astype(int)
    erro = y != pred
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    nomes = np.array(vec.get_feature_names_out())
    ordem = np.argsort(clf.coef_[0])
    texto = teste["texto"].str.lower()
    contraste = texto.str.contains(r"\b(?:mas|porém|porem|entretanto|só que|so que|apesar)\b", regex=True).to_numpy()
    negacao = texto.str.contains(r"\b(?:não|nao|nunca|jamais)\b", regex=True).to_numpy()

    def exemplos(mask, k=8):  # os erros em que o modelo estava MAIS convicto (candidatos a rótulo ruidoso)
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-np.abs(score[idx] - limiar))][:k]
        return [{"nota": int(teste["overall_rating"].iloc[i]), "texto": teste["texto"].iloc[i][:220]} for i in idx]

    return {
        "palavras_que_mais_indicam_negativo": nomes[ordem[:15]].tolist(),
        "palavras_que_mais_indicam_positivo": nomes[ordem[-15:]][::-1].tolist(),
        "taxa_de_erro": {"geral": round(float(erro.mean()), 4),
                         "textos_com_contraste(mas/porem/...)": round(float(erro[contraste].mean()), 4),
                         "textos_sem_contraste": round(float(erro[~contraste].mean()), 4),
                         "textos_com_negacao": round(float(erro[negacao].mean()), 4),
                         "textos_sem_negacao": round(float(erro[~negacao].mean()), 4)},
        "n_com_contraste": int(contraste.sum()), "n_com_negacao": int(negacao.sum()),
        "falsos_positivos_mais_confiantes(real=negativo, previsto=positivo)": exemplos(erro & (y == 0)),
        "falsos_negativos_mais_confiantes(real=positivo, previsto=negativo)": exemplos(erro & (y == 1)),
        "aviso": "Erros muito confiantes podem ser ROTULO RUIDOSO (nota incoerente com o texto), nao falha do modelo.",
    }


def por_subgrupo(teste: pd.DataFrame, score: np.ndarray, limiar: float) -> dict:
    faixa = teste["texto"].str.split().str.len().map(faixa_tamanho)
    t = teste.assign(_pred=(score >= limiar).astype(int), _faixa=faixa)
    def f(g):
        return {"n": int(len(g)), "f1_macro": round(f1_score(g["rotulo"], g["_pred"], average="macro"), 4)}
    top = t["site_category_lv1"].value_counts().head(6).index
    return {"por_tamanho_do_texto": {k: f(g) for k, g in t.groupby("_faixa")},
            "por_categoria_do_produto(top6)": {
                k: f(g) for k, g in t[t["site_category_lv1"].isin(top)].groupby("site_category_lv1")}}


def main() -> dict:
    t0 = time.time()
    SAIDA.mkdir(exist_ok=True)
    bruto = carregar(baixar())
    d, log_prep = preparar(bruto)
    d = dividir(d)
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    rel: dict = {"preparacao": log_prep, "divisao": diagnostico_split(d), "modelos": {}}

    vec = novo_vetorizador()
    X_tr = vec.fit_transform(tr["texto"])                 # vocabulário aprendido SÓ no treino
    X_va, X_te = vec.transform(va["texto"]), vec.transform(te["texto"])
    y_tr, y_va, y_te = (x["rotulo"].to_numpy() for x in (tr, va, te))
    rel["vocabulario"] = int(X_tr.shape[1])

    candidatos: dict[str, object] = {"majoritaria": DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)}
    escolha: dict[str, dict] = {}
    for nome, grade in GRADES.items():
        melhor = (-1.0, None, None)
        tabela = []
        for c in grade:
            clf = (LogisticRegression(C=c, solver="liblinear", class_weight="balanced", random_state=SEED)
                   if nome == "logreg" else LinearSVC(C=c, class_weight="balanced", random_state=SEED))
            clf.fit(X_tr, y_tr)
            s, lim = pontuar(clf, X_va)
            f1v = metricas(y_va, s, lim)["f1_macro"]
            tabela.append({"C": c, "f1_macro_validacao": f1v})
            print(f"  {nome} C={c}: F1 macro (validacao) = {f1v}   [{time.time() - t0:.0f}s]", flush=True)
            if f1v > melhor[0]:
                melhor = (f1v, c, clf)
        candidatos[nome] = melhor[2]
        escolha[nome] = {"C_escolhido": melhor[1], "grade_validacao": tabela}

    preds, scores = {}, {}
    for nome, clf in candidatos.items():
        s, lim = pontuar(clf, X_te)
        scores[nome] = (s, lim)
        preds[nome] = (s >= lim).astype(int)
        rel["modelos"][nome] = {**escolha.get(nome, {}), "validacao": metricas(y_va, *pontuar(clf, X_va)),
                                "teste": metricas(y_te, s, lim)}
    rel["bootstrap_teste_f1_macro"] = bootstrap_f1(y_te, preds)

    melhor_nome = max(("logreg", "linearsvm"), key=lambda k: rel["modelos"][k]["validacao"]["f1_macro"])
    rel["modelo_escolhido"] = {"nome": melhor_nome, "criterio": "maior F1 macro na VALIDACAO (teste nao participa)"}
    pipe = Pipeline([("tfidf", vec), ("clf", candidatos[melhor_nome])])
    s_best, lim_best = scores[melhor_nome]
    rel["subgrupos_teste"] = por_subgrupo(te, s_best, lim_best)
    rel["analise_de_erros"] = analise_de_erros(pipe, te, s_best, lim_best)  # LogReg e LinearSVC têm coef_
    rel["tempo_total_s"] = round(time.time() - t0)

    (SAIDA / "metrics.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(pipe, SAIDA / "model.joblib")
    d.drop(columns=["texto"]).to_csv(SAIDA / "split_indices.csv", index=True)  # só ids/rótulos/split p/ reprodução
    return rel


if __name__ == "__main__":
    r = main()
    print("\nESCOLHIDO:", r["modelo_escolhido"]["nome"], "| vocabulario:", r["vocabulario"],
          "| tempo:", r["tempo_total_s"], "s")
    for k, m in r["modelos"].items():
        t = m["teste"]
        print(f"  {k:<12} TESTE  acc={t['acuracia']}  F1macro={t['f1_macro']}  AUC={t['roc_auc']}  "
              f"recall_neg={t['negativo']['recall']}  prec_neg={t['negativo']['precision']}")
    print("  bootstrap:", json.dumps(r["bootstrap_teste_f1_macro"], ensure_ascii=False))
