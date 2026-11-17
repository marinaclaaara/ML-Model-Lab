"""Gráficos em SVG, um arquivo por gráfico (charts/). Lê results/metrics.json: nada é digitado à mão."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import baixar, carregar  # noqa: E402
from .prep import dividir, preparar  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
AZUL, LARANJA, CINZA, VERDE = "#2b5fd9", "#d98a00", "#8a94a3", "#1f9d55"
SAIDA = Path("charts")


def salvar(fig, nome: str) -> None:
    SAIDA.mkdir(exist_ok=True)
    fig.savefig(SAIDA / f"{nome}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> list[str]:
    rel = json.loads(Path("results/metrics.json").read_text(encoding="utf-8"))
    bruto = carregar(baixar())

    # 1) notas originais: mostra o desbalanceamento e a faixa descartada (nota 3)
    cont = bruto["overall_rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    cores = [LARANJA, LARANJA, CINZA, AZUL, AZUL]
    ax.bar(cont.index.astype(str), cont.values, color=cores)
    for x, v in zip(cont.index.astype(str), cont.values, strict=True):
        ax.text(x, v + 600, f"{v:,}".replace(",", "."), ha="center", fontsize=9)
    ax.set(title="Notas originais (B2W-Reviews01)", xlabel="nota", ylabel="reviews")
    ax.text(0.5, -0.28, "laranja = negativo (1-2), azul = positivo (4-5), cinza = nota 3 (descartada)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    salvar(fig, "01_distribuicao_notas")

    # 2) tamanho do texto por classe (já preparado)
    d, _ = preparar(bruto)
    d = dividir(d)
    pal = d["texto"].str.split().str.len().clip(upper=80)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for rot, cor, nome in [(0, LARANJA, "negativo"), (1, AZUL, "positivo")]:
        ax.hist(pal[d["rotulo"] == rot], bins=40, alpha=0.65, color=cor, label=nome, density=True)
    ax.set(title="Tamanho do texto por classe", xlabel="palavras (cortado em 80)", ylabel="densidade")
    ax.legend(frameon=False)
    salvar(fig, "02_tamanho_texto")

    # 3) comparação de modelos com IC95 por bootstrap
    bs = rel["bootstrap_teste_f1_macro"]
    nomes = ["majoritaria", "logreg", "linearsvm"]
    rotulos = ["Classe\nmajoritária", "TF-IDF +\nLogReg", "TF-IDF +\nLinear SVM"]
    med = [bs[n]["f1_macro"] for n in nomes]
    err = [[bs[n]["f1_macro"] - bs[n]["ic95"][0] for n in nomes], [bs[n]["ic95"][1] - bs[n]["f1_macro"] for n in nomes]]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(rotulos, med, yerr=err, capsize=5, color=[CINZA, AZUL, AZUL])
    for i, v in enumerate(med):
        ax.text(i, v + 0.035, f"{v:.3f}", ha="center", fontsize=10)
    ax.set(title="F1 macro no teste (barra = IC 95% por bootstrap)", ylim=(0, 1.1), ylabel="F1 macro")
    salvar(fig, "03_comparacao_modelos")

    # 4) matriz de confusão normalizada do modelo escolhido
    escolhido = rel["modelo_escolhido"]["nome"]
    mc = rel["modelos"][escolhido]["teste"]["matriz_confusao"]
    m = np.array([[mc["tn"], mc["fp"]], [mc["fn"], mc["tp"]]], dtype=float)
    mn = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mn[i, j]:.1%}\n({int(m[i, j]):,})".replace(",", "."), ha="center", va="center",
                    color="white" if mn[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["prev. negativo", "prev. positivo"],
           yticklabels=["real negativo", "real positivo"], title=f"Matriz de confusão (teste), {escolhido}")
    salvar(fig, "04_matriz_confusao")

    # 5) F1 por tamanho do texto: onde o modelo é mais frágil
    sub = rel["subgrupos_teste"]["por_tamanho_do_texto"]
    ordem = ["ate 5 palavras", "6 a 20", "21 a 60", "mais de 60"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(ordem, [sub[k]["f1_macro"] for k in ordem], color=AZUL)
    for i, k in enumerate(ordem):
        ax.text(sub[k]["f1_macro"] + 0.005, i, f"{sub[k]['f1_macro']:.3f}  (n={sub[k]['n']})", va="center", fontsize=9)
    ax.set(title="F1 macro por tamanho do texto (teste)", xlim=(0.5, 1.08), xlabel="F1 macro")
    salvar(fig, "05_f1_por_tamanho")
    return sorted(p.name for p in SAIDA.glob("*.svg"))


if __name__ == "__main__":
    print(main())
