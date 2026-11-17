import numpy as np
import pandas as pd

from model_lab.prep import diagnostico_split, dividir, preparar
from model_lab.train import bootstrap_f1, metricas


def _df(linhas):
    colunas = ["reviewer_id", "product_id", "site_category_lv1", "overall_rating", "review_text"]
    return pd.DataFrame(linhas, columns=colunas)


def test_rotulo_vem_da_nota_e_nota_3_e_descartada():
    d, log = preparar(_df([("a", "p", "c", 1, "pessimo produto"), ("b", "p", "c", 2, "ruim demais"),
                           ("c", "p", "c", 3, "mais ou menos"), ("d", "p", "c", 4, "bom produto"),
                           ("e", "p", "c", 5, "excelente compra")]))
    assert dict(zip(d["texto"], d["rotulo"], strict=True)) == {
        "pessimo produto": 0, "ruim demais": 0, "bom produto": 1, "excelente compra": 1}
    assert log["apos_remover_nota_3"] == 4


def test_remove_vazio_curto_duplicata_e_conflito():
    d, log = preparar(_df([("a", "p", "c", 5, None), ("b", "p", "c", 5, "ok"), ("c", "p", "c", 5, "Muito  bom"),
                           ("d", "p", "c", 4, "muito bom"),                       # duplicata (caixa/espaços)
                           ("e", "p", "c", 5, "bom"), ("f", "p", "c", 1, "bom"),  # mesmo texto, rótulos opostos
                           ("g", "p", "c", 1, "horrivel")]))
    assert sorted(d["texto"]) == ["Muito  bom", "horrivel"]
    assert log["textos_com_rotulos_conflitantes_removidos"] == 1 and log["duplicatas_removidas"] == 1


def test_divisao_nunca_repete_autor_entre_conjuntos():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"reviewer_id": [f"r{i % 300}" for i in range(3000)], "product_id": "p",
                      "rotulo": rng.integers(0, 2, 3000), "texto": "x", "overall_rating": 5, "site_category_lv1": "c"})
    out = dividir(d)
    diag = diagnostico_split(out)
    assert set(out["split"]) == {"train", "val", "test"}
    assert diag["autores_compartilhados"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert 0.75 < (out["split"] == "train").mean() < 0.85


def test_metricas_f1_macro_pune_o_modelo_que_so_diz_positivo():
    y = np.array([1] * 90 + [0] * 10)
    m = metricas(y, np.ones(100), 0.5)
    assert m["acuracia"] == 0.9 and m["f1_macro"] < 0.5 and m["negativo"]["recall"] == 0.0


def test_bootstrap_ic_contem_a_media_e_e_reprodutivel():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.where(rng.random(500) < 0.9, y, 1 - y)
    a = bootstrap_f1(y, {"logreg": p, "linearsvm": p}, b=200)
    b = bootstrap_f1(y, {"logreg": p, "linearsvm": p}, b=200)
    assert a == b and a["logreg"]["ic95"][0] <= a["logreg"]["f1_macro"] <= a["logreg"]["ic95"][1]
    assert a["diferenca_logreg_menos_svm"]["media"] == 0.0  # modelos idênticos: diferença nula
