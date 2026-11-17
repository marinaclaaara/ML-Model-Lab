---
language: pt
license: cc-by-nc-sa-4.0
tags: [sentiment-analysis, text-classification, tf-idf, scikit-learn]
datasets: [B2W-Reviews01]
---
# Sentimento PT-BR: TF-IDF + Linear SVM (baseline)  [NÃO PUBLICADO]

**Uso pretendido:** classificar reviews de e-commerce em português como positivas ou negativas, em estudo e portfólio.
**Fora de escopo:** uso comercial (licença do dataset), outros domínios, decisões sobre pessoas.

**Dados:** B2W-Reviews01 (B2W Digital, 2019), CC BY-NC-SA 4.0. 110.859 reviews após limpeza; rótulo derivado da nota
(1-2 negativo, 4-5 positivo); divisão 80/10/10 por autor.

**Modelo:** `TfidfVectorizer` (1-2 gramas, sem acentos, `min_df=3`, vocabulário de 89.927 termos) + `LinearSVC` (C=0,3, `class_weight=balanced`).

**Resultado (teste, n=11.074):** acurácia 0,952; F1 macro 0,943 (IC 95%: 0,938 a 0,947); ROC-AUC 0,986; recall da classe negativa 0,957.
Empata estatisticamente com a Regressão Logística (diferença de F1 macro com IC 95% incluindo zero).

**Limitações:** rótulo por nota é ruidoso (casos de elogio com nota baixa e vice-versa); erra mais com contraste ("mas…") e negação;
produtos compartilhados entre treino e teste; sem avaliação fora de e-commerce; sem classe neutra.
**Vieses:** colunas demográficas (idade, gênero, estado) não foram usadas, mas o texto pode refletir variação regional ou de escrita não auditada.
