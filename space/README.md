---
title: Sentimento PT-BR (B2W)
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---
# Sentimento em português (reviews de e-commerce)

Demo do baseline **TF-IDF + Linear SVM** treinado no B2W-Reviews01 (B2W Digital, CC BY-NC-SA 4.0: **uso não comercial**).
Código, métricas e limitações: repositório **ML Model Lab** (coloque aqui o link do seu GitHub).

Resultado no teste (n=11.074): F1 macro 0,943 (IC 95%: 0,938 a 0,947). Erra mais com contraste ("mas", "porém") e negação.
