"""Demo do ML Model Lab no Hugging Face Spaces: sentimento PT-BR com TF-IDF + Linear SVM.

O modelo (model.joblib) é o MESMO treinado em `python -m model_lab.train`. Nada é treinado aqui.
A "margem" é a distância do texto à fronteira do SVM: NÃO é probabilidade (por isso não mostramos porcentagem).
"""
from pathlib import Path

import gradio as gr
import joblib

MODELO = joblib.load(Path(__file__).parent / "model.joblib")

AVISO = """**Como interpretar.** A margem é a distância à fronteira de decisão (quanto mais longe de 0, mais "certo" o modelo está),
**não uma probabilidade**. O modelo lê palavras e pares de palavras: erra mais com **contraste** ("mas", "porém": 11,4% de erro
contra 3,7% sem contraste) e **negação** (8,1% contra 3,1%). Treinado só em reviews de e-commerce brasileiro: não
generaliza para outros assuntos. Uso **não comercial** (licença do dataset)."""

EXEMPLOS = [
    "Produto excelente, chegou antes do prazo. Recomendo!",
    "Péssima qualidade, quebrou em dois dias e ninguém responde.",
    "Gostei do design, mas a bateria dura muito pouco.",
    "Entrega rápida, porém o produto veio com defeito.",
    "Não recomendo, não funcionou como esperado.",
    "Nao gostei",
]


def classificar(texto: str) -> tuple[str, float | None]:
    if not texto or not texto.strip():
        return "Escreva uma review.", None
    margem = float(MODELO.decision_function([texto])[0])
    return ("Positivo" if margem >= 0 else "Negativo"), round(margem, 3)


demo = gr.Interface(
    fn=classificar,
    inputs=gr.Textbox(lines=4, label="Review em português", placeholder="Escreva uma review de produto..."),
    outputs=[gr.Textbox(label="Sentimento"), gr.Number(label="Margem do SVM (não é probabilidade)")],
    examples=EXEMPLOS,
    title="Sentimento em português (B2W-Reviews01)",
    description=AVISO,
    api_name="classificar",
)

if __name__ == "__main__":
    demo.launch()
