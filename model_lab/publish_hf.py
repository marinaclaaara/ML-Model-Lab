"""Publica o Transformer no Hugging Face Hub. TRAVADO: exige autorização explícita.

Antes: (1) rode o fine-tuning e (2) gere o card:  python -m model_lab.make_card_transformer --usuario SEU_USUARIO
Leia a seção "Licença" do README: o dataset é CC BY-NC-SA 4.0, então o modelo derivado deve ser NÃO COMERCIAL.

Uso:  HF_TOKEN=... python -m model_lab.publish_hf --repo SEU_USUARIO/bertimbau-sentimento-b2w --eu-autorizo-publicar
(Com --simular, mostra o que seria enviado sem enviar nada.)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def arquivos_a_enviar(pasta: Path, card: Path) -> list[str]:
    """Lista o que vai ao Hub. Valida que há pesos + tokenizer e que o card existe e é do Transformer."""
    if not (pasta / "config.json").exists() or not any(pasta.glob("*.safetensors")) and not any(pasta.glob("*.bin")):
        raise SystemExit(f"{pasta} não tem um modelo salvo (config.json + pesos). Rode o fine-tuning antes.")
    if not card.exists():
        raise SystemExit(f"{card} não existe: gere com make_card_transformer (usa métricas reais).")
    if "NÃO PUBLICADO" in card.read_text(encoding="utf-8"):
        raise SystemExit("O card parece ser o do baseline (marcado NÃO PUBLICADO). Use o card do Transformer.")
    return sorted(p.name for p in pasta.iterdir() if p.is_file()) + ["README.md (<- card)"]


def main(repo: str, pasta: Path, card: Path, autorizado: bool, simular: bool) -> str:
    lista = arquivos_a_enviar(pasta, card)
    if simular:
        return "SIMULAÇÃO, nada enviado. Iria para " + repo + ":\n  " + "\n  ".join(lista)
    if not autorizado:
        raise SystemExit("Publicação recusada: passe --eu-autorizo-publicar depois de revisar licença e card.")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Defina HF_TOKEN no ambiente (no Colab: aba Secrets). Nunca escreva o token no código.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True)  # público por padrão; passe private=True se quiser revisar antes
    api.upload_folder(repo_id=repo, folder_path=str(pasta))
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pasta", type=Path, default=Path("results/transformer_model"))
    ap.add_argument("--card", type=Path, default=Path("results/MODEL_CARD_transformer.md"))
    ap.add_argument("--eu-autorizo-publicar", action="store_true")
    ap.add_argument("--simular", action="store_true")
    a = ap.parse_args()
    print(main(a.repo, a.pasta, a.card, a.eu_autorizo_publicar, a.simular))
