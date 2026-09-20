"""API FastAPI do INOVAAPPS 2026: upload da base, inspeção, mapeamento, treino e resultados.

Camada fina sobre o pacote `motor/`: recebe arquivos, chama o motor e devolve JSON. Sem banco —
tudo em `dados/bases/<base_id>/`.

Cada pedido de treino cria uma **execução** (versão do modelo) com os seus próprios resultados e um
`resumo.json` para a tela de comparação; uma delas é a **ativa** e responde pelas rotas sem `execucoes/`.

Subir:  uv run uvicorn api.main:app --reload
Docs:   http://localhost:8000/docs
Demos:  uv run python -m api.preparar_demos
"""

VERSAO = "0.2.0"

__all__ = ["VERSAO"]
