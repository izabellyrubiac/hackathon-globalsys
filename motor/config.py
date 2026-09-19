"""Parâmetros do motor (todos genéricos — nenhum depende da base). Os padrões valem para bases pequenas
(dezenas a milhares de clientes, meses de histórico) e ficam registrados em pesos.json/validacao.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # variáveis
    janela: int = 3                  # nível = média dos últimos 3 meses
    janela_tendencia: int = 6        # tendência = inclinação dos últimos 6 meses (mín. 3 pontos)
    min_hist: int = 4                # linha treina só com ≥ 4 meses com dados (janela + 1 mês de referência)
    densidade_minima: float = 0.6    # coluna temporal com dados em < 60% dos meses = esparsa (ex.: pesquisa trimestral)
    max_categorias: int = 12         # categorias por coluna (as mais frequentes)
    min_frac_categoria: float = 0.03 # categoria precisa aparecer em ≥ 3% das linhas
    n_persistencia: int = 5          # variáveis de persistência criadas (as de maior AUC)
    # seleção
    max_nulos: float = 0.5           # descarta variável com > 50% de nulos nas linhas de treino
    max_moda: float = 0.98           # descarta variável com ≥ 98% das linhas no mesmo valor
    max_spearman: float = 0.9        # |Spearman| > 0,9 → redundante (fica a de maior AUC)
    auc_vazamento: float = 0.98      # AUC univariada acima disso = suspeita de vazamento (excluída)
    rodadas: int = 50                # reamostragens de clientes na stability selection
    frac_subamostra: float = 0.5     # fração dos clientes em cada rodada (estratificada por cancelou)
    limiar_frequencia: float = 0.6   # selecionada se escolhida em ≥ 60% das rodadas
    consistencia_sinal: float = 0.9  # e com o mesmo sinal em ≥ 90% das rodadas em que entrou
    l1_ratio: float = 0.5            # 1 = L1 (lasso); 0 < r < 1 = elastic-net na stability selection
    regra_1ep: bool = False          # True: C = o mais esparso a 1 erro-padrão do melhor (AUC agrupada)
    linhas_por_variavel: int = 5     # limite de variáveis = mín(meses-cliente positivos/5, cancelados/2), mínimo 2
    clientes_por_variavel: int = 2
    dobras_internas: int = 5         # GroupKFold para C e AUC out-of-fold
    # validação
    dobras: int = 10                 # validação externa agrupada por cliente (seleção refeita em cada dobra)
    validar: bool = True
    n_jobs: int = -1
    semente: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
