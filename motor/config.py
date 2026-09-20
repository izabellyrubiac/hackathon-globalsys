"""Parâmetros do motor (todos genéricos — nenhum depende da base). Os padrões valem para bases pequenas
(dezenas a milhares de clientes, meses de histórico) e ficam registrados em pesos.json/validacao.json.
Base grande: `perfil_automatico()` reduz dobras e rodadas (ver `perfil`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

MODELOS_VALIDOS = ("auto", "logistica", "random_forest", "lightgbm")


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
    max_candidatas: int = 300        # pré-filtro de escala: com mais candidatas que isso (base larga) …
    pre_filtro_k: int = 100          # … só as K de maior AUC univariada seguem para redundância e seleção
    rodadas: int = 50                # reamostragens de clientes na stability selection
    frac_subamostra: float = 0.5     # fração dos clientes em cada rodada (estratificada por cancelou)
    razao_ativos_rodada: float | None = None  # base grande: no máx. N ativos por cancelado em cada rodada
    limiar_frequencia: float = 0.6   # selecionada se escolhida em ≥ 60% das rodadas
    consistencia_sinal: float = 0.9  # e com o mesmo sinal em ≥ 90% das rodadas em que entrou
    l1_ratio: float = 0.5            # 1 = L1 (lasso); 0 < r < 1 = elastic-net na stability selection
    regra_1ep: bool = False          # True: C = o mais esparso a 1 erro-padrão do melhor (AUC agrupada)
    linhas_por_variavel: int = 5     # limite de variáveis = mín(meses-cliente positivos/5, cancelados/2), mínimo 2
    clientes_por_variavel: int = 2
    dobras_internas: int = 5         # GroupKFold para C, parada antecipada, calibração e AUC out-of-fold
    min_frac_dobras: float = 0.5     # modelo final: consistência = selecionada em ≥ 50% das dobras externas
    modo_dobras: str = "priorizar"   # "priorizar": no limite, as consistentes vêm antes (inconsistente só ocupa vaga
                                     # que sobrar) · "excluir": inconsistente nunca entra
    # modelos (escolha automática pela validação agrupada por cliente)
    modelo: str = "auto"             # "auto" | "logistica" | "random_forest" | "lightgbm" (opção avançada)
    min_pos_arvores: int = 200       # árvores elegíveis só com ≥ 200 meses-cliente positivos no treino
    min_canc_arvores: int = 50       # … e ≥ 50 clientes cancelados
    avaliar_todos: bool = False      # valida também os modelos não elegíveis (só comparação; nunca escolhidos no auto)
    rodadas_arvores: int = 20        # rodadas da seleção por sombras (estilo Boruta)
    limiar_sombra: float = 0.6       # fica se a importância superar a maior sombra em ≥ 60% das rodadas
    max_arvores: int = 500           # LightGBM: nº máximo de árvores (parada antecipada por GroupKFold)
    arvores_rf: int = 300            # Random Forest: nº de árvores
    taxa_aprendizado: float = 0.05
    folhas: int = 15                 # num_leaves
    min_linhas_folha: int = 20       # min_data_in_leaf
    # delta de risco (motor em dois estágios — ver motor/delta.py)
    delta_risco: bool = False        # True: acrescenta a candidata "risco de hoje − risco do mês passado"
    delta_risco_3m: bool = True      # … e também "risco de hoje − média dos 3 meses anteriores"
    delta_dobras: int = 5            # estágio 1 cross-fit por cliente nas linhas de treino (0 = sem cross-fit)
    delta_placebo: bool = False      # diagnóstico: embaralha o delta (mesma distribuição, informação zero).
                                     # Braço de controle — mede quanto da diferença entre "com" e "sem" delta é
                                     # só o ruído de refazer a seleção com duas candidatas a mais.
    # validação
    dobras: int = 10                 # validação externa agrupada por cliente (seleção refeita em cada dobra)
    validar: bool = True
    n_jobs: int = -1
    semente: int = 0
    # escala
    perfil: str = "auto"             # "auto": base grande → perfil rápido · "fixo": usa os valores acima como estão
    grande_clientes: int = 2000      # base grande: > 2.000 clientes …
    grande_linhas: int = 200_000     # … ou > 200 mil linhas no painel cliente × mês

    def to_dict(self) -> dict:
        return asdict(self)

    def perfil_automatico(self, n_clientes: int, n_linhas: int) -> tuple[Config, str | None]:
        """(config, aviso). Base grande com `perfil="auto"`: 5 dobras externas, 20 rodadas de stability
        selection, 12 de sombras e no máximo 4 ativos por cancelado em cada rodada de seleção."""
        if self.perfil != "auto" or (n_clientes <= self.grande_clientes and n_linhas <= self.grande_linhas):
            return self, None
        novo = replace(self, dobras=min(self.dobras, 5), rodadas=min(self.rodadas, 20),
                       rodadas_arvores=min(self.rodadas_arvores, 12),
                       razao_ativos_rodada=self.razao_ativos_rodada or 4.0)
        return novo, (f"Base grande ({n_clientes} clientes, {n_linhas} linhas no painel): perfil rápido — "
                      f"{novo.dobras} dobras, {novo.rodadas} rodadas de stability selection, {novo.rodadas_arvores} "
                      f"de sombras e até {novo.razao_ativos_rodada:g} ativos por cancelado em cada rodada "
                      "(Config.perfil='fixo' desliga).")
