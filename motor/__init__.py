"""Motor genérico de risco de cancelamento (churn) para qualquer base com uma tabela de clientes.

Uso:
    from motor import ler_arquivos, inspecionar, Mapeamento, treinar
    tabelas = ler_arquivos(["base.xlsx"])          # ou vários .csv
    insp = inspecionar(tabelas)                     # tipos, papéis e sugestão de mapeamento (JSON)
    m = Mapeamento.de_sugestao(insp)                # ou Mapeamento(...) confirmado pelo usuário
    m.validar(tabelas)                              # ErroMapeamento com mensagens pt-BR
    r = treinar(tabelas, m, progresso=lambda etapa, fracao: ...)
    r.clientes, r.validacao, r.pesos                # contratos clientes.json / validacao.json / pesos.json
    r.salvar("pasta/")                              # grava os três JSON

Módulos: leitura (arquivos), esquema (tipos/papéis/sugestões), mapeamento, painel (tempo e rótulo),
variaveis (candidatas), delta (candidata de variação do risco, `Config.delta_risco`),
selecao (seleção da logística + pré-seleção comum), selecao_arvores (sombras),
modelos_previsao/ (um arquivo por modelo: logística, Random Forest, LightGBM), modelos_classificacao/
(validação cruzada, métricas e escolha automática do modelo), explicar, saida. `modelo` e `validar` só
reexportam (compatibilidade).
"""

from .config import Config
from .esquema import inspecionar
from .leitura import Tabelas, de_dataframes, ler_arquivos
from .mapeamento import ErroMapeamento, Mapeamento
from .treino import Resultado, treinar

__all__ = ["Config", "ErroMapeamento", "Mapeamento", "Resultado", "Tabelas", "de_dataframes", "inspecionar",
           "ler_arquivos", "treinar"]
