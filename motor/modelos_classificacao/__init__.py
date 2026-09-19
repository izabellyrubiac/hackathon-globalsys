"""Avaliação da qualidade dos modelos e escolha — um arquivo por método.

* `validacao_cruzada.py` validação out-of-fold agrupada por cliente, seleção aninhada, vários modelos
                         nas MESMAS dobras externas;
* `metricas.py`          log-loss, AUC por mês, detecção × alarme falso, curva, cortes das faixas,
                         precisão no topo, antecedência e comparação dos modelos;
* `escolha.py`           elegibilidade, critério (log-loss fora da amostra, desempate por 1 erro-padrão)
                         e opção avançada `Config.modelo`.
"""

from .escolha import Candidato, Escolha, a_validar, candidatos, escolher
from .metricas import (KS, antecedencia, auc_por_mes, comparar_modelos, curva, desempenho, diferenca_pareada,
                       escolher_cortes, faixa, log_loss_dobras, precisao_topo)
from .validacao_cruzada import ResultadoModelo, ValidacaoOOF, dobras_externas, validar

__all__ = ["KS", "Candidato", "Escolha", "ResultadoModelo", "ValidacaoOOF", "a_validar", "antecedencia", "auc_por_mes",
           "candidatos", "comparar_modelos", "curva", "desempenho", "diferenca_pareada", "dobras_externas", "escolher",
           "escolher_cortes", "faixa", "log_loss_dobras", "precisao_topo", "validar"]
