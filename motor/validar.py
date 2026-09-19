"""Compatibilidade: a validação e as métricas agora moram em `motor/modelos_classificacao/`
(`validacao_cruzada.py`, `metricas.py` e `escolha.py`, um arquivo por método)."""

from .modelos_classificacao.metricas import (KS, antecedencia, auc_por_mes, comparar_modelos, curva, desempenho,
                                             diferenca_pareada, escolher_cortes, faixa, log_loss_dobras,
                                             precisao_topo)
from .modelos_classificacao.validacao_cruzada import ResultadoModelo, ValidacaoOOF, _dobra, dobras_externas, validar

__all__ = ["KS", "ResultadoModelo", "ValidacaoOOF", "_dobra", "antecedencia", "auc_por_mes", "comparar_modelos", "curva",
           "desempenho", "diferenca_pareada", "dobras_externas", "escolher_cortes", "faixa", "log_loss_dobras",
           "precisao_topo", "validar"]
