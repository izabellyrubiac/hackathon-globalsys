"""Mapeamento confirmado pelo usuário: tabela de clientes, alvo (cancelamento) e valor do contrato.

Campos
------
tabela_clientes, coluna_id   tabela com uma linha por cliente e a coluna que identifica o cliente
alvo_tabela                  tabela onde está o cancelamento (pode ser a própria tabela de clientes)
coluna_situacao              coluna de situação (ex.: "situacao"); None = cancelado é quem tem data de saída
valor_cancelado              valor de `coluna_situacao` que significa cancelado (ex.: "Cancelado", 1)
coluna_data_saida            mês/data da saída; None = saída no mês seguinte ao último registro do cliente
valor_tabela, coluna_valor   valor mensal do contrato (opcional; sem ele a fila usa só a probabilidade)
horizonte_meses              rótulo = cancela nos próximos N meses (1–12, padrão 3)
colunas_ignoradas            colunas que não podem virar variável ("coluna" ou "tabela.coluna")
acoes                        opcional: {coluna (ou "tabela.coluna" ou id da variável): ação sugerida}
rotulos                      opcional: {coluna: nome legível} (senão o nome vem do snake_case)

As colunas do alvo nunca viram variável; se o alvo está numa tabela separada, a tabela inteira sai.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import pandas as pd

from .esquema import coluna_chave
from .util import eh_numerica, norm_id, norm_valor, para_mes


class ErroMapeamento(ValueError):
    """Mapeamento inválido. `problemas` lista cada erro em pt-BR; str(erro) junta todos."""

    def __init__(self, problemas: list[str]):
        self.problemas = list(problemas)
        super().__init__(" ".join(self.problemas) if len(self.problemas) == 1
                         else "Mapeamento inválido: " + " | ".join(self.problemas))


@dataclass
class Mapeamento:
    tabela_clientes: str
    coluna_id: str
    alvo_tabela: str
    coluna_situacao: str | None
    valor_cancelado: object
    coluna_data_saida: str | None
    valor_tabela: str | None = None
    coluna_valor: str | None = None
    horizonte_meses: int = 3
    colunas_ignoradas: list[str] = field(default_factory=list)
    acoes: dict[str, str] = field(default_factory=dict)
    rotulos: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ serialização
    def to_dict(self) -> dict:
        d = asdict(self)
        v = d["valor_cancelado"]
        if hasattr(v, "item"):
            d["valor_cancelado"] = v.item()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mapeamento":
        nomes = {f.name for f in fields(cls)}
        extras = set(d) - nomes
        if extras:
            raise ErroMapeamento([f"Campo(s) desconhecido(s) no mapeamento: {', '.join(sorted(extras))}."])
        faltam = [n for n in ("tabela_clientes", "coluna_id", "alvo_tabela") if not d.get(n)]
        if faltam:
            raise ErroMapeamento([f"Informe o campo '{n}' do mapeamento." for n in faltam])
        d = dict(d)
        for n in ("coluna_situacao", "valor_cancelado", "coluna_data_saida"):
            d.setdefault(n, None)
        d["colunas_ignoradas"] = list(d.get("colunas_ignoradas") or [])
        d["acoes"] = dict(d.get("acoes") or {})
        d["rotulos"] = dict(d.get("rotulos") or {})
        return cls(**d)

    @classmethod
    def de_sugestao(cls, insp: dict, **extras) -> "Mapeamento":
        """Monta o mapeamento a partir de `inspecionar(...)['sugestao']` (ou do dict completo)."""
        s = insp.get("sugestao", insp)
        alvo = s.get("alvo") or {}
        val = s.get("coluna_valor") or {}
        return cls(tabela_clientes=s["tabela_clientes"], coluna_id=s["coluna_id"],
                   alvo_tabela=alvo.get("tabela") or s["tabela_clientes"],
                   coluna_situacao=alvo.get("coluna_situacao"), valor_cancelado=alvo.get("valor_cancelado"),
                   coluna_data_saida=alvo.get("coluna_data_saida"), valor_tabela=val.get("tabela"),
                   coluna_valor=val.get("coluna"), horizonte_meses=int(s.get("horizonte_meses") or 3), **extras)

    # ------------------------------------------------------------------ regras
    def colunas_alvo(self) -> set[tuple[str, str]]:
        return {(self.alvo_tabela, c) for c in (self.coluna_situacao, self.coluna_data_saida) if c}

    def ignorada(self, tabela: str, coluna: str) -> bool:
        if (tabela, coluna) in self.colunas_alvo():
            return True
        return coluna in self.colunas_ignoradas or f"{tabela}.{coluna}" in self.colunas_ignoradas

    def validar(self, tabelas: dict[str, pd.DataFrame]) -> list[str]:
        """Levanta `ErroMapeamento` (mensagens pt-BR acionáveis) ou devolve avisos não fatais."""
        erros: list[str] = []
        avisos: list[str] = []
        nomes = ", ".join(f"'{t}'" for t in tabelas)

        def tab(nome, papel):
            if nome not in tabelas:
                erros.append(f"A tabela {papel} '{nome}' não existe na base (tabelas: {nomes}).")
                return None
            return tabelas[nome]

        def col(df, tabela, nome, papel):
            if nome not in df.columns:
                lista = ", ".join(f"'{c}'" for c in df.columns)
                erros.append(f"A coluna {papel} '{nome}' não existe na tabela '{tabela}' (colunas: {lista}).")
                return None
            return df[nome]

        cli = tab(self.tabela_clientes, "de clientes")
        ids = None
        if cli is not None:
            s = col(cli, self.tabela_clientes, self.coluna_id, "de ID do cliente")
            if s is not None:
                v = norm_id(s)
                if v.isna().any():
                    erros.append(f"A coluna de ID '{self.coluna_id}' tem {int(v.isna().sum())} linha(s) vazia(s) "
                                 f"na tabela '{self.tabela_clientes}'.")
                dup = v.dropna().duplicated().sum()
                if dup:
                    ex = ", ".join(v[v.duplicated(keep=False)].dropna().unique()[:3])
                    erros.append(f"A coluna '{self.coluna_id}' tem {int(dup)} ID(s) repetido(s) na tabela "
                                 f"'{self.tabela_clientes}' (ex.: {ex}): escolha a tabela com uma linha por cliente.")
                ids = set(v.dropna())
        if not (1 <= int(self.horizonte_meses) <= 12):
            erros.append("O horizonte de previsão precisa ficar entre 1 e 12 meses.")
        if not self.coluna_situacao and not self.coluna_data_saida:
            erros.append("Informe a coluna de situação (com o valor que significa cancelado) ou a coluna de data de saída: "
                         "sem cancelamento não há como aprender.")

        alvo = tab(self.alvo_tabela, "do cancelamento")
        if alvo is not None:
            if ids is None:
                chave = None
            elif self.alvo_tabela == self.tabela_clientes:
                chave = self.coluna_id
            else:
                chave = coluna_chave(alvo, ids, self.coluna_id)
            if chave is None and ids is not None:
                erros.append(f"A tabela do cancelamento '{self.alvo_tabela}' não tem coluna com o ID do cliente "
                             f"('{self.coluna_id}').")
            cancel = None
            if self.coluna_situacao:
                s = col(alvo, self.alvo_tabela, self.coluna_situacao, "de situação")
                if s is not None:
                    vals = s.dropna().unique()
                    presentes = {norm_valor(v) for v in vals}
                    if self.valor_cancelado is None or norm_valor(self.valor_cancelado) not in presentes:
                        lista = ", ".join(f"'{v}'" for v in sorted(map(str, vals))[:10])
                        erros.append(f"O valor '{self.valor_cancelado}' não aparece na coluna '{self.coluna_situacao}' "
                                     f"(valores: {lista}). Informe o valor que significa cancelado.")
                    else:
                        cancel = s.map(norm_valor) == norm_valor(self.valor_cancelado)
            if self.coluna_data_saida:
                s = col(alvo, self.alvo_tabela, self.coluna_data_saida, "de data de saída")
                if s is not None:
                    mes = para_mes(s)
                    bruto = s.notna() & (s.astype("string").str.strip() != "")
                    ruins = int((bruto & mes.isna()).sum())
                    if ruins > 0.1 * max(int(bruto.sum()), 1):
                        erros.append(f"A coluna '{self.coluna_data_saida}' não parece uma data: {ruins} valor(es) não "
                                     "reconhecido(s). Use AAAA-MM, AAAA-MM-DD ou DD/MM/AAAA.")
                    if cancel is None and self.coluna_situacao is None:
                        cancel = mes.notna()
                    elif cancel is not None:
                        sem = int((cancel & mes.isna()).sum())
                        if sem:
                            avisos.append(f"{sem} cliente(s) cancelado(s) sem data de saída: a saída deles é "
                                          "considerada o mês seguinte ao último registro.")
            if cancel is not None and chave is not None:
                n_c, n_a = int(cancel.sum()), int((~cancel).sum())
                if n_c == 0:
                    erros.append(f"A coluna de cancelamento não tem nenhum cliente cancelado "
                                 f"('{self.valor_cancelado}') — não há como aprender.")
                elif n_c < 5:
                    erros.append(f"Só {n_c} cliente(s) cancelado(s): são necessários pelo menos 5 para aprender.")
                elif n_c < 15:
                    avisos.append(f"Só {n_c} clientes cancelados: pesos e validação ficam instáveis.")
                if n_a == 0:
                    erros.append("Todos os clientes estão cancelados: não há ativos para comparar.")
                ligados = norm_id(alvo[chave]).isin(ids).sum() if chave in alvo.columns else 0
                if ligados < 0.5 * len(ids):
                    avisos.append(f"Só {int(ligados)} de {len(ids)} clientes têm linha na tabela do cancelamento; "
                                  "os demais são considerados ativos.")

        if self.coluna_valor:
            vt = self.valor_tabela or self.tabela_clientes
            dfv = tab(vt, "do valor do contrato")
            if dfv is not None:
                s = col(dfv, vt, self.coluna_valor, "de valor do contrato")
                if s is not None and not eh_numerica(s):
                    erros.append(f"A coluna de valor '{self.coluna_valor}' não é numérica.")
                if (vt, self.coluna_valor) in self.colunas_alvo():
                    erros.append("A coluna de valor não pode ser uma das colunas do cancelamento.")
        if self.coluna_valor is None and self.valor_tabela:
            avisos.append("Tabela de valor informada sem coluna de valor: a fila usa só a probabilidade.")
        for c in self.colunas_ignoradas:
            t, _, cc = c.rpartition(".")
            existe = any(cc in df.columns for df in tabelas.values()) if not t else (t in tabelas and cc in tabelas[t].columns)
            if not existe:
                avisos.append(f"Coluna ignorada '{c}' não existe na base.")
        if erros:
            raise ErroMapeamento(erros)
        return avisos
