"""Contratos de entrada e saída da API (Pydantic). Nomes de campo em pt-BR, como o resto do projeto.

Os modelos de saída aceitam campos extras (`extra="allow"`): o contrato do motor pode ganhar campos
novos sem quebrar a API nem exigir alteração aqui.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from motor.config import MODELOS_VALIDOS, Config

FAIXAS = ("alto", "atencao", "baixo")


class Aberto(BaseModel):
    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- inspeção
class Coluna(Aberto):
    nome: str
    tipo: str = Field(description='"id" | "numerica" | "categorica" | "data" | "texto" | "binaria"')
    unicos: int = 0
    nulos_pct: float = 0.0
    exemplos: list[Any] = []


class Tabela(Aberto):
    nome: str
    linhas: int = 0
    papel_sugerido: str = Field(description='"clientes" | "estatica" | "temporal" | "eventos" | "ignorada"')
    chave: str | None = None
    coluna_data: str | None = None
    motivo: str | None = None
    colunas: list[Coluna] = []


class AlvoSugerido(Aberto):
    tabela: str | None = None
    coluna_situacao: str | None = None
    valor_cancelado: Any = None
    coluna_data_saida: str | None = None


class ValorSugerido(Aberto):
    tabela: str | None = None
    coluna: str | None = None


class Sugestao(Aberto):
    tabela_clientes: str | None = None
    coluna_id: str | None = None
    alvo: AlvoSugerido | None = None
    coluna_valor: ValorSugerido | None = None
    horizonte_meses: int = 3


class Inspecao(Aberto):
    tabelas: list[Tabela] = []
    sugestao: Sugestao = Sugestao()
    avisos: list[str] = []


# --------------------------------------------------------------------------- mapeamento
class MapeamentoEntrada(BaseModel):
    """Mapeamento confirmado pelo usuário (mesmos campos de `motor.Mapeamento`)."""

    model_config = ConfigDict(extra="forbid")

    tabela_clientes: str = Field(description="Tabela com uma linha por cliente.")
    coluna_id: str = Field(description="Coluna que identifica o cliente.")
    alvo_tabela: str = Field(description="Tabela onde está o cancelamento (pode ser a de clientes).")
    coluna_situacao: str | None = Field(None, description="Coluna de situação (ex.: 'situacao').")
    valor_cancelado: Any = Field(None, description="Valor que significa cancelado (ex.: 'Cancelado', 1).")
    coluna_data_saida: str | None = Field(None, description="Mês/data da saída (AAAA-MM, AAAA-MM-DD, DD/MM/AAAA).")
    valor_tabela: str | None = Field(None, description="Tabela do valor do contrato (padrão: a de clientes).")
    coluna_valor: str | None = Field(None, description="Valor mensal do contrato; sem ela a fila usa só o risco.")
    horizonte_meses: int = Field(3, ge=1, le=12, description="Cancela nos próximos N meses.")
    colunas_ignoradas: list[str] = Field(default_factory=list, description='"coluna" ou "tabela.coluna".')
    acoes: dict[str, str] = Field(default_factory=dict)
    rotulos: dict[str, str] = Field(default_factory=dict)


class Opcoes(BaseModel):
    """Opções avançadas do treino. `None` = padrão do motor (`motor.Config`)."""

    model_config = ConfigDict(extra="forbid")

    modelo: str = Field("auto", description=f"Um de: {', '.join(MODELOS_VALIDOS)}.")
    horizonte_meses: int | None = Field(None, ge=1, le=12,
                                        description="Sobrescreve o horizonte do mapeamento, se enviado.")
    min_hist: int | None = Field(None, ge=1, le=36, description="Mínimo de meses com dados para a linha treinar.")
    delta_risco: bool = Field(
        False,
        description="Opção avançada, desligada por padrão. Só o modelo: o delta ordena a fila e vai no JSON de qualquer jeito. Motor em dois estágios: acrescenta a candidata "
                    "'risco de hoje − risco do mês passado'. Testado nas bases INOVAAPPS e redes: não melhora o "
                    "acerto (um braço placebo com o delta embaralhado entrega o mesmo ganho) e custa 3,2× o tempo "
                    "de treino; mantido como opção avançada.")
    validar: bool | None = Field(None, description="Ligar/desligar a validação cruzada externa.")
    dobras: int | None = Field(None, ge=2, le=50, description="Dobras da validação externa (agrupadas por cliente).")
    semente: int | None = Field(None, ge=0)
    avancado: dict[str, Any] = Field(default_factory=dict,
                                     description="Outros campos de `motor.Config` (lidos dinamicamente).")

    @field_validator("modelo")
    @classmethod
    def _modelo_valido(cls, v: str) -> str:
        if v not in MODELOS_VALIDOS:
            raise ValueError(f"Modelo '{v}' não existe. Use um de: {', '.join(MODELOS_VALIDOS)}.")
        return v

    @field_validator("avancado")
    @classmethod
    def _avancado_conhecido(cls, v: dict) -> dict:
        nomes = {f.name for f in fields(Config)}
        extras = sorted(set(v) - nomes)
        if extras:
            raise ValueError(f"Opção avançada desconhecida: {', '.join(extras)}. "
                             f"Campos aceitos: {', '.join(sorted(nomes))}.")
        return v


class PedidoTreino(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mapeamento: MapeamentoEntrada
    opcoes: Opcoes = Field(default_factory=Opcoes)
    rotulo: str | None = Field(None, max_length=80,
                               description="Nome curto desta versão do modelo, livre (ex.: 'só NPS, horizonte 2'). "
                                           "Aparece na lista de execuções e na comparação.")


# --------------------------------------------------------------------------- saídas
class BaseCriada(Aberto):
    base_id: str
    nome: str
    criada_em: str
    arquivos: list[str] = []
    bytes_total: int = 0
    inspecao: Inspecao
    status: "Status"


class BaseResumo(Aberto):
    base_id: str
    nome: str
    criada_em: str | None = None
    bytes_total: int | None = None
    arquivos: list[str] = []
    estado: str
    etapa: str | None = None
    n_clientes: int | None = None
    n_execucoes: int = 0
    execucao_ativa: str | None = None
    demonstracao: bool = False


class Status(Aberto):
    """Status derivado da base: o que está rodando, senão a execução ativa, senão a mais recente."""

    base_id: str
    estado: str = Field(description='"inspecionada" | "na_fila" | "treinando" | "pronta" | "erro"')
    etapa: str | None = None
    fracao: float = 0.0
    mensagem: str | None = Field(None, description="Erro legível em pt-BR quando estado = 'erro'.")
    problemas: list[str] = []
    avisos: list[str] = []
    n_clientes: int | None = None
    execucao_id: str | None = Field(None, description="De qual execução veio este status.")
    execucao_ativa: str | None = Field(None, description="Execução que as rotas sem `execucoes/` servem.")
    n_execucoes: int = 0
    iniciado_em: str | None = None
    atualizado_em: str | None = None
    segundos: float = 0.0
    opcoes: dict[str, Any] | None = None


class StatusExecucao(Aberto):
    """Status de uma execução (versão do modelo)."""

    base_id: str
    execucao_id: str
    rotulo: str | None = None
    estado: str = Field(description='"na_fila" | "treinando" | "pronta" | "erro"')
    etapa: str | None = None
    fracao: float = 0.0
    mensagem: str | None = None
    problemas: list[str] = []
    avisos: list[str] = []
    n_clientes: int | None = None
    criada_em: str | None = None
    iniciado_em: str | None = None
    concluido_em: str | None = None
    atualizado_em: str | None = None
    segundos: float = 0.0
    opcoes: dict[str, Any] | None = None


class Execucao(StatusExecucao):
    """Uma versão do modelo: status + `resumo.json` (null enquanto não fica pronta)."""

    ativa: bool = False
    resumo: dict[str, Any] | None = None


class ListaExecucoes(Aberto):
    base_id: str
    execucao_ativa: str | None = None
    execucoes: list[Execucao] = []


class LinhaComparacao(Aberto):
    campo: str
    rotulo: str
    valores: list[Any] = []


class ColunaComparacao(Aberto):
    execucao_id: str
    rotulo: str | None = None
    criada_em: str | None = None
    ativa: bool = False
    modelo: str | None = None


class Comparacao(Aberto):
    base_id: str
    execucoes: list[ColunaComparacao] = []
    linhas: list[LinhaComparacao] = []
    resumos: list[dict[str, Any]] = []


class BaseDetalhe(Aberto):
    base_id: str
    nome: str
    criada_em: str | None = None
    arquivos: list[str] = []
    bytes_total: int | None = None
    demonstracao: bool = False
    inspecao: Inspecao
    mapeamento: dict[str, Any] | None = None
    opcoes: dict[str, Any] | None = None
    status: Status
    execucao_ativa: str | None = None
    execucoes: list[Execucao] = []


class TreinoAceito(Aberto):
    base_id: str
    execucao_id: str
    execucao: StatusExecucao
    status: Status
    avisos: list[str] = []


class PaginaClientes(Aberto):
    base_id: str
    execucao_id: str | None = None
    gerado_em: str | None = None
    mes_referencia: str | None = None
    modelo: dict[str, Any] | None = None
    resumo: dict[str, Any] | None = None
    total: int = Field(0, description="Clientes ativos na fila.")
    total_filtrado: int = Field(0, description="Depois do filtro de faixa.")
    desde: int = 0
    limite: int = 0
    clientes: list[dict[str, Any]] = Field(default_factory=list,
                                           description="Na ordem da fila (perda anual ajustada pelo delta de risco); nunca reordenados.")


class Saude(Aberto):
    ok: bool = True
    versao: str
    bases: int = 0


BaseCriada.model_rebuild()
