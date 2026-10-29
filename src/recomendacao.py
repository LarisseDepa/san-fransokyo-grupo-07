"""
Etapa 7: risco e recomendações.

A demanda é prevista UMA vez, a partir dos hábitos (modelo da etapa 6). Depois, a MESMA demanda é
comparada com cada plano: trocar o plano muda a adequação e o custo, nunca o hábito.
Os planos são os 8 planos fictícios das planilhas. Entre os que cobrem a demanda, a ordem vem do modelo
híbrido da etapa 8: soma ponderada da propensão do Random Forest e da nota normalizada da filtragem
colaborativa. O peso vem da validação; a saída é uma pontuação, não uma probabilidade calibrada.
O JavaScript do site (src/paginas/claro_guia.js) repete estas regras; um teste confere que dão o
mesmo resultado. Aqui só há números e decisões; os textos para as pessoas ficam no site.

Classes:
  AtributosDaSimulacao -> monta as variáveis do modelo a partir das poucas respostas do visitante
  AnalisadorDeRisco    -> saldo, ritmo previsto, dias até a franquia acabar e antecipação
  RecomendadorDePlanos -> até 3 planos que cobrem a demanda, na ordem da pontuação híbrida
  CasosDemonstrativos  -> a Cliente Claro (exemplo fictício) e os cenários do site, com todos os números rastreáveis
"""
import math
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

import formulas
from modelo_de_recomendacao import AtributosDaOferta, ExportadorDaColaborativa
from tabelas import COLUNA_ORIGEM

DIAS_DO_CICLO_NA_SIMULACAO = 30
NIVEL_ESGOTADA, NIVEL_ALTO, NIVEL_ATENCAO, NIVEL_BAIXO = "esgotada", "alto", "atencao", "baixo"
PROPORCAO_DE_ATENCAO = 0.85  # demanda acima de 85% do saldo: "atenção", mesmo sem acabar antes


# ------------------------------------------------------------------ variáveis a partir do visitante
class AtributosDaSimulacao:
    """O visitante informa poucos dados (GB usados, dias desde a renovação e uso típico por mês).
    As demais variáveis do modelo são estimadas por esta REGRA (hipótese da simulação):
    antes da última semana, o ritmo era o típico (uso típico / 30); a última semana leva o resto.
    Os hábitos que o visitante não informa vêm de um grupo de clientes fictícios da base."""

    def __init__(self, data_de_referencia, dias_de_um_mes=30, dias_do_ciclo=DIAS_DO_CICLO_NA_SIMULACAO):
        self.data_de_referencia = data_de_referencia
        self.dias_de_um_mes = dias_de_um_mes
        self.dias_do_ciclo = dias_do_ciclo

    def montar(self, gb_usados, dias_desde_renovacao, uso_tipico_mes_gb, habitos):
        dias, usados = int(dias_desde_renovacao), float(gb_usados)
        ritmo_tipico = float(uso_tipico_mes_gb) / self.dias_de_um_mes
        dias_restantes = self.dias_do_ciclo - dias
        if dias >= 7:
            ritmo_recente = max(0.0, (usados - ritmo_tipico * (dias - 7)) / 7)
        else:  # a última semana inclui dias do ciclo anterior, no ritmo típico
            ritmo_recente = (usados + ritmo_tipico * (7 - dias)) / 7
        ritmo_do_ciclo = float(formulas.media_diaria(usados, dias))
        return {
            **habitos,
            "dados_gb": float(uso_tipico_mes_gb),
            "consumo_ciclo_atual_gb": usados,
            "dias_decorridos": dias,
            "dias_ate_renovacao": dias_restantes,
            "media_ultimos_7_dias_gb": ritmo_recente,
            "media_7_dias_anteriores_gb": ritmo_tipico,
            "tendencia_consumo_gb_dia": ritmo_recente - ritmo_tipico,
            "consumo_ciclo_anterior_gb": float(uso_tipico_mes_gb),
            "dias_fim_de_semana_restantes": self.fins_de_semana(dias_restantes),
            "projecao_pelo_ritmo_do_ciclo_gb": formulas.projecao_pelo_ritmo(ritmo_do_ciclo, dias_restantes),
            "projecao_pelo_ritmo_recente_gb": formulas.projecao_pelo_ritmo(ritmo_recente, dias_restantes),
            "projecao_pelo_habito_gb": formulas.projecao_pelo_ritmo(ritmo_tipico, dias_restantes),
        }

    def fins_de_semana(self, dias_restantes):
        """Sábados e domingos entre a data de referência e a véspera da renovação."""
        return sum(1 for dia in range(dias_restantes)
                   if (self.data_de_referencia + timedelta(days=dia)).weekday() >= 5)


# ------------------------------------------------------------------ risco
@dataclass
class AnaliseDeRisco:
    capacidade_gb: float
    consumo_informado_gb: float
    consumo_excede_capacidade: bool  # o consumo informado passa da franquia (a internet já teria acabado)
    consumo_efetivo_gb: float
    percentual_utilizado: float
    saldo_gb: float
    demanda_prevista_gb: float
    dias_ate_renovacao: int
    ritmo_previsto_gb_dia: float
    dias_ate_acabar: float | None
    dias_de_antecipacao: int  # dias inteiros antes da renovação ("cerca de N dias")
    nivel: str


class AnalisadorDeRisco:
    """Compara a demanda prevista com o saldo de uso geral. Supõe a demanda distribuída igualmente
    pelos dias que faltam (é assim que se estima o dia em que a franquia acaba)."""

    def analisar(self, capacidade_gb, consumo_gb, demanda_prevista_gb, dias_ate_renovacao):
        consumo_efetivo = min(float(consumo_gb), float(capacidade_gb))
        saldo = float(formulas.saldo_geral(capacidade_gb, consumo_efetivo))
        ritmo = float(demanda_prevista_gb) / dias_ate_renovacao
        dias_ate_acabar = float(formulas.dias_ate_esgotamento(saldo, ritmo))
        antecipacao = float(formulas.dias_de_antecipacao(dias_ate_renovacao, dias_ate_acabar))
        return AnaliseDeRisco(
            capacidade_gb=float(capacidade_gb), consumo_informado_gb=float(consumo_gb),
            consumo_excede_capacidade=float(consumo_gb) > float(capacidade_gb), consumo_efetivo_gb=consumo_efetivo,
            percentual_utilizado=float(formulas.percentual_utilizado(consumo_efetivo, capacidade_gb)), saldo_gb=saldo,
            demanda_prevista_gb=float(demanda_prevista_gb), dias_ate_renovacao=int(dias_ate_renovacao),
            ritmo_previsto_gb_dia=ritmo, dias_ate_acabar=None if math.isnan(dias_ate_acabar) else dias_ate_acabar,
            dias_de_antecipacao=int(math.floor(antecipacao)), nivel=self._nivel(saldo, antecipacao, demanda_prevista_gb))

    @staticmethod
    def _nivel(saldo, antecipacao, demanda):
        if saldo <= 0:
            return NIVEL_ESGOTADA
        if antecipacao >= 1:
            return NIVEL_ALTO
        if demanda > PROPORCAO_DE_ATENCAO * saldo:
            return NIVEL_ATENCAO
        return NIVEL_BAIXO


# ------------------------------------------------------------------ planos
class ChancesDoPlano(NamedTuple):
    """Pontuações de 0 a 1; nome legado preservado para compatibilidade."""
    hibrida: float
    random_forest: float
    filtragem_colaborativa: float


@dataclass
class OpcaoDePlano:
    nome_do_plano: str
    tipo: str  # manter, economia ou mais_gb
    gb_do_plano: float  # GB principais
    mensalidade_brl: float
    diferenca_de_preco_brl: float  # em relação ao plano atual (negativo = mais barato)
    folga_gb: float  # GB que sobram depois da necessidade estimada
    folga_pequena: bool  # sobra menos que a margem de segurança: se o ritmo subir, pode faltar
    chance_de_aceitar: float | None  # alias legado da pontuação híbrida, sem interpretação probabilística
    chance_random_forest: float | None
    chance_filtragem_colaborativa: float | None  # nota prevista ÷ 10
    pontuacao_recomendacao: float | None = field(init=False)
    propensao_random_forest: float | None = field(init=False)
    pontuacao_colaborativa: float | None = field(init=False)

    def __post_init__(self):
        self.pontuacao_recomendacao = self.chance_de_aceitar
        self.propensao_random_forest = self.chance_random_forest
        self.pontuacao_colaborativa = self.chance_filtragem_colaborativa


@dataclass
class Recomendacao:
    necessidade_gb: float  # demanda estimada de um ciclo inteiro
    margem_de_seguranca: float
    plano_atual_atende: bool
    opcoes: list = field(default_factory=list)
    nenhum_plano_cobre: bool = False


class RecomendadorDePlanos:
    """Até 3 planos que COBREM a mesma demanda estimada (GB principais >= necessidade), na ordem da
    pontuação de recomendação (modelo híbrido da etapa 8). Quem cobre com folga menor que a margem
    recebe o aviso "folga pequena". Se o plano atual cobre, ele aparece ("manter") com as alternativas
    MAIS BARATAS que também cobrem; as mais caras não são sugeridas."""

    MAXIMO_DE_OPCOES = 3

    def __init__(self, planos, modelo_hibrido, atributos_da_oferta, margem_de_seguranca):
        self.planos = planos
        self.modelo_hibrido = modelo_hibrido
        self.atributos_da_oferta = atributos_da_oferta
        self.margem = margem_de_seguranca

    def necessidade(self, consumo_gb, demanda_prevista_gb, uso_tipico_mes_gb):
        """Demanda de um ciclo: o maior valor entre (usado + previsto) e o uso típico de um mês inteiro."""
        return max(float(consumo_gb) + float(demanda_prevista_gb), float(uso_tipico_mes_gb))

    def recomendar(self, necessidade_gb, nome_do_plano_atual, cliente):
        """cliente: as variáveis do cliente para o modelo de propensão (ver AtributosDaOferta), o id_cliente
        (None para quem não tem histórico) e o grupo."""
        atual = self.planos.por_nome(nome_do_plano_atual)
        atual_atende = atual.gb_principais >= necessidade_gb
        cobrem = [plano for plano in self.planos.outros(atual.nome) if plano.gb_principais >= necessidade_gb
                  and (not atual_atende or plano.mensalidade_brl < atual.mensalidade_brl)]
        ordenados = sorted(zip(cobrem, self._chances(cobrem, atual, cliente)), key=self._ordem)
        limite = self.MAXIMO_DE_OPCOES - 1 if atual_atende else self.MAXIMO_DE_OPCOES
        opcoes = [self._opcao(plano, atual, necessidade_gb, chances) for plano, chances in ordenados[:limite]]
        if atual_atende:
            opcoes.append(self._opcao(atual, atual, necessidade_gb, None))
        return Recomendacao(necessidade_gb=necessidade_gb, margem_de_seguranca=self.margem,
                            plano_atual_atende=atual_atende, opcoes=opcoes, nenhum_plano_cobre=not opcoes)

    def _chances(self, planos, atual, cliente):
        if not planos:
            return []
        pares = [{**self.atributos_da_oferta.de_um_par(cliente, plano.nome, atual.nome), "id_cliente": cliente["id_cliente"]}
                 for plano in planos]
        floresta, colaborativa, hibrida = self.modelo_hibrido.notas_por_modelo(pares)
        return [ChancesDoPlano(float(media), float(da_floresta), float(da_colaborativa))
                for da_floresta, da_colaborativa, media in zip(floresta, colaborativa, hibrida)]

    @staticmethod
    def _ordem(plano_e_chances):
        """Maior pontuação híbrida primeiro; no empate, o mais barato; depois, o nome."""
        plano, chances = plano_e_chances
        return -chances.hibrida, plano.mensalidade_brl, plano.nome

    def _opcao(self, plano, atual, necessidade_gb, chances):
        if plano.nome == atual.nome:
            tipo = "manter"
        elif plano.mensalidade_brl < atual.mensalidade_brl:
            tipo = "economia"
        else:
            tipo = "mais_gb"
        folga = plano.gb_principais - necessidade_gb
        return OpcaoDePlano(nome_do_plano=plano.nome, tipo=tipo, gb_do_plano=plano.gb_principais,
                            mensalidade_brl=plano.mensalidade_brl,
                            diferenca_de_preco_brl=round(plano.mensalidade_brl - atual.mensalidade_brl, 2),
                            folga_gb=folga, folga_pequena=folga < self.margem * necessidade_gb,
                            chance_de_aceitar=chances.hibrida if chances else None,
                            chance_random_forest=chances.random_forest if chances else None,
                            chance_filtragem_colaborativa=chances.filtragem_colaborativa if chances else None)


# ------------------------------------------------------------------ casos do site e da etapa 7
class CasosDemonstrativos:
    """Escolhe a Cliente Claro (o exemplo fictício) na base enriquecida e monta os cenários do site, com todos os números."""

    PLANO_DO_EXEMPLO = "Digital"
    PERFIL_DO_EXEMPLO = "Mobilidade"

    def __init__(self, base_final, previsor, planos, modelo_hibrido, configuracao_do_projeto):
        self.base = base_final
        self.previsor = previsor
        self.planos = planos
        self.config = configuracao_do_projeto
        self.analisador = AnalisadorDeRisco()
        self.recomendador = RecomendadorDePlanos(planos, modelo_hibrido, AtributosDaOferta(planos),
                                                 configuracao_do_projeto.MARGEM_DE_SEGURANCA)
        self.atributos = AtributosDaSimulacao(configuracao_do_projeto.DATA_DE_REFERENCIA,
                                              configuracao_do_projeto.DIAS_DE_UM_MES_TIPICO)

    # -------------------------------------------------------------- Cliente Claro (o exemplo fictício)
    def escolher_cliente_do_exemplo(self):
        """Uma cliente fictícia com o plano Digital (35 GB principais; bônus de uso livre já vencido):
        ritmo em alta, franquia ainda não esgotada, meio do ciclo e risco previsto de acabar antes."""
        capacidade = self.planos.por_nome(self.PLANO_DO_EXEMPLO).gb_principais
        candidatas = self.base[(self.base.plano_atual == self.PLANO_DO_EXEMPLO)
                               & (self.base.franquia_geral_gb == capacidade) & (self.base.franquia_ja_esgotada == 0)
                               & (self.base.cenario_de_ritmo_simulado == "aumento")
                               & self.base.dias_decorridos.between(8, 20)].copy()
        candidatas["previsao"] = self.previsor.prever(candidatas)
        candidatas["antecipacao"] = [
            self.analisador.analisar(capacidade, linha.consumo_ciclo_atual_gb, linha.previsao,
                                     linha.dias_ate_renovacao).dias_de_antecipacao
            for linha in candidatas.itertuples()]
        com_risco = candidatas[candidatas.antecipacao >= 2]
        preferidas = com_risco[com_risco[COLUNA_ORIGEM] == self.PERFIL_DO_EXEMPLO]
        escolhida = (preferidas if len(preferidas) else com_risco).sort_values("id_cliente")
        if escolhida.empty:
            raise ValueError("Nenhum registro da base serve para o exemplo da Cliente Claro: revise os critérios.")
        return escolhida.iloc[0]

    @staticmethod
    def cliente_para_recomendacao(uso_tipico_mes_gb, habitos, grupo, id_cliente):
        """As variáveis do cliente que o modelo híbrido usa: as do Random Forest (ver AtributosDaOferta) e o
        id_cliente da filtragem colaborativa (None para quem não tem histórico: aí vale a média do grupo)."""
        return {"id_cliente": id_cliente, "dados_gb": float(uso_tipico_mes_gb), "minutos": float(habitos["minutos"]),
                "streaming_horas": float(habitos["streaming_horas"]),
                "deslocamento_dias": float(habitos["deslocamento_dias"]), "grupo": str(grupo)}

    def caso_do_exemplo(self):
        registro = self.escolher_cliente_do_exemplo()
        atributos = {variavel: _valor_simples(registro[variavel]) for variavel in self.previsor.configuracao.variaveis}
        demanda = self.previsor.prever_um(atributos)
        plano = self.planos.por_nome(registro.plano_atual)
        risco = self.analisador.analisar(plano.gb_principais, registro.consumo_ciclo_atual_gb, demanda,
                                         int(registro.dias_ate_renovacao))
        necessidade = self.recomendador.necessidade(registro.consumo_ciclo_atual_gb, demanda, registro.dados_gb)
        cliente = self.cliente_para_recomendacao(registro.dados_gb, registro, registro[COLUNA_ORIGEM],
                                                 int(registro.id_cliente))
        return {
            "id_cliente": int(registro.id_cliente), "perfil_grupo": registro.perfil_grupo,
            "plano_atual": plano.nome, "atributos": atributos, "cliente": cliente,
            "dias_decorridos": int(registro.dias_decorridos), "dias_ate_renovacao": int(registro.dias_ate_renovacao),
            "data_proxima_renovacao": registro.data_proxima_renovacao,
            "consumo_ciclo_atual_gb": float(registro.consumo_ciclo_atual_gb),
            "media_diaria_gb": _valor_simples(registro.media_diaria_gb),
            "baseline_demanda_restante_gb": _valor_simples(registro.baseline_demanda_restante_gb),
            "demanda_prevista_gb": demanda,
            "target_demanda_restante_gb": float(registro.target_demanda_restante_gb),
            "risco": asdict(risco),
            "recomendacao": asdict(self.recomendador.recomendar(necessidade, plano.nome, cliente)),
        }

    # -------------------------------------------------------------- cenários do painel "Quer simular outra situação?"
    def habitos_do_grupo(self, nome_da_planilha):
        """Hábitos que o visitante não informa: medianas e categorias mais comuns de um grupo da base."""
        grupo = self.base[self.base[COLUNA_ORIGEM] == nome_da_planilha]
        return {"minutos": float(grupo.minutos.median()), "streaming_horas": float(grupo.streaming_horas.median()),
                "deslocamento_dias": float(grupo.deslocamento_dias.median()),
                "idade_faixa": str(grupo.idade_faixa.mode().iloc[0]), "regiao": str(grupo.regiao.mode().iloc[0])}

    def cenarios(self):
        return {
            "mais_internet": {"plano": "Streaming", "gb_usados": 26.0, "dias_desde_renovacao": 15,
                              "uso_tipico_mes_gb": 30.0, "grupo_dos_habitos": "Excesso_de_Consumo",
                              "habitos": self.habitos_do_grupo("Excesso_de_Consumo")},
            "economia": {"plano": "Total", "gb_usados": 6.0, "dias_desde_renovacao": 15,
                         "uso_tipico_mes_gb": 12.0, "grupo_dos_habitos": "Economia",
                         "habitos": self.habitos_do_grupo("Economia")},
        }

    def resultado_movel(self, cenario):
        """O mesmo cálculo do site para um cenário (usado nos testes e na etapa 7)."""
        atributos = self.atributos.montar(cenario["gb_usados"], cenario["dias_desde_renovacao"],
                                          cenario["uso_tipico_mes_gb"], cenario["habitos"])
        demanda = self.previsor.prever_um(atributos)
        plano = self.planos.por_nome(cenario["plano"])
        risco = self.analisador.analisar(plano.gb_principais, cenario["gb_usados"], demanda,
                                         atributos["dias_ate_renovacao"])
        necessidade = self.recomendador.necessidade(cenario["gb_usados"], demanda, cenario["uso_tipico_mes_gb"])
        cliente = self.cliente_para_recomendacao(cenario["uso_tipico_mes_gb"], cenario["habitos"],
                                                 cenario["grupo_dos_habitos"], id_cliente=None)
        return {"atributos": atributos, "demanda_prevista_gb": demanda, "risco": asdict(risco),
                "recomendacao": asdict(self.recomendador.recomendar(necessidade, plano.nome, cliente))}

    def resumo_por_perfil(self):
        """Evidência de como cada perfil é atendido: o modelo prevê a demanda de todos os clientes,
        e o risco é calculado contra a franquia de uso geral (acadêmica) de cada um."""
        previsoes = self.previsor.prever(self.base)
        niveis = [self.analisador.analisar(linha.franquia_geral_gb, linha.consumo_ciclo_atual_gb, previsao,
                                           int(linha.dias_ate_renovacao)).nivel
                  for linha, previsao in zip(self.base.itertuples(), previsoes)]
        tabela = pd.DataFrame({"perfil_grupo": self.base.perfil_grupo.to_numpy(), "nivel": niveis,
                               "uso_sobre_franquia": (self.base.dados_gb / self.base.franquia_geral_gb).to_numpy()})
        percentuais = pd.crosstab(tabela.perfil_grupo, tabela.nivel, normalize="index").mul(100)
        linhas = []
        for perfil, grupo in tabela.groupby("perfil_grupo"):
            linhas.append({"perfil_grupo": perfil, "clientes": len(grupo),
                           "uso_tipico_sobre_franquia_mediana": float(grupo.uso_sobre_franquia.median()),
                           **{f"percentual_{nivel}": float(percentuais.loc[perfil].get(nivel, 0.0))
                              for nivel in (NIVEL_ESGOTADA, NIVEL_ALTO, NIVEL_ATENCAO, NIVEL_BAIXO)}})
        return linhas

    def para_o_site(self, modelo_de_demanda_em, modelo_de_propensao_em):
        """Tudo o que o JavaScript do site precisa, mais os resultados do Python (para conferência)."""
        cenarios = self.cenarios()
        exemplo = self.caso_do_exemplo()
        return {
            "data_de_referencia": self.config.DATA_DE_REFERENCIA.isoformat(),
            "dias_do_ciclo": DIAS_DO_CICLO_NA_SIMULACAO, "dias_de_um_mes": self.config.DIAS_DE_UM_MES_TIPICO,
            "margem_de_seguranca": self.config.MARGEM_DE_SEGURANCA,
            "proporcao_de_atencao": PROPORCAO_DE_ATENCAO,
            "modelo": modelo_de_demanda_em,
            "modelo_de_propensao": modelo_de_propensao_em,
            "peso_random_forest": self.recomendador.modelo_hibrido.peso_random_forest,
            "planos": self.planos.como_lista_de_dicionarios(),
            "filtragem_colaborativa": ExportadorDaColaborativa().exportar(self.recomendador.modelo_hibrido.colaborativa,
                                                                          [exemplo["id_cliente"]]),
            "cliente_do_exemplo": exemplo,
            "cenarios": cenarios,
            "resumo_por_perfil": self.resumo_por_perfil(),
            "resultados_do_python": {nome: self.resultado_movel(cenario) for nome, cenario in cenarios.items()},
        }


def _valor_simples(valor):
    """Tipos do numpy/pandas para tipos do Python (para o JSON); NaN vira None."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        return None if math.isnan(valor) else float(valor)
    return valor
