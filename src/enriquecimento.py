"""
Etapa 5: enriquecimento da base unida com dados SINTÉTICOS de contrato, franquia e consumo diário,
atributos históricos, baseline e targets futuros. A base de entrada não é limpa nem alterada:
todas as colunas recebidas (inclusive "origem") são preservadas, e só se acrescentam colunas novas.

Classes, na ordem de uso:
  ConfiguracaoDoEnriquecimento -> parâmetros da simulação (valores no config.py)
  ValidadorDaEntrada           -> confere as pré-condições da base unida (não corrige nada)
  SimuladorDeContratos         -> datas de contrato e renovação e a franquia de cada cliente
  SimuladorDeConsumo           -> consumo diário sintético: histórico observado e demanda futura
  ConstrutorDeAtributos        -> atributos históricos e baseline (só dados ATÉ a data de referência)
  ConstrutorDeAlvos            -> targets futuros (só dados DEPOIS da data de referência)
  ValidadorDaBaseFinal         -> confere as regras da base pronta (datas, franquias, colunas)
  EnriquecimentoDaBase         -> executa tudo, na ordem, e devolve a base final de modelagem
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import formulas
from tabelas import COLUNA_ORIGEM

COLUNAS_ORIGINAIS = ("id_cliente", "idade_faixa", "regiao", "dados_gb", "minutos", "streaming_horas",
                     "deslocamento_dias", "plano_atual", "mensalidade_atual", "risco_cancelamento", "perfil_grupo")
COLUNAS_NUMERICAS_ORIGINAIS = ("dados_gb", "minutos", "streaming_horas", "deslocamento_dias",
                               "mensalidade_atual", "risco_cancelamento")
COLUNAS_DE_DATA = ("data_referencia", "data_contratacao", "data_inicio_plano_atual", "data_ultima_renovacao",
                   "data_proxima_renovacao", "data_validade_bonus_livre", "target_data_esgotamento")
MAIOR_NUMERO_DE_DIAS_NO_MES = 31


class ErroNaBaseDeEntrada(ValueError):
    """A base unida não atende às pré-condições: corrija na etapa anterior (preparação dos dados)."""


class ErroNaBaseFinal(ValueError):
    """A base enriquecida quebrou alguma regra (datas, franquias ou colunas): é um erro do código."""


@dataclass(frozen=True)
class ConfiguracaoDoEnriquecimento:
    """Parâmetros da simulação. Os principais vêm do config.py; os demais são fixos e documentados aqui."""
    data_de_referencia: date
    semente: int
    franquias_dos_planos: dict  # {plano acadêmico: (GB principais, bônus de uso livre, bônus só para apps)}
    meses_de_validade_do_bonus: int = 12
    dias_de_um_mes: int = 30  # hipótese: dados_gb é o consumo típico de um mês
    linhas_esperadas: int = 960
    dias_de_historico: int = 90  # janela de consumo diário simulado antes da data de referência
    dias_minimos_de_contrato: int = 75  # garante histórico do ciclo anterior completo
    dias_maximos_de_contrato: int = 6 * 365
    chance_de_manter_o_plano_da_contratacao: float = 0.6
    variabilidade_diaria: float = 0.35  # desvio do ruído diário (lognormal com média 1)
    chance_base_de_aumento: float = 0.15  # chance de o ritmo ter aumentado nas últimas semanas...
    peso_do_deslocamento_no_aumento: float = 0.45  # ...maior para quem se desloca mais
    faixa_do_fator_de_aumento: tuple = (1.25, 1.9)
    chance_de_queda: float = 0.12
    faixa_do_fator_de_queda: tuple = (0.55, 0.85)
    faixa_de_dias_desde_a_mudanca: tuple = (3, 25)
    chance_de_mudanca_futura: float = 0.25  # imprevisto depois da data de referência (não está nas features)
    faixa_do_fator_de_mudanca_futura: tuple = (0.7, 1.4)
    peso_do_deslocamento_nos_dias_uteis: float = 0.25
    peso_do_streaming_no_fim_de_semana: float = 0.35
    maior_streaming_horas: float = 25.0  # para normalizar streaming_horas entre 0 e 1

    @classmethod
    def a_partir_do_config(cls, config):
        return cls(data_de_referencia=config.DATA_DE_REFERENCIA, semente=config.SEMENTE,
                   franquias_dos_planos=config.FRANQUIAS_DOS_PLANOS_ACADEMICOS,
                   meses_de_validade_do_bonus=config.MESES_DE_VALIDADE_DO_BONUS,
                   dias_de_um_mes=config.DIAS_DE_UM_MES_TIPICO)

    @property
    def referencia(self):
        return pd.Timestamp(self.data_de_referencia)


@dataclass
class ResultadoDoEnriquecimento:
    base_final: pd.DataFrame  # base de entrada + colunas novas (uma linha por cliente)
    consumo_diario: pd.DataFrame  # consumo diário sintético (uma linha por cliente e dia)


# ------------------------------------------------------------------ validação da entrada
class ValidadorDaEntrada:
    """Confere se a base unida pode ser enriquecida. Não corrige nada: só aponta os problemas."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def validar(self, base):
        problemas = self._problemas(base)
        if problemas:
            raise ErroNaBaseDeEntrada("A base unida não pode ser enriquecida:\n- " + "\n- ".join(problemas))

    def _problemas(self, base):
        faltando = [coluna for coluna in COLUNAS_ORIGINAIS if coluna not in base]
        if faltando:
            return [f"colunas obrigatórias ausentes: {', '.join(faltando)}"]
        problemas = []
        if len(base) != self.configuracao.linhas_esperadas:
            problemas.append(f"{len(base)} linhas; eram esperadas {self.configuracao.linhas_esperadas}")
        if not base["id_cliente"].is_unique:
            problemas.append("id_cliente tem valores repetidos")
        for coluna in COLUNAS_NUMERICAS_ORIGINAIS:
            if not pd.api.types.is_numeric_dtype(base[coluna]):
                problemas.append(f"{coluna} não é numérica")
            elif base[coluna].isna().any():
                problemas.append(f"{coluna} tem {int(base[coluna].isna().sum())} valores vazios")
            elif (base[coluna] < 0).any():
                problemas.append(f"{coluna} tem valores negativos")
        planos_sem_franquia = sorted(set(base["plano_atual"]) - set(self.configuracao.franquias_dos_planos))
        if planos_sem_franquia:
            problemas.append(f"planos sem franquia em FRANQUIAS_DOS_PLANOS_ACADEMICOS: {', '.join(planos_sem_franquia)}")
        return problemas


# ------------------------------------------------------------------ contratos e franquias
class SimuladorDeContratos:
    """Datas de contrato e de renovação e a franquia de cada cliente (tudo sintético)."""

    DIAS_MINIMOS_NO_PLANO_ATUAL = 15

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def simular(self, base, gerador):
        configuracao, referencia, quantidade = self.configuracao, self.configuracao.referencia, len(base)
        dias_desde_contratacao = gerador.integers(configuracao.dias_minimos_de_contrato,
                                                  configuracao.dias_maximos_de_contrato + 1, quantidade)
        mantem_o_plano = gerador.random(quantidade) < configuracao.chance_de_manter_o_plano_da_contratacao
        dias_no_plano_atual = np.where(
            mantem_o_plano, dias_desde_contratacao,
            gerador.integers(self.DIAS_MINIMOS_NO_PLANO_ATUAL, dias_desde_contratacao))
        dia_de_renovacao = gerador.integers(1, 29, quantidade)  # 1 a 28: existe em todos os meses

        contratos = pd.DataFrame({
            "data_referencia": referencia,
            "data_contratacao": referencia - pd.to_timedelta(dias_desde_contratacao, unit="D"),
            "data_inicio_plano_atual": referencia - pd.to_timedelta(dias_no_plano_atual, unit="D"),
        })
        contratos["data_ultima_renovacao"] = [self._ultima_renovacao(referencia, dia) for dia in dia_de_renovacao]
        contratos["data_proxima_renovacao"] = [data + relativedelta(months=1) for data in contratos.data_ultima_renovacao]
        contratos["_data_renovacao_anterior"] = [data - relativedelta(months=1) for data in contratos.data_ultima_renovacao]
        contratos["dias_desde_contratacao"] = dias_desde_contratacao
        contratos["dias_no_plano_atual"] = dias_no_plano_atual
        contratos["dias_total_ciclo"] = (contratos.data_proxima_renovacao - contratos.data_ultima_renovacao).dt.days
        contratos["dias_decorridos"] = (referencia - contratos.data_ultima_renovacao).dt.days
        contratos["dias_ate_renovacao"] = (contratos.data_proxima_renovacao - referencia).dt.days
        return self._com_franquia(contratos, base)

    @staticmethod
    def _ultima_renovacao(referencia, dia):
        """A renovação mais recente que não passa da data de referência."""
        neste_mes = pd.Timestamp(referencia.year, referencia.month, int(dia))
        return neste_mes if neste_mes <= referencia else neste_mes - relativedelta(months=1)

    def _com_franquia(self, contratos, base):
        franquias = base["plano_atual"].map(self.configuracao.franquias_dos_planos)
        contratos["franquia_principal_gb"] = [float(franquia[0]) for franquia in franquias]
        contratos["bonus_livre_gb"] = [float(franquia[1]) for franquia in franquias]
        contratos["bonus_apps_gb"] = [float(franquia[2]) for franquia in franquias]
        tem_bonus = contratos.bonus_livre_gb > 0
        validade = [inicio + relativedelta(months=self.configuracao.meses_de_validade_do_bonus)
                    for inicio in contratos.data_inicio_plano_atual]
        contratos["data_validade_bonus_livre"] = pd.Series(validade).where(tem_bonus)
        # regra: o bônus vale para o ciclo inteiro se ainda estiver válido no início do ciclo
        contratos["bonus_livre_ativo"] = (tem_bonus & (contratos.data_validade_bonus_livre
                                                       > contratos.data_ultima_renovacao)).astype(int)
        contratos["franquia_geral_gb"] = formulas.franquia_geral(
            contratos.franquia_principal_gb, contratos.bonus_livre_gb, contratos.bonus_livre_ativo)
        return contratos


# ------------------------------------------------------------------ consumo diário
@dataclass
class ConsumoSimulado:
    diario: pd.DataFrame  # id_cliente, data, periodo (historico/futuro), demanda_latente_gb, consumo_observado_gb
    ritmo: pd.DataFrame  # cenario_de_ritmo_simulado e fator_de_ritmo_simulado de cada cliente


class SimuladorDeConsumo:
    """Consumo diário sintético de cada cliente.

    demanda do dia = hábito diário (dados_gb / 30) x padrão da semana x mudança de ritmo x ruído.
    No histórico, o consumo OBSERVADO para quando a franquia de uso geral do ciclo acaba.
    No futuro, guarda-se a demanda LATENTE (o que o cliente usaria se tivesse franquia): ela não é
    uma medição da operadora, só a base dos targets. Não há aplicativos, sites nem localização."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def simular(self, base, contratos, gerador):
        partes, ritmos = [], []
        for cliente, contrato in zip(base.itertuples(index=False), contratos.itertuples(index=False)):
            diario, ritmo = self._simular_cliente(cliente, contrato, gerador)
            partes.append(diario)
            ritmos.append(ritmo)
        return ConsumoSimulado(pd.concat(partes, ignore_index=True), pd.DataFrame(ritmos))

    def _simular_cliente(self, cliente, contrato, gerador):
        configuracao, referencia = self.configuracao, self.configuracao.referencia
        dias_observados = min(int(contrato.dias_desde_contratacao), configuracao.dias_de_historico)
        datas = pd.date_range(referencia - pd.Timedelta(days=dias_observados),
                              contrato.data_proxima_renovacao - pd.Timedelta(days=1), freq="D")
        cenario, fator, regime = self._mudanca_de_ritmo(cliente, datas, gerador)
        regime = regime * self._mudanca_futura(datas, contrato, gerador)
        demanda = (cliente.dados_gb / configuracao.dias_de_um_mes) * self._padrao_da_semana(cliente, datas) \
            * regime * self._ruido(len(datas), gerador)
        historico = datas < referencia
        observado = np.full(len(datas), np.nan)
        observado[historico] = self._consumo_observado(demanda[historico], datas[historico], contrato)
        diario = pd.DataFrame({"id_cliente": cliente.id_cliente, "data": datas,
                               "periodo": np.where(historico, "historico", "futuro"),
                               "demanda_latente_gb": demanda, "consumo_observado_gb": observado})
        return diario, {"cenario_de_ritmo_simulado": cenario, "fator_de_ritmo_simulado": fator}

    def _mudanca_de_ritmo(self, cliente, datas, gerador):
        """Aumento ou queda de ritmo nas últimas semanas (mais provável para quem se desloca mais)."""
        configuracao = self.configuracao
        chance_de_aumento = min(0.9, configuracao.chance_base_de_aumento + configuracao.peso_do_deslocamento_no_aumento
                                * cliente.deslocamento_dias / MAIOR_NUMERO_DE_DIAS_NO_MES)
        sorteio = gerador.random()
        dias_desde_a_mudanca = gerador.integers(configuracao.faixa_de_dias_desde_a_mudanca[0],
                                                configuracao.faixa_de_dias_desde_a_mudanca[1] + 1)
        if sorteio < chance_de_aumento:
            cenario, fator = "aumento", gerador.uniform(*configuracao.faixa_do_fator_de_aumento)
        elif sorteio < chance_de_aumento + configuracao.chance_de_queda:
            cenario, fator = "queda", gerador.uniform(*configuracao.faixa_do_fator_de_queda)
        else:
            cenario, fator = "estavel", 1.0
        inicio_da_mudanca = configuracao.referencia - pd.Timedelta(days=int(dias_desde_a_mudanca))
        return cenario, float(fator), np.where(datas >= inicio_da_mudanca, fator, 1.0)

    def _mudanca_futura(self, datas, contrato, gerador):
        """Imprevisto depois da data de referência: deixa o futuro incerto (o modelo não o vê)."""
        configuracao = self.configuracao
        acontece = gerador.random() < configuracao.chance_de_mudanca_futura
        dia = gerador.integers(0, int(contrato.dias_ate_renovacao))
        fator = gerador.uniform(*configuracao.faixa_do_fator_de_mudanca_futura)
        if not acontece:
            return np.ones(len(datas))
        return np.where(datas >= configuracao.referencia + pd.Timedelta(days=int(dia)), fator, 1.0)

    def _padrao_da_semana(self, cliente, datas):
        """Dias úteis pesam mais para quem se desloca; fins de semana, para quem usa mais streaming.
        Os pesos são normalizados para a média da semana ficar em 1."""
        configuracao = self.configuracao
        peso_util = 1 + configuracao.peso_do_deslocamento_nos_dias_uteis * cliente.deslocamento_dias / MAIOR_NUMERO_DE_DIAS_NO_MES
        peso_fim_de_semana = 1 + configuracao.peso_do_streaming_no_fim_de_semana * min(
            1.0, cliente.streaming_horas / configuracao.maior_streaming_horas)
        escala = 7 / (5 * peso_util + 2 * peso_fim_de_semana)
        return np.where(datas.day_of_week >= 5, peso_fim_de_semana, peso_util) * escala

    def _ruido(self, quantidade, gerador):
        """Ruído lognormal com média 1 (dias acima e abaixo do normal)."""
        desvio = self.configuracao.variabilidade_diaria
        return gerador.lognormal(mean=-desvio ** 2 / 2, sigma=desvio, size=quantidade)

    def _consumo_observado(self, demanda, datas, contrato):
        """Em cada ciclo, o consumo acumulado para na franquia de uso geral (depois, a internet acaba)."""
        inicios_dos_ciclos = self._inicios_dos_ciclos(datas[0], contrato)
        ciclo_de_cada_dia = np.searchsorted(inicios_dos_ciclos.values, datas.values, side="right") - 1
        observado = np.zeros(len(demanda))
        for ciclo in np.unique(ciclo_de_cada_dia):
            dias_do_ciclo = ciclo_de_cada_dia == ciclo
            franquia = self._franquia_do_ciclo(inicios_dos_ciclos[ciclo], contrato)
            acumulado = np.minimum(np.cumsum(demanda[dias_do_ciclo]), franquia)
            observado[dias_do_ciclo] = np.diff(acumulado, prepend=0.0)
        return observado

    @staticmethod
    def _inicios_dos_ciclos(primeiro_dia, contrato):
        inicios = [contrato.data_ultima_renovacao]
        while inicios[-1] > primeiro_dia:
            inicios.append(inicios[-1] - relativedelta(months=1))
        return pd.DatetimeIndex(sorted(inicios))

    @staticmethod
    def _franquia_do_ciclo(inicio_do_ciclo, contrato):
        """Mesma regra da franquia atual: o bônus conta se estiver válido no início do ciclo.
        Simplificação: ciclos anteriores ao plano atual usam a mesma franquia."""
        bonus_valido = pd.notna(contrato.data_validade_bonus_livre) and contrato.data_validade_bonus_livre > inicio_do_ciclo
        return contrato.franquia_principal_gb + (contrato.bonus_livre_gb if bonus_valido else 0.0)


# ------------------------------------------------------------------ atributos e baseline
class ConstrutorDeAtributos:
    """Atributos históricos e baseline. Usa SÓ o consumo observado até a véspera da data de referência."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def construir(self, base, contratos, diario):
        historico = diario[diario.periodo == "historico"].merge(
            contratos.assign(id_cliente=base["id_cliente"].values)[
                ["id_cliente", "data_ultima_renovacao", "_data_renovacao_anterior"]], on="id_cliente")
        referencia = self.configuracao.referencia

        def somar(condicao):
            return historico[condicao].groupby("id_cliente")["consumo_observado_gb"].sum().reindex(
                base["id_cliente"], fill_value=0.0).to_numpy()

        atributos = pd.DataFrame(index=base.index)
        atributos["consumo_ciclo_atual_gb"] = somar(historico.data >= historico.data_ultima_renovacao)
        atributos["saldo_geral_gb"] = formulas.saldo_geral(contratos.franquia_geral_gb, atributos.consumo_ciclo_atual_gb)
        atributos["percentual_franquia_utilizada"] = formulas.percentual_utilizado(
            atributos.consumo_ciclo_atual_gb, contratos.franquia_geral_gb)
        atributos["franquia_ja_esgotada"] = (atributos.saldo_geral_gb <= 1e-9).astype(int)
        atributos["consumo_ultimos_7_dias_gb"] = somar(historico.data >= referencia - pd.Timedelta(days=7))
        atributos["consumo_ultimos_14_dias_gb"] = somar(historico.data >= referencia - pd.Timedelta(days=14))
        atributos["media_diaria_gb"] = formulas.media_diaria(atributos.consumo_ciclo_atual_gb, contratos.dias_decorridos)
        atributos["media_ultimos_7_dias_gb"] = atributos.consumo_ultimos_7_dias_gb / 7
        atributos["media_7_dias_anteriores_gb"] = (atributos.consumo_ultimos_14_dias_gb
                                                   - atributos.consumo_ultimos_7_dias_gb) / 7
        atributos["tendencia_consumo_gb_dia"] = atributos.media_ultimos_7_dias_gb - atributos.media_7_dias_anteriores_gb
        atributos["consumo_ciclo_anterior_gb"] = somar((historico.data >= historico._data_renovacao_anterior)
                                                       & (historico.data < historico.data_ultima_renovacao))
        atributos["dias_observados"] = historico.groupby("id_cliente").size().reindex(base["id_cliente"]).to_numpy()
        atributos["dias_fim_de_semana_restantes"] = self._fins_de_semana_restantes(contratos)
        atributos = atributos.join(self._projecoes(base, contratos, atributos))
        return atributos.join(self._baseline(atributos, contratos))

    def _projecoes(self, base, contratos, atributos):
        """Projeções "ritmo x dias que faltam" (a demanda tem essa forma, e as árvores lidam mal com
        multiplicações). Diferente do baseline, a do ciclo nunca fica vazia: sem dias no ciclo,
        usa o ritmo dos últimos 7 dias."""
        dias = contratos.dias_ate_renovacao
        ritmo_do_ciclo = atributos.media_diaria_gb.fillna(atributos.media_ultimos_7_dias_gb)
        return pd.DataFrame({
            "projecao_pelo_ritmo_do_ciclo_gb": formulas.projecao_pelo_ritmo(ritmo_do_ciclo, dias),
            "projecao_pelo_ritmo_recente_gb": formulas.projecao_pelo_ritmo(atributos.media_ultimos_7_dias_gb, dias),
            "projecao_pelo_habito_gb": formulas.projecao_pelo_ritmo(
                base["dados_gb"] / self.configuracao.dias_de_um_mes, dias),
        }, index=atributos.index)

    def _fins_de_semana_restantes(self, contratos):
        """Sábados e domingos entre a data de referência e a renovação (calendário, conhecido de antemão)."""
        inicio = np.datetime64(self.configuracao.data_de_referencia)
        fins = contratos.data_proxima_renovacao.values.astype("datetime64[D]")
        return contratos.dias_ate_renovacao.to_numpy() - np.busday_count(inicio, fins)

    @staticmethod
    def _baseline(atributos, contratos):
        """Projeção pela média diária do ciclo (indefinida quando o ciclo ainda não tem dias)."""
        baseline = pd.DataFrame(index=atributos.index)
        baseline["baseline_demanda_restante_gb"] = formulas.baseline_demanda_restante(
            atributos.media_diaria_gb, contratos.dias_ate_renovacao)
        baseline["baseline_dias_ate_esgotamento"] = formulas.dias_ate_esgotamento(
            atributos.saldo_geral_gb, atributos.media_diaria_gb)
        risco = (baseline.baseline_demanda_restante_gb > atributos.saldo_geral_gb).astype(float)
        indefinido = baseline.baseline_demanda_restante_gb.isna() | (atributos.franquia_ja_esgotada == 1)
        baseline["baseline_risco_esgotamento"] = risco.mask(indefinido)
        return baseline


# ------------------------------------------------------------------ targets futuros
class ConstrutorDeAlvos:
    """Targets do modelo, tirados da trajetória FUTURA de demanda (nunca entram como features).

    target_demanda_restante_gb: soma da demanda diária futura até a véspera da renovação.
    Quem já estava com a franquia esgotada na data de referência fica separado: os targets de
    esgotamento ficam vazios ("não se aplica"), porque o esgotamento já aconteceu."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def construir(self, base, contratos, atributos, diario):
        futuro = diario[diario.periodo == "futuro"]
        trajetorias = dict(tuple(futuro.groupby("id_cliente")))
        linhas = [self._alvos_do_cliente(trajetorias[id_cliente], saldo, ja_esgotada, proxima)
                  for id_cliente, saldo, ja_esgotada, proxima in zip(
                      base["id_cliente"], atributos.saldo_geral_gb, atributos.franquia_ja_esgotada,
                      contratos.data_proxima_renovacao)]
        return pd.DataFrame(linhas, index=base.index)

    @staticmethod
    def _alvos_do_cliente(trajetoria, saldo, ja_esgotada, proxima_renovacao):
        demanda = trajetoria["demanda_latente_gb"].to_numpy()
        alvos = {"target_demanda_restante_gb": float(demanda.sum()), "target_esgotamento_antecipado": np.nan,
                 "target_data_esgotamento": pd.NaT, "target_dias_antecipacao": np.nan}
        if ja_esgotada:
            return alvos
        acumulado = np.cumsum(demanda)
        esgota = acumulado >= saldo
        alvos["target_esgotamento_antecipado"] = int(esgota.any())
        alvos["target_dias_antecipacao"] = 0
        if esgota.any():
            data = trajetoria["data"].iloc[int(np.argmax(esgota))]
            alvos["target_data_esgotamento"] = data
            alvos["target_dias_antecipacao"] = int((proxima_renovacao - data).days)
        return alvos


# ------------------------------------------------------------------ validação da saída
class ValidadorDaBaseFinal:
    """Confere a base enriquecida: mesmas linhas e colunas de entrada, datas coerentes e franquias certas."""

    def validar(self, base_de_entrada, base_final):
        problemas = self.problemas(base_de_entrada, base_final)
        if problemas:
            raise ErroNaBaseFinal("A base enriquecida quebrou regras:\n- " + "\n- ".join(problemas))

    @staticmethod
    def problemas(base_de_entrada, base_final):
        problemas = []
        if len(base_final) != len(base_de_entrada):
            problemas.append("o número de linhas mudou")
        if list(base_final.columns[:len(base_de_entrada.columns)]) != list(base_de_entrada.columns):
            problemas.append("as colunas de entrada não estão preservadas no começo da base")
        elif not base_final[base_de_entrada.columns].reset_index(drop=True).equals(base_de_entrada.reset_index(drop=True)):
            problemas.append("algum valor das colunas de entrada foi alterado")
        if base_final.columns.duplicated().any():
            problemas.append("há colunas com o mesmo nome")
        if base_final.duplicated().any():
            problemas.append("há linhas duplicadas")
        datas = {coluna: pd.to_datetime(base_final[coluna]) for coluna in COLUNAS_DE_DATA}
        if not (datas["data_contratacao"] <= datas["data_inicio_plano_atual"]).all():
            problemas.append("data_contratacao depois de data_inicio_plano_atual")
        if not (datas["data_inicio_plano_atual"] <= datas["data_referencia"]).all():
            problemas.append("data_inicio_plano_atual depois da data de referência")
        if not ((datas["data_ultima_renovacao"] <= datas["data_referencia"])
                & (datas["data_referencia"] < datas["data_proxima_renovacao"])).all():
            problemas.append("renovações fora da ordem última <= referência < próxima")
        franquia_esperada = formulas.franquia_geral(base_final.franquia_principal_gb, base_final.bonus_livre_gb,
                                                    base_final.bonus_livre_ativo)
        if not np.allclose(base_final.franquia_geral_gb, franquia_esperada):
            problemas.append("franquia_geral_gb diferente de principal + bônus livre válido")
        if (base_final.consumo_ciclo_atual_gb > base_final.franquia_geral_gb + 1e-9).any():
            problemas.append("consumo observado acima da franquia de uso geral")
        return problemas


# ------------------------------------------------------------------ orquestração
class EnriquecimentoDaBase:
    """Executa a etapa 5 inteira: valida, simula, calcula atributos, baseline e targets, e valida de novo.

    Uso:  resultado = EnriquecimentoDaBase(ConfiguracaoDoEnriquecimento.a_partir_do_config(config)).executar(uniao)
          resultado.base_final   # base_final_modelagem_claro.csv
    """

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def executar(self, base_unida):
        ValidadorDaEntrada(self.configuracao).validar(base_unida)
        base = base_unida.reset_index(drop=True)
        gerador = np.random.default_rng(self.configuracao.semente)
        contratos = SimuladorDeContratos(self.configuracao).simular(base, gerador)
        consumo = SimuladorDeConsumo(self.configuracao).simular(base, contratos, gerador)
        atributos = ConstrutorDeAtributos(self.configuracao).construir(base, contratos, consumo.diario)
        alvos = ConstrutorDeAlvos(self.configuracao).construir(base, contratos, atributos, consumo.diario)

        colunas_novas = pd.concat([contratos.drop(columns=["_data_renovacao_anterior"]), consumo.ritmo,
                                   atributos, alvos], axis=1)
        sobrepostas = set(colunas_novas.columns) & set(base.columns)
        if sobrepostas:
            raise ErroNaBaseFinal(f"colunas novas com nome de coluna de entrada: {sorted(sobrepostas)}")
        base_final = self._datas_em_texto(pd.concat([base, colunas_novas], axis=1))
        ValidadorDaBaseFinal().validar(base_unida, base_final)
        return ResultadoDoEnriquecimento(base_final, consumo.diario)

    @staticmethod
    def _datas_em_texto(base_final):
        """Datas no formato ISO (AAAA-MM-DD); vazio quando não se aplica."""
        base_final = base_final.copy()
        for coluna in COLUNAS_DE_DATA:
            datas = pd.to_datetime(base_final[coluna])
            base_final[coluna] = datas.dt.strftime("%Y-%m-%d").where(datas.notna(), None)
        return base_final


def colunas_novas(base_final):
    """As colunas acrescentadas pelo enriquecimento (tudo o que não veio da base unida)."""
    return [coluna for coluna in base_final.columns if coluna not in (*COLUNAS_ORIGINAIS, COLUNA_ORIGEM)]
