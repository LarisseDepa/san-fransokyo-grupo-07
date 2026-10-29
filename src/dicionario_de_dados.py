"""
Dicionário de dados da base final de modelagem: uma fonte única de metadados para TODAS as colunas.
É usado no notebook, no site e no arquivo dicionario_de_dados.csv. Um teste confere que toda coluna
da base tem descrição e que toda descrição corresponde a uma coluna.

As fórmulas descritas aqui são as de src/formulas.py e src/enriquecimento.py.
O papel no modelo (variável, target, baseline ou metadado) vem da configuração do modelo, não de texto.
"""
from dataclasses import asdict, dataclass

import pandas as pd

from tabelas import formatar_numero_br

# Nomes curtos das variáveis do modelo, para gráficos e para a explicação "Como calculamos?"
NOMES_CURTOS = {
    "dados_gb": "Uso típico por mês", "minutos": "Minutos de ligação", "streaming_horas": "Horas de streaming",
    "deslocamento_dias": "Dias em deslocamento", "consumo_ciclo_atual_gb": "GB usados no ciclo",
    "dias_decorridos": "Dias desde a renovação", "dias_ate_renovacao": "Dias até renovar",
    "media_ultimos_7_dias_gb": "Ritmo da última semana", "media_7_dias_anteriores_gb": "Ritmo da semana anterior",
    "tendencia_consumo_gb_dia": "Mudança de ritmo", "consumo_ciclo_anterior_gb": "GB usados no ciclo anterior",
    "dias_fim_de_semana_restantes": "Fins de semana até renovar",
    "projecao_pelo_ritmo_do_ciclo_gb": "Projeção pelo ritmo do ciclo",
    "projecao_pelo_ritmo_recente_gb": "Projeção pelo ritmo recente",
    "projecao_pelo_habito_gb": "Projeção pelo uso típico", "idade_faixa": "Faixa de idade", "regiao": "Região",
}

ORIGINAL, CONTROLE, SINTETICA, CALCULADA, TARGET = "original", "controle (criada na união)", "sintética", "calculada", "target"


@dataclass(frozen=True)
class ColunaDoDicionario:
    nome: str
    descricao: str
    tipo: str
    unidade: str
    origem: str  # original, controle, sintética, calculada ou target
    formula_ou_regra: str
    disponivel_na_previsao: str  # a operadora conheceria este valor na data de referência?
    papel_no_modelo: str = ""


# (descrição, tipo, unidade, origem, fórmula ou regra, disponível na previsão)
METADADOS = {
    "origem": ("Nome da planilha de onde veio a linha.", "texto", "—", CONTROLE,
               "Criada na união das planilhas (etapa 3).", "sim"),
    "id_cliente": (None, "inteiro", "—", ORIGINAL, "Valor original da planilha.", "sim"),
    "idade_faixa": (None, "texto (categoria)", "—", ORIGINAL, "Valor original da planilha.", "sim"),
    "regiao": (None, "texto (categoria)", "—", ORIGINAL, "Valor original da planilha.", "sim"),
    "dados_gb": (None, "decimal", "GB por mês (hipótese)", ORIGINAL,
                 "Valor original. Hipótese da simulação: é o consumo típico de um mês "
                 "(a periodicidade não está documentada nas planilhas).", "sim"),
    "minutos": (None, "inteiro", "minutos", ORIGINAL, "Valor original da planilha.", "sim"),
    "streaming_horas": (None, "decimal", "horas", ORIGINAL, "Valor original da planilha.", "sim"),
    "deslocamento_dias": (None, "inteiro", "dias", ORIGINAL, "Valor original da planilha.", "sim"),
    "plano_atual": (None, "texto (categoria)", "—", ORIGINAL,
                    "Plano ACADÊMICO (fictício), não uma oferta comercial da Claro.", "sim"),
    "mensalidade_atual": (None, "decimal", "R$ por mês", ORIGINAL,
                          "Valor original; nunca é trocado pelos preços comerciais.", "sim"),
    "risco_cancelamento": (None, "binário (0/1)", "—", ORIGINAL, "Valor original da planilha.", "sim"),
    "perfil_grupo": (None, "texto (categoria)", "—", ORIGINAL, "Valor original da planilha.", "sim"),
    "data_referencia": ("Data em que a análise é feita: antes dela é histórico; depois, futuro.", "data",
                        "AAAA-MM-DD", SINTETICA, "DATA_DE_REFERENCIA do config.py.", "sim"),
    "data_contratacao": ("Quando o cliente contratou a linha.", "data", "AAAA-MM-DD", SINTETICA,
                         "data_referencia − sorteio entre 75 dias e 6 anos.", "sim"),
    "data_inicio_plano_atual": ("Quando começou o plano atual.", "data", "AAAA-MM-DD", SINTETICA,
                                "Em 60% dos clientes, igual à contratação; nos outros, sorteada entre a "
                                "contratação e 15 dias antes da referência.", "sim"),
    "data_ultima_renovacao": ("Início do ciclo atual da franquia.", "data", "AAAA-MM-DD", SINTETICA,
                              "Dia de renovação sorteado (1 a 28); é a renovação mais recente que não passa "
                              "da data de referência.", "sim"),
    "data_proxima_renovacao": ("Quando a franquia renova.", "data", "AAAA-MM-DD", CALCULADA,
                               "data_ultima_renovacao + 1 mês.", "sim"),
    "dias_desde_contratacao": ("Dias desde a contratação.", "inteiro", "dias", CALCULADA,
                               "data_referencia − data_contratacao.", "sim"),
    "dias_no_plano_atual": ("Dias no plano atual.", "inteiro", "dias", CALCULADA,
                            "data_referencia − data_inicio_plano_atual.", "sim"),
    "dias_total_ciclo": ("Tamanho do ciclo atual.", "inteiro", "dias", CALCULADA,
                         "data_proxima_renovacao − data_ultima_renovacao.", "sim"),
    "dias_decorridos": ("Dias completos já observados no ciclo atual.", "inteiro", "dias", CALCULADA,
                        "data_referencia − data_ultima_renovacao.", "sim"),
    "dias_ate_renovacao": ("Dias que faltam para renovar, contando o dia da referência.", "inteiro", "dias",
                           CALCULADA, "data_proxima_renovacao − data_referencia.", "sim"),
    "franquia_principal_gb": ("GB principais do plano acadêmico.", "decimal", "GB", SINTETICA,
                              "FRANQUIAS_DOS_PLANOS_ACADEMICOS[plano_atual] no config.py.", "sim"),
    "bonus_livre_gb": ("GB de bônus de uso livre do plano acadêmico.", "decimal", "GB", SINTETICA,
                       "FRANQUIAS_DOS_PLANOS_ACADEMICOS[plano_atual] no config.py.", "sim"),
    "bonus_apps_gb": ("GB de bônus só para redes sociais e apps. Não entram no uso geral.", "decimal", "GB",
                      SINTETICA, "FRANQUIAS_DOS_PLANOS_ACADEMICOS[plano_atual] no config.py.", "sim"),
    "data_validade_bonus_livre": ("Até quando vale o bônus de uso livre.", "data", "AAAA-MM-DD", CALCULADA,
                                  "data_inicio_plano_atual + 12 meses; vazio se o plano não tem bônus livre.", "sim"),
    "bonus_livre_ativo": ("1 se o bônus de uso livre vale no ciclo atual.", "binário (0/1)", "—", CALCULADA,
                          "1 se data_validade_bonus_livre > data_ultima_renovacao (o bônus vale o ciclo "
                          "inteiro se estiver válido no início dele).", "sim"),
    "franquia_geral_gb": ("GB de uso geral do ciclo: internet para qualquer app.", "decimal", "GB", CALCULADA,
                          "franquia_principal_gb + bonus_livre_gb × bonus_livre_ativo (os GB só para apps "
                          "não entram).", "sim"),
    "cenario_de_ritmo_simulado": ("Mudança de ritmo sorteada na simulação: aumento, queda ou estável.",
                                  "texto (categoria)", "—", SINTETICA,
                                  "Aumento com chance 15% + 45% × deslocamento_dias/31; queda com chance 12%. "
                                  "É latente: a operadora não conheceria.", "não (latente)"),
    "fator_de_ritmo_simulado": ("Multiplicador do ritmo depois da mudança.", "decimal", "—", SINTETICA,
                                "Aumento: 1,25 a 1,9; queda: 0,55 a 0,85; estável: 1. É latente.", "não (latente)"),
    "consumo_ciclo_atual_gb": ("GB de uso geral consumidos no ciclo atual até a véspera da referência.",
                               "decimal", "GB", SINTETICA,
                               "Soma do consumo diário simulado desde data_ultima_renovacao. O consumo para "
                               "quando a franquia de uso geral do ciclo acaba.", "sim"),
    "saldo_geral_gb": ("GB de uso geral que ainda restam no ciclo.", "decimal", "GB", CALCULADA,
                       "max(0, franquia_geral_gb − consumo_ciclo_atual_gb).", "sim"),
    "percentual_franquia_utilizada": ("Quanto da franquia de uso geral já foi usado.", "decimal", "% (0 a 100)",
                                      CALCULADA, "100 × consumo_ciclo_atual_gb / franquia_geral_gb.", "sim"),
    "franquia_ja_esgotada": ("1 se a franquia de uso geral já tinha acabado na data de referência.",
                             "binário (0/1)", "—", CALCULADA, "1 se saldo_geral_gb = 0.", "sim"),
    "consumo_ultimos_7_dias_gb": ("GB consumidos nos 7 dias antes da referência.", "decimal", "GB", SINTETICA,
                                  "Soma do consumo diário observado; pode incluir dias do ciclo anterior.", "sim"),
    "consumo_ultimos_14_dias_gb": ("GB consumidos nos 14 dias antes da referência.", "decimal", "GB", SINTETICA,
                                   "Soma do consumo diário observado; pode incluir dias do ciclo anterior.", "sim"),
    "media_diaria_gb": ("Ritmo médio do ciclo atual.", "decimal", "GB/dia", CALCULADA,
                        "consumo_ciclo_atual_gb / dias_decorridos; vazio se dias_decorridos = 0.", "sim"),
    "media_ultimos_7_dias_gb": ("Ritmo médio da última semana.", "decimal", "GB/dia", CALCULADA,
                                "consumo_ultimos_7_dias_gb / 7.", "sim"),
    "media_7_dias_anteriores_gb": ("Ritmo médio da semana anterior à última.", "decimal", "GB/dia", CALCULADA,
                                   "(consumo_ultimos_14_dias_gb − consumo_ultimos_7_dias_gb) / 7.", "sim"),
    "tendencia_consumo_gb_dia": ("Quanto o ritmo mudou de uma semana para a outra (positivo = acelerou).",
                                 "decimal", "GB/dia", CALCULADA,
                                 "media_ultimos_7_dias_gb − media_7_dias_anteriores_gb.", "sim"),
    "consumo_ciclo_anterior_gb": ("GB consumidos no ciclo anterior completo.", "decimal", "GB", SINTETICA,
                                  "Soma do consumo diário observado entre a renovação anterior e a última.", "sim"),
    "dias_observados": ("Dias de consumo diário simulado antes da referência.", "inteiro", "dias", CALCULADA,
                        "min(dias_desde_contratacao, 90).", "sim"),
    "dias_fim_de_semana_restantes": ("Sábados e domingos até a véspera da renovação.", "inteiro", "dias",
                                     CALCULADA, "Contagem no calendário (conhecido de antemão).", "sim"),
    "projecao_pelo_ritmo_do_ciclo_gb": ("GB até a renovação mantendo o ritmo do ciclo.", "decimal", "GB",
                                        CALCULADA, "media_diaria_gb × dias_ate_renovacao; sem dias no ciclo, usa "
                                        "media_ultimos_7_dias_gb no lugar da média do ciclo.", "sim"),
    "projecao_pelo_ritmo_recente_gb": ("GB até a renovação mantendo o ritmo da última semana.", "decimal", "GB",
                                       CALCULADA, "media_ultimos_7_dias_gb × dias_ate_renovacao.", "sim"),
    "projecao_pelo_habito_gb": ("GB até a renovação mantendo o consumo típico mensal.", "decimal", "GB",
                                CALCULADA, "dados_gb / 30 × dias_ate_renovacao.", "sim"),
    "baseline_demanda_restante_gb": ("Previsão de comparação: projeção pela média diária do ciclo.", "decimal",
                                     "GB", CALCULADA, "media_diaria_gb × dias_ate_renovacao; vazio se "
                                     "dias_decorridos = 0.", "sim"),
    "baseline_dias_ate_esgotamento": ("Dias até o saldo acabar no ritmo médio do ciclo.", "decimal", "dias",
                                      CALCULADA, "saldo_geral_gb / media_diaria_gb; 0 se não há saldo; vazio se "
                                      "a média é zero ou indefinida.", "sim"),
    "baseline_risco_esgotamento": ("1 se a projeção pela média diária passa do saldo.", "binário (0/1)", "—",
                                   CALCULADA, "1 se baseline_demanda_restante_gb > saldo_geral_gb; vazio se "
                                   "indefinido ou se a franquia já acabou.", "sim"),
    "target_demanda_restante_gb": ("GB que o cliente vai querer usar até a renovação (o que o modelo prevê).",
                                   "decimal", "GB", TARGET,
                                   "Soma da demanda diária FUTURA sintética (latente, sem o limite da franquia) "
                                   "da data de referência até a véspera da renovação.", "não (futuro)"),
    "target_esgotamento_antecipado": ("1 se a franquia acaba antes da renovação.", "binário (0/1)", "—", TARGET,
                                      "1 se a demanda futura acumulada alcança o saldo antes da renovação; vazio "
                                      "se a franquia já tinha acabado na referência.", "não (futuro)"),
    "target_data_esgotamento": ("Dia em que a franquia acaba.", "data", "AAAA-MM-DD", TARGET,
                                "Primeiro dia futuro em que a demanda acumulada alcança o saldo; vazio se não "
                                "acaba ou se já tinha acabado.", "não (futuro)"),
    "target_dias_antecipacao": ("Quantos dias antes da renovação a franquia acaba.", "inteiro", "dias", TARGET,
                                "data_proxima_renovacao − target_data_esgotamento; 0 se não acaba; vazio se "
                                "já tinha acabado.", "não (futuro)"),
}


class DicionarioDeDados:
    """Monta o dicionário a partir dos METADADOS e da configuração do modelo treinado."""

    def __init__(self, configuracao_do_modelo, descricoes_das_colunas_originais):
        self.configuracao_do_modelo = configuracao_do_modelo
        self.descricoes_originais = descricoes_das_colunas_originais

    def colunas(self, ordem=None):
        nomes = list(ordem) if ordem is not None else list(METADADOS)
        return [self._coluna(nome) for nome in nomes]

    def como_tabela(self, ordem=None):
        return pd.DataFrame([asdict(coluna) for coluna in self.colunas(ordem)])

    def problemas(self, base_final):
        """Colunas da base sem descrição e descrições sem coluna na base."""
        sem_descricao = [coluna for coluna in base_final.columns if coluna not in METADADOS]
        sem_coluna = [coluna for coluna in METADADOS if coluna not in base_final.columns]
        return ([f"coluna sem descrição: {coluna}" for coluna in sem_descricao]
                + [f"descrição de coluna inexistente: {coluna}" for coluna in sem_coluna])

    def _coluna(self, nome):
        descricao, tipo, unidade, origem, regra, disponivel = METADADOS[nome]
        descricao = descricao or self.descricoes_originais.get(nome, "")
        return ColunaDoDicionario(nome, descricao, tipo, unidade, origem, regra, disponivel, self._papel(nome))

    def _papel(self, nome):
        configuracao = self.configuracao_do_modelo
        if nome == configuracao.alvo:
            return "target do modelo"
        if nome in configuracao.variaveis_categoricas:
            return "variável do modelo (One-Hot dentro do pipeline)"
        if nome in configuracao.variaveis_numericas:
            return "variável do modelo"
        if nome.startswith("target_"):
            return "target derivado (avaliação do risco)"
        if nome.startswith("baseline_"):
            return "baseline (comparação)"
        return "metadado (não entra no modelo)"


def exemplos_de_calculo(registro):
    """As principais fórmulas com os valores reais de um cliente da base (uma linha)."""
    def gb(valor):
        return f"{formatar_numero_br(valor, 2)} GB"

    def ritmo(valor):
        return f"{formatar_numero_br(valor, 3)} GB/dia"

    r = registro
    exemplos = [
        ("franquia_geral_gb", "franquia_principal_gb + bonus_livre_gb × bonus_livre_ativo",
         f"{gb(r.franquia_principal_gb)} + {gb(r.bonus_livre_gb)} × {int(r.bonus_livre_ativo)}",
         gb(r.franquia_geral_gb)),
        ("saldo_geral_gb", "max(0, franquia_geral_gb − consumo_ciclo_atual_gb)",
         f"max(0, {gb(r.franquia_geral_gb)} − {gb(r.consumo_ciclo_atual_gb)})", gb(r.saldo_geral_gb)),
        ("dias_ate_renovacao", "data_proxima_renovacao − data_referencia",
         f"{r.data_proxima_renovacao} − {r.data_referencia}", f"{int(r.dias_ate_renovacao)} dias"),
        ("media_ultimos_7_dias_gb", "consumo_ultimos_7_dias_gb / 7", f"{gb(r.consumo_ultimos_7_dias_gb)} / 7",
         ritmo(r.media_ultimos_7_dias_gb)),
        ("tendencia_consumo_gb_dia", "media_ultimos_7_dias_gb − media_7_dias_anteriores_gb",
         f"{ritmo(r.media_ultimos_7_dias_gb)} − {ritmo(r.media_7_dias_anteriores_gb)}",
         ritmo(r.tendencia_consumo_gb_dia)),
    ]
    if pd.notna(r.media_diaria_gb):
        exemplos += [
            ("media_diaria_gb", "consumo_ciclo_atual_gb / dias_decorridos",
             f"{gb(r.consumo_ciclo_atual_gb)} / {int(r.dias_decorridos)}", ritmo(r.media_diaria_gb)),
            ("baseline_demanda_restante_gb", "media_diaria_gb × dias_ate_renovacao",
             f"{ritmo(r.media_diaria_gb)} × {int(r.dias_ate_renovacao)}", gb(r.baseline_demanda_restante_gb)),
        ]
    return pd.DataFrame(exemplos, columns=["coluna", "formula", "conta_com_os_valores", "resultado"])
