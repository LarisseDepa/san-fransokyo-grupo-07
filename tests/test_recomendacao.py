"""Etapa 7: demanda independente do plano, cobertura, ordenação por pontuação e casos rastreáveis."""
import math

import pytest

import config
from modelo_de_recomendacao import AtributosDaOferta
from recomendacao import AnalisadorDeRisco, AtributosDaSimulacao, CasosDemonstrativos, RecomendadorDePlanos

HABITOS = {"minutos": 400.0, "streaming_horas": 12.0, "deslocamento_dias": 6.0, "idade_faixa": "26-40",
           "regiao": "urbana"}
CLIENTE_ECONOMICO = {"id_cliente": None, "dados_gb": 12.0, "minutos": 150.0, "streaming_horas": 3.0, "deslocamento_dias": 2.0,
                     "grupo": "Economia"}
CLIENTE_QUE_USA_MUITO = {"id_cliente": None, "dados_gb": 30.0, "minutos": 500.0, "streaming_horas": 16.0, "deslocamento_dias": 5.0,
                         "grupo": "Excesso_de_Consumo"}


@pytest.fixture(scope="module")
def recomendador(planos, recomendacao_treinada):
    return RecomendadorDePlanos(planos, recomendacao_treinada.hibrida, AtributosDaOferta(planos),
                                config.MARGEM_DE_SEGURANCA)


def test_demanda_nao_muda_ao_trocar_de_plano(previsor, recomendador, planos):
    atributos = AtributosDaSimulacao(config.DATA_DE_REFERENCIA).montar(26, 15, 30, HABITOS)
    demanda = previsor.prever_um(atributos)
    resultados = {plano.nome: recomendador.recomendar(recomendador.necessidade(26, demanda, 30), plano.nome,
                                                      CLIENTE_QUE_USA_MUITO)
                  for plano in planos.planos}
    assert len({round(resultado.necessidade_gb, 9) for resultado in resultados.values()}) == 1
    assert "plano" not in " ".join(previsor.configuracao.variaveis)


def test_atributos_da_simulacao_sao_coerentes():
    atributos = AtributosDaSimulacao(config.DATA_DE_REFERENCIA).montar(26, 15, 30, HABITOS)
    assert atributos["dias_ate_renovacao"] == 15
    assert math.isclose(atributos["media_7_dias_anteriores_gb"], 1.0)
    assert math.isclose(atributos["media_ultimos_7_dias_gb"], (26 - 1.0 * 8) / 7)
    assert atributos["dias_fim_de_semana_restantes"] == 4  # de 28/10/2026 (quarta) a 11/11/2026


def test_ordem_pela_chance_e_manter_quando_atende(recomendador):
    recomendacao = recomendador.recomendar(12.4, "Total", CLIENTE_ECONOMICO)
    alternativas = recomendacao.opcoes[:-1]
    chances = [opcao.chance_de_aceitar for opcao in alternativas]
    assert recomendacao.plano_atual_atende and recomendacao.opcoes[-1].tipo == "manter"
    assert chances == sorted(chances, reverse=True) and len(recomendacao.opcoes) <= 3
    assert all(opcao.gb_do_plano >= 12.4 and opcao.tipo == "economia" for opcao in alternativas)


def test_nao_sugere_plano_mais_caro_quando_o_atual_basta(recomendador):
    recomendacao = recomendador.recomendar(20, "Digital", CLIENTE_QUE_USA_MUITO)
    assert all(opcao.diferenca_de_preco_brl <= 0 for opcao in recomendacao.opcoes)


def test_nenhum_plano_cobre(recomendador):
    recomendacao = recomendador.recomendar(100, "Digital", CLIENTE_QUE_USA_MUITO)
    assert recomendacao.nenhum_plano_cobre and recomendacao.opcoes == []


def test_folga_pequena(recomendador):
    opcoes = recomendador.recomendar(46, "Digital", CLIENTE_QUE_USA_MUITO).opcoes
    com_50_gb = [opcao for opcao in opcoes if opcao.gb_do_plano == 50]
    assert com_50_gb and all(opcao.folga_pequena for opcao in com_50_gb)


def test_risco_e_antecipacao():
    risco = AnalisadorDeRisco().analisar(35, 18.78, 25.36, 17)
    assert risco.nivel == "alto" and risco.dias_de_antecipacao == 6
    tranquilo = AnalisadorDeRisco().analisar(60, 6, 6, 15)
    assert tranquilo.nivel == "baixo" and tranquilo.dias_de_antecipacao == 0


def test_consumo_acima_da_capacidade():
    risco = AnalisadorDeRisco().analisar(35, 40, 10, 10)
    assert risco.consumo_excede_capacidade and risco.nivel == "esgotada"
    assert risco.consumo_efetivo_gb == 35 and risco.percentual_utilizado == 100


def test_cliente_do_exemplo_e_coerente(base_final, previsor, planos, recomendacao_treinada):
    caso = CasosDemonstrativos(base_final, previsor, planos, recomendacao_treinada.hibrida, config).caso_do_exemplo()
    registro = base_final[base_final.id_cliente == caso["id_cliente"]].iloc[0]
    assert caso["plano_atual"] == registro.plano_atual == "Digital"
    assert registro.franquia_geral_gb == planos.por_nome("Digital").gb_principais == 35
    assert registro.origem == "Mobilidade" and registro.franquia_ja_esgotada == 0
    assert caso["risco"]["dias_de_antecipacao"] >= 2
    opcoes = caso["recomendacao"]["opcoes"]
    assert all(opcao["gb_do_plano"] >= caso["recomendacao"]["necessidade_gb"] for opcao in opcoes)
    chances = [opcao["chance_de_aceitar"] for opcao in opcoes]
    assert chances == sorted(chances, reverse=True)
    peso = recomendacao_treinada.hibrida.peso_random_forest
    for opcao in opcoes:
        assert math.isclose(opcao["chance_de_aceitar"],
                            peso * opcao["chance_random_forest"] + (1 - peso) * opcao["chance_filtragem_colaborativa"])
        assert opcao["pontuacao_recomendacao"] == opcao["chance_de_aceitar"]  # alias legado

