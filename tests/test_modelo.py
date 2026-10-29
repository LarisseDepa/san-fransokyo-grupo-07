"""Etapa 6: sem vazamento, One-Hot ajustado só no treino, métricas reais e exportação fiel ao scikit-learn."""
import json
import math

import numpy as np
import pytest

import config
from dicionario_de_dados import DicionarioDeDados
from modelo_de_demanda import COLUNAS_PROIBIDAS, ExportadorParaNavegador, FlorestaExportada, PrevisorDeConsumo


def test_sem_vazamento_de_dados(treino, base_final):
    variaveis = treino.configuracao.variaveis
    assert not [variavel for variavel in variaveis if variavel.startswith(("target_", "baseline_"))]
    assert not set(variaveis) & set(COLUNAS_PROIBIDAS)
    disponivel = DicionarioDeDados(treino.configuracao, config.DESCRICAO_DAS_COLUNAS).como_tabela(base_final.columns)
    disponivel = dict(zip(disponivel.nome, disponivel.disponivel_na_previsao))
    assert all(disponivel[variavel] == "sim" for variavel in variaveis)


def test_treino_e_teste_por_cliente_sem_sobreposicao(treino):
    assert not set(treino.treino.id_cliente) & set(treino.teste.id_cliente)
    assert len(treino.treino) + len(treino.teste) == 960


def test_one_hot_ajustado_so_no_treino(treino):
    codificador = treino.floresta.named_steps["preparo"].named_transformers_["categoricas"]
    for variavel, categorias in zip(treino.configuracao.variaveis_categoricas, codificador.categories_):
        assert list(categorias) == sorted(treino.treino[variavel].unique())
    assert codificador.handle_unknown == "ignore"


def test_metricas_medidas_e_finitas(treino):
    modelos = set(treino.metricas.modelo)
    assert {"Baseline (média diária do ciclo)", "Ridge (regressão linear)", "Random Forest"} <= modelos
    for coluna in ["mae_gb", "rmse_gb", "r2"]:
        assert treino.metricas[coluna].map(math.isfinite).all()
    assert (treino.metricas.mae_gb > 0).all()
    assert treino.selecao_de_variaveis.escolhido.sum() == 1


def test_registro_do_treino_completo(treino):
    registro = treino.registro
    for campo in ["versao_do_modelo", "treinado_em", "versao_dos_dados", "semente", "parametros", "alvo",
                  "variaveis_numericas", "bibliotecas", "metricas_no_teste", "limitacoes", "selecao_de_variaveis"]:
        assert campo in registro
    assert registro["alvo"] == "target_demanda_restante_gb"


def test_exportacao_para_o_navegador_igual_ao_scikit_learn(treino, base_final):
    modelo = json.loads(json.dumps(ExportadorParaNavegador().exportar(treino.floresta, treino.registro)))
    exportada = FlorestaExportada(modelo)
    variaveis = treino.configuracao.variaveis
    esperado = treino.floresta.predict(base_final[variaveis])
    obtido = np.array([exportada.prever_um(linha) for linha in base_final[variaveis].to_dict(orient="records")])
    assert np.abs(esperado - obtido).max() < 1e-6


def test_artefatos_salvos_sao_integros(pastas, base_final):
    if not (pastas.modelo_joblib.exists() and pastas.modelo_json.exists()):
        pytest.skip("Rode antes o notebook claro_guia_etapas_0_a_8.py (ou pixi run site)")
    previsor = PrevisorDeConsumo.carregar(pastas.modelo_joblib)
    modelo = json.loads(pastas.modelo_json.read_text(encoding="utf-8"))
    registro = json.loads(pastas.registro_do_treino.read_text(encoding="utf-8"))
    assert modelo["versao_do_modelo"] == registro["versao_do_modelo"]
    amostra = base_final.sample(50, random_state=1)
    obtido = [FlorestaExportada(modelo).prever_um(linha) for linha in amostra[previsor.configuracao.variaveis]
              .to_dict(orient="records")]
    assert np.allclose(previsor.prever(amostra), obtido, atol=1e-6)
