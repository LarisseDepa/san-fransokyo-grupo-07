"""Etapa 8: histórico de ofertas simulado e modelos de recomendação (Random Forest e filtragem colaborativa)."""
from itertools import permutations
import re
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import config
from avaliacao_da_recomendacao import ComparadorPareado, DiagnosticoDaClassificacao, MetricasDeOrdenacao
from historico_de_ofertas import SimuladorDeHistoricoDeOfertas
from graficos_da_recomendacao import GraficosDaRecomendacao
from modelo_de_demanda import FlorestaExportada
from modelo_de_recomendacao import (ConfiguracaoDaRecomendacao, ExportadorDaPropensao, FiltragemColaborativa,
                                   PrevisorDePropensao, RecomendacaoHibrida, RegraDePreco, TreinadorDaRecomendacao)
from referencias_cientificas import BibliografiaDaRecomendacao

CLIENTES = 960


def test_historico_bem_formado(historico, planos):
    assert len(historico) == CLIENTES * config.CICLOS_DO_HISTORICO * config.OFERTAS_POR_CICLO
    assert (historico.plano_ofertado != historico.plano_atual).all()
    assert not historico.duplicated(["id_cliente", "ciclo", "plano_ofertado"]).any()
    assert set(historico.plano_ofertado) <= {plano.nome for plano in planos.planos}
    assert set(historico.aceitou) == {0, 1}
    assert historico.nota_do_cliente.between(0, config.NOTA_MAXIMA).all()


def test_nota_acompanha_a_resposta(historico):
    medias = historico.groupby("aceitou").nota_do_cliente.mean()
    assert medias[1] > medias[0]


def test_historico_reprodutivel(base_final, planos, configuracao_do_historico, historico):
    de_novo = SimuladorDeHistoricoDeOfertas(configuracao_do_historico, planos).executar(base_final)
    assert de_novo.equals(historico)


def test_regra_do_gosto_aparece_nos_dados(recomendacao_treinada):
    aceitacao = recomendacao_treinada.aceitacao_por_grupo.set_index("grupo")
    assert aceitacao.loc["Economia", "Essencial"] > aceitacao.loc["Economia", "Total"]
    assert aceitacao.loc["Excesso_de_Consumo", "Total"] > aceitacao.loc["Excesso_de_Consumo", "Essencial"]
    assert aceitacao.loc["Mobilidade", "Viagem"] == aceitacao.loc["Mobilidade"].max()


def test_chance_simulada_nunca_e_variavel(propensao):
    assert "chance_simulada" not in propensao.configuracao.variaveis
    assert "aceitou" not in propensao.configuracao.variaveis
    assert "nota_do_cliente" not in propensao.configuracao.variaveis


def test_teste_e_o_ultimo_ciclo(recomendacao_treinada):
    resultado = recomendacao_treinada
    assert resultado.treino.ciclo.max() < resultado.validacao.ciclo.min() < resultado.teste.ciclo.min()
    assert resultado.treino_final.ciclo.max() == resultado.validacao.ciclo.max()
    assert set(resultado.treino_final.index) == set(resultado.treino.index) | set(resultado.validacao.index)


def test_respostas_do_teste_nao_mudam_peso_nem_modelos(base_final, historico, planos, configuracao_do_historico,
                                                       recomendacao_treinada):
    alterado = historico.copy()
    teste = alterado.ciclo == alterado.ciclo.max()
    alterado.loc[teste, "aceitou"] = 1 - alterado.loc[teste, "aceitou"]
    alterado.loc[teste, "nota_do_cliente"] = config.NOTA_MAXIMA - alterado.loc[teste, "nota_do_cliente"]
    resultado = TreinadorDaRecomendacao(ConfiguracaoDaRecomendacao.a_partir_do_config(config)).treinar(
        base_final, alterado, planos, configuracao_do_historico, pd.Timestamp("2026-10-28"))
    assert resultado.hibrida.peso_random_forest == recomendacao_treinada.hibrida.peso_random_forest
    pd.testing.assert_frame_equal(resultado.comparacao_dos_pesos, recomendacao_treinada.comparacao_dos_pesos)
    assert np.allclose(resultado.hibrida.notas(resultado.teste),
                       recomendacao_treinada.hibrida.notas(recomendacao_treinada.teste))


def test_hibrido_preserva_media_padrao_e_aceita_componentes_isolados(recomendacao_treinada):
    resultado = recomendacao_treinada
    amostra = resultado.teste.head(20)
    floresta = resultado.propensao.notas(amostra)
    colaborativa = resultado.colaborativa.notas(amostra)
    for peso in [0, 0.5, 1]:
        modelo = RecomendacaoHibrida(resultado.propensao, resultado.colaborativa, peso)
        assert np.allclose(modelo.notas(amostra), peso * floresta + (1 - peso) * colaborativa)
    media_original = RecomendacaoHibrida(resultado.propensao, resultado.colaborativa)
    assert np.allclose(media_original.notas(amostra), (floresta + colaborativa) / 2)
    with pytest.raises(ValueError):
        RecomendacaoHibrida(resultado.propensao, resultado.colaborativa, 1.1)


def test_bootstrap_pareia_por_cliente_e_e_reprodutivel():
    hibrido = pd.Series([1, 0, 1], index=[10, 20, 30])
    floresta = pd.Series([0, 1, 1], index=[20, 30, 10])  # mesmos acertos, outra ordem
    comparador = ComparadorPareado(200, 2026)
    resultado = comparador.executar(hibrido, floresta)
    assert resultado == comparador.executar(hibrido, floresta)
    assert resultado["diferenca_de_acerto"] == resultado["ic95_inferior"] == resultado["ic95_superior"] == 0
    assert not resultado["evidencia_de_ganho"]


def test_bootstrap_identifica_diferenca_constante_e_rejeita_pares_incompletos():
    comparador = ComparadorPareado(100, 2026)
    resultado = comparador.executar(pd.Series([1, 1]), pd.Series([0, 0]))
    assert resultado["ic95_inferior"] == resultado["ic95_superior"] == 1
    assert resultado["evidencia_de_ganho"]
    with pytest.raises(ValueError, match="mesmos clientes"):
        comparador.executar(pd.Series([1], index=[10]), pd.Series([0], index=[20]))
    with pytest.raises(ValueError, match="ausentes"):
        comparador.executar(pd.Series([1, np.nan]), pd.Series([0, 1]))


def test_metricas_de_ordenacao_incluem_quem_recusou_todas():
    pares = pd.DataFrame({"id_cliente": np.repeat([1, 2, 3], 3),
                          "plano_ofertado": ["A", "B", "C"] * 3,
                          "aceitou": [0, 1, 0, 0, 0, 0, 1, 1, 1]})
    resultado = MetricasDeOrdenacao().executar([3, 2, 1] * 3, pares)
    assert resultado["clientes_avaliados"] == 3
    assert np.isclose(resultado["acerto_primeira_oferta_todos"], 1 / 3)
    assert np.isclose(resultado["mrr_observado"], 0.5)
    assert np.isclose(resultado["ndcg_3_observado"], (1 / np.log2(3) + 1) / 3)


@pytest.mark.parametrize("respostas", [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0]])
def test_sorteio_de_ordenacao_igual_a_enumeracao_exaustiva(respostas):
    pares = pd.DataFrame({"id_cliente": [1] * len(respostas),
                          "plano_ofertado": list("ABC")[:len(respostas)], "aceitou": respostas})
    avaliador = MetricasDeOrdenacao()
    esperado = pd.DataFrame([avaliador.executar(list(ordem), pares)
                             for ordem in permutations(range(len(respostas)))]).mean()
    resultado = avaliador.sorteio(pares)
    for metrica in ["acerto_primeira_oferta_todos", "ndcg_3_observado", "mrr_observado"]:
        assert np.isclose(resultado[metrica], esperado[metrica])


def test_bibliografia_tem_recorte_doi_e_fontes_da_versao_final():
    bibliografia = BibliografiaDaRecomendacao()
    registro = bibliografia.como_registro()
    assert len(registro["artigos"]) == 2
    for artigo in bibliografia.ARTIGOS:
        assert bibliografia.INICIO <= artigo.publicacao_final_online <= bibliografia.CONFERIDO_EM
        assert artigo.doi.startswith("10.") and "arxiv" not in artigo.fonte_da_versao_final
        assert bibliografia.REVISTAS[artigo.revista]["jif"] > 5
        assert artigo.titulo in bibliografia.como_html() and artigo.titulo in bibliografia.como_markdown()


def test_rejeita_historico_sem_ciclo_de_validacao(base_final, historico, planos, configuracao_do_historico):
    with pytest.raises(ValueError, match="três ciclos"):
        TreinadorDaRecomendacao(ConfiguracaoDaRecomendacao()).treinar(
            base_final, historico[historico.ciclo <= 2], planos, configuracao_do_historico, pd.Timestamp("2026-10-28"))


def test_diagnostico_preserva_todas_as_ofertas(recomendacao_treinada):
    resultado = recomendacao_treinada
    curva = resultado.curva_de_calibracao
    assert curva.ofertas.sum() == len(resultado.teste)
    assert np.isclose(np.average(curva.frequencia_de_aceitacao, weights=curva.ofertas), resultado.teste.aceitou.mean())
    assert resultado.diagnostico_de_propensao["brier"] >= 0
    assert resultado.diagnostico_de_propensao["log_loss"] >= 0


def test_matriz_de_confusao_tem_linhas_observadas_e_limiar_inclusivo():
    respostas = [0, 0, 0, 1, 1, 1]
    notas = {"RF": [0.1, 0.7, 0.5, 0.2, 0.8, 0.5]}
    diagnostico = DiagnosticoDaClassificacao(0.5).executar(respostas, notas, "RF")
    assert diagnostico["matriz_de_confusao"] == [[1, 2], [1, 2]]
    assert diagnostico["ofertas_avaliadas"] == 6 and diagnostico["acuracia"] == 0.5
    assert diagnostico["rotulos"] == ["Recusa", "Aceitação"]
    positiva = diagnostico["relatorio_por_classe"][1]
    assert positiva["suporte"] == 3 and np.isclose(positiva["precisao"], 0.5)
    assert np.isclose(positiva["sensibilidade"], 2 / 3) and np.isclose(positiva["f1"], 4 / 7)
    assert np.isclose(diagnostico["comparacao_de_curvas"][0]["roc_auc"], 11 / 18)
    assert np.isclose(diagnostico["comparacao_de_curvas"][0]["average_precision"], 0.7)
    com_outro_limiar = DiagnosticoDaClassificacao(0.75).executar(respostas, notas, "RF")
    assert com_outro_limiar["matriz_de_confusao"] != diagnostico["matriz_de_confusao"]
    assert com_outro_limiar["comparacao_de_curvas"] == diagnostico["comparacao_de_curvas"]


def test_diagnostico_binario_rejeita_classes_ausentes_e_pontuacoes_invalidas():
    with pytest.raises(ValueError, match="aceitas e recusadas"):
        DiagnosticoDaClassificacao(0.5).executar([0, 0], {"RF": [0.1, 0.2]}, "RF")
    with pytest.raises(ValueError, match="pontuação finita"):
        DiagnosticoDaClassificacao(0.5).executar([0, 1], {"RF": [0.1, np.nan]}, "RF")
    with pytest.raises(ValueError, match="entre 0 e 1"):
        DiagnosticoDaClassificacao(1.5)


def test_classificacao_usa_todas_as_ofertas_e_preserva_auc_da_ordenacao(recomendacao_treinada):
    resultado = recomendacao_treinada
    diagnostico = resultado.classificacao_no_teste
    matriz = np.array(diagnostico["matriz_de_confusao"])
    assert diagnostico["limiar"] == config.LIMIAR_DA_CLASSIFICACAO_DA_OFERTA
    assert matriz.sum() == diagnostico["ofertas_avaliadas"] == len(resultado.teste)
    assert matriz[1].sum() == diagnostico["aceitacoes_observadas"] == resultado.teste.aceitou.sum()
    assert matriz[0].sum() == (resultado.teste.aceitou == 0).sum()
    metricas = resultado.metricas.set_index("modelo")
    for curva in diagnostico["comparacao_de_curvas"]:
        assert np.isclose(curva["roc_auc"], metricas.loc[curva["modelo"], "auc"])
        assert 0 <= curva["average_precision"] <= 1


def test_graficos_sao_svg_acessiveis_com_ids_independentes(recomendacao_treinada):
    graficos = GraficosDaRecomendacao().executar(recomendacao_treinada)
    assert set(graficos) == {"roc", "precisao_revocacao", "matriz_de_confusao", "importancia", "arvore"}
    identificadores_usados = set()
    for nome, svg in graficos.items():
        raiz = ElementTree.fromstring(svg)
        assert raiz.tag.endswith("}svg") and "viewBox" in raiz.attrib
        assert raiz.attrib["role"] == "img" and raiz.attrib["aria-label"]
        assert "<?xml" not in svg and "<!DOCTYPE" not in svg and "<script" not in svg
        identificadores = {elemento.attrib["id"] for elemento in raiz.iter() if "id" in elemento.attrib}
        assert not identificadores_usados.intersection(identificadores), f"IDs repetidos em {nome}"
        referencias = set(re.findall(r'url\(#([^)]*)\)', svg))
        assert referencias <= identificadores, f"Referências SVG ausentes em {nome}"
        identificadores_usados.update(identificadores)


def test_random_forest_supera_as_referencias(recomendacao_treinada):
    metricas = recomendacao_treinada.metricas.set_index("modelo")
    floresta = metricas.loc[PrevisorDePropensao.NOME]
    for referencia in ["Popularidade (referência)", RegraDePreco.NOME, "Sorteio (referência)"]:
        assert floresta.auc > metricas.loc[referencia, "auc"]
        assert floresta.acerto_da_primeira_oferta > metricas.loc[referencia, "acerto_da_primeira_oferta"]
    assert metricas.loc[FiltragemColaborativa.NOME, "auc"] > metricas.loc["Popularidade (referência)", "auc"]


def test_colaborativa_cliente_novo_recebe_a_media_do_grupo(recomendacao_treinada):
    colaborativa = recomendacao_treinada.colaborativa
    assert colaborativa.nota_prevista(-1, "Viagem", "Mobilidade") == colaborativa.media_por_grupo.loc["Mobilidade", "Viagem"]
    assert colaborativa.nota_prevista(-1, "Essencial", "Economia") > colaborativa.nota_prevista(-1, "Total", "Economia")
    assert np.allclose(np.diag(colaborativa.semelhanca), 0.0)


def test_semelhanca_e_a_correlacao_de_pearson(recomendacao_treinada):
    """A mesma conta do scipy.stats.pearsonr (usado no vídeo de referência), só nos planos que os dois avaliaram."""
    colaborativa = recomendacao_treinada.colaborativa
    notas = colaborativa.notas_por_cliente
    for primeiro, segundo in [(notas.index[0], notas.index[1]), (notas.index[10], notas.index[500])]:
        em_comum = notas.loc[primeiro].notna() & notas.loc[segundo].notna()
        esperado = pearsonr(notas.loc[primeiro][em_comum], notas.loc[segundo][em_comum])[0]
        assert np.isclose(colaborativa.semelhanca.loc[primeiro, segundo], esperado)


def test_exportacao_da_propensao_igual_ao_scikit_learn(recomendacao_treinada):
    modelo = ExportadorDaPropensao().exportar(recomendacao_treinada.propensao.pipeline, recomendacao_treinada.registro)
    amostra = recomendacao_treinada.teste.head(150)
    pelo_json = [FlorestaExportada(modelo).prever_um(linha) for linha in amostra.to_dict("records")]
    assert np.allclose(pelo_json, recomendacao_treinada.propensao.notas(amostra), atol=1e-8)
