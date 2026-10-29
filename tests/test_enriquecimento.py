"""Etapa 5: a base enriquecida preserva a entrada e respeita as regras de datas, franquias e targets."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

import config
import formulas
from dicionario_de_dados import DicionarioDeDados
from enriquecimento import (COLUNAS_ORIGINAIS, EnriquecimentoDaBase, ErroNaBaseDeEntrada, ValidadorDaBaseFinal,
                            colunas_novas)


def test_mantem_960_clientes_e_todas_as_colunas_de_entrada(base_unida, base_final):
    assert len(base_final) == 960 == len(base_unida)
    assert list(base_final.columns[:base_unida.shape[1]]) == list(base_unida.columns)
    assert "origem" in base_final
    pd.testing.assert_frame_equal(base_final[base_unida.columns], base_unida)


def test_nao_ha_duplicadas_nem_colunas_sobrescritas(base_final):
    assert not base_final.duplicated().any()
    assert base_final["id_cliente"].is_unique
    assert not base_final.columns.duplicated().any()
    assert not set(colunas_novas(base_final)) & set(COLUNAS_ORIGINAIS)
    assert len(colunas_novas(base_final)) > 30


def test_validador_detecta_valor_de_entrada_alterado(base_unida, base_final):
    alterada = base_final.copy()
    alterada.loc[0, "dados_gb"] += 1
    assert "algum valor das colunas de entrada foi alterado" in ValidadorDaBaseFinal.problemas(base_unida, alterada)
    assert ValidadorDaBaseFinal.problemas(base_unida, base_final) == []


def test_datas_respeitam_a_ordem(base_final):
    datas = {coluna: pd.to_datetime(base_final[coluna]) for coluna in
             ["data_contratacao", "data_inicio_plano_atual", "data_referencia", "data_ultima_renovacao", "data_proxima_renovacao"]}
    assert (datas["data_contratacao"] <= datas["data_inicio_plano_atual"]).all()
    assert (datas["data_inicio_plano_atual"] <= datas["data_referencia"]).all()
    assert (datas["data_ultima_renovacao"] <= datas["data_referencia"]).all()
    assert (datas["data_referencia"] < datas["data_proxima_renovacao"]).all()
    assert (base_final.dias_decorridos + base_final.dias_ate_renovacao == base_final.dias_total_ciclo).all()


def test_franquia_geral_nao_inclui_bonus_de_apps(base_final):
    esperado = base_final.franquia_principal_gb + base_final.bonus_livre_gb * base_final.bonus_livre_ativo
    assert np.allclose(base_final.franquia_geral_gb, esperado)
    com_apps = base_final[base_final.bonus_apps_gb > 0]
    assert (com_apps.franquia_geral_gb < com_apps.franquia_principal_gb + com_apps.bonus_livre_gb
            * com_apps.bonus_livre_ativo + com_apps.bonus_apps_gb).all()


def test_bonus_so_vale_dentro_da_validade(base_final):
    validade = pd.to_datetime(base_final.data_validade_bonus_livre)
    ultima = pd.to_datetime(base_final.data_ultima_renovacao)
    ativos = base_final.bonus_livre_ativo == 1
    assert (validade[ativos] > ultima[ativos]).all()
    assert (base_final.loc[base_final.bonus_livre_gb == 0, "bonus_livre_ativo"] == 0).all()


def test_saldo_percentual_e_consumo(base_final):
    assert np.allclose(base_final.saldo_geral_gb, formulas.saldo_geral(base_final.franquia_geral_gb,
                                                                       base_final.consumo_ciclo_atual_gb))
    assert base_final.percentual_franquia_utilizada.between(0, 100).all()
    assert (base_final.consumo_ciclo_atual_gb <= base_final.franquia_geral_gb + 1e-9).all()
    assert (base_final.consumo_ultimos_14_dias_gb >= base_final.consumo_ultimos_7_dias_gb - 1e-9).all()


def test_baseline_segue_a_formula_e_fica_vazio_sem_dias(base_final):
    definido = base_final.dias_decorridos > 0
    esperado = base_final.consumo_ciclo_atual_gb / base_final.dias_decorridos * base_final.dias_ate_renovacao
    assert np.allclose(base_final.loc[definido, "baseline_demanda_restante_gb"], esperado[definido])
    assert base_final.loc[~definido, "baseline_demanda_restante_gb"].isna().all()


def test_targets_coerentes_com_a_trajetoria(base_final):
    ja_esgotada = base_final.franquia_ja_esgotada == 1
    assert base_final.loc[ja_esgotada, "target_esgotamento_antecipado"].isna().all()
    nao_esgotada = base_final[~ja_esgotada]
    com_data = nao_esgotada.target_data_esgotamento.notna()
    assert ((nao_esgotada.target_esgotamento_antecipado == 1) == com_data).all()
    datas = nao_esgotada[com_data]
    antecipacao = (pd.to_datetime(datas.data_proxima_renovacao) - pd.to_datetime(datas.target_data_esgotamento)).dt.days
    assert (antecipacao == datas.target_dias_antecipacao).all()
    assert (pd.to_datetime(datas.target_data_esgotamento) >= pd.to_datetime(datas.data_referencia)).all()
    assert (base_final.target_demanda_restante_gb > 0).all()


def test_simulacao_reprodutivel(base_unida, configuracao_do_enriquecimento, base_final):
    de_novo = EnriquecimentoDaBase(configuracao_do_enriquecimento).executar(base_unida).base_final
    pd.testing.assert_frame_equal(de_novo, base_final)
    outra_semente = dataclasses.replace(configuracao_do_enriquecimento, semente=configuracao_do_enriquecimento.semente + 1)
    diferente = EnriquecimentoDaBase(outra_semente).executar(base_unida).base_final
    assert not diferente.target_demanda_restante_gb.equals(base_final.target_demanda_restante_gb)


def test_entrada_invalida_e_recusada_sem_corrigir(base_unida, configuracao_do_enriquecimento):
    enriquecimento = EnriquecimentoDaBase(configuracao_do_enriquecimento)
    with pytest.raises(ErroNaBaseDeEntrada, match="colunas obrigatórias ausentes"):
        enriquecimento.executar(base_unida.drop(columns=["dados_gb"]))
    with pytest.raises(ErroNaBaseDeEntrada, match="959 linhas"):
        enriquecimento.executar(base_unida.iloc[:-1])
    com_vazio = base_unida.copy()
    com_vazio.loc[0, "minutos"] = np.nan
    with pytest.raises(ErroNaBaseDeEntrada, match="minutos tem 1 valores vazios"):
        enriquecimento.executar(com_vazio)


def test_dicionario_cobre_todas_as_colunas(base_final, treino):
    dicionario = DicionarioDeDados(treino.configuracao, config.DESCRICAO_DAS_COLUNAS)
    assert dicionario.problemas(base_final) == []
    tabela = dicionario.como_tabela(base_final.columns)
    assert len(tabela) == base_final.shape[1]
    assert (tabela.descricao.str.len() > 0).all()
    assert set(tabela.papel_no_modelo[tabela.nome.isin(treino.configuracao.variaveis)].str[:8]) == {"variável"}
