"""Fórmulas puras (src/formulas.py): resultados conhecidos e divisões por zero."""
import math

import formulas


def test_franquia_geral_soma_so_o_bonus_valido():
    assert formulas.franquia_geral(15, 20, 1) == 35
    assert formulas.franquia_geral(15, 20, 0) == 15


def test_saldo_nunca_fica_negativo():
    assert formulas.saldo_geral(35, 20) == 15
    assert formulas.saldo_geral(35, 50) == 0


def test_media_diaria_indefinida_sem_dias():
    assert formulas.media_diaria(10, 5) == 2
    assert math.isnan(formulas.media_diaria(10, 0))


def test_dias_ate_esgotamento():
    assert formulas.dias_ate_esgotamento(10, 2) == 5
    assert formulas.dias_ate_esgotamento(0, 2) == 0  # já acabou
    assert math.isnan(formulas.dias_ate_esgotamento(10, 0))  # nesse ritmo, não acaba
    assert math.isnan(formulas.dias_ate_esgotamento(10, float("nan")))


def test_percentual_fica_entre_0_e_100():
    assert formulas.percentual_utilizado(17.5, 35) == 50
    assert formulas.percentual_utilizado(50, 35) == 100
    assert math.isnan(formulas.percentual_utilizado(5, 0))


def test_antecipacao():
    assert formulas.dias_de_antecipacao(17, 10.9) == 17 - 10.9
    assert formulas.dias_de_antecipacao(17, 30) == 0
    assert formulas.dias_de_antecipacao(17, float("nan")) == 0


def test_projecao_e_baseline():
    assert formulas.projecao_pelo_ritmo(1.5, 10) == 15
    assert formulas.baseline_demanda_restante(1.5, 10) == 15
    assert math.isnan(formulas.baseline_demanda_restante(float("nan"), 10))
