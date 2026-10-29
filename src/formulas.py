"""
Fórmulas do consumo de internet. São funções puras (a mesma entrada dá sempre a mesma saída)
e aceitam um número ou uma coluna inteira (numpy/pandas).

São a fonte única das contas: o enriquecimento, o recomendador e os testes usam estas funções.
O JavaScript do site (src/paginas/claro_guia.js) repete as mesmas contas, e um teste confere
que os dois dão o mesmo resultado.
"""
import numpy as np


def _como_resultado(valor):
    """Devolve float quando a entrada era um número só; senão, o array."""
    valor = np.asarray(valor, dtype=float)
    return float(valor) if valor.ndim == 0 else valor


def franquia_geral(principal_gb, bonus_livre_gb, bonus_livre_ativo):
    """GB de uso geral = GB principais + bônus de uso livre válido.
    Os GB de bônus só para apps NUNCA entram aqui."""
    principal_gb, bonus_livre_gb = np.asarray(principal_gb, float), np.asarray(bonus_livre_gb, float)
    return _como_resultado(principal_gb + bonus_livre_gb * np.asarray(bonus_livre_ativo, float))


def saldo_geral(franquia_geral_gb, consumo_ciclo_atual_gb):
    """GB de uso geral que ainda restam no ciclo: max(0, franquia - consumo)."""
    return _como_resultado(np.maximum(0.0, np.asarray(franquia_geral_gb, float)
                                      - np.asarray(consumo_ciclo_atual_gb, float)))


def media_diaria(consumo_gb, dias):
    """GB por dia. Indefinida (NaN) quando não há dias: não esconde a divisão por zero."""
    consumo_gb, dias = np.asarray(consumo_gb, float), np.asarray(dias, float)
    com_dias = dias > 0
    return _como_resultado(np.where(com_dias, consumo_gb / np.where(com_dias, dias, 1.0), np.nan))


def projecao_pelo_ritmo(ritmo_gb_por_dia, dias_ate_renovacao):
    """GB que seriam usados até a renovação mantendo um ritmo: ritmo x dias que faltam."""
    return _como_resultado(np.asarray(ritmo_gb_por_dia, float) * np.asarray(dias_ate_renovacao, float))


def baseline_demanda_restante(media_diaria_gb, dias_ate_renovacao):
    """Baseline (projeção pela média diária do ciclo). NaN se o ciclo ainda não tem dias."""
    return projecao_pelo_ritmo(media_diaria_gb, dias_ate_renovacao)


def dias_ate_esgotamento(saldo_gb, ritmo_gb_por_dia):
    """Quantos dias o saldo dura no ritmo informado.
    0 se não há saldo; NaN se o ritmo é indefinido ou zero (o saldo não acaba nesse ritmo)."""
    saldo_gb, ritmo = np.asarray(saldo_gb, float), np.asarray(ritmo_gb_por_dia, float)
    ritmo_valido = np.isfinite(ritmo) & (ritmo > 0)
    dias = np.where(ritmo_valido, saldo_gb / np.where(ritmo_valido, ritmo, 1.0), np.nan)
    return _como_resultado(np.where(saldo_gb <= 0, 0.0, dias))


def percentual_utilizado(consumo_gb, franquia_geral_gb):
    """Percentual da franquia de uso geral já usado, de 0 a 100 (NaN se a franquia é zero)."""
    consumo_gb, franquia = np.asarray(consumo_gb, float), np.asarray(franquia_geral_gb, float)
    com_franquia = franquia > 0
    percentual = np.where(com_franquia, 100.0 * consumo_gb / np.where(com_franquia, franquia, 1.0), np.nan)
    return _como_resultado(np.clip(percentual, 0.0, 100.0))


def dias_de_antecipacao(dias_ate_renovacao, dias_ate_esgotar):
    """Quantos dias antes da renovação a franquia acaba.
    0 quando ela não acaba antes (inclusive quando o ritmo é zero e dias_ate_esgotar é NaN)."""
    dias_ate_renovacao, dias_ate_esgotar = np.asarray(dias_ate_renovacao, float), np.asarray(dias_ate_esgotar, float)
    return _como_resultado(np.where(np.isnan(dias_ate_esgotar), 0.0,
                                    np.maximum(0.0, dias_ate_renovacao - dias_ate_esgotar)))
