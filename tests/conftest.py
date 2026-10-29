"""Peças comuns dos testes: caminhos, base unida, base enriquecida, modelo de demanda, planos das planilhas,
histórico de ofertas simulado e modelo de recomendação.
As peças mais lentas (enriquecimento, treinos e histórico) rodam uma vez só por sessão de testes."""
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_DO_PROJETO / "src"))

import config  # noqa: E402
from enriquecimento import ConfiguracaoDoEnriquecimento, EnriquecimentoDaBase  # noqa: E402
from historico_de_ofertas import (ConfiguracaoDoHistorico, PlanosDasPlanilhas,  # noqa: E402
                                  SimuladorDeHistoricoDeOfertas)
from modelo_de_demanda import ConfiguracaoDoModelo, PrevisorDeConsumo, TreinadorDoModelo, agora_em  # noqa: E402
from modelo_de_recomendacao import ConfiguracaoDaRecomendacao, TreinadorDaRecomendacao  # noqa: E402
from pastas_do_projeto import PastasDoProjeto  # noqa: E402


@pytest.fixture(scope="session")
def pastas():
    return PastasDoProjeto(RAIZ_DO_PROJETO)


@pytest.fixture(scope="session")
def base_unida(pastas):
    if not pastas.uniao_csv.exists():
        pytest.skip("Falta a união das planilhas: rode antes pixi run python notebooks/claro_guia_etapas_0_a_8.py")
    return pd.read_csv(pastas.uniao_csv)


@pytest.fixture(scope="session")
def configuracao_do_enriquecimento():
    return ConfiguracaoDoEnriquecimento.a_partir_do_config(config)


@pytest.fixture(scope="session")
def enriquecimento(base_unida, configuracao_do_enriquecimento):
    return EnriquecimentoDaBase(configuracao_do_enriquecimento).executar(base_unida)


@pytest.fixture(scope="session")
def base_final(enriquecimento):
    return enriquecimento.base_final


@pytest.fixture(scope="session")
def treino(base_final):
    return TreinadorDoModelo(ConfiguracaoDoModelo.a_partir_do_config(config)).treinar(
        base_final, "versao-de-teste", agora_em(config.FUSO_HORARIO))


@pytest.fixture(scope="session")
def previsor(treino):
    return PrevisorDeConsumo(treino.floresta, treino.configuracao)


@pytest.fixture(scope="session")
def planos(base_final):
    return PlanosDasPlanilhas.a_partir_da_base(base_final, config.FRANQUIAS_DOS_PLANOS_ACADEMICOS)


@pytest.fixture(scope="session")
def configuracao_do_historico():
    return ConfiguracaoDoHistorico.a_partir_do_config(config)


@pytest.fixture(scope="session")
def historico(base_final, planos, configuracao_do_historico):
    return SimuladorDeHistoricoDeOfertas(configuracao_do_historico, planos).executar(base_final)


@pytest.fixture(scope="session")
def recomendacao_treinada(base_final, historico, planos, configuracao_do_historico):
    return TreinadorDaRecomendacao(ConfiguracaoDaRecomendacao.a_partir_do_config(config)).treinar(
        base_final, historico, planos, configuracao_do_historico, agora_em(config.FUSO_HORARIO))


@pytest.fixture(scope="session")
def propensao(recomendacao_treinada):
    return recomendacao_treinada.propensao


@pytest.fixture(scope="session")
def projeto():
    return config
