"""
Etapa 8 (parte 1): histórico de ofertas SIMULADO.

As planilhas mostram só o plano atual de cada cliente: não há registro de ofertas feitas, aceitas ou
recusadas, e sem esse histórico não dá para treinar um modelo de recomendação. Aqui ele é simulado
por uma regra (o "gosto" de cada grupo de clientes, em config.py). Os modelos da etapa 8 aprendem essa
regra: as métricas mostram o que eles aprenderam da simulação, não a preferência de clientes reais.

Os planos são os 8 planos ACADÊMICOS das planilhas (fictícios). Os GB vêm de config.py; a
mensalidade de cada plano vem das próprias planilhas (todos os clientes de um plano pagam o mesmo).
Para comparar planos, contam só os GB principais: o bônus de uso livre é temporário e o bônus para
apps não serve para qualquer aplicativo.

Classes:
  PlanoDaPlanilha               -> um dos 8 planos: GB e mensalidade
  PlanosDasPlanilhas            -> os 8 planos, montados a partir da base e de config.py
  ConfiguracaoDoHistorico       -> ciclos, ofertas por ciclo, semente e os pesos da regra
  RegraDeAceitacao              -> a chance de um cliente aceitar uma oferta (a regra simulada)
  SimuladorDeHistoricoDeOfertas -> sorteia as ofertas de cada ciclo, as respostas (aceitou ou recusou) e a
                                   nota de 0 a 10 que a pessoa dá a cada plano oferecido
"""
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from tabelas import COLUNA_ORIGEM

COLUNA_DO_PLANO_ATUAL = "plano_atual"
COLUNA_DA_MENSALIDADE = "mensalidade_atual"
LIMITE_DA_COBERTURA = 1.0  # a cobertura vai de −1 (o plano tem bem menos GB que o uso) a 1 (bem mais)
FLUXO_DO_SORTEIO = 8  # separa o sorteio desta etapa do sorteio da etapa 5, com a mesma semente
FLUXO_DAS_NOTAS = 2  # sorteio próprio das notas: as respostas (aceitou ou recusou) não mudam por causa delas


class PlanoDaPlanilha(NamedTuple):
    nome: str
    gb_principais: float  # valem o mês todo, para qualquer aplicativo
    bonus_livre_gb: float  # temporário (vale por um tempo a partir do início do plano)
    bonus_para_apps_gb: float  # só redes sociais e apps do benefício
    mensalidade_brl: float


class PlanosDasPlanilhas:
    """Os 8 planos acadêmicos, do mais barato ao mais caro."""

    def __init__(self, planos):
        self.planos = sorted(planos, key=lambda plano: (plano.mensalidade_brl, plano.nome))
        self._por_nome = {plano.nome: plano for plano in self.planos}

    @classmethod
    def a_partir_da_base(cls, base, franquias):
        """GB de config.py e mensalidade das planilhas (a mediana; cada plano tem um preço só)."""
        mensalidades = base.groupby(COLUNA_DO_PLANO_ATUAL)[COLUNA_DA_MENSALIDADE].median()
        return cls([PlanoDaPlanilha(nome, float(principais), float(livre), float(apps), float(mensalidades[nome]))
                    for nome, (principais, livre, apps) in franquias.items()])

    def por_nome(self, nome):
        return self._por_nome[nome]

    def outros(self, nome):
        """Os planos que o cliente ainda não tem (os que podem ser oferecidos)."""
        return [plano for plano in self.planos if plano.nome != nome]

    def como_tabela(self):
        return pd.DataFrame([plano._asdict() for plano in self.planos])

    def como_lista_de_dicionarios(self):
        return [plano._asdict() for plano in self.planos]


@dataclass(frozen=True)
class ConfiguracaoDoHistorico:
    ciclos: int = 6
    ofertas_por_ciclo: int = 3
    semente: int = 2026
    intercepto: float = -1.0
    valor_dos_gb: dict = field(default_factory=dict)  # por grupo
    sensibilidade_ao_preco: dict = field(default_factory=dict)  # por grupo
    preferencia_pelo_plano: dict = field(default_factory=dict)  # {plano: (hábito, referência, peso)}
    nota_maxima: int = 10
    desvio_padrao_da_nota: float = 1.5

    @classmethod
    def a_partir_do_config(cls, config):
        return cls(ciclos=config.CICLOS_DO_HISTORICO, ofertas_por_ciclo=config.OFERTAS_POR_CICLO,
                   semente=config.SEMENTE, intercepto=config.INTERCEPTO_DA_ACEITACAO,
                   valor_dos_gb=dict(config.VALOR_DOS_GB_POR_GRUPO),
                   sensibilidade_ao_preco=dict(config.SENSIBILIDADE_AO_PRECO_POR_GRUPO),
                   preferencia_pelo_plano=dict(config.PREFERENCIA_PELO_PLANO),
                   nota_maxima=config.NOTA_MAXIMA, desvio_padrao_da_nota=config.DESVIO_PADRAO_DA_NOTA)


class RegraDeAceitacao:
    """O "gosto" simulado: chance = logística(utilidade). Ver a fórmula em config.py."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    @staticmethod
    def cobertura(plano_ofertado, uso_tipico_gb):
        relativa = (plano_ofertado.gb_principais - uso_tipico_gb) / max(uso_tipico_gb, 1.0)
        return float(np.clip(relativa, -LIMITE_DA_COBERTURA, LIMITE_DA_COBERTURA))

    @staticmethod
    def diferenca_de_preco_relativa(plano_ofertado, plano_atual):
        return (plano_ofertado.mensalidade_brl - plano_atual.mensalidade_brl) / plano_atual.mensalidade_brl

    def preferencia(self, cliente, plano_ofertado):
        if plano_ofertado.nome not in self.configuracao.preferencia_pelo_plano:
            return 0.0
        habito, referencia, peso = self.configuracao.preferencia_pelo_plano[plano_ofertado.nome]
        return peso * min(1.0, getattr(cliente, habito) / referencia)

    def utilidade(self, cliente, plano_ofertado, plano_atual):
        grupo = getattr(cliente, COLUNA_ORIGEM)
        return (self.configuracao.intercepto
                + self.configuracao.valor_dos_gb[grupo] * self.cobertura(plano_ofertado, cliente.dados_gb)
                - self.configuracao.sensibilidade_ao_preco[grupo]
                * self.diferenca_de_preco_relativa(plano_ofertado, plano_atual)
                + self.preferencia(cliente, plano_ofertado))

    def chance(self, cliente, plano_ofertado, plano_atual):
        return 1.0 / (1.0 + math.exp(-self.utilidade(cliente, plano_ofertado, plano_atual)))


class SimuladorDeHistoricoDeOfertas:
    """Em cada ciclo passado, cada cliente recebe algumas ofertas sorteadas entre os planos que ainda não
    tem, responde a cada uma (aceita ou recusa) com a chance da regra e dá uma nota de 0 a 10 ao plano
    (10 × chance + variação aleatória). A coluna chance_simulada é a "verdade" escondida da simulação: fica no
    arquivo para conferência e nunca entra nos modelos."""

    def __init__(self, configuracao, planos):
        self.configuracao = configuracao
        self.planos = planos
        self.regra = RegraDeAceitacao(configuracao)

    def executar(self, base):
        gerador = np.random.default_rng([self.configuracao.semente, FLUXO_DO_SORTEIO])
        gerador_das_notas = np.random.default_rng([self.configuracao.semente, FLUXO_DO_SORTEIO, FLUXO_DAS_NOTAS])
        linhas = []
        for cliente in base.sort_values("id_cliente").itertuples():
            plano_atual = self.planos.por_nome(getattr(cliente, COLUNA_DO_PLANO_ATUAL))
            candidatos = [plano.nome for plano in self.planos.outros(plano_atual.nome)]
            for ciclo in range(1, self.configuracao.ciclos + 1):
                for nome in gerador.choice(candidatos, size=self.configuracao.ofertas_por_ciclo, replace=False):
                    ofertado = self.planos.por_nome(str(nome))
                    chance = self.regra.chance(cliente, ofertado, plano_atual)
                    linhas.append({"id_cliente": int(cliente.id_cliente), "ciclo": ciclo,
                                   "plano_atual": plano_atual.nome, "plano_ofertado": ofertado.nome,
                                   "chance_simulada": chance, "aceitou": int(gerador.random() < chance),
                                   "nota_do_cliente": self._nota(chance, gerador_das_notas)})
        return pd.DataFrame(linhas)

    def _nota(self, chance, gerador):
        maxima = self.configuracao.nota_maxima
        nota = round(maxima * chance + gerador.normal(0.0, self.configuracao.desvio_padrao_da_nota))
        return int(min(maxima, max(0, nota)))
