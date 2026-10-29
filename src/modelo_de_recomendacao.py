"""
Etapa 8 (parte 2): modelos de recomendação treinados com o histórico de ofertas SIMULADO.

Cada modelo dá uma nota a um par (cliente, plano); os planos são recomendados em ordem de nota.
  - Random Forest de propensão: estima a chance de o cliente aceitar o plano, a partir dos hábitos do
    cliente e das características do plano. Também dá nota a quem não tem histórico (como o visitante do site).
  - Filtragem colaborativa usuário a usuário, como no vídeo de referência ("Algoritmo de recomendação na
    prática com Python"): a semelhança entre dois clientes é a correlação de Pearson das notas (0 a 10) que os
    dois deram aos mesmos planos; a nota prevista de um plano é a média das notas dos clientes parecidos,
    ponderada pela semelhança. Para um cliente sem histórico, vira a média das notas do grupo dele.
O híbrido combina os dois por soma ponderada. O peso é escolhido no penúltimo ciclo (validação),
antes de avaliar no último ciclo (teste). A média original de pesos iguais continua como referência.
A saída híbrida é uma pontuação para ordenação, não uma probabilidade calibrada de aceitação.
Duas referências simples entram na comparação: a popularidade (a taxa de aceitação de cada plano) e a
regra de preço (o plano mais barato que cobre o uso típico, como na etapa 7).

Separação no tempo: treino inicial, validação e teste; depois da seleção, reajuste em treino + validação.
Lembrete: o histórico foi simulado por uma regra; as métricas mostram o quanto cada modelo recupera
essa regra, não a preferência de clientes reais.

Classes:
  AtributosDaOferta          -> as variáveis de um par (cliente, plano) para o Random Forest
  ConfiguracaoDaRecomendacao -> variáveis, semente e parâmetros
  PrevisorDePropensao        -> chance de aceitar (Random Forest); salva e carrega o .joblib
  FiltragemColaborativa      -> usuário a usuário: Pearson entre clientes e média ponderada das notas
  RecomendacaoHibrida        -> soma ponderada do Random Forest e da filtragem colaborativa
  Popularidade, RegraDePreco -> referências simples
  AvaliadorDaRecomendacao    -> AUC e acerto da 1ª oferta, no ciclo de teste
  TreinadorDaRecomendacao    -> separa no tempo, treina, avalia e monta o registro
  ExportadorDaPropensao      -> grava o Random Forest em JSON para o site (a chance de aceitar em cada folha)
  ExportadorDaColaborativa   -> grava o que o site precisa para repetir a filtragem colaborativa no navegador
"""
import hashlib
import sys
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from avaliacao_da_recomendacao import (ComparadorPareado, DiagnosticoDaClassificacao, DiagnosticoDePropensao,
                                      MetricasDeOrdenacao, SeletorDePeso)
from modelo_de_demanda import ExportadorParaNavegador
from referencias_cientificas import BibliografiaDaRecomendacao
from tabelas import COLUNA_ORIGEM

COLUNAS_DO_CLIENTE = ("dados_gb", "minutos", "streaming_horas", "deslocamento_dias", "grupo")
VARIAVEIS_NUMERICAS = ("dados_gb", "minutos", "streaming_horas", "deslocamento_dias", "mensalidade_atual",
                       "gb_do_plano_ofertado", "mensalidade_ofertada", "diferenca_de_preco_brl",
                       "diferenca_de_preco_relativa", "cobertura_gb")
VARIAVEIS_CATEGORICAS = ("grupo", "plano_ofertado")
ALVO = "aceitou"
NOTA = "nota_do_cliente"  # 0 a 10, simulada (ver historico_de_ofertas.py)
NOTA_DE_QUEM_COBRE = 1000.0  # na regra de preço, cobrir o uso típico vem antes de qualquer diferença de preço


class AtributosDaOferta:
    """Variáveis de um par (cliente, plano): hábitos do cliente, preço do plano atual e características
    do plano ofertado. A chance simulada (a "verdade" escondida da regra) nunca entra."""

    def __init__(self, planos):
        self.planos = planos

    def de_um_par(self, cliente, nome_do_plano_ofertado, nome_do_plano_atual):
        """cliente: dados_gb (uso típico de um mês), minutos, streaming_horas, deslocamento_dias e grupo."""
        ofertado = self.planos.por_nome(nome_do_plano_ofertado)
        atual = self.planos.por_nome(nome_do_plano_atual)
        diferenca = ofertado.mensalidade_brl - atual.mensalidade_brl
        return {**{coluna: cliente[coluna] for coluna in COLUNAS_DO_CLIENTE},
                "mensalidade_atual": atual.mensalidade_brl, "plano_ofertado": ofertado.nome,
                "gb_do_plano_ofertado": ofertado.gb_principais, "mensalidade_ofertada": ofertado.mensalidade_brl,
                "diferenca_de_preco_brl": diferenca, "diferenca_de_preco_relativa": diferenca / atual.mensalidade_brl,
                "cobertura_gb": ofertado.gb_principais - cliente["dados_gb"]}

    def do_historico(self, base, historico):
        """Uma linha por oferta do histórico, com id_cliente, ciclo e a resposta (aceitou)."""
        clientes = (base.rename(columns={COLUNA_ORIGEM: "grupo"}).set_index("id_cliente")
                    [list(COLUNAS_DO_CLIENTE)].to_dict("index"))
        atributos = pd.DataFrame([self.de_um_par(clientes[oferta.id_cliente], oferta.plano_ofertado,
                                                 oferta.plano_atual) for oferta in historico.itertuples()])
        return pd.concat([historico[["id_cliente", "ciclo", ALVO, NOTA]].reset_index(drop=True), atributos], axis=1)


@dataclass(frozen=True)
class ConfiguracaoDaRecomendacao:
    variaveis_numericas: tuple = VARIAVEIS_NUMERICAS
    variaveis_categoricas: tuple = VARIAVEIS_CATEGORICAS
    alvo: str = ALVO
    semente: int = 2026
    nota_maxima: int = 10
    pesos_do_random_forest: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    reamostragens: int = 2000
    limiar_da_classificacao: float = 0.5
    parametros_da_floresta: dict = field(default_factory=lambda: {
        "n_estimators": 100, "min_samples_leaf": 20, "max_features": 0.6})

    @classmethod
    def a_partir_do_config(cls, config):
        return cls(semente=config.SEMENTE, nota_maxima=config.NOTA_MAXIMA,
                   pesos_do_random_forest=config.PESOS_DO_RANDOM_FOREST,
                   reamostragens=config.REAMOSTRAGENS_DA_RECOMENDACAO,
                   limiar_da_classificacao=config.LIMIAR_DA_CLASSIFICACAO_DA_OFERTA)

    def __post_init__(self):
        if not self.pesos_do_random_forest or any(not 0 <= peso <= 1 for peso in self.pesos_do_random_forest):
            raise ValueError("Informe pesos entre 0 e 1 para a validação.")
        if not 0 <= self.limiar_da_classificacao <= 1:
            raise ValueError("O limiar da classificação deve estar entre 0 e 1.")

    @property
    def variaveis(self):
        return [*self.variaveis_numericas, *self.variaveis_categoricas]


class PrevisorDePropensao:
    """Chance de aceitar cada oferta (0 a 1), com o pipeline treinado."""

    NOME = "Random Forest (propensão)"

    def __init__(self, pipeline, configuracao):
        self.pipeline = pipeline
        self.configuracao = configuracao

    def notas(self, pares):
        coluna_de_quem_aceitou = list(self.pipeline.classes_).index(1)
        return self.pipeline.predict_proba(pd.DataFrame(pares)[self.configuracao.variaveis])[:, coluna_de_quem_aceitou]

    def salvar(self, caminho):
        """Arquivo .joblib para uso LOCAL (Python). Ele não vai para o site: o navegador usa o JSON."""
        joblib.dump({"pipeline": self.pipeline, "configuracao": asdict(self.configuracao)}, caminho)

    @classmethod
    def carregar(cls, caminho):
        conteudo = joblib.load(caminho)
        configuracao = ConfiguracaoDaRecomendacao(**{chave: tuple(valor) if isinstance(valor, list) else valor
                                                     for chave, valor in conteudo["configuracao"].items()})
        return cls(conteudo["pipeline"], configuracao)


class FiltragemColaborativa:
    """Usuário a usuário, como no vídeo de referência:
    1. a nota de um cliente para um plano é a média das notas (0 a 10) que ele deu ao plano no treino; plano
       nunca oferecido fica sem nota (como as séries "não assistidas" do vídeo);
    2. a semelhança entre dois clientes é a correlação de Pearson das notas dos planos que os DOIS avaliaram
       (de −1, gostos opostos, a 1, gostos iguais); com menos de 2 planos em comum, não há semelhança;
    3. a nota prevista de um plano = soma(semelhança × nota do vizinho) ÷ soma(semelhança), entre os clientes
       parecidos (semelhança > 0) que avaliaram o plano.
    Um cliente sem histórico não tem como calcular Pearson: seus "parecidos" passam a ser os clientes do mesmo
    grupo (a mesma planilha), e a nota prevista é a média das notas deles para o plano. Um cliente com histórico
    mas sem vizinhos parecidos que avaliaram o plano recebe a média das notas do plano entre todos."""

    NOME = "Filtragem colaborativa (usuário a usuário, Pearson)"
    MINIMO_DE_PLANOS_EM_COMUM = 2  # a correlação de Pearson precisa de pelo menos 2 pares de notas

    def __init__(self, nota_maxima):
        self.nota_maxima = nota_maxima

    def ajustar(self, pares_de_treino):
        self.notas_por_cliente = pares_de_treino.pivot_table(index="id_cliente", columns="plano_ofertado", values=NOTA,
                                                 aggfunc="mean")
        self.media_dos_planos = self.notas_por_cliente.mean()
        grupo_de_cada_cliente = pares_de_treino.groupby("id_cliente").grupo.first()
        self.media_por_grupo = self.notas_por_cliente.groupby(grupo_de_cada_cliente).mean()
        self.semelhanca = pd.DataFrame(self._pearson_entre_clientes(self.notas_por_cliente.to_numpy()),
                                       index=self.notas_por_cliente.index, columns=self.notas_por_cliente.index)
        return self

    def _pearson_entre_clientes(self, notas):
        """Matriz cliente × cliente com a correlação de Pearson nos planos avaliados pelos dois (0 = sem semelhança)."""
        avaliou = ~np.isnan(notas)
        valores = np.nan_to_num(notas)
        semelhanca = np.zeros((len(notas), len(notas)))
        for cliente in range(len(notas)):
            em_comum = avaliou & avaliou[cliente]
            quantidade = em_comum.sum(axis=1)
            com_dados = quantidade >= self.MINIMO_DE_PLANOS_EM_COMUM
            meu = np.where(em_comum, valores[cliente], 0.0)
            dos_outros = np.where(em_comum, valores, 0.0)
            com_divisor = np.maximum(quantidade, 1)[:, None]
            meu_centrado = np.where(em_comum, meu - meu.sum(axis=1, keepdims=True) / com_divisor, 0.0)
            dos_outros_centrado = np.where(em_comum, dos_outros - dos_outros.sum(axis=1, keepdims=True) / com_divisor, 0.0)
            divisor = np.sqrt((meu_centrado ** 2).sum(axis=1) * (dos_outros_centrado ** 2).sum(axis=1))
            correlacao = np.divide((meu_centrado * dos_outros_centrado).sum(axis=1), divisor,
                                   out=np.zeros(len(notas)), where=com_dados & (divisor > 0))
            semelhanca[cliente] = correlacao
        np.fill_diagonal(semelhanca, 0.0)  # o próprio cliente não conta como vizinho
        return semelhanca

    def nota_prevista(self, id_cliente, plano, grupo):
        """Nota de 0 a 10 que o cliente daria ao plano."""
        if id_cliente not in self.semelhanca.index:  # cliente novo: a média das notas do grupo dele
            return float(self.media_por_grupo.loc[grupo, plano])
        semelhancas = self.semelhanca.loc[id_cliente]
        notas_do_plano = self.notas_por_cliente[plano]
        vizinhos = (semelhancas > 0) & notas_do_plano.notna()
        if not vizinhos.any():
            return float(self.media_dos_planos[plano])
        return float((semelhancas[vizinhos] * notas_do_plano[vizinhos]).sum() / semelhancas[vizinhos].sum())

    def notas(self, pares):
        """O percentual previsto (nota prevista ÷ nota máxima), de 0 a 1."""
        return np.array([self.nota_prevista(par.id_cliente, par.plano_ofertado, par.grupo) / self.nota_maxima
                         for par in pd.DataFrame(pares).itertuples()])


class RecomendacaoHibrida:
    """Soma ponderada da propensão e da nota normalizada: pontuação, não probabilidade.

    Sem id_cliente, a componente colaborativa usa a média das notas do grupo.
    O peso padrão 0,5 preserva a combinação original para consumidores existentes.
    """

    NOME = "Híbrido ponderado (peso escolhido na validação)"
    NOME_DA_MEDIA = "Híbrido de pesos iguais (referência 50/50)"

    def __init__(self, propensao, colaborativa, peso_random_forest=0.5):
        if not 0 <= peso_random_forest <= 1:
            raise ValueError("O peso do Random Forest deve estar entre 0 e 1.")
        self.propensao = propensao
        self.colaborativa = colaborativa
        self.peso_random_forest = peso_random_forest

    def notas_por_modelo(self, pares):
        """(propensão, nota colaborativa normalizada, pontuação ponderada) de cada par."""
        pares = pd.DataFrame(pares)
        floresta, colaborativa = self.propensao.notas(pares), self.colaborativa.notas(pares)
        pontuacao = self.peso_random_forest * floresta + (1 - self.peso_random_forest) * colaborativa
        return floresta, colaborativa, pontuacao

    def notas(self, pares):
        return self.notas_por_modelo(pares)[2]


class Popularidade:
    """Referência: todo cliente recebe a mesma nota para um plano (a taxa de aceitação dele no treino)."""

    NOME = "Popularidade (referência)"

    def ajustar(self, pares_de_treino):
        self.taxa_dos_planos = pares_de_treino.groupby("plano_ofertado")[ALVO].mean()
        return self

    def notas(self, pares):
        return pares.plano_ofertado.map(self.taxa_dos_planos).to_numpy()


class RegraDePreco:
    """Referência: primeiro os planos que cobrem o uso típico; entre eles, o mais barato (como na etapa 7)."""

    NOME = "Regra de preço (referência)"

    @staticmethod
    def notas(pares):
        cobre = (pares.gb_do_plano_ofertado >= pares.dados_gb).astype(float)
        return (cobre * NOTA_DE_QUEM_COBRE - pares.mensalidade_ofertada).to_numpy()


class AvaliadorDaRecomendacao:
    """No ciclo de teste:
    - AUC: a chance de uma oferta aceita receber nota maior que uma recusada (0,5 = sorteio; 1 = sempre);
    - acerto da 1ª oferta: entre as ofertas do ciclo, a de maior nota foi aceita? Só entram clientes que
      aceitaram uma e recusaram outra (nos demais, a ordem não muda nada)."""

    NOME_DO_SORTEIO = "Sorteio (referência)"

    def avaliar(self, nome, notas, pares_de_teste):
        com_nota = pares_de_teste.assign(nota=notas)
        acertos = self.acertos_por_cliente(notas, pares_de_teste)
        return {"modelo": nome, "auc": float(roc_auc_score(com_nota[ALVO], com_nota.nota)),
                "acerto_da_primeira_oferta": float(acertos.mean()), "clientes_elegiveis": len(acertos),
                **MetricasDeOrdenacao().executar(notas, pares_de_teste)}

    def acertos_por_cliente(self, notas, pares):
        com_nota = pares.assign(nota=notas)
        elegiveis = com_nota.groupby("id_cliente")[ALVO].transform("nunique") == 2
        primeiras = (com_nota[elegiveis].sort_values(["nota", "plano_ofertado"], ascending=[False, True])
                     .drop_duplicates("id_cliente").set_index("id_cliente"))
        if primeiras.empty:
            raise ValueError("O ciclo precisa de clientes com ofertas aceitas e recusadas para avaliar o ranking.")
        return primeiras[ALVO].sort_index()

    def sorteio(self, pares_de_teste):
        """O que se espera escolhendo a 1ª oferta ao acaso: a proporção de ofertas aceitas de cada cliente."""
        clientes = self._clientes_com_respostas_diferentes(pares_de_teste)
        return {"modelo": self.NOME_DO_SORTEIO, "auc": 0.5, "clientes_elegiveis": len(clientes),
                "acerto_da_primeira_oferta": float(np.mean([
            ofertas[ALVO].mean() for ofertas in clientes])), **MetricasDeOrdenacao().sorteio(pares_de_teste)}

    @staticmethod
    def _clientes_com_respostas_diferentes(pares):
        return [ofertas for _, ofertas in pares.groupby("id_cliente") if ofertas[ALVO].nunique() == 2]


@dataclass
class ResultadoDaRecomendacao:
    configuracao: ConfiguracaoDaRecomendacao
    propensao: PrevisorDePropensao
    colaborativa: FiltragemColaborativa
    hibrida: RecomendacaoHibrida
    treino: pd.DataFrame
    validacao: pd.DataFrame
    treino_final: pd.DataFrame
    teste: pd.DataFrame
    comparacao_dos_pesos: pd.DataFrame
    comparacao_pareada: dict
    curva_de_calibracao: pd.DataFrame
    diagnostico_de_propensao: dict
    metricas: pd.DataFrame  # comparação no ciclo de teste
    aceitacao_por_grupo: pd.DataFrame  # % de ofertas aceitas, por grupo e plano (todo o histórico)
    classificacao_no_teste: dict = field(default_factory=dict)
    registro: dict = field(default_factory=dict)  # vai para registro_da_recomendacao.json


class TreinadorDaRecomendacao:
    """Seleciona o peso na validação; reajusta antes do teste, que não participa dos ajustes."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def treinar(self, base, historico, planos, configuracao_do_historico, agora):
        pares = AtributosDaOferta(planos).do_historico(base, historico)
        ciclos = sorted(pares.ciclo.unique())
        if len(ciclos) < 3:
            raise ValueError("São necessários pelo menos três ciclos: treino, validação e teste.")
        ciclo_de_validacao, ultimo_ciclo = ciclos[-2:]
        treino = pares[pares.ciclo < ciclo_de_validacao]
        validacao = pares[pares.ciclo == ciclo_de_validacao]
        treino_final = pares[pares.ciclo < ultimo_ciclo]
        teste = pares[pares.ciclo == ultimo_ciclo]
        avaliador = AvaliadorDaRecomendacao()
        propensao_inicial, colaborativa_inicial = self._ajustar(treino)
        peso, comparacao_dos_pesos = SeletorDePeso().executar(
            self.configuracao.pesos_do_random_forest, propensao_inicial.notas(validacao),
            colaborativa_inicial.notas(validacao), validacao, avaliador)
        propensao, colaborativa = self._ajustar(treino_final)
        hibrida = RecomendacaoHibrida(propensao, colaborativa, peso)
        modelos = [propensao, colaborativa, hibrida, Popularidade().ajustar(treino_final), RegraDePreco()]
        metricas = pd.DataFrame([avaliador.avaliar(modelo.NOME, modelo.notas(teste), teste) for modelo in modelos]
                                + [avaliador.avaliar(RecomendacaoHibrida.NOME_DA_MEDIA,
                                   RecomendacaoHibrida(propensao, colaborativa).notas(teste), teste)]
                                + [avaliador.sorteio(teste)])
        comparacao_pareada = ComparadorPareado(self.configuracao.reamostragens, self.configuracao.semente).executar(
            avaliador.acertos_por_cliente(hibrida.notas(teste), teste),
            avaliador.acertos_por_cliente(propensao.notas(teste), teste))
        diagnostico, curva = DiagnosticoDePropensao().executar(teste[ALVO], propensao.notas(teste))
        classificacao = DiagnosticoDaClassificacao(self.configuracao.limiar_da_classificacao).executar(
            teste[ALVO], {modelo.NOME: modelo.notas(teste) for modelo in [propensao, colaborativa, hibrida]},
            propensao.NOME)
        aceitacao = (pares.pivot_table(index="grupo", columns="plano_ofertado", values=ALVO).mul(100)
                     [[plano.nome for plano in planos.planos]].reset_index())
        resultado = ResultadoDaRecomendacao(configuracao=self.configuracao, propensao=propensao,
                                            colaborativa=colaborativa, hibrida=hibrida, treino=treino, validacao=validacao,
                                            treino_final=treino_final, teste=teste, comparacao_dos_pesos=comparacao_dos_pesos,
                                            comparacao_pareada=comparacao_pareada, curva_de_calibracao=curva,
                                            diagnostico_de_propensao=diagnostico,
                                            metricas=metricas, aceitacao_por_grupo=aceitacao,
                                            classificacao_no_teste=classificacao)
        resultado.registro = self._registro(resultado, historico, configuracao_do_historico, ultimo_ciclo, agora)
        return resultado

    def _ajustar(self, treino):
        floresta = self._pipeline_da_floresta().fit(treino[self.configuracao.variaveis], treino[self.configuracao.alvo])
        if list(floresta.classes_) != [0, 1]:
            raise ValueError(f"Respostas inesperadas no histórico: {list(floresta.classes_)}")
        return (PrevisorDePropensao(floresta, self.configuracao),
                FiltragemColaborativa(self.configuracao.nota_maxima).ajustar(treino))

    def _pipeline_da_floresta(self):
        preparo = ColumnTransformer([
            ("numericas", "passthrough", list(self.configuracao.variaveis_numericas)),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(self.configuracao.variaveis_categoricas))])
        floresta = RandomForestClassifier(random_state=self.configuracao.semente,
                                          n_jobs=1 if sys.platform == "emscripten" else -1,
                                          **self.configuracao.parametros_da_floresta)
        return Pipeline([("preparo", preparo), ("floresta", floresta)])

    def _registro(self, resultado, historico, configuracao_do_historico, ultimo_ciclo, agora):
        impressao_do_historico = hashlib.sha256(historico.to_csv(index=False).encode("utf-8")).hexdigest()[:10]
        return {
            "versao_do_modelo": f"recomendacao-v2-{impressao_do_historico}-rf{resultado.hibrida.peso_random_forest:.2f}",
            "versao_do_protocolo": 2,
            "treinado_em": agora.isoformat(),
            "semente": self.configuracao.semente,
            "algoritmo": "RandomForestClassifier (scikit-learn), dentro de um Pipeline com One-Hot",
            "parametros": self.configuracao.parametros_da_floresta,
            "alvo": self.configuracao.alvo,
            "variaveis_numericas": list(self.configuracao.variaveis_numericas),
            "variaveis_categoricas": list(self.configuracao.variaveis_categoricas),
            "ofertas_no_treino": len(resultado.treino), "ofertas_na_validacao": len(resultado.validacao),
            "ofertas_no_treino_final": len(resultado.treino_final), "ofertas_no_teste": len(resultado.teste),
            "separacao": f"no tempo: ciclos anteriores a {int(resultado.validacao.ciclo.min())} no treino inicial; "
                         f"ciclo {int(resultado.validacao.ciclo.min())} na validação; ciclo {ultimo_ciclo} no teste. "
                         "Após escolher o peso, reajuste em treino + validação, sem acessar o teste",
            "selecao_do_peso": {"peso_random_forest": resultado.hibrida.peso_random_forest,
                               "criterio": "maior acerto condicional da primeira oferta na validação; "
                                           "desempate por AUC, proximidade de 0,5 e menor peso",
                               "comparacao_na_validacao": resultado.comparacao_dos_pesos.to_dict("records")},
            "comparacao_pareada_no_teste": resultado.comparacao_pareada,
            "diagnostico_de_propensao_no_teste": resultado.diagnostico_de_propensao,
            "classificacao_no_teste": resultado.classificacao_no_teste,
            "saida_do_hibrido": "pontuação de recomendação em [0, 1]; não é probabilidade calibrada",
            "bibliografia_cientifica": BibliografiaDaRecomendacao().como_registro(),
            "protocolo_de_ordenacao": {
                "candidatos": "somente ofertas observadas no ciclo, iguais para todos os modelos; não é o catálogo completo",
                "metricas_adicionais": "NDCG@3, MRR e acerto da primeira oferta em todos os clientes do ciclo",
                "clientes_sem_aceitacao": "incluídos com NDCG e MRR iguais a zero",
                "desempate": "nome do plano, em ordem alfabética",
                "selecao_preservada": "somente o acerto condicional original seleciona o peso; métricas novas são diagnósticas",
            },
            "taxa_de_aceitacao": float(historico[ALVO].mean()),
            "regra_do_historico_simulado": asdict(configuracao_do_historico),
            "metricas_no_teste": resultado.metricas.to_dict("records"),
            "bibliotecas": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "limitacoes": [
                "A avaliação mede a ordenação entre as ofertas sorteadas de cada ciclo, antes dos filtros da "
                "demonstração; não mede o desempenho de todo o fluxo de previsão de consumo e recomendação.",
                "O acerto da primeira oferta é condicional aos clientes com aceitação e recusa no ciclo; "
                "não é a taxa geral de aceitação dos clientes.",
                "O peso é escolhido para clientes com histórico. O uso da média do grupo para visitantes "
                "sem histórico não foi validado em uma amostra independente de novos clientes.",
                "O teste existente já foi consultado em decisões anteriores do projeto. Esta separação melhora "
                "o protocolo, mas uma confirmação independente exige novos dados ainda não examinados.",
                "O bootstrap é condicional aos modelos treinados e à regra sintética; não comprova superioridade "
                "geral. Notas e aceitações são produzidas pela mesma preferência latente simulada.",
                "O histórico de ofertas é simulado por uma regra (config.py): os modelos aprendem essa regra; "
                "as métricas não medem a preferência de clientes reais.",
                "Os planos são os 8 planos fictícios das planilhas, com o preço que aparece nelas.",
                "As notas de 0 a 10 também são simuladas (10 × a chance da regra + variação aleatória).",
                "A filtragem colaborativa precisa do histórico do próprio cliente; para um cliente novo, ela usa "
                "a média das notas do grupo dele (a mesma planilha).",
            ],
        }


class ExportadorDaPropensao(ExportadorParaNavegador):
    """Como o da etapa 6, mas cada folha guarda a chance de aceitar (a proporção de "aceitou" na folha,
    que é o que o predict_proba do scikit-learn usa). A média das árvores é a chance final."""

    @staticmethod
    def _valores_das_folhas(arvore):
        contagens = arvore.value[:, 0, :]
        return contagens[:, 1] / contagens.sum(axis=1)


class ExportadorDaColaborativa:
    """O que o navegador precisa para fazer a MESMA conta da filtragem colaborativa: as médias (por plano e por
    grupo, para quem não tem histórico) e, para cada cliente pedido, os vizinhos parecidos (semelhança > 0) com
    as notas deles, na mesma ordem do Python."""

    def exportar(self, colaborativa, ids_de_clientes):
        return {
            "nota_maxima": colaborativa.nota_maxima,
            "media_dos_planos": {plano: float(media) for plano, media in colaborativa.media_dos_planos.items()},
            "media_por_grupo": {grupo: {plano: float(media) for plano, media in linha.items()}
                                for grupo, linha in colaborativa.media_por_grupo.iterrows()},
            "clientes": {str(id_cliente): self._vizinhos(colaborativa, id_cliente) for id_cliente in ids_de_clientes},
        }

    @staticmethod
    def _vizinhos(colaborativa, id_cliente):
        semelhancas = colaborativa.semelhanca.loc[id_cliente]
        parecidos = semelhancas[semelhancas > 0]
        notas = colaborativa.notas_por_cliente.loc[parecidos.index]
        return [{"semelhanca": float(semelhanca),
                 "notas": {plano: (None if np.isnan(nota) else float(nota)) for plano, nota in notas.loc[vizinho].items()}}
                for vizinho, semelhanca in parecidos.items()]
