"""
Etapa 6: modelo que prevê a DEMANDA de internet móvel até a renovação (regressão, em GB).

O modelo usa só hábitos e consumo observado até a data de referência. O plano NÃO entra como
variável: ele define a capacidade (quantos GB cabem), não o hábito. Assim, ao comparar outro plano,
a demanda prevista continua a mesma, e só a adequação muda (ver src/recomendacao.py).

Classes:
  ConfiguracaoDoModelo     -> conjuntos de variáveis candidatos, target, semente e parâmetros
  TreinadorDoModelo        -> separa treino/teste, escolhe as variáveis por validação cruzada NO TREINO,
                              treina Random Forest e Ridge e compara com o baseline no teste
  PrevisorDeConsumo        -> usa o modelo treinado para prever (e salva/carrega o arquivo .joblib)
  ExportadorParaNavegador  -> grava as árvores do Random Forest em JSON, para o site prever sem servidor
  FlorestaExportada        -> lê esse JSON e prevê em Python, do mesmo jeito que o JavaScript do site
"""
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VARIAVEIS_BASICAS = (
    "dados_gb", "minutos", "streaming_horas", "deslocamento_dias",  # hábitos (base original)
    "consumo_ciclo_atual_gb", "dias_decorridos", "dias_ate_renovacao",  # momento do ciclo
    "media_ultimos_7_dias_gb", "media_7_dias_anteriores_gb", "tendencia_consumo_gb_dia",  # ritmo recente
    "consumo_ciclo_anterior_gb", "dias_fim_de_semana_restantes",  # ciclo anterior e calendário
)
PROJECOES_RECENTE_E_HABITO = ("projecao_pelo_ritmo_recente_gb", "projecao_pelo_habito_gb")
PROJECAO_DO_CICLO = ("projecao_pelo_ritmo_do_ciclo_gb",)
CONJUNTOS_DE_VARIAVEIS = {
    "A: hábitos, momento do ciclo e ritmo recente": VARIAVEIS_BASICAS,
    "B: A + projeções pelo ritmo recente e pelo hábito": VARIAVEIS_BASICAS + PROJECOES_RECENTE_E_HABITO,
    "C: B + projeção pelo ritmo do ciclo": VARIAVEIS_BASICAS + PROJECOES_RECENTE_E_HABITO + PROJECAO_DO_CICLO,
}
TODAS_AS_VARIAVEIS_NUMERICAS = CONJUNTOS_DE_VARIAVEIS["C: B + projeção pelo ritmo do ciclo"]
VARIAVEIS_CATEGORICAS = ("idade_faixa", "regiao")
ALVO = "target_demanda_restante_gb"
COLUNA_DO_BASELINE = "baseline_demanda_restante_gb"
COLUNA_DO_PERFIL = "perfil_grupo"
COLUNAS_PROIBIDAS = ("id_cliente", "origem", COLUNA_DO_PERFIL, "risco_cancelamento", "plano_atual",
                     "mensalidade_atual", "cenario_de_ritmo_simulado", "fator_de_ritmo_simulado")


@dataclass(frozen=True)
class ConfiguracaoDoModelo:
    variaveis_numericas: tuple = TODAS_AS_VARIAVEIS_NUMERICAS
    variaveis_categoricas: tuple = VARIAVEIS_CATEGORICAS
    conjuntos_candidatos: dict = field(default_factory=lambda: dict(CONJUNTOS_DE_VARIAVEIS))
    alvo: str = ALVO
    semente: int = 2026
    proporcao_de_teste: float = 0.2
    dobras_da_validacao: int = 5
    parametros_da_floresta: dict = field(default_factory=lambda: {
        "n_estimators": 100, "min_samples_leaf": 5, "max_features": 0.6})
    alfa_do_ridge: float = 1.0
    repeticoes_da_importancia: int = 20

    @classmethod
    def a_partir_do_config(cls, config):
        return cls(semente=config.SEMENTE, proporcao_de_teste=config.PROPORCAO_DE_TESTE)

    @property
    def variaveis(self):
        return [*self.variaveis_numericas, *self.variaveis_categoricas]


@dataclass
class ResultadoDoTreino:
    configuracao: ConfiguracaoDoModelo  # com as variáveis escolhidas
    floresta: Pipeline
    ridge: Pipeline
    treino: pd.DataFrame
    teste: pd.DataFrame
    selecao_de_variaveis: pd.DataFrame  # validação cruzada no treino de cada conjunto candidato
    metricas: pd.DataFrame  # comparação no teste entre baseline, Ridge e Random Forest
    metricas_por_perfil: pd.DataFrame
    importancia: pd.DataFrame  # importância por permutação, no teste
    previsoes_no_teste: pd.DataFrame
    registro: dict = field(default_factory=dict)  # vai para registro_do_treino.json


def calcular_metricas(observado, previsto):
    """MAE e RMSE (em GB) e R² (até 1; quanto mais perto de 1, melhor)."""
    return {"mae_gb": float(mean_absolute_error(observado, previsto)),
            "rmse_gb": float(np.sqrt(mean_squared_error(observado, previsto))),
            "r2": float(r2_score(observado, previsto))}


class TreinadorDoModelo:
    """Treina e avalia o modelo de demanda. O One-Hot e a padronização ficam dentro do Pipeline do
    scikit-learn: são ajustados SÓ no treino. O teste é usado uma única vez, no fim."""

    def __init__(self, configuracao):
        self.configuracao = configuracao

    def treinar(self, base, versao_dos_dados, agora):
        treino, teste = train_test_split(base, test_size=self.configuracao.proporcao_de_teste,
                                         random_state=self.configuracao.semente, stratify=base[COLUNA_DO_PERFIL])
        selecao = self._comparar_conjuntos(treino)
        escolhido = selecao.dropna(subset=["variaveis"]).sort_values("mae_validacao_cruzada_gb").iloc[0]
        configuracao = replace(self.configuracao,
                               variaveis_numericas=self.configuracao.conjuntos_candidatos[escolhido.conjunto])
        self._conferir_variaveis(base, configuracao)

        floresta = self._pipeline_da_floresta(configuracao).fit(treino[configuracao.variaveis], treino[configuracao.alvo])
        ridge = self._pipeline_do_ridge(configuracao).fit(treino[configuracao.variaveis], treino[configuracao.alvo])
        previsoes = teste[["id_cliente", COLUNA_DO_PERFIL, configuracao.alvo, COLUNA_DO_BASELINE]].copy()
        previsoes["previsao_floresta_gb"] = floresta.predict(teste[configuracao.variaveis])
        previsoes["previsao_ridge_gb"] = ridge.predict(teste[configuracao.variaveis])
        previsoes["residuo_floresta_gb"] = previsoes[configuracao.alvo] - previsoes.previsao_floresta_gb

        resultado = ResultadoDoTreino(
            configuracao=configuracao, floresta=floresta, ridge=ridge, treino=treino, teste=teste,
            selecao_de_variaveis=selecao.assign(escolhido=selecao.conjunto == escolhido.conjunto),
            metricas=self._comparacao(previsoes, configuracao),
            metricas_por_perfil=self._metricas_por_perfil(previsoes, configuracao),
            importancia=self._importancia(floresta, teste, configuracao), previsoes_no_teste=previsoes)
        resultado.registro = self._registro(resultado, versao_dos_dados, agora)
        return resultado

    def _comparar_conjuntos(self, treino):
        """MAE médio em validação cruzada (só no treino) de cada conjunto de variáveis candidato."""
        dobras = KFold(self.configuracao.dobras_da_validacao, shuffle=True, random_state=self.configuracao.semente)
        linhas = []
        for nome, numericas in self.configuracao.conjuntos_candidatos.items():
            configuracao = replace(self.configuracao, variaveis_numericas=numericas)
            erros = -cross_val_score(self._pipeline_da_floresta(configuracao), treino[configuracao.variaveis],
                                     treino[configuracao.alvo], cv=dobras, scoring="neg_mean_absolute_error")
            linhas.append({"conjunto": nome, "variaveis": len(configuracao.variaveis),
                           "mae_validacao_cruzada_gb": float(erros.mean())})
        com_baseline = treino[treino[COLUNA_DO_BASELINE].notna()]
        linhas.append({"conjunto": "Referência: baseline (média diária), no treino", "variaveis": None,
                       "mae_validacao_cruzada_gb": float(mean_absolute_error(
                           com_baseline[self.configuracao.alvo], com_baseline[COLUNA_DO_BASELINE]))})
        return pd.DataFrame(linhas)

    @staticmethod
    def _conferir_variaveis(base, configuracao):
        """Nenhum target, baseline, identificador ou informação latente da simulação pode ser variável."""
        proibidas = [variavel for variavel in configuracao.variaveis
                     if variavel.startswith(("target_", "baseline_")) or variavel in COLUNAS_PROIBIDAS]
        faltando = [variavel for variavel in configuracao.variaveis if variavel not in base]
        if proibidas or faltando:
            raise ValueError(f"Variáveis inválidas no modelo: proibidas={proibidas}, faltando={faltando}")

    @staticmethod
    def _codificador():
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _pipeline_da_floresta(self, configuracao):
        preparo = ColumnTransformer([("numericas", "passthrough", list(configuracao.variaveis_numericas)),
                                     ("categoricas", self._codificador(), list(configuracao.variaveis_categoricas))])
        floresta = RandomForestRegressor(random_state=configuracao.semente,
                                         n_jobs=1 if sys.platform == "emscripten" else -1,
                                         **configuracao.parametros_da_floresta)
        return Pipeline([("preparo", preparo), ("floresta", floresta)])

    def _pipeline_do_ridge(self, configuracao):
        preparo = ColumnTransformer([("numericas", StandardScaler(), list(configuracao.variaveis_numericas)),
                                     ("categoricas", self._codificador(), list(configuracao.variaveis_categoricas))])
        return Pipeline([("preparo", preparo), ("ridge", Ridge(alpha=configuracao.alfa_do_ridge))])

    @staticmethod
    def _comparacao(previsoes, configuracao):
        """Os três modelos nos MESMOS clientes de teste: aqueles em que o baseline existe
        (ele não existe quando o ciclo ainda não tem dias). O Random Forest também é avaliado
        em todos os clientes de teste."""
        alvo = configuracao.alvo
        comparaveis = previsoes[previsoes[COLUNA_DO_BASELINE].notna()]
        linhas = [
            {"modelo": "Baseline (média diária do ciclo)", "clientes_avaliados": len(comparaveis),
             **calcular_metricas(comparaveis[alvo], comparaveis[COLUNA_DO_BASELINE])},
            {"modelo": "Ridge (regressão linear)", "clientes_avaliados": len(comparaveis),
             **calcular_metricas(comparaveis[alvo], comparaveis.previsao_ridge_gb)},
            {"modelo": "Random Forest", "clientes_avaliados": len(comparaveis),
             **calcular_metricas(comparaveis[alvo], comparaveis.previsao_floresta_gb)},
            {"modelo": "Random Forest (todos os clientes de teste)", "clientes_avaliados": len(previsoes),
             **calcular_metricas(previsoes[alvo], previsoes.previsao_floresta_gb)},
        ]
        return pd.DataFrame(linhas)

    @staticmethod
    def _metricas_por_perfil(previsoes, configuracao):
        alvo = configuracao.alvo
        linhas = []
        for perfil, grupo in previsoes.groupby(COLUNA_DO_PERFIL):
            comparaveis = grupo[grupo[COLUNA_DO_BASELINE].notna()]
            linhas.append({"perfil_grupo": perfil, "clientes_de_teste": len(grupo),
                           "mae_floresta_gb": mean_absolute_error(grupo[alvo], grupo.previsao_floresta_gb),
                           "mae_baseline_gb": mean_absolute_error(comparaveis[alvo], comparaveis[COLUNA_DO_BASELINE]),
                           "demanda_media_real_gb": grupo[alvo].mean()})
        return pd.DataFrame(linhas)

    @staticmethod
    def _importancia(floresta, teste, configuracao):
        """Quanto o erro (MAE) piora quando os valores de uma variável são embaralhados no teste."""
        resultado = permutation_importance(
            floresta, teste[configuracao.variaveis], teste[configuracao.alvo],
            scoring="neg_mean_absolute_error", n_repeats=configuracao.repeticoes_da_importancia,
            random_state=configuracao.semente, n_jobs=1)
        return pd.DataFrame({"variavel": configuracao.variaveis, "aumento_do_mae_gb": resultado.importances_mean,
                             "desvio_gb": resultado.importances_std}).sort_values(
            "aumento_do_mae_gb", ascending=False, ignore_index=True)

    @staticmethod
    def _registro(resultado, versao_dos_dados, agora):
        configuracao = resultado.configuracao
        parametros = {**configuracao.parametros_da_floresta, "random_state": configuracao.semente}
        assinatura = json.dumps({"dados": versao_dos_dados, "parametros": parametros,
                                 "variaveis": configuracao.variaveis}, sort_keys=True)
        return {
            "versao_do_modelo": "demanda-rf-" + hashlib.sha256(assinatura.encode()).hexdigest()[:10],
            "treinado_em": agora.isoformat(timespec="seconds"),
            "versao_dos_dados": versao_dos_dados,
            "semente": configuracao.semente,
            "algoritmo": "RandomForestRegressor (scikit-learn)",
            "parametros": parametros,
            "alvo": configuracao.alvo,
            "variaveis_numericas": list(configuracao.variaveis_numericas),
            "variaveis_categoricas_one_hot": list(configuracao.variaveis_categoricas),
            "variaveis_depois_do_one_hot": list(resultado.floresta.named_steps["preparo"].get_feature_names_out()),
            "selecao_de_variaveis": resultado.selecao_de_variaveis.to_dict(orient="records"),
            "clientes_no_treino": len(resultado.treino), "clientes_no_teste": len(resultado.teste),
            "separacao": f"aleatória por cliente, {configuracao.proporcao_de_teste:.0%} para teste, "
                         "estratificada por perfil_grupo; variáveis escolhidas por validação cruzada "
                         f"({configuracao.dobras_da_validacao} dobras) só no treino",
            "bibliotecas": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
            "metricas_no_teste": resultado.metricas.to_dict(orient="records"),
            "limitacoes": [
                "O target é sintético: as métricas medem o quanto o modelo aprendeu a simulação, "
                "não a acurácia com clientes reais.",
                "dados_gb foi tratado como consumo típico de um mês (hipótese: a periodicidade não está documentada).",
                "Uma mudança de ritmo depois da data de referência (sorteada na simulação) não pode ser prevista.",
                "Quem já esgotou a franquia tem o consumo observado travado, o que esconde parte da demanda.",
            ],
        }


class PrevisorDeConsumo:
    """Prevê a demanda restante (GB) com o pipeline treinado."""

    def __init__(self, pipeline, configuracao):
        self.pipeline = pipeline
        self.configuracao = configuracao

    def prever(self, linhas):
        """linhas: DataFrame (ou lista de dicionários) com as variáveis do modelo."""
        return self.pipeline.predict(pd.DataFrame(linhas)[self.configuracao.variaveis])

    def prever_um(self, atributos):
        return float(self.prever([atributos])[0])

    def salvar(self, caminho):
        """Arquivo .joblib para uso LOCAL (Python). Ele não vai para o site: o navegador usa o JSON."""
        joblib.dump({"pipeline": self.pipeline, "configuracao": asdict(self.configuracao)}, caminho)

    @classmethod
    def carregar(cls, caminho):
        conteudo = joblib.load(caminho)
        configuracao = ConfiguracaoDoModelo(**{chave: tuple(valor) if isinstance(valor, list) else valor
                                               for chave, valor in conteudo["configuracao"].items()})
        return cls(conteudo["pipeline"], configuracao)


class ExportadorParaNavegador:
    """Grava o Random Forest treinado num JSON que o JavaScript do site percorre.

    Cada árvore vira 4 listas: esquerda, direita, variavel e valor. Nos nós de decisão, "valor" é o
    limiar; nas folhas (esquerda = -1), é a previsão da folha. O site faz o mesmo One-Hot (com as
    categorias aprendidas no treino) e compara como o scikit-learn: valor em float32 <= limiar.

    Os limiares vão com precisão total: às vezes o limiar é exatamente um valor float32 dos dados,
    e qualquer arredondamento mandaria o cliente para o lado errado da árvore. Só as folhas são
    arredondadas (9 algarismos significativos, diferença menor que 0,000001 GB)."""

    CASAS_DA_FOLHA = 9

    def exportar(self, pipeline, registro):
        preparo, floresta = pipeline.named_steps["preparo"], pipeline.named_steps["floresta"]
        codificador = preparo.named_transformers_["categoricas"]
        categoricas = list(preparo.transformers_[1][2])
        return {
            "versao_do_modelo": registro["versao_do_modelo"],
            "treinado_em": registro["treinado_em"],
            "alvo": registro["alvo"],
            "variaveis_numericas": list(preparo.transformers_[0][2]),
            "variaveis_categoricas": {variavel: [str(categoria) for categoria in categorias]
                                      for variavel, categorias in zip(categoricas, codificador.categories_)},
            "ordem_das_variaveis": list(preparo.get_feature_names_out()),
            "arvores": [self._arvore(estimador.tree_) for estimador in floresta.estimators_],
        }

    def _arvore(self, arvore):
        folha = arvore.children_left == -1
        valores = np.where(folha, self._valores_das_folhas(arvore), arvore.threshold)
        return {"esquerda": arvore.children_left.tolist(), "direita": arvore.children_right.tolist(),
                "variavel": np.where(folha, -1, arvore.feature).tolist(),
                "valor": [float(f"{valor:.{self.CASAS_DA_FOLHA}g}") if eh_folha else float(valor)
                          for valor, eh_folha in zip(valores, folha)]}

    @staticmethod
    def _valores_das_folhas(arvore):
        """Regressão: a previsão de cada folha (a média do treino que caiu nela)."""
        return arvore.value[:, 0, 0]


class FlorestaExportada:
    """Prevê a partir do JSON exportado, com a MESMA lógica do JavaScript do site
    (usado nos testes para conferir que o site e o scikit-learn dão o mesmo resultado)."""

    def __init__(self, modelo_json):
        self.modelo = modelo_json

    def vetor(self, atributos):
        vetor = [float(np.float32(atributos[variavel])) for variavel in self.modelo["variaveis_numericas"]]
        for variavel, categorias in self.modelo["variaveis_categoricas"].items():
            vetor += [1.0 if str(atributos[variavel]) == categoria else 0.0 for categoria in categorias]
        return vetor

    def prever_um(self, atributos):
        vetor = self.vetor(atributos)
        soma = 0.0
        for arvore in self.modelo["arvores"]:
            no = 0
            while arvore["esquerda"][no] != -1:
                no = (arvore["esquerda"][no] if vetor[arvore["variavel"][no]] <= arvore["valor"][no]
                      else arvore["direita"][no])
            soma += arvore["valor"][no]
        return soma / len(self.modelo["arvores"])


def versao_dos_dados(caminho_da_base):
    """Impressão digital (sha256, 16 caracteres) do CSV usado no treino."""
    return hashlib.sha256(caminho_da_base.read_bytes()).hexdigest()[:16]


def agora_em(fuso_horario):
    return datetime.now(fuso_horario).replace(microsecond=0)
