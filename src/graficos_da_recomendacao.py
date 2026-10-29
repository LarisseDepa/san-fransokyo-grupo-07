"""Diagnósticos da classificação de ofertas no teste temporal, em SVG.

Os Displays do scikit-learn usam as respostas simuladas e os escores dos modelos
já ajustados. Nenhum gráfico treina, seleciona pesos ou otimiza um limiar no teste.
"""
import html
import io
import re
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, roc_auc_score)
from sklearn.tree import plot_tree

import config


class GraficosDaRecomendacao:
    """Curvas comparáveis, erros de classificação e explicações do Random Forest."""

    CORES = ("#2a78d6", "#9264a3", "#528275")
    TEXTO, GRADE, REFERENCIA = "#343330", "#e3e1db", "#6b6a66"
    CLASSES = ("Recusa", "Aceitação")
    NOMES_DAS_VARIAVEIS = {
        "dados_gb": "Uso típico mensal (GB)",
        "minutos": "Minutos de ligação",
        "streaming_horas": "Horas de streaming",
        "deslocamento_dias": "Dias em deslocamento",
        "mensalidade_atual": "Preço atual (R$)",
        "gb_do_plano_ofertado": "Franquia ofertada (GB)",
        "mensalidade_ofertada": "Preço ofertado (R$)",
        "diferenca_de_preco_brl": "Variação do preço (R$)",
        "diferenca_de_preco_relativa": "Variação relativa do preço",
        "cobertura_gb": "Folga da franquia (GB)",
        "grupo": "Grupo de origem",
        "plano_ofertado": "Plano ofertado",
    }

    def executar(self, resultado_da_recomendacao):
        """Devolve cinco SVGs independentes; as figuras são fechadas após a exportação."""
        resultado = resultado_da_recomendacao
        respostas = resultado.teste[resultado.configuracao.alvo].to_numpy()
        series = [
            ("Random Forest", resultado.propensao.notas(resultado.teste)),
            ("Híbrido", resultado.hibrida.notas(resultado.teste)),
            ("Pearson", resultado.colaborativa.notas(resultado.teste)),
        ]
        estilo = {
            "font.family": "DejaVu Sans", "font.size": 11,
            "text.color": self.TEXTO, "axes.labelcolor": self.TEXTO,
            "xtick.color": self.TEXTO, "ytick.color": self.TEXTO,
            "axes.edgecolor": self.GRADE, "axes.spines.top": False,
            "axes.spines.right": False, "grid.color": self.GRADE,
            "grid.linewidth": 0.6, "legend.frameon": False,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
        with mpl.rc_context(estilo):
            return {
                "roc": self._curva_roc(respostas, series),
                "precisao_revocacao": self._curva_de_precisao_e_revocacao(respostas, series),
                "matriz_de_confusao": self._matriz_de_confusao(
                    respostas, series[0][1], resultado.configuracao.limiar_da_classificacao),
                "importancia": self._importancia_por_permutacao(resultado, respostas),
                "arvore": self._arvore_ilustrativa(resultado),
            }

    def _curva_roc(self, respostas, series):
        figura, eixo = self._figura((6.4, 5.2))
        try:
            for (nome, notas), cor in zip(series, self.CORES):
                curva = RocCurveDisplay.from_predictions(
                    respostas, notas, pos_label=1, name=nome, ax=eixo)
                curva.line_.set(color=cor, linewidth=2)
                curva.line_.set_label(f"{nome} · AUC = {self._numero(roc_auc_score(respostas, notas))}")
            eixo.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=self.REFERENCIA,
                      label="Sorteio · AUC = 0,500")
            eixo.set(xlabel="Taxa de falsas aceitações (falsos positivos)",
                     ylabel="Taxa de aceitações detectadas (sensibilidade)",
                     xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            eixo.grid(alpha=0.7)
            eixo.legend(loc="lower right", fontsize=10)
            return self._svg(figura, "roc", "Curvas ROC no ciclo de teste simulado")
        finally:
            plt.close(figura)

    def _curva_de_precisao_e_revocacao(self, respostas, series):
        figura, eixo = self._figura((6.4, 5.2))
        try:
            for (nome, notas), cor in zip(series, self.CORES):
                curva = PrecisionRecallDisplay.from_predictions(
                    respostas, notas, pos_label=1, name=nome, ax=eixo)
                curva.line_.set(color=cor, linewidth=2)
                precisao_media = average_precision_score(respostas, notas)
                curva.line_.set_label(f"{nome} · AP = {self._numero(precisao_media)}")
            taxa = float(np.mean(respostas))
            eixo.axhline(taxa, linestyle="--", linewidth=1.2, color=self.REFERENCIA,
                         label=f"Referência · aceitação = {self._numero(taxa)}")
            eixo.set(xlabel="Revocação (proporção das aceitações detectadas)",
                     ylabel="Precisão (acerto entre as aceitações previstas)",
                     xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            eixo.grid(alpha=0.7)
            eixo.legend(loc="lower left", fontsize=10)
            return self._svg(figura, "precisao-revocacao", "Precisão e revocação no ciclo de teste simulado")
        finally:
            plt.close(figura)

    def _matriz_de_confusao(self, respostas, probabilidades, limiar):
        figura, eixo = self._figura((5.8, 5.0))
        try:
            previsoes = (np.asarray(probabilidades) >= limiar).astype(int)
            ConfusionMatrixDisplay.from_predictions(
                respostas, previsoes, labels=[0, 1], display_labels=self.CLASSES, ax=eixo,
                normalize=None, values_format="d", colorbar=False, cmap="Blues",
                text_kw={"fontsize": 16})
            eixo.set(xlabel="Resposta prevista pelo Random Forest", ylabel="Resposta observada na simulação")
            return self._svg(figura, "matriz-confusao", "Matriz de confusão do Random Forest, em número de ofertas")
        finally:
            plt.close(figura)

    def _importancia_por_permutacao(self, resultado, respostas):
        """Permuta as colunas originais, mantendo o pré-processamento dentro do pipeline."""
        variaveis = resultado.configuracao.variaveis
        importancia = permutation_importance(
            resultado.propensao.pipeline, resultado.teste[variaveis], respostas,
            scoring="roc_auc", n_repeats=config.REPETICOES_DA_IMPORTANCIA_DA_OFERTA,
            random_state=resultado.configuracao.semente, n_jobs=1)
        ordem = np.argsort(importancia.importances_mean)
        rotulos = [textwrap.fill(self.NOMES_DAS_VARIAVEIS.get(variaveis[i], variaveis[i]), 26)
                   for i in ordem]
        figura, eixo = self._figura((8.0, 6.5))
        try:
            eixo.barh(rotulos, importancia.importances_mean[ordem],
                      xerr=importancia.importances_std[ordem], color=self.CORES[0], height=0.6,
                      error_kw={"ecolor": self.REFERENCIA, "capsize": 3, "linewidth": 1})
            eixo.axvline(0, color=self.REFERENCIA, linewidth=0.8)
            eixo.set(xlabel="Queda da AUC ao embaralhar cada variável no teste")
            eixo.set_axisbelow(True)
            eixo.grid(axis="x", alpha=0.7)
            return self._svg(figura, "importancia", "Importância por permutação do Random Forest no teste")
        finally:
            plt.close(figura)

    def _arvore_ilustrativa(self, resultado):
        pipeline = resultado.propensao.pipeline
        nomes = [self._nome_transformado(nome)
                 for nome in pipeline.named_steps["preparo"].get_feature_names_out()]
        figura, eixo = self._figura((13.5, 6.3))
        try:
            anotacoes = plot_tree(
                pipeline.named_steps["floresta"].estimators_[0],
                max_depth=config.PROFUNDIDADE_DA_ARVORE_ILUSTRATIVA,
                feature_names=nomes, class_names=list(self.CLASSES),
                filled=True, rounded=True, impurity=False, proportion=True,
                precision=2, fontsize=10, ax=eixo)
            for anotacao in anotacoes:
                texto = (anotacao.get_text().replace("samples =", "amostras =")
                         .replace("value =", "proporções =").replace("class =", "classe =")
                         .replace("True", "Sim").replace("False", "Não"))
                anotacao.set_text(texto)
            return self._svg(figura, "arvore", "Primeiros níveis da primeira árvore do Random Forest de propensão")
        finally:
            plt.close(figura)

    def _nome_transformado(self, nome):
        coluna = nome.split("__", 1)[-1]
        if coluna.startswith("grupo_"):
            return "Grupo: " + coluna.removeprefix("grupo_").replace("_", " ")
        if coluna.startswith("plano_ofertado_"):
            return "Plano: " + coluna.removeprefix("plano_ofertado_")
        return self.NOMES_DAS_VARIAVEIS.get(coluna, coluna)

    @staticmethod
    def _numero(valor):
        return f"{valor:.3f}".replace(".", ",")

    @staticmethod
    def _figura(tamanho):
        """Canvas sem janela; mantém o backend do notebook e exporta em SVG vetorial."""
        figura = Figure(figsize=tamanho, layout="constrained")
        FigureCanvasAgg(figura)
        return figura, figura.subplots()

    @staticmethod
    def _svg(figura, nome, descricao):
        """Sem declaração XML/DOCTYPE: serve como arquivo SVG e como fragmento HTML."""
        saida = io.StringIO()
        with mpl.rc_context({"svg.hashsalt": f"claro-guia-{nome}"}):
            figura.savefig(saida, format="svg", metadata={"Date": None, "Title": descricao})
        svg = saida.getvalue()
        svg = svg[svg.index("<svg"):].strip()
        # Além dos clipPaths, o Matplotlib repete IDs como figure_1 e axes_1.
        # Prefixar todos eles permite reunir os gráficos na mesma página HTML.
        svg = re.sub(r'\bid="([^"]+)"', lambda trecho: f'id="{nome}-{trecho[1]}"', svg)
        svg = svg.replace('href="#', f'href="#{nome}-').replace("url(#", f"url(#{nome}-")
        svg = svg.replace('<style type="text/css">*{',
                          '<style type="text/css">.grafico-recomendacao *{')
        # Inter já é carregada pelo site; o arquivo avulso mantém uma fonte de reserva.
        svg = re.sub(r'font-family:[^;"]+', "font-family: Inter, 'DejaVu Sans', sans-serif", svg)
        abertura, restante = svg.split(">", 1)
        abertura = re.sub(r'\s(?:width|height)="[^"]*"', "", abertura)
        return (f'{abertura} class="grafico grafico-recomendacao" role="img" '
                f'aria-label="{html.escape(descricao, quote=True)}" '
                f'style="width:100%;height:auto">{restante}')
