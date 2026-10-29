"""Seleção na validação e incerteza no teste; nenhum ajuste usa as respostas de teste."""
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, log_loss, roc_auc_score)


class MetricasDeOrdenacao:
    """Médias por cliente entre ofertas observadas, incluindo quem recusou todas.

    NDCG e MRR valem zero quando não há aceitação; não se atribui recusa a plano não ofertado.
    Os candidatos e os desempates são os mesmos para todos os modelos.
    """

    LIMITE = 3

    def executar(self, notas, pares):
        ordenados = (pares.assign(nota=notas)
                     .sort_values(["nota", "plano_ofertado"], ascending=[False, True]))
        metricas = pd.DataFrame([self._de_um_cliente(ofertas.aceitou.to_numpy())
                                 for _, ofertas in ordenados.groupby("id_cliente")])
        return {**metricas.mean().to_dict(), "clientes_avaliados": len(metricas)}

    def _de_um_cliente(self, respostas):
        descontos = 1 / np.log2(np.arange(2, min(len(respostas), self.LIMITE) + 2))
        ganho = np.dot(respostas[:self.LIMITE], descontos)
        ideal = np.dot(np.sort(respostas)[::-1][:self.LIMITE], descontos)
        aceitas = np.flatnonzero(respostas)
        return {"acerto_primeira_oferta_todos": float(respostas[0]),
                "ndcg_3_observado": float(ganho / ideal) if ideal else 0.0,
                "mrr_observado": float(1 / (aceitas[0] + 1)) if len(aceitas) else 0.0}

    def sorteio(self, pares):
        """Esperanças exatas de uma permutação uniforme dos mesmos candidatos."""
        metricas = []
        for _, ofertas in pares.groupby("id_cliente"):
            quantidade, positivas = len(ofertas), int(ofertas.aceitou.sum())
            descontos = 1 / np.log2(np.arange(2, min(quantidade, self.LIMITE) + 2))
            ideal = descontos[:positivas].sum()
            reciprocidade = (sum(comb(quantidade - posicao, positivas - 1) / comb(quantidade, positivas) / posicao
                                  for posicao in range(1, quantidade - positivas + 2)) if positivas else 0.0)
            metricas.append({"acerto_primeira_oferta_todos": positivas / quantidade,
                             "ndcg_3_observado": float(positivas / quantidade * descontos.sum() / ideal) if ideal else 0.0,
                             "mrr_observado": reciprocidade})
        return {**pd.DataFrame(metricas).mean().to_dict(), "clientes_avaliados": len(metricas)}


class SeletorDePeso:
    """Maximiza o acerto condicional da primeira oferta na validação.

    Desempates previamente definidos: maior AUC e depois peso mais próximo de 0,5.
    Os extremos permitem verificar se combinar realmente ajuda na validação.
    """

    def executar(self, pesos, floresta, colaborativa, pares, avaliador):
        comparacao = pd.DataFrame([
            {"peso_random_forest": peso,
             **avaliador.avaliar("Candidato", peso * floresta + (1 - peso) * colaborativa, pares)}
            for peso in pesos
        ])
        ordenada = (comparacao.assign(distancia_da_media=(comparacao.peso_random_forest - 0.5).abs())
                    .sort_values(["acerto_da_primeira_oferta", "auc", "distancia_da_media", "peso_random_forest"],
                                 ascending=[False, False, True, True]))
        return float(ordenada.iloc[0].peso_random_forest), comparacao.drop(columns="modelo")


class ComparadorPareado:
    """IC percentil de 95% da diferença de acerto (híbrido − RF), por cliente elegível.

    O intervalo é condicional aos modelos ajustados e a este cenário sintético; não mede
    a variação do treino nem corrige a escolha histórica feita com este mesmo conjunto.
    """

    def __init__(self, reamostragens, semente):
        if reamostragens < 1:
            raise ValueError("O bootstrap precisa de pelo menos uma reamostragem.")
        self.reamostragens = reamostragens
        self.semente = semente

    def executar(self, acertos_do_hibrido, acertos_da_floresta):
        if (not acertos_do_hibrido.index.is_unique or not acertos_da_floresta.index.is_unique
                or len(acertos_do_hibrido.index.difference(acertos_da_floresta.index))
                or len(acertos_da_floresta.index.difference(acertos_do_hibrido.index))):
            raise ValueError("A comparação precisa dos mesmos clientes, com um acerto por modelo e cliente.")
        acertos = pd.concat([acertos_do_hibrido.rename("hibrido"),
                             acertos_da_floresta.rename("floresta")], axis=1)
        if acertos.isna().any().any():
            raise ValueError("A comparação pareada não admite acertos ausentes.")
        if acertos.empty:
            raise ValueError("Não há clientes elegíveis para a comparação pareada.")
        diferencas = (acertos.hibrido - acertos.floresta).to_numpy(dtype=float)
        gerador = np.random.default_rng(self.semente)
        medias = np.array([gerador.choice(diferencas, len(diferencas), replace=True).mean()
                           for _ in range(self.reamostragens)])
        inferior, superior = np.quantile(medias, [0.025, 0.975])
        return {"comparacao": "híbrido selecionado − Random Forest",
                "diferenca_de_acerto": float(diferencas.mean()), "ic95_inferior": float(inferior),
                "ic95_superior": float(superior), "clientes_elegiveis": len(diferencas),
                "reamostragens": self.reamostragens, "semente": self.semente,
                "metodo": "bootstrap percentil pareado por cliente; intervalo de 95%",
                "evidencia_de_ganho": bool(inferior > 0)}


class DiagnosticoDePropensao:
    """Avalia o RF sem recalibrá-lo no teste. Brier/log loss não isolam calibração."""

    QUANTIDADE_DE_FAIXAS = 10

    def executar(self, respostas, probabilidades):
        dados = pd.DataFrame({"aceitou": np.asarray(respostas), "previsao": probabilidades})
        dados["faixa"] = pd.cut(dados.previsao, np.linspace(0, 1, self.QUANTIDADE_DE_FAIXAS + 1),
                                include_lowest=True)
        curva = (dados.groupby("faixa", observed=True)
                 .agg(ofertas=("aceitou", "size"), probabilidade_media=("previsao", "mean"),
                      frequencia_de_aceitacao=("aceitou", "mean")).reset_index())
        curva["faixa"] = curva.faixa.astype(str)
        metricas = {"brier": float(brier_score_loss(respostas, probabilidades)),
                    "log_loss": float(log_loss(respostas, probabilidades, labels=[0, 1])),
                    "observacao": "Diagnóstico no teste sintético; não comprova calibração em clientes reais."}
        return metricas, curva


class DiagnosticoDaClassificacao:
    """Diagnóstico por oferta no teste, com limiar fixado antes de consultar as respostas.

    As curvas comparam pontuações; a matriz e o relatório de classes pertencem apenas ao RF.
    Esses diagnósticos não selecionam peso, limiar nem parâmetros dos modelos.
    """

    CLASSES = (0, 1)
    ROTULOS = ("Recusa", "Aceitação")

    def __init__(self, limiar):
        if not 0 <= limiar <= 1:
            raise ValueError("O limiar da classificação deve estar entre 0 e 1.")
        self.limiar = float(limiar)

    def executar(self, respostas, notas_por_modelo, nome_do_random_forest):
        respostas = np.asarray(respostas)
        if respostas.ndim != 1 or set(np.unique(respostas)) != set(self.CLASSES):
            raise ValueError("O diagnóstico precisa de ofertas aceitas e recusadas, codificadas como 1 e 0.")
        if nome_do_random_forest not in notas_por_modelo:
            raise ValueError("Informe as previsões do Random Forest para calcular a matriz de confusão.")
        curvas = []
        for nome, valores in notas_por_modelo.items():
            notas = np.asarray(valores, dtype=float)
            if notas.shape != respostas.shape or not np.isfinite(notas).all():
                raise ValueError("Cada modelo deve ter uma pontuação finita para cada oferta do teste.")
            curvas.append({"modelo": nome, "roc_auc": float(roc_auc_score(respostas, notas)),
                           "average_precision": float(average_precision_score(respostas, notas))})
        notas_da_floresta = np.asarray(notas_por_modelo[nome_do_random_forest], dtype=float)
        if not ((notas_da_floresta >= 0) & (notas_da_floresta <= 1)).all():
            raise ValueError("A propensão do Random Forest deve estar entre 0 e 1.")
        previstas = (notas_da_floresta >= self.limiar).astype(int)
        relatorio = classification_report(respostas, previstas, labels=self.CLASSES,
                                         target_names=self.ROTULOS, output_dict=True, zero_division=0)
        return {"modelo": nome_do_random_forest, "limiar": self.limiar,
                "classes": list(self.CLASSES), "rotulos": list(self.ROTULOS),
                "ofertas_avaliadas": len(respostas), "aceitacoes_observadas": int(respostas.sum()),
                "prevalencia_de_aceitacao": float(respostas.mean()),
                "matriz_de_confusao": confusion_matrix(respostas, previstas, labels=self.CLASSES).tolist(),
                "acuracia": float(np.mean(respostas == previstas)),
                "relatorio_por_classe": [self._classe(rotulo, relatorio[rotulo]) for rotulo in self.ROTULOS],
                "media_macro": self._classe("Média macro", relatorio["macro avg"]),
                "media_ponderada": self._classe("Média ponderada", relatorio["weighted avg"]),
                "comparacao_de_curvas": curvas,
                "unidade_de_avaliacao": "oferta observada no ciclo de teste sintético",
                "criterio_do_limiar": "valor fixo de diagnóstico, não selecionado pelas respostas do teste",
                "interpretacao_de_ap": "average precision; não é a área trapezoidal da curva precisão–revocação"}

    @staticmethod
    def _classe(rotulo, valores):
        return {"classe": rotulo, "precisao": float(valores["precision"]),
                "sensibilidade": float(valores["recall"]), "f1": float(valores["f1-score"]),
                "suporte": int(valores["support"])}
