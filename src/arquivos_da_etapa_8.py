"""
Grava os resultados da etapa 8 nas pastas do projeto (ver PastasDoProjeto).
Os nomes dos arquivos da avaliação ficam aqui, num lugar só (o site também os usa).
"""
import json

import pandas as pd

from referencias_cientificas import BibliografiaDaRecomendacao


class ArquivosDaEtapa8:
    """Salva o histórico de ofertas simulado, o modelo de propensão e a comparação dos modelos."""

    METRICAS = "metricas_da_recomendacao.csv"
    ACEITACAO_POR_GRUPO = "aceitacao_por_grupo_e_plano.csv"
    PLANOS = "planos_das_planilhas.csv"
    PESOS = "comparacao_dos_pesos_na_validacao.csv"
    INCERTEZA = "comparacao_pareada_no_teste.json"
    CALIBRACAO = "calibracao_da_propensao_no_teste.csv"
    REFERENCIAS = "referencias_cientificas_verificadas.json"
    CLASSIFICACAO = "classificacao_no_teste.json"
    MATRIZ_DE_CONFUSAO = "matriz_de_confusao.csv"
    RELATORIO_DE_CLASSIFICACAO = "relatorio_de_classificacao.csv"
    CURVAS = "comparacao_das_curvas.csv"
    GRAFICOS = {"roc": "curva_roc.svg", "precisao_revocacao": "curva_precisao_revocacao.svg",
                "matriz_de_confusao": "matriz_de_confusao.svg", "importancia": "importancia_por_permutacao.svg",
                "arvore": "arvore_ilustrativa.svg"}

    def __init__(self, pastas):
        self.pastas = pastas

    def salvar_historico(self, historico):
        self.pastas.pasta_do_historico_de_ofertas.mkdir(parents=True, exist_ok=True)
        historico.to_csv(self.pastas.historico_de_ofertas, index=False)

    def salvar_treino(self, resultado, modelo_json, planos, graficos=None):
        self.pastas.pasta_do_modelo_de_recomendacao.mkdir(parents=True, exist_ok=True)
        resultado.propensao.salvar(self.pastas.modelo_de_propensao_joblib)
        self._json(self.pastas.modelo_de_propensao_json, modelo_json, compacto=True)
        self._json(self.pastas.registro_da_recomendacao, resultado.registro)

        pasta = self.pastas.avaliacao_da_recomendacao
        pasta.mkdir(parents=True, exist_ok=True)
        resultado.metricas.to_csv(pasta / self.METRICAS, index=False)
        resultado.comparacao_dos_pesos.to_csv(pasta / self.PESOS, index=False)
        resultado.curva_de_calibracao.to_csv(pasta / self.CALIBRACAO, index=False)
        self._json(pasta / self.INCERTEZA, resultado.comparacao_pareada)
        self._json(pasta / self.REFERENCIAS, BibliografiaDaRecomendacao().como_registro())
        resultado.aceitacao_por_grupo.to_csv(pasta / self.ACEITACAO_POR_GRUPO, index=False)
        planos.como_tabela().to_csv(pasta / self.PLANOS, index=False)
        self._salvar_classificacao(pasta, resultado.classificacao_no_teste)
        for nome, svg in (graficos or {}).items():
            (pasta / self.GRAFICOS[nome]).write_text(svg, encoding="utf-8")

    def _salvar_classificacao(self, pasta, classificacao):
        self._json(pasta / self.CLASSIFICACAO, classificacao)
        matriz = pd.DataFrame(classificacao["matriz_de_confusao"],
                              index=pd.Index(classificacao["rotulos"], name="classe_observada"),
                              columns=[f"Previsto: {rotulo}" for rotulo in classificacao["rotulos"]])
        matriz.to_csv(pasta / self.MATRIZ_DE_CONFUSAO)
        linhas = [*classificacao["relatorio_por_classe"], classificacao["media_macro"], classificacao["media_ponderada"]]
        pd.DataFrame(linhas).to_csv(pasta / self.RELATORIO_DE_CLASSIFICACAO, index=False)
        pd.DataFrame(classificacao["comparacao_de_curvas"]).to_csv(pasta / self.CURVAS, index=False)

    @staticmethod
    def _json(caminho, conteudo, compacto=False):
        texto = (json.dumps(conteudo, ensure_ascii=False, separators=(",", ":")) if compacto
                 else json.dumps(conteudo, ensure_ascii=False, indent=1))
        caminho.write_text(texto, encoding="utf-8")
