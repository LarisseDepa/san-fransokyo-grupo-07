"""
Grava os resultados das etapas 5 a 7 nas pastas do projeto (ver PastasDoProjeto).
Os nomes dos arquivos da avaliação ficam aqui, num lugar só (o site também os usa).
"""
import json


class ArquivosDasEtapas5a7:
    """Salva a base enriquecida, o modelo treinado, a avaliação e os dados da demonstração."""

    METRICAS = "metricas_no_teste.csv"
    METRICAS_POR_PERFIL = "metricas_por_perfil.csv"
    SELECAO_DE_VARIAVEIS = "selecao_de_variaveis.csv"
    IMPORTANCIA = "importancia_das_variaveis.csv"
    PREVISOES_NO_TESTE = "previsoes_no_teste.csv"
    GRAFICOS = {"previsto_vs_real": "grafico_previsto_vs_real.svg",
                "distribuicao_dos_erros": "grafico_distribuicao_dos_erros.svg",
                "importancia": "grafico_importancia_das_variaveis.svg"}

    def __init__(self, pastas):
        self.pastas = pastas

    def salvar_base(self, resultado_do_enriquecimento, dicionario):
        self.pastas.pasta_da_base_enriquecida.mkdir(parents=True, exist_ok=True)
        resultado_do_enriquecimento.base_final.to_csv(self.pastas.base_final, index=False)
        dicionario.to_csv(self.pastas.dicionario_de_dados, index=False)
        diario = resultado_do_enriquecimento.consumo_diario.assign(
            data=resultado_do_enriquecimento.consumo_diario.data.dt.strftime("%Y-%m-%d"))
        diario.to_csv(self.pastas.consumo_diario_sintetico, index=False)

    def salvar_treino(self, resultado_do_treino, previsor, modelo_json, graficos):
        self.pastas.pasta_do_modelo.mkdir(parents=True, exist_ok=True)
        previsor.salvar(self.pastas.modelo_joblib)
        self._json(self.pastas.modelo_json, modelo_json, compacto=True)
        self._json(self.pastas.registro_do_treino, resultado_do_treino.registro)

        pasta = self.pastas.avaliacao_do_modelo
        pasta.mkdir(parents=True, exist_ok=True)
        resultado_do_treino.metricas.to_csv(pasta / self.METRICAS, index=False)
        resultado_do_treino.metricas_por_perfil.to_csv(pasta / self.METRICAS_POR_PERFIL, index=False)
        resultado_do_treino.selecao_de_variaveis.to_csv(pasta / self.SELECAO_DE_VARIAVEIS, index=False)
        resultado_do_treino.importancia.to_csv(pasta / self.IMPORTANCIA, index=False)
        resultado_do_treino.previsoes_no_teste.to_csv(pasta / self.PREVISOES_NO_TESTE, index=False)
        for nome, svg in graficos.items():
            (pasta / self.GRAFICOS[nome]).write_text(svg, encoding="utf-8")

    def salvar_demonstracao(self, dados):
        self.pastas.recomendacoes.mkdir(parents=True, exist_ok=True)
        self._json(self.pastas.dados_da_demonstracao, dados)

    @staticmethod
    def _json(caminho, conteudo, compacto=False):
        texto = (json.dumps(conteudo, ensure_ascii=False, separators=(",", ":")) if compacto
                 else json.dumps(conteudo, ensure_ascii=False, indent=1))
        caminho.write_text(texto, encoding="utf-8")
