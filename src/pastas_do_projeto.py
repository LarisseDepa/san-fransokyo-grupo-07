"""
Onde fica cada pasta e cada arquivo do projeto. É o único lugar com esses nomes:
para mudar um caminho, mude aqui.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PastasDoProjeto:
    """Caminhos do projeto, a partir da pasta raiz (a que tem o pixi.toml).
    O número em `etapaN` é a etapa do notebook notebooks/claro_guia_etapas_0_a_8.py."""

    raiz: Path

    # ---------------------------------------------------------------- código
    @property
    def notebook_completo(self):
        """Etapas 0 a 8: preparação dos dados, EDA, enriquecimento, modelo, recomendações e modelo de recomendação."""
        return self.raiz / "notebooks" / "claro_guia_etapas_0_a_8.py"

    # ---------------------------------------------------------------- dados
    @property
    def originais(self):
        """As planilhas recebidas. Nunca alterar, mover ou apagar."""
        return self.raiz / "dados" / "originais"

    @property
    def copia_dos_originais(self):
        return self.raiz / "dados" / "etapa0_copia_dos_originais"

    @property
    def planilhas_limpas(self):
        return self.raiz / "dados" / "etapa1_planilhas_limpas"

    def planilha_limpa(self, nome_da_planilha):
        return self.planilhas_limpas / f"{nome_da_planilha}.csv"

    @property
    def pasta_da_uniao(self):
        return self.raiz / "dados" / "etapa3_uniao"

    @property
    def uniao_csv(self):
        return self.pasta_da_uniao / "uniao_planilhas_limpas.csv"

    @property
    def uniao_excel(self):
        return self.pasta_da_uniao / "uniao_planilhas_limpas.xlsx"

    @property
    def pasta_da_base_enriquecida(self):
        return self.raiz / "dados" / "etapa5_base_enriquecida"

    @property
    def base_final(self):
        """A base de modelagem: base unida + dados sintéticos, atributos, baseline e targets."""
        return self.pasta_da_base_enriquecida / "base_final_modelagem_claro.csv"

    @property
    def dicionario_de_dados(self):
        return self.pasta_da_base_enriquecida / "dicionario_de_dados.csv"

    @property
    def consumo_diario_sintetico(self):
        """Consumo diário simulado (uma linha por cliente e dia), para conferência."""
        return self.pasta_da_base_enriquecida / "consumo_diario_sintetico.csv"

    # ---------------------------------------------------------------- modelo
    @property
    def pasta_do_modelo(self):
        return self.raiz / "modelos" / "etapa6_modelo_de_demanda"

    @property
    def modelo_joblib(self):
        """Pipeline treinado para uso local em Python (não vai para o site)."""
        return self.pasta_do_modelo / "modelo_demanda_rf.joblib"

    @property
    def modelo_json(self):
        """Árvores do Random Forest em JSON, para o site prever no navegador."""
        return self.pasta_do_modelo / "modelo_demanda_rf.json"

    @property
    def registro_do_treino(self):
        return self.pasta_do_modelo / "registro_do_treino.json"

    # ---------------------------------------------------------------- etapa 8: histórico e recomendação
    @property
    def pasta_do_historico_de_ofertas(self):
        return self.raiz / "dados" / "etapa8_historico_de_ofertas"

    @property
    def historico_de_ofertas(self):
        """Histórico SIMULADO de ofertas (aceitas ou recusadas), uma linha por oferta."""
        return self.pasta_do_historico_de_ofertas / "historico_de_ofertas_simulado.csv"

    @property
    def pasta_do_modelo_de_recomendacao(self):
        return self.raiz / "modelos" / "etapa8_modelo_de_recomendacao"

    @property
    def modelo_de_propensao_joblib(self):
        """Pipeline treinado para uso local em Python (não vai para o site)."""
        return self.pasta_do_modelo_de_recomendacao / "modelo_propensao_rf.joblib"

    @property
    def modelo_de_propensao_json(self):
        """Árvores do Random Forest de propensão em JSON, para o site ordenar os planos no navegador."""
        return self.pasta_do_modelo_de_recomendacao / "modelo_propensao_rf.json"

    @property
    def registro_da_recomendacao(self):
        return self.pasta_do_modelo_de_recomendacao / "registro_da_recomendacao.json"

    # ---------------------------------------------------------------- relatórios
    @property
    def relatorios(self):
        return self.raiz / "relatorios"

    @property
    def conferencia_da_copia(self):
        return self.relatorios / "etapa0_conferencia_da_copia.csv"

    @property
    def registro_da_preparacao(self):
        return self.relatorios / "registro_da_preparacao_dos_dados.csv"

    def eda_da_planilha(self, nome_da_planilha):
        return self.relatorios / f"etapa2_eda_{nome_da_planilha}"

    @property
    def eda_da_uniao(self):
        return self.relatorios / "etapa4_eda_uniao"

    @property
    def avaliacao_do_modelo(self):
        return self.relatorios / "etapa6_avaliacao_do_modelo"

    @property
    def recomendacoes(self):
        return self.relatorios / "etapa7_recomendacoes"

    @property
    def dados_da_demonstracao(self):
        """Cliente Claro (o exemplo), cenários e planos: tudo o que a demonstração do site usa."""
        return self.recomendacoes / "dados_da_demonstracao.json"

    @property
    def avaliacao_da_recomendacao(self):
        return self.relatorios / "etapa8_avaliacao_da_recomendacao"

    @property
    def app_eda(self):
        return self.relatorios / "app_eda"

    @property
    def uniao_explorada_no_app(self):
        """Junção (sem limpeza) das planilhas abertas no app de EDA. Não é a união limpa."""
        return self.app_eda / "uniao_das_planilhas_exploradas.csv"

    # ---------------------------------------------------------------- site
    @property
    def site(self):
        """Site estático (GitHub Pages), montado por src/site_estatico.py."""
        return self.raiz / "site"

    # ---------------------------------------------------------------- apoio
    def relativo(self, caminho):
        """Caminho a partir da raiz, para mostrar na tela: 'dados/etapa3_uniao'."""
        return Path(caminho).relative_to(self.raiz).as_posix()
