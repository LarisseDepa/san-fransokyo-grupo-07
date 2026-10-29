# /// script
# dependencies = [
#   "pandas==3.0.2; sys_platform == 'emscripten'",
#   "numpy==2.4.3; sys_platform == 'emscripten'",
#   "scipy==1.17.1; sys_platform == 'emscripten'",
#   "scikit-learn==1.8.0; sys_platform == 'emscripten'",
#   "matplotlib==3.10.8; sys_platform == 'emscripten'",
#   "skrub==0.11.0; sys_platform == 'emscripten'",
#   "openpyxl==3.1.5; sys_platform == 'emscripten'",
# ]
# [tool.marimo.runtime]
# # O relatório estatístico detalhado da união, a tabela completa e os gráficos passam de 8 MB (o limite
# # padrão do marimo) dentro do notebook; sem isto, aparece "Your output is too large".
# output_max_bytes = 30_000_000
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="CLARO GUIA - etapas 0 a 8")


@app.cell
async def _():
    import sys
    import io
    import zipfile
    from pathlib import Path

    import marimo as mo

    no_navegador = sys.platform == "emscripten"
    if no_navegador:
        from pyodide.http import pyfetch

        # Cada kernel recebe uma cópia dos módulos e dados publicados. Reexecutar
        # o notebook preserva edições; Restaurar cria um kernel com arquivos novos.
        RAIZ_DO_PROJETO = Path("/tmp/claro_guia")
        from urllib.parse import urlencode

        _marcador = RAIZ_DO_PROJETO / ".pacote_carregado"
        if not _marcador.exists():
            _versao = mo.query_params().get("v", "")
            _endereco = str(mo.notebook_location() / "public" / "projeto.zip") + "?" + urlencode({"v": _versao})
            _resposta = await pyfetch(_endereco)
            if not _resposta.ok:
                raise RuntimeError(f"Não foi possível carregar os arquivos do projeto: {_resposta.status}")
            with zipfile.ZipFile(io.BytesIO(await _resposta.bytes())) as _pacote:
                _pacote.extractall(RAIZ_DO_PROJETO)
            _marcador.touch()
    else:
        RAIZ_DO_PROJETO = Path(mo.notebook_dir()).parent
    sys.path.insert(0, str(RAIZ_DO_PROJETO / "src"))
    return RAIZ_DO_PROJETO, mo, no_navegador


@app.cell
def _(mo, no_navegador):
    aplicar_alteracoes_nos_modulos = mo.ui.run_button(label="Aplicar alterações nos arquivos Python")
    mo.vstack([
        mo.md("### Editar o projeto\n"
                  "As células abaixo usam classes de outros arquivos Python. Para editar essas implementações, "
                  "abra **Files (Arquivos) → codigo_do_projeto → src** depois da primeira execução. "
                  "Se a pasta não aparecer, use **↻ Atualizar** no painel de arquivos. "
                  "Abra um arquivo, edite e **salve**; em seguida, clique no botão abaixo. "
                  "Isso recarrega os módulos e recalcula as etapas dependentes. "
                  "**Restaurar versão original**, acima do editor, recupera células e arquivos."),
        aplicar_alteracoes_nos_modulos,
    ]) if no_navegador else mo.md("")
    return (aplicar_alteracoes_nos_modulos,)


@app.cell
def _(RAIZ_DO_PROJETO, aplicar_alteracoes_nos_modulos, no_navegador):
    from datetime import datetime
    from dataclasses import replace

    import pandas as pd
    from skrub import TableReport

    if no_navegador:
        # A leitura do botão cria a dependência reativa que reaplica os imports.
        aplicar_alteracoes_nos_modulos.value
        from edicao_de_modulos_no_navegador import EdicaoDeModulosNoNavegador as _EdicaoDeModulos

        _EdicaoDeModulos(RAIZ_DO_PROJETO).executar()

    import config
    from arquivos_da_etapa_8 import ArquivosDaEtapa8
    from arquivos_das_etapas_5_a_7 import ArquivosDasEtapas5a7
    from dicionario_de_dados import NOMES_CURTOS, DicionarioDeDados, exemplos_de_calculo
    if no_navegador:
        from eda_no_navegador import AnaliseExploratoriaNoNavegador as AnaliseExploratoria
    else:
        from eda import AnaliseExploratoria
    from enriquecimento import ConfiguracaoDoEnriquecimento, EnriquecimentoDaBase, colunas_novas
    from graficos import GraficosDoModelo
    from graficos_da_recomendacao import GraficosDaRecomendacao
    from historico_de_ofertas import ConfiguracaoDoHistorico, PlanosDasPlanilhas, SimuladorDeHistoricoDeOfertas
    from modelo_de_demanda import (ConfiguracaoDoModelo, ExportadorParaNavegador, PrevisorDeConsumo,
                                   TreinadorDoModelo, agora_em, versao_dos_dados)
    from modelo_de_recomendacao import ConfiguracaoDaRecomendacao, ExportadorDaPropensao, TreinadorDaRecomendacao
    from pastas_do_projeto import PastasDoProjeto
    from preparacao_dos_dados import (ArquivosDaPreparacao, CopiaDosOriginais, IndicadoresDeQualidade,
                                      LimpezaDePlanilha, UniaoDasPlanilhas)
    from recomendacao import CasosDemonstrativos
    from referencias_cientificas import BibliografiaDaRecomendacao
    from tabelas import COLUNA_ORIGEM, formatar_numero_br

    pastas = PastasDoProjeto(RAIZ_DO_PROJETO)
    arquivos_da_preparacao = ArquivosDaPreparacao(pastas)
    arquivos_das_etapas_5_a_7 = ArquivosDasEtapas5a7(pastas)
    arquivos_da_etapa_8 = ArquivosDaEtapa8(pastas)
    return (AnaliseExploratoria, BibliografiaDaRecomendacao, COLUNA_ORIGEM, CasosDemonstrativos, ConfiguracaoDaRecomendacao,
            ConfiguracaoDoEnriquecimento, ConfiguracaoDoHistorico, ConfiguracaoDoModelo, CopiaDosOriginais,
            DicionarioDeDados, EnriquecimentoDaBase, ExportadorDaPropensao, ExportadorParaNavegador,
            GraficosDaRecomendacao, GraficosDoModelo,
            IndicadoresDeQualidade, LimpezaDePlanilha, NOMES_CURTOS, PlanosDasPlanilhas, PrevisorDeConsumo,
            SimuladorDeHistoricoDeOfertas, TableReport, TreinadorDaRecomendacao, TreinadorDoModelo, UniaoDasPlanilhas,
            agora_em, arquivos_da_etapa_8, arquivos_da_preparacao, arquivos_das_etapas_5_a_7, colunas_novas, config,
            datetime, exemplos_de_calculo, formatar_numero_br, pastas, pd, replace, versao_dos_dados)


@app.cell
def _(config, mo, no_navegador):
    mo.vstack([
        mo.md(f"# {config.NOME_DO_PROJETO}: etapas 0 a 8\n"
              "TCC bootcamp Claro. Tudo roda a partir das planilhas originais, em código "
              "(configuração em `src/config.py`), e é reproduzível."),
        mo.callout(mo.md(
            "**Experimento no navegador.** Edite uma célula e execute-a para recalcular suas dependências. "
            "Os parâmetros das duas florestas estão na próxima célula. A execução completa pode levar "
            "alguns minutos. As alterações ficam neste ambiente de experimentação e não atualizam o site. "
            "Use **Restaurar versão original**, acima do editor, para recomeçar.\n\n"
            "A EDA portátil recalcula estatísticas e verificações de qualidade; o relatório completo de "
            "fg-data-profiling permanece nos resultados originais. As versões das bibliotecas do navegador "
            "podem produzir pequenas diferenças em relação ao ambiente local."), kind="neutral")
        if no_navegador else mo.md(""),
        mo.callout(mo.md(
            "**0. Cópia** dos originais → **1. Limpeza de cada planilha** (tudo é limpo antes da união) → "
            f"**2. EDA da planilha {config.PLANILHA_ANALISADA_SOZINHA.replace('_', ' ')}** → "
            "**3. União** das 4 planilhas limpas → **4. EDA da união**\n\n"
            "**Preparação dos dados** = limpeza (1) + união (3). **EDA** = análise exploratória de "
            "dados (2 e 4). **Alertas de qualidade** = avisos automáticos sobre os dados, com gravidade "
            "e ação sugerida; nem todo alerta é erro."), kind="neutral"),
        mo.callout(mo.md(
            "**5. Enriquecimento:** a base unida ganha dados **sintéticos** de contrato, franquia e consumo "
            "diário. → **6. Modelo:** um Random Forest aprende a prever quantos GB cada cliente ainda vai "
            "querer usar até a renovação. → **7. Recomendações:** a previsão é comparada com o saldo e com "
            "os planos das planilhas. → **8. Modelo de recomendação:** um histórico de ofertas **simulado** treina "
            "e compara dois modelos (Random Forest de propensão e filtragem colaborativa); a combinação ponderada "
            "com peso escolhido na validação ordena os planos na demonstração.\n\n"
            "Todos os dados novos são **fictícios**: as métricas mostram o que o modelo aprendeu da "
            "simulação, não a acurácia com clientes reais."), kind="warn"),
    ])
    return


@app.cell
def _(ConfiguracaoDaRecomendacao, ConfiguracaoDoModelo, config, mo, replace):
    # Edite os parâmetros abaixo e execute esta célula. O marimo recalcula os
    # modelos e os gráficos dependentes. Os valores iniciais preservam o projeto.
    parametros_da_demanda = {"n_estimators": 100, "min_samples_leaf": 5, "max_features": 0.6}
    parametros_da_propensao = {"n_estimators": 100, "min_samples_leaf": 20, "max_features": 0.6}
    configuracao_do_modelo = replace(
        ConfiguracaoDoModelo.a_partir_do_config(config), parametros_da_floresta=parametros_da_demanda)
    configuracao_da_recomendacao = replace(
        ConfiguracaoDaRecomendacao.a_partir_do_config(config), parametros_da_floresta=parametros_da_propensao)
    mo.md("### Parâmetros dos experimentos\n"
          "Altere `n_estimators`, `min_samples_leaf` ou `max_features` na célula acima. "
          "As métricas descrevem dados e ofertas simulados. Use a validação para comparar escolhas; "
          "ajustar repetidamente pelos resultados do teste compromete sua independência.")
    return configuracao_da_recomendacao, configuracao_do_modelo


@app.cell
def _(AnaliseExploratoria, TableReport, mo, no_navegador, pastas):
    def painel_da_eda(df, pasta_dos_relatorios, colunas_ignoradas=(), nota=""):
        """Faz a EDA (salvando os relatórios na pasta) e monta o painel com os resultados."""
        dados = df.drop(columns=[coluna for coluna in colunas_ignoradas if coluna in df])
        resultado = AnaliseExploratoria(dados).executar(pasta_dos_relatorios)
        alertas_por_gravidade = resultado.alertas.gravidade.value_counts()
        relatorio_detalhado = resultado.caminho_do_relatorio_detalhado.read_text(encoding="utf-8")
        return mo.vstack([
            mo.md(nota),
            mo.hstack([
                mo.stat(f"{dados.shape[0]} x {dados.shape[1]}", label="Linhas x colunas", bordered=True),
                mo.stat(int(dados.isna().sum().sum()), label="Valores ausentes", bordered=True),
                mo.stat(int(alertas_por_gravidade.get("alta", 0)), label="Alertas de gravidade alta", bordered=True),
                mo.stat(int(alertas_por_gravidade.get("media", 0)), label="Alertas de gravidade média", bordered=True),
                mo.stat(int(alertas_por_gravidade.get("baixa", 0)), label="Alertas de gravidade baixa", bordered=True),
            ], widths="equal"),
            mo.ui.tabs({
                "Alertas de qualidade": mo.ui.table(resultado.alertas, selection=None, page_size=15)
                if len(resultado.alertas) else mo.md("_Nenhum alerta de qualidade._"),
                "Estatísticas descritivas": mo.ui.table(resultado.estatisticas_descritivas.round(2), selection=None),
                "Descrição dos dados": mo.md("\n".join(f"- {frase}" for frase in resultado.descricao_dos_dados.frase)),
                "Tabela interativa (skrub)": mo.iframe(TableReport(dados, verbose=0).html(), height="800px"),
                ("Relatório da EDA portátil" if no_navegador else "Relatório estatístico detalhado"):
                    mo.iframe(relatorio_detalhado, height="900px"),
            }),
            mo.md(f"Relatórios salvos em `{pastas.relativo(pasta_dos_relatorios)}/`."),
        ])
    return (painel_da_eda,)


# ============================================================ 0. CÓPIA DOS ORIGINAIS
@app.cell
def _(CopiaDosOriginais, arquivos_da_preparacao, config, datetime, mo, pastas):
    _copia = CopiaDosOriginais(pastas, config)
    mo.stop(not _copia.listar_originais(),
            mo.callout(f"Coloque as planilhas em {pastas.originais}", kind="danger"))
    _resultado_da_copia = _copia.executar()
    arquivos_da_preparacao.salvar_conferencia_da_copia(_resultado_da_copia.conferencia)
    planilhas_brutas = _resultado_da_copia.planilhas_brutas

    mo.vstack([
        mo.md(f"## 0. Cópia dos originais\n{datetime.now(config.FUSO_HORARIO):%d/%m/%Y %H:%M} · "
              f"**{_resultado_da_copia.conferencia.linhas.sum()} linhas** em {len(planilhas_brutas)} planilhas. "
              f"O `sha256` é a impressão digital de cada original. Cópia em "
              f"`{pastas.relativo(pastas.copia_dos_originais)}/`; esta tabela em "
              f"`{pastas.relativo(pastas.conferencia_da_copia)}`."),
        mo.ui.table(_resultado_da_copia.conferencia, selection=None),
    ])
    return (planilhas_brutas,)


# ============================================================ 1. LIMPEZA DE CADA PLANILHA
@app.cell
def _(IndicadoresDeQualidade, LimpezaDePlanilha, arquivos_da_preparacao, config, mo, pastas, planilhas_brutas):
    planilhas_limpas, acoes_da_limpeza = {}, []
    for _nome, _planilha_bruta in planilhas_brutas.items():
        _limpeza = LimpezaDePlanilha(config, _nome)
        planilhas_limpas[_nome] = _limpeza.executar(_planilha_bruta)
        acoes_da_limpeza += _limpeza.registro.acoes
        arquivos_da_preparacao.salvar_planilha_limpa(_nome, planilhas_limpas[_nome])

    indicadores_de_qualidade = IndicadoresDeQualidade(config)

    def _aba_da_planilha(nome):
        comparacao = indicadores_de_qualidade.comparar(planilhas_brutas[nome], planilhas_limpas[nome])
        sem_pendencias = (comparacao.depois.iloc[1:] == 0).all()  # tudo zerado, menos o nº de linhas
        acoes_da_planilha = [acao for acao in acoes_da_limpeza if acao["planilha"] == nome]
        return mo.vstack([
            mo.callout("Planilha limpa: sem ausentes, vazios disfarçados, duplicadas, ids repetidos "
                       "ou números como texto.", kind="success") if sem_pendencias
            else mo.callout("Ainda há pendências nesta planilha.", kind="danger"),
            mo.ui.table(comparacao, selection=None),
            mo.md("**O que foi feito:**\n\n" + "\n".join(
                f"- **{acao['tipo_de_acao']}:** {acao['acao']}" for acao in acoes_da_planilha)),
        ])

    mo.vstack([
        mo.md("## 1. Limpeza de cada planilha\nCada planilha é padronizada e limpa **separadamente**, "
              f"antes da união. Resultado em `{pastas.relativo(pastas.planilhas_limpas)}/`."),
        mo.ui.tabs({_nome: _aba_da_planilha(_nome) for _nome in planilhas_limpas}),
    ])
    return acoes_da_limpeza, indicadores_de_qualidade, planilhas_limpas


# ============================================================ 2. EDA DE UMA PLANILHA
@app.cell
def _(config, mo, painel_da_eda, pastas, planilhas_limpas):
    _nome = config.PLANILHA_ANALISADA_SOZINHA
    mo.stop(_nome not in planilhas_limpas,
            mo.callout(f"Planilha '{_nome}' não encontrada. Opções: {', '.join(planilhas_limpas)}", kind="danger"))
    with mo.status.spinner(title=f"2. Fazendo a EDA de {_nome}..."):
        _painel = painel_da_eda(
            planilhas_limpas[_nome], pastas.eda_da_planilha(_nome),
            colunas_ignoradas=[config.COLUNA_GRUPO],
            nota=f"A coluna `{config.COLUNA_GRUPO}` fica de fora: numa planilha só ela tem sempre o mesmo valor.")
    mo.vstack([mo.md(f"## 2. EDA da planilha {_nome.replace('_', ' ')} (já limpa)"), _painel])
    return


# ============================================================ 3. UNIÃO
@app.cell
def _(UniaoDasPlanilhas, acoes_da_limpeza, arquivos_da_preparacao, config, mo, pastas, planilhas_limpas):
    _uniao_das_planilhas = UniaoDasPlanilhas(config)
    uniao = _uniao_das_planilhas.executar(planilhas_limpas)
    arquivos_da_preparacao.salvar_uniao(uniao, planilhas_limpas)
    arquivos_da_preparacao.salvar_registro(acoes_da_limpeza + _uniao_das_planilhas.registro.acoes)

    _diferencas = _uniao_das_planilhas.diferencas
    _linhas_das_planilhas = sum(len(planilha) for planilha in planilhas_limpas.values())
    mo.vstack([
        mo.md(f"## 3. União das {len(planilhas_limpas)} planilhas limpas\n**{len(uniao)} linhas** "
              f"(soma das planilhas: {_linhas_das_planilhas}). A coluna `origem` identifica a planilha de cada linha."),
        mo.callout(mo.md("**Diferenças entre as planilhas:**\n\n"
                         + "\n".join(f"- {diferenca}" for diferenca in _diferencas)),
                   kind="danger") if _diferencas
        else mo.callout("Planilhas compatíveis: mesmas colunas e tipos.", kind="success"),
        mo.md("\n".join(f"- {acao['acao']}" for acao in _uniao_das_planilhas.registro.acoes)),
        mo.md(f"Arquivos em `{pastas.relativo(pastas.pasta_da_uniao)}/`: `{pastas.uniao_csv.name}` e "
              f"`{pastas.uniao_excel.name}` (uma aba por planilha + aba **Uniao**). Tudo o que a preparação "
              f"fez (limpeza + união): `{pastas.relativo(pastas.registro_da_preparacao)}`."),
    ])
    return (uniao,)


# ============================================================ 4. EDA DA UNIÃO
@app.cell
def _(COLUNA_ORIGEM, config, mo, painel_da_eda, pastas, uniao):
    _colunas_numericas = [coluna for coluna in uniao.select_dtypes("number").columns if coluna != config.COLUNA_ID]
    _medias_por_grupo = uniao.groupby(config.COLUNA_GRUPO)[_colunas_numericas].mean().round(2).reset_index()
    _planos_por_grupo = (uniao.groupby([config.COLUNA_GRUPO, config.COLUNA_PLANO_ATUAL]).size()
                         .unstack(fill_value=0).reset_index()) if config.COLUNA_PLANO_ATUAL in uniao else None

    with mo.status.spinner(title="4. Fazendo a EDA da união..."):
        _painel = painel_da_eda(
            uniao, pastas.eda_da_uniao, colunas_ignoradas=[COLUNA_ORIGEM],
            nota=f"A coluna `{COLUNA_ORIGEM}` fica de fora da análise: ela repete a informação de "
                 f"`{config.COLUNA_GRUPO}`.")

    mo.vstack([
        mo.md("## 4. EDA da união"),
        mo.md(f"**Comparação entre os grupos** (média de cada coluna numérica por `{config.COLUNA_GRUPO}`):"),
        mo.ui.table(_medias_por_grupo, selection=None),
        mo.md("**Planos atuais por grupo** (quantidade de clientes):"),
        mo.ui.table(_planos_por_grupo, selection=None) if _planos_por_grupo is not None else mo.md(""),
        _painel,
    ])
    return


# ============================================================ 5. ENRIQUECIMENTO
@app.cell
def _(ConfiguracaoDoEnriquecimento, EnriquecimentoDaBase, config, mo, pastas, pd, uniao):
    # lê a união salva na etapa 3: com os tipos de coluna do CSV, o modelo é treinado sempre do mesmo jeito
    base_unida = pd.read_csv(pastas.uniao_csv)
    mo.stop(len(base_unida) != len(uniao),
            mo.callout(f"A união salva ({pastas.relativo(pastas.uniao_csv)}) não confere com a da etapa 3.",
                       kind="danger"))
    configuracao_do_enriquecimento = ConfiguracaoDoEnriquecimento.a_partir_do_config(config)
    resultado_do_enriquecimento = EnriquecimentoDaBase(configuracao_do_enriquecimento).executar(base_unida)
    base_final = resultado_do_enriquecimento.base_final
    return base_final, base_unida, configuracao_do_enriquecimento, resultado_do_enriquecimento


# ============================================================ 6. MODELO
@app.cell
def _(ExportadorParaNavegador, GraficosDoModelo, NOMES_CURTOS, PrevisorDeConsumo,
      TreinadorDoModelo, agora_em, arquivos_das_etapas_5_a_7, base_final, config, configuracao_do_modelo,
      pastas, versao_dos_dados):
    # a base é salva antes do treino: a versão do modelo leva a impressão digital deste arquivo
    pastas.pasta_da_base_enriquecida.mkdir(parents=True, exist_ok=True)
    base_final.to_csv(pastas.base_final, index=False)
    treino = TreinadorDoModelo(configuracao_do_modelo).treinar(
        base_final, versao_dos_dados(pastas.base_final), agora_em(config.FUSO_HORARIO))
    previsor = PrevisorDeConsumo(treino.floresta, treino.configuracao)
    modelo_para_o_navegador = ExportadorParaNavegador().exportar(treino.floresta, treino.registro)
    _graficos = GraficosDoModelo()
    graficos = {
        "previsto_vs_real": _graficos.previsto_vs_observado(treino.previsoes_no_teste, treino.configuracao.alvo,
                                                            "previsao_floresta_gb"),
        "distribuicao_dos_erros": _graficos.distribuicao_dos_erros(treino.previsoes_no_teste.residuo_floresta_gb),
        "importancia": _graficos.importancia(treino.importancia, NOMES_CURTOS),
    }
    arquivos_das_etapas_5_a_7.salvar_treino(treino, previsor, modelo_para_o_navegador, graficos)
    return graficos, modelo_para_o_navegador, previsor, treino


@app.cell
def _(DicionarioDeDados, arquivos_das_etapas_5_a_7, base_final, config, resultado_do_enriquecimento, treino):
    dicionario = DicionarioDeDados(treino.configuracao, config.DESCRICAO_DAS_COLUNAS)
    tabela_do_dicionario = dicionario.como_tabela(base_final.columns)
    problemas_do_dicionario = dicionario.problemas(base_final)
    arquivos_das_etapas_5_a_7.salvar_base(resultado_do_enriquecimento, tabela_do_dicionario)
    return problemas_do_dicionario, tabela_do_dicionario


# ============================================================ 8. HISTÓRICO DE OFERTAS E MODELO DE RECOMENDAÇÃO
@app.cell
def _(ConfiguracaoDoHistorico, ExportadorDaPropensao, GraficosDaRecomendacao, PlanosDasPlanilhas,
      SimuladorDeHistoricoDeOfertas, TreinadorDaRecomendacao, agora_em, arquivos_da_etapa_8, base_final, config,
      configuracao_da_recomendacao):
    # vem antes da etapa 7 no cálculo: a demonstração usa o modelo híbrido para ordenar os planos
    planos = PlanosDasPlanilhas.a_partir_da_base(base_final, config.FRANQUIAS_DOS_PLANOS_ACADEMICOS)
    configuracao_do_historico = ConfiguracaoDoHistorico.a_partir_do_config(config)
    historico_de_ofertas = SimuladorDeHistoricoDeOfertas(configuracao_do_historico, planos).executar(base_final)
    recomendacao_treinada = TreinadorDaRecomendacao(configuracao_da_recomendacao).treinar(
        base_final, historico_de_ofertas, planos, configuracao_do_historico, agora_em(config.FUSO_HORARIO))
    graficos_da_recomendacao = GraficosDaRecomendacao().executar(recomendacao_treinada)
    arquivos_da_etapa_8.salvar_historico(historico_de_ofertas)
    arquivos_da_etapa_8.salvar_treino(recomendacao_treinada, ExportadorDaPropensao().exportar(
        recomendacao_treinada.propensao.pipeline, recomendacao_treinada.registro), planos, graficos_da_recomendacao)
    return configuracao_do_historico, graficos_da_recomendacao, historico_de_ofertas, planos, recomendacao_treinada


# ============================================================ 7. RECOMENDAÇÕES
@app.cell
def _(CasosDemonstrativos, arquivos_das_etapas_5_a_7, base_final, config, planos, previsor, recomendacao_treinada):
    casos = CasosDemonstrativos(base_final, previsor, planos, recomendacao_treinada.hibrida, config)
    dados_da_demonstracao = casos.para_o_site(modelo_de_demanda_em="etapa6_modelo/modelo_demanda_rf.json",
                                              modelo_de_propensao_em="etapa8_recomendacao/modelo_propensao_rf.json")
    arquivos_das_etapas_5_a_7.salvar_demonstracao(dados_da_demonstracao)
    return (dados_da_demonstracao,)


# ============================================================ 5. EXIBIÇÃO
@app.cell
def _(base_final, base_unida, colunas_novas, configuracao_do_enriquecimento, dados_da_demonstracao,
      exemplos_de_calculo, mo, pastas, problemas_do_dicionario, tabela_do_dicionario):
    _exemplo = base_final[base_final.id_cliente == dados_da_demonstracao["cliente_do_exemplo"]["id_cliente"]].iloc[0]
    mo.vstack([
        mo.md("## 5. Enriquecimento e dicionário de dados\n"
              "A base unida ganhou dados sintéticos de contrato, franquia e consumo diário, com atributos, "
              "baseline e targets. **Nenhuma coluna de entrada foi alterada.**"),
        mo.hstack([
            mo.stat(f"{len(base_final)}", label="clientes (linhas)", bordered=True),
            mo.stat(f"{base_final.shape[1]}", label="colunas na base enriquecida", bordered=True),
            mo.stat(f"{base_unida.shape[1]}", label="colunas de entrada (11 originais + origem)", bordered=True),
            mo.stat(f"{len(colunas_novas(base_final))}", label="colunas novas", bordered=True),
        ], widths="equal"),
        mo.callout(mo.md(
            f"Data de referência: **{configuracao_do_enriquecimento.data_de_referencia:%d/%m/%Y}** · semente "
            f"**{configuracao_do_enriquecimento.semente}** (a mesma semente gera os mesmos dados). Hipótese: "
            "`dados_gb` é o consumo típico de um mês."), kind="info"),
        mo.callout("Dicionário completo: todas as colunas da base têm descrição." if not problemas_do_dicionario
                   else mo.md("**Dicionário incompleto:**\n\n" + "\n".join(f"- {p}" for p in problemas_do_dicionario)),
                   kind="success" if not problemas_do_dicionario else "danger"),
        mo.md(f"**Exemplo de cálculo com os valores reais** do cliente {int(_exemplo.id_cliente)} (a Cliente Claro da demonstração):"),
        mo.ui.table(exemplos_de_calculo(_exemplo), selection=None),
        mo.ui.tabs({
            "Dicionário de dados": mo.ui.table(tabela_do_dicionario, selection=None, page_size=15),
            "Base completa (960 linhas)": mo.ui.table(base_final, selection=None, page_size=15),
        }),
        mo.md(f"Arquivos: `{pastas.relativo(pastas.base_final)}`, `{pastas.relativo(pastas.dicionario_de_dados)}` "
              f"e `{pastas.relativo(pastas.consumo_diario_sintetico)}`."),
    ])
    return


# ============================================================ 6. EXIBIÇÃO
@app.cell
def _(NOMES_CURTOS, formatar_numero_br, graficos, mo, pastas, treino):
    _metricas = treino.metricas.copy()
    _floresta = _metricas[_metricas.modelo == "Random Forest"].iloc[0]
    _baseline = _metricas[_metricas.modelo.str.startswith("Baseline")].iloc[0]
    _melhor = "menor" if _floresta.mae_gb < _baseline.mae_gb else "maior"
    _variaveis = ", ".join(NOMES_CURTOS.get(variavel, variavel) for variavel in treino.configuracao.variaveis)
    mo.vstack([
        mo.md("## 6. Treinamento e avaliação do modelo"),
        mo.md("### O que prevemos?\nQuantos **GB de internet móvel** cada cliente ainda vai querer usar até a "
              "renovação (`target_demanda_restante_gb`). É uma **regressão**. O plano não entra no modelo: ele "
              "define quantos GB cabem, não o hábito."),
        mo.md(f"### Como treinamos?\n{len(treino.treino)} clientes no treino e {len(treino.teste)} no teste "
              "(separação por cliente, estratificada pelo perfil). Variáveis escolhidas por **validação cruzada "
              f"só no treino**: {_variaveis}. Faixa de idade e região passam por One-Hot dentro do pipeline."),
        mo.ui.table(treino.selecao_de_variaveis.round(3), selection=None),
        mo.md("### Quão bem funcionou na simulação?\n"
              f"No teste, o Random Forest errou em média **{formatar_numero_br(_floresta.mae_gb, 2)} GB** (MAE), "
              f"erro {_melhor} que o da projeção pela média diária (**{formatar_numero_br(_baseline.mae_gb, 2)} GB**). "
              f"O R² de {formatar_numero_br(_floresta.r2, 3)} indica quanto da variação da demanda o modelo explica (máximo 1)."),
        mo.ui.table(_metricas.round(3), selection=None),
        mo.ui.table(treino.metricas_por_perfil.round(2), selection=None),
        mo.hstack([mo.Html(graficos["previsto_vs_real"]), mo.Html(graficos["distribuicao_dos_erros"])], widths="equal"),
        mo.md("**Importância das variáveis** (quanto o erro piora ao embaralhar cada uma no teste):"),
        mo.Html(graficos["importancia"]),
        mo.md("### Limitações\n" + "\n".join(f"- {texto}" for texto in treino.registro["limitacoes"])),
        mo.md(f"Versão do modelo: `{treino.registro['versao_do_modelo']}` · treinado em "
              f"{treino.registro['treinado_em']} · arquivos em `{pastas.relativo(pastas.pasta_do_modelo)}/` "
              f"e `{pastas.relativo(pastas.avaliacao_do_modelo)}/`."),
    ])
    return


# ============================================================ 7. EXIBIÇÃO
@app.cell
def _(dados_da_demonstracao, formatar_numero_br, mo, pastas, pd):
    _exemplo = dados_da_demonstracao["cliente_do_exemplo"]
    _risco = _exemplo["risco"]
    _recomendacao = _exemplo["recomendacao"]
    mo.vstack([
        mo.md("## 7. Previsões, regras e recomendações\n"
              "**previsão da demanda → saldo → risco → planos que cobrem a demanda, na ordem da pontuação de recomendação "
              "(modelo da etapa 8).** A demanda é prevista uma vez; a mesma demanda é comparada com cada plano."),
        mo.md(f"### Caso Cliente Claro (cliente {_exemplo['id_cliente']}, modelo Random Forest)\n"
              f"Plano atual: {_exemplo['plano_atual']} (**{formatar_numero_br(_risco['capacidade_gb'], 0)} GB "
              f"principais**). Usou {formatar_numero_br(_exemplo['consumo_ciclo_atual_gb'])} GB em "
              f"{_exemplo['dias_decorridos']} dias; faltam {_exemplo['dias_ate_renovacao']} dias. Demanda prevista: "
              f"**{formatar_numero_br(_exemplo['demanda_prevista_gb'])} GB** (projeção pela média diária: "
              f"{formatar_numero_br(_exemplo['baseline_demanda_restante_gb'])} GB; na simulação, o futuro foi "
              f"{formatar_numero_br(_exemplo['target_demanda_restante_gb'])} GB). Saldo: "
              f"{formatar_numero_br(_risco['saldo_gb'])} GB → pode acabar cerca de "
              f"**{_risco['dias_de_antecipacao']} dias antes** da renovação."),
        mo.md(f"Necessidade de um ciclo: **{formatar_numero_br(_recomendacao['necessidade_gb'])} GB**. Planos que "
              "cobrem essa necessidade, na ordem da pontuação híbrida de recomendação (etapa 8):"),
        mo.ui.table(pd.DataFrame(_recomendacao["opcoes"]), selection=None),
        mo.md(f"Arquivo: `{pastas.relativo(pastas.dados_da_demonstracao)}`."),
    ])
    return


# ============================================================ 8. EXIBIÇÃO
@app.cell
def _(BibliografiaDaRecomendacao, config, configuracao_do_historico, formatar_numero_br, graficos_da_recomendacao,
      historico_de_ofertas, mo, pastas, pd, planos, recomendacao_treinada):
    _registro = recomendacao_treinada.registro
    _floresta = recomendacao_treinada.metricas.iloc[0]
    _colaborativa = recomendacao_treinada.metricas.iloc[1]
    _classificacao = recomendacao_treinada.classificacao_no_teste
    _matriz = pd.DataFrame(
        _classificacao["matriz_de_confusao"],
        index=["Recusa observada (0)", "Aceitação observada (1)"],
        columns=["Recusa prevista (0)", "Aceitação prevista (1)"],
    ).rename_axis("Resposta observada").reset_index()
    mo.vstack([
        mo.md("## 8. Histórico de ofertas (simulado) e modelo de recomendação\n"
              "As planilhas não têm histórico de ofertas aceitas ou recusadas. Aqui esse histórico é **simulado** "
              "por uma regra (o \"gosto\" de cada grupo, em `src/config.py`) e usado para treinar e comparar dois "
              "modelos de recomendação."),
        mo.callout("Os modelos aprendem a regra da simulação: as métricas mostram o quanto cada um recupera essa "
                   "regra, não a preferência de clientes reais.", kind="warn"),
        mo.md("### Os 8 planos das planilhas (fictícios)\nGB de `src/config.py`; mensalidade tirada das planilhas. "
              "Para comparar planos, contam só os GB principais."),
        mo.ui.table(planos.como_tabela(), selection=None),
        mo.md(f"### O histórico simulado\n{configuracao_do_historico.ciclos} ciclos × "
              f"{configuracao_do_historico.ofertas_por_ciclo} ofertas sorteadas por cliente = "
              f"**{len(historico_de_ofertas)} ofertas**, das quais "
              f"{formatar_numero_br(_registro['taxa_de_aceitacao'] * 100, 1)}% foram aceitas. "
              "% de ofertas aceitas, por grupo e plano:"),
        mo.ui.table(recomendacao_treinada.aceitacao_por_grupo.round(1), selection=None),
        mo.md("### Escolha do peso na validação\nOs ciclos anteriores treinam os modelos iniciais; o penúltimo "
              "seleciona o peso pelo acerto condicional da 1ª oferta. Desempates: maior AUC, proximidade de 0,5 e menor "
              "peso. Depois, os modelos são reajustados em treino + validação, antes de avaliar o último ciclo. "
              "A média 50/50 e os modelos isolados permanecem como referências."),
        mo.ui.table(recomendacao_treinada.comparacao_dos_pesos.round(4), selection=None),
        mo.md(f"Peso selecionado do Random Forest: **{recomendacao_treinada.hibrida.peso_random_forest:.0%}**. "
              "A pontuação ponderada não é uma probabilidade calibrada de aceitação."),
        mo.md(f"### Comparação no ciclo de teste\nSeparação {_registro['separacao']}. "
              f"O Random Forest teve AUC de **{formatar_numero_br(_floresta.auc, 3)}** e a 1ª oferta dele foi aceita em "
              f"**{formatar_numero_br(_floresta.acerto_da_primeira_oferta * 100, 1)}%** dos clientes elegíveis "
              f"(com aceitação e recusa no ciclo); a filtragem "
              f"colaborativa, {formatar_numero_br(_colaborativa.auc, 3)} e "
              f"{formatar_numero_br(_colaborativa.acerto_da_primeira_oferta * 100, 1)}%."),
        mo.ui.table(recomendacao_treinada.metricas.round(3), selection=None),
        mo.md("As colunas adicionais **NDCG@3 observado**, **MRR observado** e **acerto da primeira oferta: todos** "
              "avaliam todos os clientes do ciclo. NDCG e MRR valem zero para quem recusou todas. "
              "Todos os modelos recebem os mesmos candidatos observados; não se presume recusa para plano não ofertado. "
              "Os novos indicadores são diagnósticos e não alteram o critério anterior de seleção do peso."),
        mo.md("### Curvas ROC e precisão–revocação\n"
              f"Os gráficos de scikit-learn avaliam as **{_classificacao['ofertas_avaliadas']} ofertas** do último ciclo, "
              f"correspondentes a **{int(_floresta.clientes_avaliados)} clientes**; uma oferta é a unidade de classificação. "
              "A classe positiva é aceitação (1), e recusa é 0. As curvas comparam os mesmos dados e variam o limiar "
              "dos escores, sem reajustar os modelos ou selecionar um limiar pelo teste."),
        mo.Html(graficos_da_recomendacao["roc"]),
        mo.Html(graficos_da_recomendacao["precisao_revocacao"]),
        mo.ui.table(pd.DataFrame(_classificacao["comparacao_de_curvas"]).round(4), selection=None),
        mo.md("**ROC-AUC** resume a discriminação entre aceitação e recusa; 0,5 é a referência aleatória. "
              "**Precisão** é a proporção de aceitações entre as ofertas classificadas como aceitas; "
              "**revocação ou sensibilidade** é a proporção das aceitações observadas recuperada pelo modelo. "
              "**Average precision (AP)** resume a curva precisão–revocação por diferenças de revocação; "
              "não é ROC-AUC nem área trapezoidal dessa curva. A taxa de aceitação é a referência da curva PR."),
        mo.md("### Matriz de confusão do Random Forest\n"
              f"Limiar diagnóstico fixo **{formatar_numero_br(_classificacao['limiar'], 2)}**, definido antes desta "
              "avaliação: propensão igual ou maior que o limiar prevê aceitação. Ele não foi escolhido pelas respostas "
              "do teste e não altera a ordenação dos planos. Linhas indicam respostas observadas; colunas, previstas."),
        mo.Html(graficos_da_recomendacao["matriz_de_confusao"]),
        mo.ui.table(_matriz, selection=None),
        mo.md("Superior esquerda: **verdadeiros negativos** (recusa correta); superior direita: **falsos positivos** "
              "(previu aceitação, houve recusa). Inferior esquerda: **falsos negativos** (previu recusa, houve aceitação); "
              "inferior direita: **verdadeiros positivos** (aceitação correta)."),
        mo.ui.table(pd.DataFrame(_classificacao["relatorio_por_classe"]).round(4), selection=None),
        mo.md(f"**Acurácia:** {formatar_numero_br(_classificacao['acuracia'] * 100, 1)}% das ofertas corretamente "
              "classificadas nesse limiar. F1 combina precisão e sensibilidade; suporte é o número de respostas "
              "observadas de cada classe. Essas métricas não são o acerto da primeira oferta por cliente."),
        mo.md("### Importância das variáveis e uma árvore da floresta\n"
              "A importância por permutação mede a queda de ROC-AUC ao embaralhar cada variável no teste, "
              "mantendo o modelo treinado. As barras de erro indicam um desvio-padrão em "
              f"{config.REPETICOES_DA_IMPORTANCIA_DA_OFERTA} repetições. "
              "Variáveis relacionadas podem dividir importância; o gráfico não demonstra "
              "causalidade nem seleciona variáveis pelas respostas do teste."),
        mo.Html(graficos_da_recomendacao["importancia"]),
        mo.Html(graficos_da_recomendacao["arvore"]),
        mo.md(f"A árvore exibida é **uma das {_registro['parametros']['n_estimators']} árvores** do Random Forest, "
              f"com visualização cortada na profundidade {config.PROFUNDIDADE_DA_ARVORE_ILUSTRATIVA}. "
              "As amostras e proporções dos nós descrevem os dados de ajuste, na ordem [recusa, aceitação]. "
              "É uma ilustração do modelo ajustado; a previsão final "
              "combina as probabilidades de todas as árvores e não segue apenas esse desenho."),
        mo.md("A filtragem colaborativa é usuário a usuário, como no vídeo de referência: correlação de Pearson entre "
              "as notas (0 a 10, simuladas) de dois clientes e média ponderada das notas dos clientes parecidos. Ela "
              "precisa das notas do próprio cliente; para quem não tem histórico, usa a média das notas do grupo dele. "
              "Na demonstração, a pontuação ponderada ordena os planos, sem interpretação como probabilidade."),
        mo.md("### Incerteza no teste\nBootstrap percentil de 95%, pareado por cliente elegível, da diferença "
              "de acerto (híbrido selecionado − RF). Intervalo condicional aos modelos e ao cenário sintético; "
              "se incluir zero, não sustenta uma afirmação de ganho."),
        mo.ui.table([recomendacao_treinada.comparacao_pareada], selection=None),
        mo.md("### Diagnóstico da propensão do RF\nPrevisão média e frequência observada por faixa no teste "
              "sintético. Brier e log loss avaliam qualidade probabilística, mas não isolam calibração. "
              "Não houve recalibração com respostas do teste."),
        mo.ui.table([recomendacao_treinada.diagnostico_de_propensao], selection=None),
        mo.ui.table(recomendacao_treinada.curva_de_calibracao, selection=None),
        mo.md("### Limitações do experimento\n" + "\n".join(f"- {texto}" for texto in _registro["limitacoes"])),
        mo.md("### Referências científicas verificadas\n\n"
              + BibliografiaDaRecomendacao().como_markdown()),
        mo.md("JIF de 2025: ACM TOIS **11,2** "
              "([fonte ACM](https://www.acm.org/media-center/2026/july/impact-factors-2026)); "
              "IEEE TPAMI **20,4** ([fonte IEEE](https://www.computer.org/publications)). "
              "Indicadores conferidos em 29/10/2026; são da revista, não do artigo. "
              "Autores, DOI, datas e alcance da aplicação ficam no registro de referências verificadas."),
        mo.md(f"Arquivos: `{pastas.relativo(pastas.historico_de_ofertas)}`, "
              f"`{pastas.relativo(pastas.pasta_do_modelo_de_recomendacao)}/` e "
              f"`{pastas.relativo(pastas.avaliacao_da_recomendacao)}/`."),
    ])
    return

if __name__ == "__main__":
    app.run()
