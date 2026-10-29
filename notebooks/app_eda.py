# /// script
# [tool.marimo.runtime]
# # O relatório estatístico detalhado da união passa de 8 MB (o limite padrão do marimo)
# # dentro do notebook; sem isto, aparece "Your output is too large".
# output_max_bytes = 30_000_000
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="EDA - análise exploratória de dados")


@app.cell
def _():
    import io
    import sys
    from pathlib import Path

    import marimo as mo
    from skrub import TableReport

    RAIZ_DO_PROJETO = Path(mo.notebook_dir()).parent  # uma pasta acima de notebooks/
    sys.path.insert(0, str(RAIZ_DO_PROJETO / "src"))   # o código fica em src/

    from eda import AnaliseExploratoria
    from leitura_de_planilhas import LeitorDePlanilhas
    from pastas_do_projeto import PastasDoProjeto
    from tabelas import linhas_duplicadas

    pastas = PastasDoProjeto(RAIZ_DO_PROJETO)
    leitor = LeitorDePlanilhas()
    return AnaliseExploratoria, Path, TableReport, io, leitor, linhas_duplicadas, mo, pastas


@app.cell
def _(mo, pastas):
    mo.md(f"""
    # EDA: análise exploratória de dados
    Envie **uma ou várias planilhas** (várias são unidas com a coluna `origem`),
    **ou** informe uma pasta (já vem `{pastas.relativo(pastas.copia_dos_originais)}`) ou caminhos separados por `;`.
    """)
    return


@app.cell
def _(leitor, mo, pastas):
    arquivos_enviados = mo.ui.file(filetypes=list(leitor.EXTENSOES), multiple=True, label="Enviar planilhas")
    campo_de_caminhos = mo.ui.text(value=pastas.relativo(pastas.copia_dos_originais),
                                   placeholder="dados/etapa1_planilhas_limpas  ou  arquivo1.xlsx; arquivo2.xlsx",
                                   label="ou pasta / caminhos dos arquivos", full_width=True)
    mo.hstack([arquivos_enviados, campo_de_caminhos], widths=[1, 3], align="end")
    return arquivos_enviados, campo_de_caminhos


@app.cell
def _(Path, arquivos_enviados, campo_de_caminhos, io, leitor, mo, pastas):
    if arquivos_enviados.value:
        _escolhidas = [(arquivo.name, io.BytesIO(arquivo.contents)) for arquivo in arquivos_enviados.value]
    elif campo_de_caminhos.value.strip():
        # caminhos relativos partem da raiz do projeto (um caminho absoluto fica como está)
        _caminhos = [pastas.raiz / caminho.strip() for caminho in campo_de_caminhos.value.split(";")
                     if caminho.strip()]
        _nao_encontrados = [str(caminho) for caminho in _caminhos if not caminho.exists()]
        mo.stop(bool(_nao_encontrados),
                mo.callout(f"Arquivo não encontrado: {', '.join(_nao_encontrados)}", kind="danger"))
        _escolhidas = [(caminho, caminho) for caminho in leitor.listar(_caminhos)]
    else:
        _escolhidas = []

    mo.stop(not _escolhidas, mo.callout("Escolha um ou mais arquivos para começar.", kind="info"))

    if len(_escolhidas) > 1:
        juncao = leitor.juntar(_escolhidas)
        pastas.app_eda.mkdir(parents=True, exist_ok=True)
        juncao.tabela.to_csv(pastas.uniao_explorada_no_app, index=False)
        dados = juncao.tabela
        nome_dos_dados = (f"{len(_escolhidas)} planilhas unidas "
                          f"(salvo em {pastas.relativo(pastas.uniao_explorada_no_app)})")
    else:
        _nome_do_arquivo, _arquivo = _escolhidas[0]
        juncao = None
        dados = leitor.ler(_arquivo, _nome_do_arquivo)
        nome_dos_dados = Path(str(_nome_do_arquivo)).name
    return dados, juncao, nome_dos_dados


@app.cell
def _(juncao, mo):
    _avisos = []
    if juncao is not None:
        _avisos.append(mo.callout(mo.md(juncao.resumo), kind="success"))
    if juncao is not None and juncao.avisos:
        _lista = "\n".join(f"- {aviso}" for aviso in juncao.avisos)
        _avisos.append(mo.callout(mo.md(
            "**Atenção: as planilhas não estão iguais.** Nomes de colunas diferentes: corrija nas cópias e "
            f"envie de novo. Tipos diferentes: o preparacao_dos_dados.py trata.\n\n{_lista}"), kind="warn"))
    mo.vstack(_avisos)
    return


@app.cell
def _(AnaliseExploratoria, dados, mo, pastas):
    with mo.status.spinner(title="Fazendo a EDA..."):
        resultado_da_eda = AnaliseExploratoria(dados).executar(pastas.app_eda)  # salva os relatórios nessa pasta
    return (resultado_da_eda,)


@app.cell
def _(dados, linhas_duplicadas, mo, nome_dos_dados, resultado_da_eda):
    _alertas_por_gravidade = resultado_da_eda.alertas.gravidade.value_counts()
    mo.vstack([
        mo.md(f"**Arquivo:** `{nome_dos_dados}`"),
        mo.hstack([
            mo.stat(f"{dados.shape[0]} x {dados.shape[1]}", label="Linhas x colunas", bordered=True),
            mo.stat(int(linhas_duplicadas(dados).sum()), label="Linhas duplicadas", bordered=True),
            mo.stat(int(dados.isna().sum().sum()), label="Valores ausentes", bordered=True),
            mo.stat(len(resultado_da_eda.alertas), label="Alertas de qualidade", bordered=True),
            mo.stat(int(_alertas_por_gravidade.get("alta", 0)), label="Gravidade alta", bordered=True),
            mo.stat(int(_alertas_por_gravidade.get("media", 0)), label="Gravidade média", bordered=True),
            mo.stat(int(_alertas_por_gravidade.get("baixa", 0)), label="Gravidade baixa", bordered=True),
        ], widths="equal", wrap=True),
    ])
    return


@app.cell
def _(TableReport, dados, mo, resultado_da_eda):
    # calculados uma vez só (não refazem quando o filtro muda)
    tabela_interativa = mo.iframe(TableReport(dados, verbose=0).html(), height="800px")
    relatorio_detalhado = mo.iframe(
        resultado_da_eda.caminho_do_relatorio_detalhado.read_text(encoding="utf-8"), height="900px")
    return relatorio_detalhado, tabela_interativa


@app.cell
def _(mo):
    filtro_de_gravidade = mo.ui.multiselect(
        options=["alta", "media", "baixa"], value=["alta", "media", "baixa"], label="Gravidade"
    )
    return (filtro_de_gravidade,)


@app.cell
def _(filtro_de_gravidade, mo, relatorio_detalhado, resultado_da_eda, tabela_interativa):
    _alertas = resultado_da_eda.alertas
    _alertas_filtrados = _alertas[_alertas.gravidade.isin(filtro_de_gravidade.value)]

    mo.ui.tabs({
        "Alertas de qualidade": mo.vstack([filtro_de_gravidade,
                                           mo.ui.table(_alertas_filtrados, selection=None, page_size=20)]),
        "Estatísticas descritivas": mo.ui.table(resultado_da_eda.estatisticas_descritivas.round(2), selection=None),
        "Tabela interativa (skrub)": tabela_interativa,
        "Descrição dos dados": mo.md("\n".join(f"- {frase}" for frase in resultado_da_eda.descricao_dos_dados.frase)),
        "Relatório estatístico detalhado": relatorio_detalhado,
    })
    return


@app.cell
def _(mo):
    mo.md("""
    ## Documentação
    A descrição dos dados (aba acima) foi gerada automaticamente. Acrescente suas observações
    e baixe tudo em um arquivo.
    """)
    return


@app.cell
def _(mo):
    anotacoes = mo.ui.text_area(
        placeholder="Ex.: A coluna renda tem outliers porque inclui clientes empresariais.",
        label="Suas anotações", full_width=True, rows=6,
    )
    anotacoes
    return (anotacoes,)


@app.cell
def _(anotacoes, mo, nome_dos_dados, resultado_da_eda):
    _texto = f"Documentação do dataset: {nome_dos_dados}\n\nDescrição dos dados:\n"
    _texto += "\n".join(f"- {frase}" for frase in resultado_da_eda.descricao_dos_dados.frase)
    if anotacoes.value.strip():
        _texto += f"\n\nAnotações:\n{anotacoes.value.strip()}\n"

    mo.download(data=_texto.encode("utf-8"), filename="documentacao.txt",
                mimetype="text/plain", label="Baixar documentação")
    return


if __name__ == "__main__":
    app.run()
