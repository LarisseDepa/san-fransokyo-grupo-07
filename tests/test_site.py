"""Site montado (pasta site/): estrutura e links. Com o Chrome disponível, abre o site sem janela e confere
a demonstração, a equivalência JavaScript x Python e o layout em 320, 375, 768, 1024 e 1440 px."""
import functools
import html
import json
import re
import shutil
import subprocess
import threading
from datetime import date
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from modelo_de_demanda import FlorestaExportada
from modelo_de_recomendacao import PrevisorDePropensao
from arquivos_da_etapa_8 import ArquivosDaEtapa8
from recomendacao import AnalisadorDeRisco, AtributosDaSimulacao
from tabelas import formatar_numero_br as numero

CAMINHOS_DO_CHROME = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
LARGURAS = [320, 375, 768, 1024, 1440]
PALAVRAS_PROIBIDAS_NA_MENSAGEM = ["viag", "foto", "rede social", "redes sociais", "localiza", "aplicativo", "trabalho"]


def _chrome():
    for caminho in CAMINHOS_DO_CHROME:
        encontrado = shutil.which(caminho) or (caminho if Path(caminho).exists() else None)
        if encontrado:
            return encontrado
    return None


@pytest.fixture(scope="module")
def site(pastas):
    if not (pastas.site / "index.html").exists():
        pytest.skip("Monte o site antes: pixi run site")
    return pastas.site


@pytest.fixture(scope="module")
def pagina(site):
    """A página inicial: a demonstração e "Como fizemos isso?"."""
    return (site / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def detalhes(site):
    """A página "Detalhes sobre o projeto": os dados, a união, o passo a passo e as etapas 5 a 8."""
    return (site / "detalhes_do_projeto.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ estrutura (sem navegador)
def test_secoes_e_ancoras(pagina, detalhes):
    assert 'id="experimente"' in pagina and 'id="dados"' not in pagina
    for ancora in ["dados", "uniao", "passo-a-passo", "enriquecimento", "modelo", "recomendacoes",
                   "modelo-de-recomendacao", "simulacao", "limites", "referencias"]:
        assert f'id="{ancora}"' in detalhes
    for etapa in range(9):
        assert f"Etapa {etapa}" in detalhes
    assert detalhes.index('id="recomendacoes"') < detalhes.index('id="simulacao"') < detalhes.index('id="modelo"')
    assert detalhes.index('id="dados"') < detalhes.index('id="enriquecimento"') < detalhes.index('id="passo-a-passo"')
    for conteudo in [pagina, detalhes]:
        assert 'id="termos"' not in conteudo and "<footer" not in conteudo


def test_avisos_de_simulacao_visiveis(site, pagina):
    # o aviso fica no rodapé do cartão do exemplo (Cliente Claro), escrito pelo JavaScript (o teste no navegador confere que aparece)
    script = (site / "claro_guia.js").read_text(encoding="utf-8")
    assert "Não acessamos sua conta Claro, não enviamos WhatsApp e este não é um serviço oficial" in script
    assert "<b>Simulação acadêmica.</b>" in pagina  # também sem JavaScript (noscript)


def test_protocolo_e_pontuacao_sem_promessa_de_probabilidade(site, detalhes):
    script = (site / "claro_guia.js").read_text(encoding="utf-8")
    assert "Pontuação de recomendação:" in script
    assert "Chance estimada de aceitar:" not in script
    assert 'id="protocolo-cientifico"' in detalhes
    for texto in ["peso escolhido na validação", "IC de 95%", "clientes", "DaisyRec 2.0", "NDCG@3 observado",
                  "comparacao_dos_pesos_na_validacao.csv", "calibracao_da_propensao_no_teste.csv"]:
        assert texto in detalhes
    dados = json.loads((site / "etapa7_recomendacoes" / "dados_da_demonstracao.json").read_text(encoding="utf-8"))
    registro = json.loads((site / "etapa8_recomendacao" / "registro_da_recomendacao.json").read_text(encoding="utf-8"))
    assert dados["peso_random_forest"] == registro["selecao_do_peso"]["peso_random_forest"]
    for doi_anterior in ["10.1023/A:1021240730564", "10.1007/s10791-020-09371-3", "10.1002/aaai.12051"]:
        assert doi_anterior not in detalhes
    referencias = json.loads((site / "etapa8_recomendacao" / "referencias_cientificas_verificadas.json")
                              .read_text(encoding="utf-8"))
    assert referencias == registro["bibliografia_cientifica"]
    for artigo in referencias["artigos"]:
        assert html.escape(artigo["fonte_da_versao_final"], quote=True) in detalhes
        assert html.escape(artigo["aplicacao_no_projeto"]) in detalhes


def test_classificacao_publica_graficos_e_contagens_consistentes(site, detalhes):
    pasta = site / "etapa8_recomendacao"
    classificacao = json.loads((pasta / ArquivosDaEtapa8.CLASSIFICACAO).read_text(encoding="utf-8"))
    assert 'id="graficos-da-recomendacao"' in detalhes
    assert sum(map(sum, classificacao["matriz_de_confusao"])) == classificacao["ofertas_avaliadas"]
    assert len(classificacao["comparacao_de_curvas"]) == 3
    for arquivo in ArquivosDaEtapa8.GRAFICOS.values():
        svg = (pasta / arquivo).read_text(encoding="utf-8")
        assert svg in detalhes and f'href="etapa8_recomendacao/{arquivo}"' in detalhes
    for arquivo in [ArquivosDaEtapa8.MATRIZ_DE_CONFUSAO, ArquivosDaEtapa8.RELATORIO_DE_CLASSIFICACAO,
                    ArquivosDaEtapa8.CURVAS]:
        assert (pasta / arquivo).exists() and f'href="etapa8_recomendacao/{arquivo}"' in detalhes


def test_sem_campos_do_modelo_sem_preencher(pagina, detalhes):
    for conteudo in [pagina, detalhes]:
        assert [campo for campo in re.findall(r"\$[a-z_]+", conteudo) if campo != "$nome"] == []


def test_links_locais_existem(site, pagina, detalhes):
    for conteudo in [pagina, detalhes]:
        links = re.findall(r'(?:href|src)="([^"#:?]+)(?:\?[^"]*)?"', conteudo)
        links = [link for link in links if not link.startswith(("data:", "https:", "http:"))]
        assert links
        assert [link for link in links if not (site / link).exists()] == []


def test_script_tem_versao_do_conteudo_para_atualizar_o_cache(site, pagina):
    versao = sha256((site / "claro_guia.js").read_bytes()).hexdigest()
    assert f'<script src="claro_guia.js?v={versao}" defer></script>' in pagina


def test_acesso_discreto_a_documentacao(pagina, detalhes):
    caixa = pagina[pagina.index('<aside class="comece">'):pagina.index("</aside>")]
    assert "Como fizemos isso?" in caixa and "Primeira vez aqui?" not in pagina
    assert re.findall(r'<a\b[^>]*href="([^"]+)"', caixa) == ["detalhes_do_projeto.html"]
    assert 'class="saiba-mais"' in caixa
    assert 'href="notebook_editavel.html"' not in pagina
    for destino in ["notebook_editavel.html", "claro_guia_etapas_0_a_8.html?show-code=false"]:
        assert f'href="{destino}"' in detalhes
    assert "notebook_modelo.html" not in pagina


def test_notebook_editavel_publica_mesma_fonte_e_dados_originais(site, pastas, detalhes):
    from zipfile import ZipFile

    editor = (site / "editor" / "index.html").read_text(encoding="utf-8")
    wrapper = (site / "notebook_editavel.html").read_text(encoding="utf-8")
    assert '<marimo-code' in editor
    assert 'notebook_editavel.html' in detalhes
    assert 'data-editor="editor/index.html"' in wrapper
    assert 'Restaurar versão original' in wrapper
    assert 'fg-data-profiling' in detalhes
    with ZipFile(site / "editor" / "public" / "projeto.zip") as pacote:
        entradas = pacote.namelist()
        assert not any(nome.endswith((".joblib", ".pkl")) for nome in entradas)
        originais = sorted(pastas.originais.glob("*.csv"))
        assert len(originais) == 4
        for original in originais:
            assert pacote.read(original.relative_to(pastas.raiz).as_posix()) == original.read_bytes()
        for modulo in (pastas.raiz / "src").glob("*.py"):
            assert pacote.read(f"src/{modulo.name}") == modulo.read_bytes()


def test_parametros_iniciais_do_notebook_preservam_os_modelos(pastas):
    import ast
    from modelo_de_demanda import ConfiguracaoDoModelo
    from modelo_de_recomendacao import ConfiguracaoDaRecomendacao

    arvore = ast.parse(pastas.notebook_completo.read_text(encoding="utf-8"))
    valores = {}
    for no in ast.walk(arvore):
        if isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
            if no.targets[0].id in ("parametros_da_demanda", "parametros_da_propensao"):
                valores[no.targets[0].id] = ast.literal_eval(no.value)
    assert valores["parametros_da_demanda"] == ConfiguracaoDoModelo().parametros_da_floresta
    assert valores["parametros_da_propensao"] == ConfiguracaoDaRecomendacao().parametros_da_floresta


def test_arquivos_da_eda(site):
    for pasta_da_eda in ["etapa2_eda_Excesso_de_Consumo", "etapa4_eda_uniao"]:
        assert sorted(arquivo.name for arquivo in (site / pasta_da_eda).iterdir()) == [
            "descricao_dos_dados.txt", "fg-data-profiling-relatorio-detalhado.html"]


def test_sem_resumo_da_eda(site, pagina, detalhes):
    assert not list(site.rglob("resumo_da_eda.html"))
    assert "Resumo da EDA" not in pagina and "Resumo da EDA" not in detalhes
    assert ('href="claro_guia_etapas_0_a_8.html?show-code=false" target="_blank" rel="noopener"' in detalhes)


def test_topo_enxuto_e_sem_dicionario_na_pagina(site, pagina, detalhes):
    cabecalho = pagina[pagina.index("<header>"):pagina.index("</header>")]
    assert ('<h1><span class="titulo-destaque">CLARO</span> <span class="titulo-resto">GUIA</span></h1>'
            in cabecalho)
    for retirado_da_pagina in ["Experimentar agora", "TCC · Bootcamp Claro", 'id="titulo-experimente"', "aviso-simulacao"]:
        assert retirado_da_pagina not in pagina
    for texto_retirado in ["Sua internet ainda acompanha", "Simulação acadêmica · não oficial", "colunas originais",
                           'class="sumario"']:
        assert texto_retirado not in cabecalho
    for conteudo in [pagina, detalhes]:
        assert "busca-no-dicionario" not in conteudo and "Dicionário de dados (" not in conteudo
    assert not list(site.rglob("dicionario_de_dados.csv"))


def test_documentacao_separa_implementacao_de_proximos_passos(detalhes):
    limites = re.sub(r"\s+", " ", detalhes[detalhes.index('id="limites"'):detalhes.index('id="referencias"')])
    assert "histórico de ofertas reais" in limites and "validação do fluxo completo" in limites
    assert "já foi consultado" in limites and "novas observações" in limites
    assert "<h3>Ações futuras</h3>" not in detalhes


def test_fonte_e_tamanhos_padronizados(pagina, detalhes):
    tamanhos = {"var(--tamanho-titulo-pagina)", "var(--tamanho-titulo-secao)", "var(--tamanho-subtitulo)",
                "var(--tamanho-texto)", "var(--tamanho-pequeno)"}
    for conteudo in [detalhes]:
        estilo = conteudo[conteudo.index("<style>"):conteudo.index("</style>")]
        assert set(re.findall(r"font-size:([^;}]+)", estilo)) == tamanhos
        assert set(re.findall(r"font-family:([^;}]+)", estilo)) == {"var(--fonte-texto)", "var(--fonte-codigo)"}
        assert "fonts.googleapis.com/css2?family=Inter" in conteudo
    # A home mantém as fontes e medidas da maquete; a documentação conserva seu estilo.
    assert 'family=Fira+Sans' in pagina and 'family=Barlow' in pagina
    assert 'family=Inter' not in pagina
    assert 'height:680px' in pagina


def test_fluxo_mostra_apenas_a_primeira_tela_antes_do_javascript(pagina):
    for numero_da_etapa in range(1, 8):
        abertura = re.search(rf'<section\b(?=[^>]*\bid="etapa-{numero_da_etapa}")[^>]*>', pagina).group()
        assert (' hidden' in abertura) == (numero_da_etapa != 1)
        assert (' inert' in abertura) == (numero_da_etapa != 1)
        assert ('aria-current="step"' in abertura) == (numero_da_etapa == 1)
    assert 'class="fluxo-navegacao"' in pagina
    assert 'data-acao="voltar-etapa" hidden' in pagina
    assert 'id="progresso-fluxo">Etapa 1 de 7<' in pagina


def test_sem_dicas_para_usar_menos_dados(site):
    script = (site / "claro_guia.js").read_text(encoding="utf-8")
    assert "Dicas para usar menos dados" not in script and "dicas de economia" not in script


def test_detalhes_voltam_para_a_demonstracao(detalhes):
    assert '<h1>Detalhes sobre o projeto</h1>' in detalhes
    assert 'href="#experimente"' not in detalhes and 'href="index.html#experimente"' in detalhes


def test_site_nao_publica_pickle_nem_dados_internos(site):
    assert not list(site.rglob("*.joblib")) and not list(site.rglob("*.pkl"))
    assert not list(site.rglob("consumo_diario_sintetico.csv"))
    base = (site / "etapa5_base_enriquecida" / "base_final_modelagem_claro.csv").read_text(encoding="utf-8").splitlines()
    assert len(base) == 961


# ------------------------------------------------------------------ no navegador (Chrome sem janela)
class _ServidorSilencioso(SimpleHTTPRequestHandler):
    def log_message(self, *argumentos):
        pass


@pytest.fixture(scope="module")
def navegador(site, tmp_path_factory):
    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome não encontrado: os testes no navegador foram pulados")
    pasta = tmp_path_factory.mktemp("navegador")
    (pasta / "site").symlink_to(site, target_is_directory=True)
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_ServidorSilencioso, directory=str(pasta)))
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield chrome, pasta, f"http://127.0.0.1:{servidor.server_address[1]}"
    servidor.shutdown()


def _rodar_no_chrome(navegador, nome_da_pagina, conteudo):
    chrome, pasta, endereco = navegador
    (pasta / nome_da_pagina).write_text(conteudo, encoding="utf-8")
    processo = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={pasta / ('perfil-' + nome_da_pagina)}",
         "--virtual-time-budget=60000", "--dump-dom", f"{endereco}/{nome_da_pagina}"],
        capture_output=True, text=True, timeout=240)
    achado = re.search(r'<pre id="resultado">(.*?)</pre>', processo.stdout, re.S)
    assert achado and achado.group(1), f"O teste no navegador não terminou. {processo.stderr[-500:]}"
    resultado = json.loads(html.unescape(achado.group(1)))
    assert "erro" not in resultado, resultado.get("erro")
    return resultado


def test_restaurar_notebook_descarta_edicao_sem_afetar_o_projeto(navegador, site):
    _, pasta, _ = navegador
    # Um editor pequeno isola o controle de recuperação dos downloads de Python.
    # O teste real de execução WASM é independente deste teste de interface.
    wrapper = (site / "notebook_editavel.html").read_text(encoding="utf-8")
    wrapper = wrapper.replace('data-editor="editor/index.html"',
                              'data-editor="editor_teste.html?filename=alterado.py#code=alterado"')
    wrapper = wrapper.replace('src="notebook_editavel.js?', 'src="site/notebook_editavel.js?')
    (pasta / "laboratorio_teste.html").write_text(wrapper, encoding="utf-8")
    (pasta / "editor_teste.html").write_text(
        '<div class="cm-editor"><pre class="cm-content" id="fonte">codigo_original = 1</pre></div>',
        encoding="utf-8")
    resultado = _rodar_no_chrome(navegador, "teste_restaurar.html", """<!doctype html>
<iframe id="laboratorio" src="laboratorio_teste.html" style="width:100%;height:900px"></iframe><pre id="resultado"></pre>
<script>
(async () => {
  const esperar = async (condicao) => {
    for (let tentativa = 0; tentativa < 200; tentativa++) {
      if (condicao()) return;
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    throw new Error("O editor não terminou de carregar");
  };
  const saida = {};
  try {
    const quadro = document.getElementById("laboratorio");
    await esperar(() => quadro.contentDocument?.querySelector("#editor-python")?.contentDocument?.querySelector("#fonte"));
    const laboratorio = quadro.contentDocument;
    const primeiroEditor = laboratorio.getElementById("editor-python");
    const fonte = primeiroEditor.contentDocument.getElementById("fonte");
    saida.fonte_inicial = primeiroEditor.contentWindow.getComputedStyle(fonte).fontSize;
    const tamanho = laboratorio.getElementById("tamanho-do-codigo");
    tamanho.value = "22";
    tamanho.dispatchEvent(new Event("change"));
    saida.fonte_ampliada = primeiroEditor.contentWindow.getComputedStyle(fonte).fontSize;
    const alturaAntes = primeiroEditor.getBoundingClientRect().height;
    laboratorio.getElementById("ampliar-editor").click();
    saida.area_ampliada = primeiroEditor.getBoundingClientRect().height > alturaAntes;
    saida.estado_ampliado = laboratorio.getElementById("ampliar-editor").getAttribute("aria-pressed");
    laboratorio.getElementById("ampliar-editor").click();
    fonte.textContent = "raise RuntimeError('experimento quebrado')";
    laboratorio.getElementById("restaurar-notebook").click();
    saida.confirmacao = laboratorio.getElementById("confirmacao-da-restauracao").open;
    laboratorio.getElementById("cancelar-restauracao").click();
    saida.cancelar_preserva = laboratorio.getElementById("editor-python") === primeiroEditor
      && fonte.textContent.includes("RuntimeError");
    for (let numero = 1; numero <= 2; numero++) {
      laboratorio.getElementById("restaurar-notebook").click();
      laboratorio.getElementById("confirmar-restauracao").click();
      const novoEditor = laboratorio.getElementById("editor-python");
      await esperar(() => novoEditor.contentDocument?.getElementById("fonte"));
      saida["restauracao_" + numero] = novoEditor !== primeiroEditor
        && novoEditor.contentDocument.getElementById("fonte").textContent === "codigo_original = 1";
      const endereco = new URL(novoEditor.src);
      saida.url_limpa = endereco.hash === "" && !endereco.searchParams.has("filename")
        && endereco.searchParams.get("restauracao") === String(numero);
    }
    saida.voltar = laboratorio.querySelector(".laboratorio-voltar").getAttribute("href");
    saida.original = laboratorio.querySelector('a[target="_blank"]').getAttribute("href");
  } catch (erro) { saida.erro = String(erro.stack || erro); }
  document.getElementById("resultado").textContent = JSON.stringify(saida);
})();
</script>""")
    assert resultado["confirmacao"] and resultado["cancelar_preserva"]
    assert resultado["fonte_inicial"] == "18px" and resultado["fonte_ampliada"] == "22px"
    assert resultado["area_ampliada"] and resultado["estado_ampliado"] == "true"
    assert resultado["restauracao_1"] and resultado["restauracao_2"] and resultado["url_limpa"]
    assert resultado["voltar"] == "index.html"
    assert resultado["original"] == "claro_guia_etapas_0_a_8.html?show-code=false"


PAGINA_DE_EQUIVALENCIA = """<!DOCTYPE html><meta charset="utf-8"><pre id="resultado"></pre>
<script src="site/claro_guia.js"></script>
<script>
(async () => {
  const saida = {};
  try {
    const dados = await (await fetch("site/etapa7_recomendacoes/dados_da_demonstracao.json")).json();
    const modelo = await (await fetch("site/" + dados.modelo)).json();
    const modeloDePropensao = await (await fetch("site/" + dados.modelo_de_propensao)).json();
    const casos = await (await fetch("casos_de_teste.json")).json();
    const casosDaPropensao = await (await fetch("casos_da_propensao.json")).json();
    const G = window.ClaroGuia;
    const floresta = new G.FlorestaNoNavegador(modelo);
    const propensao = new G.FlorestaNoNavegador(modeloDePropensao);
    const atributos = new G.AtributosDaSimulacao(dados.data_de_referencia, dados.dias_de_um_mes, dados.dias_do_ciclo);
    const analisador = new G.AnalisadorDeRisco(dados.proporcao_de_atencao);
    const recomendador = new G.RecomendadorDePlanos(dados.planos, dados.margem_de_seguranca, dados.peso_random_forest);
    const colaborativa = new G.FiltragemColaborativaNoNavegador(dados.filtragem_colaborativa);
    saida.previsoes = casos.map((caso) => floresta.prever(caso));
    saida.chances = casosDaPropensao.map((caso) => propensao.prever(caso));
    const m = dados.cliente_do_exemplo;
    const demanda = floresta.prever(m.atributos);
    const plano = recomendador.porNome(m.plano_atual);
    saida.cliente_do_exemplo = { demanda,
      risco: analisador.analisar(plano.gb_principais, m.consumo_ciclo_atual_gb, demanda, m.dias_ate_renovacao),
      recomendacao: recomendador.recomendar(recomendador.necessidade(m.consumo_ciclo_atual_gb, demanda, m.atributos.dados_gb),
        m.plano_atual, m.cliente, propensao, colaborativa) };
    saida.cenarios = {};
    for (const nome of ["mais_internet", "economia"]) {
      const c = dados.cenarios[nome];
      const a = atributos.montar(c.gb_usados, c.dias_desde_renovacao, c.uso_tipico_mes_gb, c.habitos);
      const d = floresta.prever(a);
      const o = recomendador.porNome(c.plano);
      const cliente = { id_cliente: null, dados_gb: c.uso_tipico_mes_gb, minutos: c.habitos.minutos, streaming_horas: c.habitos.streaming_horas,
        deslocamento_dias: c.habitos.deslocamento_dias, grupo: c.grupo_dos_habitos };
      saida.cenarios[nome] = { atributos: a, demanda: d,
        risco: analisador.analisar(o.gb_principais, c.gb_usados, d, a.dias_ate_renovacao),
        recomendacao: recomendador.recomendar(recomendador.necessidade(c.gb_usados, d, c.uso_tipico_mes_gb), c.plano,
          cliente, propensao, colaborativa) };
    }
  } catch (erro) { saida.erro = String((erro && erro.stack) || erro); }
  document.getElementById("resultado").textContent = JSON.stringify(saida);
})();
</script>"""


def _comparar(obtido, esperado, caminho=""):
    """Compara resultados do JavaScript e do Python (números com tolerância de 1e-6)."""
    if isinstance(esperado, dict):
        for chave, valor in esperado.items():
            _comparar(obtido[chave], valor, f"{caminho}.{chave}")
    elif isinstance(esperado, list):
        assert len(obtido) == len(esperado), caminho
        for indice, (item_obtido, item_esperado) in enumerate(zip(obtido, esperado)):
            _comparar(item_obtido, item_esperado, f"{caminho}[{indice}]")
    elif isinstance(esperado, float) and not isinstance(esperado, bool):
        assert obtido is not None and abs(obtido - esperado) <= 1e-6 * max(1.0, abs(esperado)), \
            f"{caminho}: JS {obtido} x Python {esperado}"
    else:
        assert obtido == esperado, f"{caminho}: JS {obtido!r} x Python {esperado!r}"


def test_javascript_calcula_igual_ao_python(navegador, site, pastas, base_final, recomendacao_treinada):
    modelo = json.loads((site / "etapa6_modelo" / "modelo_demanda_rf.json").read_text(encoding="utf-8"))
    variaveis = modelo["variaveis_numericas"] + list(modelo["variaveis_categoricas"])
    casos = json.loads(base_final.sample(120, random_state=7)[variaveis].to_json(orient="records"))
    (navegador[1] / "casos_de_teste.json").write_text(json.dumps(casos), encoding="utf-8")
    propensao = PrevisorDePropensao.carregar(pastas.modelo_de_propensao_joblib)  # o mesmo modelo do JSON do site
    casos_da_propensao = json.loads(recomendacao_treinada.teste.head(150)[propensao.configuracao.variaveis]
                                    .to_json(orient="records"))
    (navegador[1] / "casos_da_propensao.json").write_text(json.dumps(casos_da_propensao), encoding="utf-8")
    resultado = _rodar_no_chrome(navegador, "equivalencia.html", PAGINA_DE_EQUIVALENCIA)

    esperado_python = [FlorestaExportada(modelo).prever_um(caso) for caso in casos]
    assert np.allclose(resultado["previsoes"], esperado_python, atol=1e-9)
    assert np.allclose(resultado["chances"], propensao.notas(casos_da_propensao), atol=1e-8)  # navegador x scikit-learn

    dados = json.loads((site / "etapa7_recomendacoes" / "dados_da_demonstracao.json").read_text(encoding="utf-8"))
    exemplo = dados["cliente_do_exemplo"]
    assert abs(resultado["cliente_do_exemplo"]["demanda"] - exemplo["demanda_prevista_gb"]) < 1e-6  # navegador x scikit-learn
    _comparar(resultado["cliente_do_exemplo"]["risco"], exemplo["risco"], "cliente_do_exemplo.risco")
    _comparar(resultado["cliente_do_exemplo"]["recomendacao"], exemplo["recomendacao"], "cliente_do_exemplo.recomendacao")
    for nome in ["mais_internet", "economia"]:
        python = dados["resultados_do_python"][nome]
        _comparar(resultado["cenarios"][nome]["atributos"], python["atributos"], f"{nome}.atributos")
        assert abs(resultado["cenarios"][nome]["demanda"] - python["demanda_prevista_gb"]) < 1e-6
        _comparar(resultado["cenarios"][nome]["risco"], python["risco"], f"{nome}.risco")
        _comparar(resultado["cenarios"][nome]["recomendacao"], python["recomendacao"], f"{nome}.recomendacao")


PAGINA_DA_JORNADA = r"""<!DOCTYPE html><meta charset="utf-8"><pre id="resultado"></pre>
<script>
const LARGURAS = __LARGURAS__;
const espera = (ms) => new Promise((pronto) => setTimeout(pronto, ms));
async function ate(condicao, limite = 20000) {
  const inicio = performance.now();
  while (!condicao()) { if (performance.now() - inicio > limite) throw new Error("tempo esgotado"); await espera(50); }
}
async function abrir(largura) {
  const quadro = document.createElement("iframe");
  quadro.style.width = `${largura}px`; quadro.style.height = "900px"; quadro.style.border = "0";
  const carregado = new Promise((pronto) => { quadro.onload = pronto; });
  quadro.src = "site/index.html";
  document.body.appendChild(quadro);
  await carregado;
  await ate(() => quadro.contentDocument.querySelector("#claro-guia")?.dataset.modelo === "pronto");
  return quadro;
}
const textoDe = (doc, seletor) => doc.querySelector(seletor)?.textContent.replace(/\s+/g, " ").trim() ?? null;
const transborda = (doc) => doc.documentElement.scrollWidth > doc.documentElement.clientWidth + 1;
function controleVisivel(doc, seletor) {
  const alvo = doc.querySelector(seletor);
  if (!alvo) throw new Error(`Controle ausente: ${seletor}`);
  if (!alvo.checkVisibility() || alvo.closest("[hidden], [inert]")) {
    throw new Error(`Controle fora da tela ativa: ${seletor}`);
  }
  return alvo;
}
function clicar(doc, seletor) {
  controleVisivel(doc, seletor).click();
}
function mudar(doc, seletor, valor, tipo = "change") {
  const campo = controleVisivel(doc, seletor);
  campo.focus();
  if (campo.type === "radio") campo.checked = true; else campo.value = String(valor);
  campo.dispatchEvent(new Event(tipo, { bubbles: true }));
}
function telaAtual(doc) {
  return {
    etapas: [...doc.querySelectorAll(".flow>.step")].map((etapa) => ({
      id: etapa.id, oculta: etapa.hidden, inerte: etapa.inert,
      visivel: etapa.getClientRects().length > 0, atual: etapa.getAttribute("aria-current") === "step"
    })),
    foco: doc.activeElement.id,
    foco_oculto: !!doc.activeElement.closest(".step[hidden], .step[inert]"),
    progresso: textoDe(doc, "#progresso-fluxo"),
    voltar_oculto: doc.querySelector('[data-acao="voltar-etapa"]').hidden,
    controles_ocultos_visiveis: [...doc.querySelectorAll('.step[hidden] button, .step[hidden] input, .step[hidden] select, .step[hidden] summary')]
      .filter((controle) => controle.getClientRects().length).map((controle) => controle.id || controle.tagName)
  };
}
function editar(doc) {
  if (doc.querySelector('#etapa-7[aria-current="step"]')) clicar(doc, '#etapa-7 [data-ir="6"]');
  controleVisivel(doc, "#campo-plano");
}
function enviar(doc) {
  controleVisivel(doc, "#formulario-simulacao").requestSubmit();
}
function capturar(doc) {
  const planos = (prefixo) => [...doc.querySelectorAll(`${prefixo} button[data-plano]`)].map((botao) => ({
    nome: botao.dataset.plano, texto: botao.textContent.replace(/\s+/g, " ").trim(),
    preco: botao.querySelector(".pr").textContent,
    detalhes: doc.getElementById(botao.getAttribute("aria-controls")).textContent
  }));
  return {
    modo: doc.querySelector("#claro-guia").dataset.modo,
    mensagem: textoDe(doc, "#mensagem-dinamica"),
    analise: textoDe(doc, "#analise-dinamica"),
    barra: doc.querySelector("#analise-dinamica .bar").getAttribute("aria-label"),
    risco: doc.querySelector("#analise-dinamica [data-risco]").dataset.risco,
    panorama: [...doc.querySelectorAll("#panorama-dinamico .grid b, #panorama-dinamico .grid4 b")].map((v) => v.textContent),
    resultado: doc.querySelector("#etapa-7").dataset.resultado,
    titulo_resultado: textoDe(doc, "#titulo-etapa-7"),
    positivo: {
      classe: doc.querySelector("#etapa-7").classList.contains("resultado-positivo"),
      hero: textoDe(doc, "#panorama-dinamico .ok-hero"),
      titulo: textoDe(doc, "#panorama-dinamico .h2"),
      mascote: doc.querySelector("#panorama-dinamico .ok-hero img")?.getAttribute("src") ?? null,
      usados: doc.querySelector("#panorama-dinamico .gbar .u")?.style.width ?? null,
      previstos: doc.querySelector("#panorama-dinamico .gbar .f")?.style.width ?? null,
      inicio_previstos: doc.querySelector("#panorama-dinamico .gbar .f")?.style.left ?? null,
      barra: doc.querySelector("#panorama-dinamico .gbar")?.getAttribute("aria-label") ?? null,
      manter: textoDe(doc, '#panorama-dinamico [data-acao="manter-plano"]'),
      simular: textoDe(doc, '#panorama-dinamico [data-ir="6"]'),
      economizar: textoDe(doc, '#panorama-dinamico [data-ir="4"]'),
    },
    calculo: textoDe(doc, "#panorama-dinamico .resumo-da-demanda"),
    demanda: textoDe(doc, "#panorama-dinamico .demanda-do-ciclo"),
    planos: planos("#planos-dinamicos"), panorama_planos: planos("#panorama-dinamico"),
    sem_opcao: doc.querySelector("#planos-dinamicos .sem-opcao") !== null,
    panorama_sem_opcao: doc.querySelector("#panorama-dinamico .sem-opcao") !== null,
    controle: { plano: doc.querySelector("#campo-plano").value,
      maximo: doc.querySelector("#campo-gb_usados").max, valor: doc.querySelector("#campo-gb_usados").value,
      rotulo: textoDe(doc, "#valor-gb_usados"), aria: doc.querySelector("#campo-gb_usados").getAttribute("aria-valuetext"),
      dias: doc.querySelector("#campo-dias_desde_renovacao").value },
    foco: doc.activeElement.id, tela: telaAtual(doc),
  };
}
// Depois de rolar um controle para dentro do telefone, ele deve estar inteiro em cada área que o recorta.
function controlesInacessiveis(doc) {
  const erros = [];
  for (const alvo of doc.querySelectorAll(".phone button, .phone select, .phone input, .phone summary")) {
    // O conteúdo de details fechado pode ter retângulos sem estar visível.
    if (!alvo.checkVisibility()) continue;
    alvo.scrollIntoView({ behavior: "instant", block: "nearest", inline: "nearest" });
    const r = alvo.getBoundingClientRect();
    for (let pai = alvo.parentElement; pai && pai !== doc.body; pai = pai.parentElement) {
      const css = doc.defaultView.getComputedStyle(pai);
      if (!/^(hidden|clip|auto|scroll)$/.test(css.overflowY)) continue;
      const p = pai.getBoundingClientRect();
      if (r.top < p.top - 2 || r.bottom > p.bottom + 2) {
        erros.push(`${alvo.id || alvo.dataset.acao || alvo.textContent.trim().slice(0, 30)} em ${pai.className}`);
        break;
      }
    }
  }
  return erros;
}
(async () => {
  const saida = { jornada: {}, larguras: {} };
  try {
    const quadro = await abrir(1024), doc = quadro.contentDocument, j = saida.jornada;
    const dados = JSON.parse(doc.querySelector("#dados-da-demonstracao").textContent);
    j.etapas = [...doc.querySelectorAll(".flow>.step")].map((etapa) => ({ id: etapa.id,
      titulo: etapa.querySelector(".note h2").textContent, foco: etapa.querySelector(".note h2").tabIndex }));
    j.inicial = capturar(doc);
    j.aviso = textoDe(doc, ".aviso-persistente");
    j.navegacao = [];
    for (const n of [2, 3, 4, 5]) {
      clicar(doc, `[data-ir="${n}"]`);
      j.navegacao.push({ etapa: n, tela: telaAtual(doc) });
    }
    j.cenarios_recolhidos = !doc.querySelector("details.outros-cenarios").open;
    j.cenarios = [...doc.querySelectorAll('input[name="cenario"]')].map((c) => ({
      valor: c.value, titulo: c.closest("label").textContent.trim() }));
    clicar(doc, '[data-acao="comecar-simulacao"]');
    j.comecou = telaAtual(doc);
    j.simulacao_padrao = capturar(doc);
    j.rotulos_ausentes = [...doc.querySelectorAll("#formulario-simulacao input:not([type=radio]), #formulario-simulacao select")]
      .filter((c) => !doc.querySelector(`label[for="${c.id}"]`)).map((c) => c.id);
    j.limites_dos_dias = { min: doc.querySelector("#campo-dias_desde_renovacao").min,
      max: doc.querySelector("#campo-dias_desde_renovacao").max };
    enviar(doc);
    j.enviou = { tela: telaAtual(doc),
      status: textoDe(doc, ".jornada-status") };
    j.selecoes = {};
    for (const nome of ["Essencial", "Viagem", "Total"]) {
      editar(doc);
      mudar(doc, "#campo-plano", nome);
      j.selecoes[nome] = capturar(doc);
    }
    // O plano muda a franquia; para a mesma entrada de consumo, a previsão deve permanecer igual.
    mudar(doc, "#campo-gb_usados", 26, "input");
    j.mesmo_consumo_outro_plano = capturar(doc);
    mudar(doc, "#campo-gb_usados", 10, "input");
    j.dez_gb = capturar(doc);
    mudar(doc, "#campo-dias_desde_renovacao", 16, "input");
    j.outro_dia = capturar(doc);
    mudar(doc, "#campo-plano", "Essencial");
    mudar(doc, "#campo-gb_usados", 90, "input");
    j.limitado = capturar(doc);
    mudar(doc, "#campo-gb_usados", 10, "input");
    j.essencial_dez = capturar(doc);
    j.limites_por_plano = dados.planos.map((plano) => {
      mudar(doc, "#campo-plano", plano.nome);
      return { nome: plano.nome, franquia: plano.gb_principais, ...capturar(doc).controle };
    });
    controleVisivel(doc, "details.outros-cenarios summary").click();
    mudar(doc, 'input[name="cenario"][value="economia"]', true);
    j.economia = capturar(doc);
    j.economia_cenario = doc.querySelector('input[name="cenario"]:checked').value;
    enviar(doc);
    j.resultado_adequado = capturar(doc);
    j.status_adequado = textoDe(doc, ".jornada-status");
    clicar(doc, '#panorama-dinamico [data-acao="manter-plano"]');
    j.manter = { texto: textoDe(doc, "#confirmacao-plano"), tela: telaAtual(doc),
      papel: doc.querySelector("#confirmacao-plano").getAttribute("role"),
      desabilitado: doc.querySelector('[data-acao="manter-plano"]').disabled,
      botao: textoDe(doc, '[data-acao="manter-plano"]'),
      visivel: controleVisivel(doc, "#confirmacao-plano").getClientRects().length > 0 };
    clicar(doc, '#panorama-dinamico [data-ir="4"]');
    j.economizar = { tela: telaAtual(doc), simulacao: capturar(doc) };
    const primeiroPlano = controleVisivel(doc, "#planos-dinamicos button[data-plano]");
    if (!primeiroPlano) throw new Error("O cenário de economia deveria ter recomendações");
    const detalhe = doc.getElementById(primeiroPlano.getAttribute("aria-controls"));
    primeiroPlano.click();
    j.detalhe_aberto = primeiroPlano.getAttribute("aria-expanded") === "true" && !detalhe.hidden;
    primeiroPlano.click();
    j.detalhe_fechado = primeiroPlano.getAttribute("aria-expanded") === "false" && detalhe.hidden;
    clicar(doc, '[data-acao="voltar-etapa"]');
    j.voltou_da_economia = telaAtual(doc);
    clicar(doc, '#analise-dinamica [data-ir="4"]');
    clicar(doc, '#etapa-4 [data-ir="5"]');
    clicar(doc, '[data-acao="comecar-simulacao"]');
    enviar(doc);
    j.voltar = [];
    for (const n of [6, 5, 4, 3, 2, 1]) {
      clicar(doc, '[data-acao="voltar-etapa"]');
      j.voltar.push({ etapa: n, tela: telaAtual(doc) });
    }
    for (const n of [2, 3, 4, 5]) clicar(doc, `[data-ir="${n}"]`);
    clicar(doc, '[data-acao="comecar-simulacao"]');
    clicar(doc, '[data-acao="ver-simulacao"]');
    j.botao_ver = telaAtual(doc);
    editar(doc);
    // O saldo aparente não basta: o uso típico mensal também participa da recomendação.
    mudar(doc, 'input[name="cenario"][value="mais_internet"]', true);
    mudar(doc, "#campo-plano", "Essencial");
    mudar(doc, "#campo-gb_usados", 1, "input");
    mudar(doc, "#campo-dias_desde_renovacao", 29, "input");
    enviar(doc);
    j.uso_tipico_acima_da_franquia = capturar(doc);
    editar(doc);
    // Localiza entradas reais de atenção/alto/esgotada com o modelo publicado, sem criar riscos artificiais.
    const G = quadro.contentWindow.ClaroGuia;
    const floresta = new G.FlorestaNoNavegador(await (await fetch("site/" + dados.modelo)).json());
    const atributos = new G.AtributosDaSimulacao(dados.data_de_referencia, dados.dias_de_um_mes, dados.dias_do_ciclo);
    const analisador = new G.AnalisadorDeRisco(dados.proporcao_de_atencao);
    const casosDeRisco = {};
    const cenario = dados.cenarios.mais_internet;
    for (const dias of [15, 29, 1, 10, 20]) {
      for (let consumo = 0; consumo <= 40; consumo++) {
        const a = atributos.montar(consumo, dias, cenario.uso_tipico_mes_gb, cenario.habitos);
        const risco = analisador.analisar(40, consumo, floresta.prever(a), a.dias_ate_renovacao);
        if (["atencao", "alto", "esgotada"].includes(risco.nivel) && !casosDeRisco[risco.nivel]) {
          casosDeRisco[risco.nivel] = { consumo, dias };
        }
      }
      if (Object.keys(casosDeRisco).length === 3) break;
    }
    j.riscos_sem_positivo = {};
    for (const [nivel, entrada] of Object.entries(casosDeRisco)) {
      mudar(doc, "#campo-plano", "Streaming");
      mudar(doc, "#campo-gb_usados", entrada.consumo, "input");
      mudar(doc, "#campo-dias_desde_renovacao", entrada.dias, "input");
      enviar(doc);
      j.riscos_sem_positivo[nivel] = { entrada, simulacao: capturar(doc) };
      editar(doc);
    }
    mudar(doc, 'input[name="cenario"][value="economia"]', true);
    mudar(doc, "#campo-plano", "Essencial");
    mudar(doc, "#campo-gb_usados", 1, "input");
    mudar(doc, "#campo-dias_desde_renovacao", 29, "input");
    enviar(doc);
    j.adequado_sem_economia = capturar(doc);
    j.nova_confirmacao_oculta = doc.querySelector("#confirmacao-plano").hidden;
    editar(doc);
    clicar(doc, '[data-acao="voltar-exemplo"]');
    j.restaurado = capturar(doc);
    j.foco_restaurado = doc.activeElement.id;
    j.acesso = [...doc.querySelectorAll(".comece a")].map((a) => ({ href: a.getAttribute("href"), texto: a.textContent.trim() }));
    for (const largura of LARGURAS) {
      const q = await abrir(largura), d = q.contentDocument;
      const medidas = { etapas: d.querySelectorAll(".flow>.step").length, telas: [],
        controles_inacessiveis: [], phones: [] };
      function medir(numero) {
        const ativa = controleVisivel(d, `#etapa-${numero}`);
        const telefone = ativa.querySelector(".phone").getBoundingClientRect();
        const resultado = numero === 7 ? ativa.dataset.resultado : null;
        medidas.telas.push({ etapa: numero, resultado, estado: telaAtual(d), transborda: transborda(d) });
        medidas.phones.push({ resultado, largura: telefone.width, altura: telefone.height });
        medidas.controles_inacessiveis.push(...controlesInacessiveis(d));
      }
      medir(1);
      for (const n of [2, 3, 4, 5]) {
        clicar(d, `[data-ir="${n}"]`);
        medir(n);
      }
      clicar(d, '[data-acao="comecar-simulacao"]');
      medir(6);
      d.querySelectorAll("#etapa-6 details").forEach((detalhe) => { detalhe.open = true; });
      medidas.aberto = transborda(d);
      medidas.controles_inacessiveis.push(...controlesInacessiveis(d));
      mudar(d, 'input[name="cenario"][value="economia"]', true);
      medidas.economia = transborda(d);
      enviar(d);
      medir(7);
      medidas.positivo = capturar(d);
      clicar(d, '#panorama-dinamico [data-acao="manter-plano"]');
      medidas.confirmacao = controleVisivel(d, "#confirmacao-plano").textContent;
      medidas.controles_inacessiveis.push(...controlesInacessiveis(d));
      clicar(d, '#panorama-dinamico [data-ir="4"]');
      medidas.voltou_opcoes = telaAtual(d);
      for (const botao of d.querySelectorAll("#planos-dinamicos button[data-plano]")) {
        controleVisivel(d, `#planos-dinamicos button[data-plano="${botao.dataset.plano}"]`).click();
      }
      medidas.planos_abertos = transborda(d);
      medidas.controles_inacessiveis.push(...controlesInacessiveis(d));
      clicar(d, '#etapa-4 [data-ir="5"]');
      clicar(d, '[data-acao="comecar-simulacao"]');
      mudar(d, 'input[name="cenario"][value="mais_internet"]', true);
      enviar(d);
      medir(7);
      medidas.negativo = capturar(d);
      for (const botao of d.querySelectorAll("#panorama-dinamico button[data-plano]")) {
        controleVisivel(d, `#panorama-dinamico button[data-plano="${botao.dataset.plano}"]`).click();
      }
      medidas.panorama_aberto = transborda(d);
      medidas.controles_inacessiveis.push(...controlesInacessiveis(d));
      saida.larguras[largura] = medidas;
      q.remove();
    }
  } catch (erro) { saida.erro = String(erro.stack || erro); }
  document.getElementById("resultado").textContent = JSON.stringify(saida);
})();
</script>"""

@pytest.fixture(scope="module")
def conta_da_simulacao_em_python(site):
    """Previsões do modelo exportado nesta montagem, sem fixar resultados de outra plataforma."""
    dados = json.loads((site / "etapa7_recomendacoes" / "dados_da_demonstracao.json").read_text(encoding="utf-8"))
    modelo = FlorestaExportada(json.loads((site / dados["modelo"]).read_text(encoding="utf-8")))
    atributos = AtributosDaSimulacao(date.fromisoformat(dados["data_de_referencia"]),
                                    dados["dias_de_um_mes"], dados["dias_do_ciclo"])

    def calcular(gb_usados, nome_do_cenario="mais_internet", dias_desde_renovacao=None, plano=None):
        cenario = dados["cenarios"][nome_do_cenario]
        dias = cenario["dias_desde_renovacao"] if dias_desde_renovacao is None else dias_desde_renovacao
        entrada = atributos.montar(gb_usados, dias,
                                   cenario["uso_tipico_mes_gb"], cenario["habitos"])
        previstos = modelo.prever_um(entrada)
        total = gb_usados + previstos
        capacidade = next(p["gb_principais"] for p in dados["planos"] if p["nome"] == (plano or cenario["plano"]))
        risco = AnalisadorDeRisco().analisar(capacidade, gb_usados, previstos, entrada["dias_ate_renovacao"])
        necessidade = max(total, cenario["uso_tipico_mes_gb"])
        return {"previstos": previstos, "total": total, "uso_tipico": cenario["uso_tipico_mes_gb"],
                "necessidade": necessidade, "risco": risco.nivel, "saldo": risco.saldo_gb,
                "atende": capacidade >= necessidade,
                "adequado": risco.nivel == "baixo" and capacidade >= necessidade}

    return calcular


def test_jornada_e_layout_no_navegador(navegador, site, conta_da_simulacao_em_python):
    resultado = _rodar_no_chrome(navegador, "jornada.html", PAGINA_DA_JORNADA.replace("__LARGURAS__", json.dumps(LARGURAS)))
    j = resultado["jornada"]
    dados = json.loads((site / "etapa7_recomendacoes" / "dados_da_demonstracao.json").read_text(encoding="utf-8"))
    catalogo = {plano["nome"]: plano for plano in dados["planos"]}

    def gb(valor):
        return f"{numero(valor, 0 if float(valor).is_integer() else 1)} GB"

    def nomes(opcoes):
        return [opcao["nome"] for opcao in opcoes]

    def conferir_tela(tela, numero_da_etapa, foco=None):
        assert tela["etapas"] == [
            {"id": f"etapa-{n}", "oculta": n != numero_da_etapa, "inerte": n != numero_da_etapa,
             "visivel": n == numero_da_etapa, "atual": n == numero_da_etapa}
            for n in range(1, 8)
        ], "somente a etapa atual pode aparecer e receber interação"
        assert tela["progresso"] == f"Etapa {numero_da_etapa} de 7"
        assert tela["voltar_oculto"] == (numero_da_etapa == 1)
        assert not tela["foco_oculto"] and tela["controles_ocultos_visiveis"] == []
        if foco:
            assert tela["foco"] == foco

    def conferir_planos(exibidos, necessidade, nome_atual):
        atual = catalogo[nome_atual]
        notas = []
        assert len(exibidos) <= 3 and len(set(nomes(exibidos))) == len(exibidos)
        for opcao in exibidos:
            plano = catalogo[opcao["nome"]]
            assert plano["gb_principais"] >= necessidade
            assert opcao["preco"] == f"R$ {numero(plano['mensalidade_brl'], 2)}/mês"
            assert gb(plano["gb_principais"]) in opcao["texto"]
            if atual["gb_principais"] >= necessidade and plano["nome"] != nome_atual:
                assert plano["mensalidade_brl"] < atual["mensalidade_brl"]
            if plano["nome"] != nome_atual:
                nota = re.search(r"Pontuação de recomendação: (\d+) pontos de 100", opcao["detalhes"])
                assert nota, "a recomendação deve explicar a pontuação, sem prometer uma probabilidade"
                notas.append(int(nota.group(1)))
        assert notas == sorted(notas, reverse=True)

    def conferir_simulacao(exibido, plano, usados, cenario="mais_internet", dias=15, etapa=6):
        esperado = conta_da_simulacao_em_python(usados, cenario, dias, plano)
        franquia = catalogo[plano]["gb_principais"]
        conferir_tela(exibido["tela"], etapa)
        assert exibido["modo"] == "simulacao"
        assert exibido["controle"]["plano"] == plano and float(exibido["controle"]["valor"]) == usados
        assert exibido["controle"]["rotulo"] == exibido["controle"]["aria"] == gb(usados)
        usado_na_mensagem = numero(usados, 0 if float(usados).is_integer() else 1)
        assert f"Você já usou {usado_na_mensagem} dos {gb(franquia)}" in exibido["mensagem"]
        assert (f"{gb(usados)} usados; {gb(esperado['previstos'])} previstos até renovar; franquia de {gb(franquia)}"
                == exibido["barra"])
        assert f"Plano: {gb(franquia)}" in exibido["analise"]
        percentual = int(usados / franquia * 100 + 0.5)  # Math.round para consumos não negativos
        assert exibido["risco"] == esperado["risco"]
        conferir_planos(exibido["planos"], esperado["necessidade"], plano)
        assert exibido["sem_opcao"] == (not exibido["planos"])
        positivo = exibido["positivo"]
        assert exibido["resultado"] == ("adequado" if esperado["adequado"] else "panorama")
        assert positivo["classe"] == esperado["adequado"]
        if esperado["adequado"]:
            assert exibido["titulo_resultado"] == "Resultado: plano adequado"
            assert positivo["titulo"] == "Resultado da sua simulação"
            assert positivo["hero"] and positivo["mascote"] == "mascote-claro-guia.svg"
            assert exibido["panorama"] == [gb(franquia), f"{gb(usados)} ({percentual}%)", gb(esperado["previstos"]),
                                            "~" + gb(max(0, esperado["saldo"] - esperado["previstos"]))]
            # O CSSOM arredonda os percentuais ao serializar style.width e style.left.
            assert float(positivo["usados"].removesuffix("%")) == pytest.approx(usados / franquia * 100, rel=0, abs=1e-4)
            assert float(positivo["previstos"].removesuffix("%")) == pytest.approx(esperado["previstos"] / franquia * 100, rel=0, abs=1e-4)
            assert float(positivo["inicio_previstos"].removesuffix("%")) == pytest.approx(usados / franquia * 100, rel=0, abs=1e-4)
            assert positivo["barra"], "a barra deve expor a previsão também como texto acessível"
            assert exibido["calculo"] is None and exibido["demanda"] is None
            assert exibido["panorama_planos"] == [] and not exibido["panorama_sem_opcao"]
            assert positivo["manter"] == "Continuar com meu plano"
            assert positivo["simular"] == "Simular outro cenário"
            tem_economia = any(p["nome"] != plano for p in exibido["planos"])
            assert positivo["economizar"] == ("Ver opções para economizar" if tem_economia else None)
        else:
            assert exibido["titulo_resultado"] == "Panorama completo"
            assert positivo["hero"] is None and positivo["manter"] is None
            assert exibido["panorama"] == [gb(franquia), f"{gb(usados)} ({percentual}%)", gb(franquia - usados),
                                            gb(esperado["previstos"])]
            assert (f"{numero(usados)} GB usados + {numero(esperado['previstos'])} GB previstos até renovar = "
                    f"{numero(esperado['total'])} GB") in exibido["calculo"]
            assert f"Seu uso típico mensal é de {numero(esperado['uso_tipico'])} GB" in exibido["calculo"]
            assert exibido["demanda"].startswith(f"Demanda estimada para um ciclo: {numero(esperado['necessidade'])} GB.")
            assert exibido["planos"] == exibido["panorama_planos"]
            assert exibido["panorama_sem_opcao"] == (not exibido["planos"])
        if usados == franquia:
            assert exibido["risco"] == "esgotada" and "A franquia acabou" in exibido["analise"]
        else:
            assert exibido["risco"] != "esgotada"
        return esperado

    titulos = ["Mensagem no WhatsApp", "Abre a conversa", "Tela da análise", "Recomendação de planos",
               "Tela do Claro Guia", "Recomendador de planos no Claro Guia", "Panorama completo"]
    assert j["etapas"] == [{"id": f"etapa-{i}", "titulo": titulo, "foco": -1} for i, titulo in enumerate(titulos, 1)]
    conferir_tela(j["inicial"]["tela"], 1)
    assert j["inicial"]["modo"] == "exemplo"
    assert j["inicial"]["mensagem"].startswith("Olá, cliente Claro! 👋")
    assert "antes da renovação" in j["inicial"]["mensagem"]
    assert not [p for p in PALAVRAS_PROIBIDAS_NA_MENSAGEM if p in j["inicial"]["mensagem"].lower()]
    assert j["aviso"] == "Não acessamos sua conta Claro, não enviamos WhatsApp e este não é um serviço oficial."
    exemplo = dados["cliente_do_exemplo"]
    assert (f"{gb(exemplo['consumo_ciclo_atual_gb'])} usados; {gb(exemplo['demanda_prevista_gb'])} previstos até renovar; "
            f"franquia de {gb(exemplo['risco']['capacidade_gb'])}") == j["inicial"]["barra"]
    assert nomes(j["inicial"]["planos"]) == [o["nome_do_plano"] for o in exemplo["recomendacao"]["opcoes"]]
    assert [passo["etapa"] for passo in j["navegacao"]] == [2, 3, 4, 5]
    for passo in j["navegacao"]:
        conferir_tela(passo["tela"], passo["etapa"], f"titulo-etapa-{passo['etapa']}")
    assert j["cenarios_recolhidos"]
    assert j["cenarios"] == [{"valor": "mais_internet", "titulo": "Estou usando mais internet"},
                              {"valor": "economia", "titulo": "Quero avaliar se posso economizar"}]
    conferir_tela(j["comecou"], 6, "titulo-etapa-6")
    assert j["rotulos_ausentes"] == [] and j["limites_dos_dias"] == {"min": "1", "max": "29"}
    conferir_simulacao(j["simulacao_padrao"], "Streaming", 26)
    assert nomes(j["simulacao_padrao"]["planos"]) == [o["nome_do_plano"] for o in dados["resultados_do_python"]["mais_internet"]["recomendacao"]["opcoes"]]
    conferir_tela(j["enviou"]["tela"], 7, "titulo-etapa-7")
    assert "Panorama completo da simulação" in j["enviou"]["status"]
    for nome, franquia in [("Essencial", 15), ("Viagem", 50), ("Total", 70)]:
        caso = j["selecoes"][nome]
        assert caso["controle"]["maximo"] == caso["controle"]["valor"] == str(franquia)
        assert caso["foco"] == "campo-plano", "trocar plano deve preservar o foco do seletor"
        conferir_simulacao(caso, nome, franquia)
    conferir_simulacao(j["mesmo_consumo_outro_plano"], "Total", 26)
    assert j["mesmo_consumo_outro_plano"]["barra"].split(";")[1] == j["simulacao_padrao"]["barra"].split(";")[1], "o plano não pode alterar a previsão para o mesmo consumo"
    conferir_simulacao(j["dez_gb"], "Total", 10)
    assert j["dez_gb"]["foco"] == "campo-gb_usados", "a atualização não pode tirar o foco do controle"
    conferir_simulacao(j["outro_dia"], "Total", 10, dias=16)
    assert j["outro_dia"]["foco"] == "campo-dias_desde_renovacao"
    conferir_simulacao(j["limitado"], "Essencial", 15, dias=16)
    conferir_simulacao(j["essencial_dez"], "Essencial", 10, dias=16)
    assert len(j["limites_por_plano"]) == len(catalogo) == 8
    for plano in j["limites_por_plano"]:
        assert float(plano["maximo"]) == float(plano["valor"]) == plano["franquia"]
        assert plano["rotulo"] == plano["aria"] == gb(plano["franquia"])
    conferir_simulacao(j["economia"], "Total", 6, cenario="economia")
    assert nomes(j["economia"]["planos"]) == [o["nome_do_plano"] for o in dados["resultados_do_python"]["economia"]["recomendacao"]["opcoes"]]
    conferir_simulacao(j["resultado_adequado"], "Total", 6, cenario="economia", etapa=7)
    conferir_tela(j["resultado_adequado"]["tela"], 7, "titulo-etapa-7")
    assert "plano adequado" in j["status_adequado"].lower()
    assert j["manter"]["texto"] == "Você escolheu manter o plano atual nesta simulação."
    assert j["manter"]["visivel"] and j["manter"]["papel"] == "status"
    assert j["manter"]["desabilitado"] and j["manter"]["botao"] == "Plano mantido nesta simulação"
    conferir_tela(j["manter"]["tela"], 7)
    conferir_tela(j["economizar"]["tela"], 4, "titulo-etapa-4")
    assert j["economizar"]["simulacao"]["planos"] == j["economia"]["planos"]
    conferir_tela(j["voltou_da_economia"], 3, "titulo-etapa-3")
    assert j["economia_cenario"] == "economia" and j["detalhe_aberto"] and j["detalhe_fechado"]
    assert [passo["etapa"] for passo in j["voltar"]] == [6, 5, 4, 3, 2, 1]
    for passo in j["voltar"]:
        conferir_tela(passo["tela"], passo["etapa"], f"titulo-etapa-{passo['etapa']}")
    conferir_tela(j["botao_ver"], 7, "titulo-etapa-7")
    esperado = conferir_simulacao(j["uso_tipico_acima_da_franquia"], "Essencial", 1, dias=29, etapa=7)
    assert esperado["saldo"] > esperado["previstos"] and esperado["risco"] == "baixo"
    assert esperado["uso_tipico"] > catalogo["Essencial"]["gb_principais"] and not esperado["atende"]
    assert j["uso_tipico_acima_da_franquia"]["resultado"] == "panorama"
    assert set(j["riscos_sem_positivo"]) == {"atencao", "alto", "esgotada"}
    for nivel, caso in j["riscos_sem_positivo"].items():
        esperado = conferir_simulacao(caso["simulacao"], "Streaming", caso["entrada"]["consumo"],
                                      dias=caso["entrada"]["dias"], etapa=7)
        assert esperado["risco"] == nivel and not esperado["adequado"]
        assert caso["simulacao"]["resultado"] == "panorama"
    esperado = conferir_simulacao(j["adequado_sem_economia"], "Essencial", 1, cenario="economia", dias=29, etapa=7)
    assert esperado["adequado"] and j["adequado_sem_economia"]["positivo"]["economizar"] is None
    assert j["nova_confirmacao_oculta"], "uma nova simulação não deve herdar a confirmação anterior"
    assert j["restaurado"]["modo"] == "exemplo" and j["restaurado"]["mensagem"] == j["inicial"]["mensagem"]
    assert j["restaurado"]["barra"] == j["inicial"]["barra"] and j["restaurado"]["controle"] == j["inicial"]["controle"]
    assert j["foco_restaurado"] == "titulo-etapa-1"
    conferir_tela(j["restaurado"]["tela"], 1, "titulo-etapa-1")
    assert len(j["acesso"]) == 1 and j["acesso"][0]["href"] == "detalhes_do_projeto.html"
    assert j["acesso"][0]["texto"].startswith("Como fizemos isso?")
    assert len(resultado["larguras"]) == len(LARGURAS)
    for largura, medidas in resultado["larguras"].items():
        assert medidas["etapas"] == 7
        assert [tela["etapa"] for tela in medidas["telas"]] == [1, 2, 3, 4, 5, 6, 7, 7]
        for tela in medidas["telas"]:
            conferir_tela(tela["estado"], tela["etapa"], None if tela["etapa"] == 1 else f"titulo-etapa-{tela['etapa']}")
            assert not tela["transborda"], f"{largura}px: rolagem horizontal na etapa {tela['etapa']}"
        assert [tela["resultado"] for tela in medidas["telas"] if tela["etapa"] == 7] == ["adequado", "panorama"]
        conferir_simulacao(medidas["positivo"], "Total", 6, cenario="economia", etapa=7)
        conferir_simulacao(medidas["negativo"], "Streaming", 26, etapa=7)
        conferir_tela(medidas["voltou_opcoes"], 4, "titulo-etapa-4")
        assert medidas["confirmacao"] == "Você escolheu manter o plano atual nesta simulação."
        for estado in ["aberto", "economia", "planos_abertos", "panorama_aberto"]:
            assert not medidas[estado], f"{largura}px: rolagem horizontal em '{estado}'"
        assert medidas["controles_inacessiveis"] == [], f"{largura}px: controles cortados ou inacessíveis por rolagem"
        assert len(medidas["phones"]) == 8
        for telefone in medidas["phones"]:
            assert 0 < telefone["largura"] <= int(largura)
            assert telefone["altura"] == (760 if telefone["resultado"] == "adequado" else 680)


PAGINA_DA_DOCUMENTACAO = r"""<!doctype html><html><head><meta charset="utf-8"></head><body>
<pre id="resultado"></pre><script>
(async () => {
  const saida = { telas: {} };
  try {
    for (const largura of __LARGURAS__) {
      const quadro = document.createElement("iframe");
      quadro.style.width = `${largura}px`; quadro.style.height = "900px"; quadro.style.border = "0";
      const carregado = new Promise((pronto) => { quadro.onload = pronto; });
      quadro.src = "site/detalhes_do_projeto.html";
      document.body.appendChild(quadro);
      await carregado;
      const doc = quadro.contentDocument;
      const transborda = () => doc.documentElement.scrollWidth > doc.documentElement.clientWidth + 1;
      const detalhes = [...doc.querySelectorAll("details")];
      saida.telas[largura] = {
        fechado: transborda(),
        recolhidos: detalhes.every((detalhe) => !detalhe.open),
        palavras: doc.querySelector("main").innerText.split(/\s+/).length,
        colunas: doc.defaultView.getComputedStyle(doc.querySelector(".arquitetura-trilhas")).gridTemplateColumns.split(" ").length,
        alvos_ausentes: [...doc.querySelectorAll('a[href^="#"]')]
          .filter((link) => !doc.getElementById(link.hash.slice(1))).map((link) => link.hash),
        resumos_baixos: detalhes.filter((detalhe) => detalhe.querySelector("summary").getBoundingClientRect().height < 44).length,
        etapas: doc.querySelectorAll(".registro-das-etapas details").length,
        exemplo: doc.querySelector(".conta-da-simulacao").textContent.replace(/\s+/g, " ").trim()
      };
      detalhes.forEach((detalhe) => { detalhe.open = true; });
      saida.telas[largura].aberto = transborda();
      quadro.remove();
    }
  } catch (erro) { saida.erro = String(erro.stack || erro); }
  document.getElementById("resultado").textContent = JSON.stringify(saida);
})();
</script></body></html>"""


def test_documentacao_objetiva_e_responsiva_no_navegador(navegador, conta_da_simulacao_em_python):
    resultado = _rodar_no_chrome(navegador, "documentacao.html", PAGINA_DA_DOCUMENTACAO.replace(
        "__LARGURAS__", json.dumps(LARGURAS)))
    esperado = conta_da_simulacao_em_python(40)
    assert len(resultado["telas"]) == len(LARGURAS)
    for largura, medidas in resultado["telas"].items():
        assert not medidas["fechado"] and not medidas["aberto"], f"{largura}px: documentação com rolagem horizontal"
        assert medidas["recolhidos"] and medidas["palavras"] < 1100, "a leitura principal deve ser curta"
        assert medidas["resumos_baixos"] == 0 and medidas["alvos_ausentes"] == []
        assert medidas["colunas"] == (1 if int(largura) <= 760 else 2)
        assert medidas["etapas"] == 9
        assert (f"40 GB usados + {numero(esperado['previstos'])} GB previstos até renovar = "
                f"{numero(esperado['total'])} GB") in medidas["exemplo"]
