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
from recomendacao import AtributosDaSimulacao
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
        assert links
        assert [link for link in links if not (site / link).exists()] == []


def test_script_tem_versao_do_conteudo_para_atualizar_o_cache(site, pagina):
    versao = sha256((site / "claro_guia.js").read_bytes()).hexdigest()
    assert f'<script src="claro_guia.js?v={versao}" defer></script>' in pagina


def test_botoes_de_como_fizemos_isso(pagina):
    caixa = pagina[pagina.index('<aside class="comece">'):pagina.index("</aside>")]
    assert "<h2>Como fizemos isso?</h2>" in caixa and "Primeira vez aqui?" not in pagina
    nova_aba = 'target="_blank" rel="noopener"'
    botoes = re.findall(rf'<a class="botao[^"]*" href="([^"]+)" {nova_aba}>([^<]+)</a>', caixa)
    assert botoes == [("detalhes_do_projeto.html", "Detalhes sobre o projeto"),
                      ("notebook_editavel.html", "Editar e executar no navegador"),
                      ("claro_guia_etapas_0_a_8.html?show-code=false", "Notebook completo (código oculto)")]
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
    for conteudo in [pagina, detalhes]:
        estilo = conteudo[conteudo.index("<style>"):conteudo.index("</style>")]
        assert set(re.findall(r"font-size:([^;}]+)", estilo)) == tamanhos
        assert set(re.findall(r"font-family:([^;}]+)", estilo)) == {"var(--fonte-texto)", "var(--fonte-codigo)"}
        assert "fonts.googleapis.com/css2?family=Inter" in conteudo


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


PAGINA_DA_JORNADA = """<!DOCTYPE html><meta charset="utf-8"><pre id="resultado"></pre>
<script>
const LARGURAS = __LARGURAS__;
const espera = (ms) => new Promise((pronto) => setTimeout(pronto, ms));
async function ate(condicao, limite = 20000) {
  const inicio = performance.now();
  while (!condicao()) { if (performance.now() - inicio > limite) throw new Error("tempo esgotado"); await espera(50); }
}
function abrir(largura) {
  return new Promise((pronto) => {
    const quadro = document.createElement("iframe");
    quadro.style.width = `${largura}px`; quadro.style.height = "900px"; quadro.style.border = "0";
    quadro.onload = () => pronto(quadro);
    quadro.src = "site/index.html";
    document.body.appendChild(quadro);
  });
}
const titulo = (doc) => (doc.querySelector(".jornada-tela h3") || {}).textContent || "";
const transborda = (doc) => doc.documentElement.scrollWidth > doc.documentElement.clientWidth + 1;
function clicar(doc, acao) { doc.querySelector(`[data-acao="${acao}"]`).click(); }
function mudar(doc, seletor, valor) {
  const campo = doc.querySelector(seletor);
  if (campo.type === "radio" || campo.type === "checkbox") campo.checked = valor; else campo.value = valor;
  campo.dispatchEvent(new Event("change", { bubbles: true }));
}
function deslizar(doc, seletor, valor) {
  const controle = doc.querySelector(seletor);
  controle.value = valor;
  controle.dispatchEvent(new Event("input", { bubbles: true }));
}
const textoDe = (doc, seletor) => doc.querySelector(seletor).textContent.replace(/\\s+/g, " ").trim();
(async () => {
  const saida = { jornada: {}, larguras: {} };
  try {
    const quadro = await abrir(1024);
    const doc = quadro.contentDocument;
    await ate(() => titulo(doc) === "Mensagem simulada");
    const j = saida.jornada;
    j.mensagem = doc.querySelector(".balao").textContent.replace(/\\s+/g, " ").trim();
    j.selo = doc.querySelector(".selo-simulada").textContent;
    j.historia_antes_da_mensagem = doc.querySelector(".historia") !== null && doc.querySelector(".como-ajudaria") === null;
    j.cartao = doc.querySelector(".historia").textContent.replace(/\\s+/g, " ").trim();
    j.rodape_da_historia = doc.querySelector(".historia .historia-rodape").textContent.replace(/\\s+/g, " ").trim();
    j.botoes_passo1 = [...doc.querySelectorAll(".jornada-tela button")].map((b) => b.textContent.trim());
    j.abas = [...doc.querySelectorAll(".passos button")].map((b) => b.textContent.trim());
    j.convite = textoDe(doc, ".simulacao h3");
    j.convite_fechado = doc.querySelector('[data-acao="simular"]') !== null && doc.querySelector(".cenarios") === null;
    j.botao_simular = textoDe(doc, '[data-acao="simular"]');
    doc.querySelector('.passos [data-passo="3"]').click();
    j.aba_planos = titulo(doc);
    j.aba_marcada = doc.querySelector('.passos [aria-current="step"]').textContent;
    doc.querySelector('.passos [data-passo="1"]').click();
    j.aba_whatsapp = titulo(doc);
    clicar(doc, "avancar");
    j.titulo2 = titulo(doc);
    j.foco_no_titulo = doc.activeElement === doc.querySelector(".jornada-tela h3");
    j.analise = doc.querySelector(".jornada-tela").textContent.replace(/\\s+/g, " ");
    j.demanda_antes = [...doc.querySelectorAll(".numeros-da-analise dd")][3].textContent;
    clicar(doc, "avancar");
    j.titulo3 = titulo(doc);
    j.opcoes = doc.querySelectorAll(".jornada-tela .opcao").length;
    j.botoes_passo3 = [...doc.querySelectorAll(".jornada-tela button")].map((b) => b.textContent.trim());
    j.chances = [...doc.querySelectorAll(".jornada-tela .opcao-chance")]
      .map((chance) => Number(chance.textContent.match(/(\\d+) pontos/)[1]));
    j.planos_recomendados = [...doc.querySelectorAll(".jornada-tela .opcao h4")].map((h) => h.textContent.trim());
    clicar(doc, "simular");
    j.cenarios = [...doc.querySelectorAll('input[name="cenario"]')].map((opcao) => opcao.closest("label").textContent.trim());
    j.cenario_inicial = doc.querySelector('input[name="cenario"]:checked').value;
    j.plano_inicial = doc.querySelector("#campo-plano").value;
    j.limite_inicial_de_gb = doc.querySelector("#campo-gb_usados").max;
    j.controles = [...doc.querySelectorAll(".simulacao input[type=range]")].map((controle) => controle.id);
    j.valores_iniciais = [...doc.querySelectorAll(".simulacao output")].map((valor) => valor.textContent);
    j.titulo_antes_de_ver = titulo(doc);
    clicar(doc, "ver-simulacao");
    j.titulo_simulacao = titulo(doc);
    j.aba_da_simulacao = doc.querySelector('.passos [aria-current="step"]').textContent;
    j.foco_da_simulacao = doc.activeElement === doc.querySelector(".jornada-tela h3");
    j.contexto_da_simulacao = textoDe(doc, ".contexto-curto");
    j.demanda_plano_original = [...doc.querySelectorAll(".numeros-da-analise dd")][3].textContent;
    mudar(doc, "#campo-plano", "Total");
    j.controle_apos_ampliar_plano = { maximo: doc.querySelector("#campo-gb_usados").max,
      valor: doc.querySelector("#campo-gb_usados").value };
    j.demanda_outro_plano = [...doc.querySelectorAll(".numeros-da-analise dd")][3].textContent;
    j.consumo_apos_ampliar_plano = textoDe(doc, ".consumo-principal");
    await espera(200);  // o aviso para leitores de tela entra 50 ms depois, de propósito
    j.status_plano = doc.querySelector(".jornada-status").textContent;
    // Com o mesmo consumo, o plano continua sem influenciar a previsão do modelo.
    deslizar(doc, "#campo-gb_usados", "26");
    await espera(400);
    j.demanda_com_o_mesmo_consumo = [...doc.querySelectorAll(".numeros-da-analise dd")][3].textContent;
    deslizar(doc, "#campo-gb_usados", "40");
    j.valor_ao_deslizar = doc.querySelector("#valor-gb_usados").textContent;
    await espera(400);  // a análise acompanha o controle 150 ms depois do último movimento
    j.consumo_depois_de_deslizar = textoDe(doc, ".consumo-principal");
    j.campos_sem_rotulo = [...doc.querySelectorAll(".simulacao input:not([type=radio]), .simulacao select")]
      .filter((campo) => !doc.querySelector(`label[for="${campo.id}"]`)).map((campo) => campo.id);
    clicar(doc, "voltar-exemplo");
    j.voltou = doc.querySelector(".historia") !== null && doc.querySelector('[data-acao="simular"]') !== null;

    // O primeiro ajuste já precisa sair do exemplo e acompanhar os valores selecionados.
    clicar(doc, "simular");
    deslizar(doc, "#campo-gb_usados", "30");
    await espera(400);
    j.primeiro_ajuste = { titulo: titulo(doc), consumo: textoDe(doc, ".consumo-principal"),
      aba: textoDe(doc, '.passos [aria-current="step"]') };
    mudar(doc, 'input[value="economia"]', true);
    j.aba_apos_mudar_cenario = textoDe(doc, '.passos [aria-current="step"]');
    deslizar(doc, "#campo-gb_usados", "60");
    await espera(400);
    mudar(doc, "#campo-plano", "Essencial");
    await espera(200);
    j.plano_menor = { maximo: doc.querySelector("#campo-gb_usados").max,
      valor: doc.querySelector("#campo-gb_usados").value, rotulo: textoDe(doc, "#valor-gb_usados"),
      aria: doc.querySelector("#campo-gb_usados").getAttribute("aria-valuetext"),
      consumo: textoDe(doc, ".consumo-principal"), saldo: doc.querySelectorAll(".numeros-da-analise dd")[0].textContent,
      status: textoDe(doc, ".jornada-status") };
    mudar(doc, "#campo-dias_desde_renovacao", "16");
    // O próprio controle impede informar um valor acima da franquia selecionada.
    deslizar(doc, "#campo-gb_usados", "90");
    await espera(400);
    j.consumo_limitado_ao_plano = { consumo: textoDe(doc, ".consumo-principal"),
      legenda: textoDe(doc, ".legenda-da-barra"), barra: doc.querySelector(".barra span").style.width,
      dias: doc.querySelectorAll(".numeros-da-analise dd")[1].textContent,
      valor_do_controle: textoDe(doc, "#valor-gb_usados") };
    doc.querySelector('.passos [data-passo="1"]').click();
    clicar(doc, "ver-simulacao");
    j.aba_apos_ver_novamente = textoDe(doc, '.passos [aria-current="step"]');
    mudar(doc, "#campo-plano", "Total");
    j.plano_maior = { maximo: doc.querySelector("#campo-gb_usados").max,
      valor: doc.querySelector("#campo-gb_usados").value, consumo: textoDe(doc, ".consumo-principal"),
      saldo: doc.querySelectorAll(".numeros-da-analise dd")[0].textContent };
    mudar(doc, "#campo-plano", "Essencial");
    deslizar(doc, "#campo-gb_usados", "10");
    await espera(400);
    j.consumo_abaixo_da_franquia = { consumo: textoDe(doc, ".consumo-principal"),
      saldo: doc.querySelectorAll(".numeros-da-analise dd")[0].textContent,
      esgotada: doc.querySelector(".risco-esgotada") !== null };
    const planos = JSON.parse(doc.getElementById("dados-da-demonstracao").textContent).planos;
    j.limites_por_plano = planos.map((plano) => {
      mudar(doc, "#campo-plano", plano.nome);
      return { plano: plano.nome, esperado: plano.gb_principais, obtido: Number(doc.querySelector("#campo-gb_usados").max),
        valor: Number(doc.querySelector("#campo-gb_usados").value), rotulo: textoDe(doc, "#valor-gb_usados"),
        aria: doc.querySelector("#campo-gb_usados").getAttribute("aria-valuetext"),
        consumo: textoDe(doc, ".consumo-principal") };
    });
    // Caso da imagem: a aba Planos acompanha plano, consumo e previsão, inclusive sem opções elegíveis.
    mudar(doc, 'input[value="mais_internet"]', true);
    mudar(doc, "#campo-plano", "Viagem");
    doc.querySelector('.passos [data-passo="3"]').click();
    const resumoDosPlanos = () => ({ referencia: textoDe(doc, ".plano-de-referencia"),
      calculo: textoDe(doc, ".resumo-da-demanda"), demanda: textoDe(doc, ".demanda-do-ciclo"),
      valor: doc.querySelector("#campo-gb_usados").value,
      aba: textoDe(doc, '.passos [aria-current="step"]'),
      sem_opcao: doc.querySelector(".sem-opcao") !== null, opcoes: doc.querySelectorAll(".opcao").length });
    j.planos_viagem = resumoDosPlanos();
    mudar(doc, "#campo-plano", "Total");
    j.planos_total = resumoDosPlanos();
    deslizar(doc, "#campo-gb_usados", "10");
    await espera(400);
    j.planos_com_menos_uso = resumoDosPlanos();
    clicar(doc, "ver-simulacao");
    j.analise_apos_planos = { aba: textoDe(doc, '.passos [aria-current="step"]'),
      consumo: textoDe(doc, ".consumo-principal") };
    // Voltar ao exemplo também cancela a atualização adiada de um controle.
    deslizar(doc, "#campo-gb_usados", "90");
    clicar(doc, "voltar-exemplo");
    await espera(400);
    j.voltou_sem_atualizacao_pendente = doc.querySelector(".historia") !== null && titulo(doc) === "Mensagem simulada";

    for (const largura of LARGURAS) {
      const q = await abrir(largura);
      const d = q.contentDocument;
      await ate(() => titulo(d) === "Mensagem simulada");
      const medidas = { inicio: transborda(d) };
      medidas.botoes_baixos = [...d.querySelectorAll("#claro-guia .botao, #claro-guia .aba, #claro-guia .botao-mensagem")]
        .filter((b) => b.getBoundingClientRect().height < 44).length;
      clicar(d, "avancar"); medidas.passo2 = transborda(d);
      clicar(d, "avancar"); medidas.passo3 = transborda(d);
      clicar(d, "simular"); medidas.painel = transborda(d);
      medidas.botoes_baixos += [...d.querySelectorAll(".simulacao .botao")]
        .filter((b) => b.getBoundingClientRect().height < 44).length;
      mudar(d, 'input[value="economia"]', true); medidas.painel_economia = transborda(d);
      clicar(d, "ver-simulacao"); clicar(d, "avancar");
      medidas.planos_economia = transborda(d);
      d.querySelectorAll("details").forEach((detalhe) => { detalhe.open = true; });
      medidas.tudo_aberto = transborda(d);
      saida.larguras[largura] = medidas;
    }
  } catch (erro) { saida.erro = String((erro && erro.stack) || erro); }
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

    def calcular(gb_usados, nome_do_cenario="mais_internet"):
        cenario = dados["cenarios"][nome_do_cenario]
        entrada = atributos.montar(gb_usados, cenario["dias_desde_renovacao"],
                                   cenario["uso_tipico_mes_gb"], cenario["habitos"])
        previstos = modelo.prever_um(entrada)
        total = gb_usados + previstos
        return {"previstos": previstos, "total": total, "uso_tipico": cenario["uso_tipico_mes_gb"],
                "necessidade": max(total, cenario["uso_tipico_mes_gb"])}

    return calcular


def test_jornada_e_layout_no_navegador(navegador, conta_da_simulacao_em_python):
    resultado = _rodar_no_chrome(navegador, "jornada.html", PAGINA_DA_JORNADA.replace("__LARGURAS__", json.dumps(LARGURAS)))
    j = resultado["jornada"]
    assert j["mensagem"].startswith("Olá, Cliente Claro!") and j["selo"].startswith("Mensagem simulada — não enviada")
    assert "cerca de" in j["mensagem"] and "antes da renovação" in j["mensagem"]
    assert not [palavra for palavra in PALAVRAS_PROIBIDAS_NA_MENSAGEM if palavra in j["mensagem"].lower()]
    assert j["historia_antes_da_mensagem"] and j["botoes_passo1"] == ["Ver minha análise"]
    assert j["rodape_da_historia"].startswith("Simulação acadêmica. Não acessamos sua conta Claro")
    # o cartão do exemplo usa os mesmos números do WhatsApp e da análise (nada de informação desencontrada)
    assert "Cliente Claro" in j["cartao"] and "Mariana" not in j["cartao"]
    dias_no_cartao = re.search(r"cerca de (\d+) dias antes do esperado", j["cartao"]).group(1)
    assert f"cerca de {dias_no_cartao} dias antes da renovação" in j["mensagem"]
    consumo, total = re.search(r"consumiu ([\d,]+) dos (\d+) GB", j["cartao"]).groups()
    assert j["abas"] == ["WhatsApp", "Análise", "Planos"]
    assert j["aba_planos"] == "Seu plano ainda combina com você?" and j["aba_marcada"] == "Planos"
    assert j["aba_whatsapp"] == "Mensagem simulada"
    assert j["titulo2"] == "Análise do consumo da Cliente Claro" and j["foco_no_titulo"]
    assert j["analise"].strip().startswith("Minha Claro APP")
    assert "Random Forest" in j["analise"] and "treinado com dados fictícios" in j["analise"]
    assert "Você usou" in j["analise"] and "cerca de" in j["analise"]
    assert f"Você usou {consumo} GB de {total} GB" in j["analise"]
    assert j["titulo3"] == "Seu plano ainda combina com você?" and 1 <= j["opcoes"] <= 3
    assert j["chances"] and j["chances"] == sorted(j["chances"], reverse=True), "planos na ordem da pontuação"
    assert all(nome.split(" ")[0] in {"Essencial", "Conecta", "Flex", "Digital", "Streaming", "Max", "Viagem", "Total"}
               for nome in j["planos_recomendados"])
    assert j["convite"] == "Quer simular outra situação?" and j["convite_fechado"] and j["botao_simular"] == "Simular"
    assert j["botoes_passo3"] == ["Voltar"]
    assert j["cenarios"] == ["Estou usando mais internet", "Quero avaliar se posso economizar"]
    assert j["cenario_inicial"] == "mais_internet" and j["plano_inicial"] == "Streaming"
    assert j["limite_inicial_de_gb"] == "40"
    assert j["controle_apos_ampliar_plano"] == {"maximo": "70", "valor": "70"}
    assert "Você usou 70,0 GB de 70 GB" in j["consumo_apos_ampliar_plano"]
    assert j["controles"] == ["campo-gb_usados", "campo-dias_desde_renovacao"]
    assert j["valores_iniciais"] == ["26 GB", "15 dias"]
    assert j["titulo_antes_de_ver"] == "Seu plano ainda combina com você?", "abrir o painel não troca a tela"
    assert j["titulo_simulacao"] == "Sua análise" and j["aba_da_simulacao"] == "Análise" and j["foco_da_simulacao"]
    assert "Estou usando mais internet" in j["contexto_da_simulacao"]
    assert j["demanda_plano_original"] != j["demanda_outro_plano"], "o consumo preenchido também muda a previsão"
    assert j["demanda_plano_original"] == j["demanda_com_o_mesmo_consumo"], "a mesma entrada deve manter a previsão"
    assert "Plano Total selecionado. Consumo ajustado para 70 GB" in j["status_plano"]
    assert j["valor_ao_deslizar"] == "40 GB" and "Você usou 40,0 GB" in j["consumo_depois_de_deslizar"]
    assert j["campos_sem_rotulo"] == [] and j["voltou"]
    assert j["primeiro_ajuste"]["titulo"] == "Sua análise" and j["primeiro_ajuste"]["aba"] == "Análise"
    assert "30,0 GB" in j["primeiro_ajuste"]["consumo"]
    assert j["aba_apos_mudar_cenario"] == j["aba_apos_ver_novamente"] == "Análise"
    menor = j["plano_menor"]
    assert menor["maximo"] == menor["valor"] == "15" and menor["rotulo"] == menor["aria"] == "15 GB"
    assert "Você usou 15,0 GB de 15 GB" in menor["consumo"] and menor["saldo"] == "0,0 GB"
    assert "Consumo ajustado para 15 GB" in menor["status"]
    caso = j["consumo_limitado_ao_plano"]
    assert "Você usou 15,0 GB de 15 GB" in caso["consumo"]
    assert caso["valor_do_controle"] == "15 GB" and caso["dias"] == "14"
    assert caso["barra"] == "100%" and "0,0 GB restantes" in caso["legenda"]
    assert j["plano_maior"]["maximo"] == "70" and j["plano_maior"]["valor"] == "70"
    assert "Você usou 70,0 GB de 70 GB" in j["plano_maior"]["consumo"] and j["plano_maior"]["saldo"] == "0,0 GB"
    assert len(j["limites_por_plano"]) == 8
    for plano in j["limites_por_plano"]:
        franquia = plano["esperado"]
        assert plano["obtido"] == plano["valor"] == franquia
        assert plano["rotulo"] == plano["aria"] == f"{franquia} GB"
        assert f"Você usou {franquia},0 GB de {franquia} GB" in plano["consumo"]
    assert "Você usou 10,0 GB de 15 GB" in j["consumo_abaixo_da_franquia"]["consumo"]
    assert j["consumo_abaixo_da_franquia"]["saldo"] == "5,0 GB" and not j["consumo_abaixo_da_franquia"]["esgotada"]
    viagem, total, menor_uso = j["planos_viagem"], j["planos_total"], j["planos_com_menos_uso"]
    assert "Viagem · 50 GB de uso geral" in viagem["referencia"] and viagem["valor"] == "50"
    for exibido, utilizados in [(viagem, 50), (total, 70), (menor_uso, 10)]:
        esperado = conta_da_simulacao_em_python(utilizados)
        assert (f"{numero(utilizados)} GB usados + {numero(esperado['previstos'])} GB previstos até renovar = "
                f"{numero(esperado['total'])} GB") in exibido["calculo"]
        assert f"Seu uso típico mensal é de {numero(esperado['uso_tipico'])} GB" in exibido["calculo"]
        assert exibido["demanda"].startswith(
            f"Demanda estimada para um ciclo: {numero(esperado['necessidade'])} GB.")
    assert viagem["sem_opcao"] and viagem["opcoes"] == 0
    assert "Total · 70 GB de uso geral" in total["referencia"] and total["valor"] == "70"
    assert "70,0 GB usados" in total["calculo"] and total["demanda"] != viagem["demanda"]
    assert menor_uso["valor"] == "10" and "10,0 GB usados" in menor_uso["calculo"]
    assert not menor_uso["sem_opcao"] and menor_uso["opcoes"] > 0
    assert viagem["aba"] == total["aba"] == menor_uso["aba"] == "Planos"
    assert j["analise_apos_planos"]["aba"] == "Análise"
    assert "Você usou 10,0 GB de 70 GB" in j["analise_apos_planos"]["consumo"]
    assert j["voltou_sem_atualizacao_pendente"]
    for largura, medidas in resultado["larguras"].items():
        assert medidas["botoes_baixos"] == 0, f"{largura}px: botões com menos de 44 px de altura"
        for estado in ["inicio", "passo2", "passo3", "painel", "painel_economia", "planos_economia", "tudo_aberto"]:
            assert not medidas[estado], f"{largura}px: rolagem horizontal em '{estado}'"


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
