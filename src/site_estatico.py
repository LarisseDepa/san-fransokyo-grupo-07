"""
Monta a pasta site/: um site estático (só arquivos HTML, JavaScript e de dados), pronto para o GitHub Pages.

  site/index.html               -> demonstração do CLARO GUIA (Cliente Claro e simulação) e "Como fizemos isso?"
  site/detalhes_do_projeto.html -> os dados, a união, o passo a passo (arquivos de cada etapa) e as etapas 5 a 8
  site/claro_guia.js            -> a demonstração: Random Forest no navegador e regras de recomendação
  site/claro_guia_etapas_0_a_8.html -> o notebook completo (etapas 0 a 8), exportado
  site/etapa2_eda_<planilha>/, site/etapa4_eda_uniao/ -> as EDAs
  site/etapa5_base_enriquecida/, site/etapa6_modelo/, site/etapa7_recomendacoes/, site/etapa8_recomendacao/
                                -> base, modelo de demanda, casos e modelo de recomendação
  site/arquivos/                -> união limpa, registro da preparação e conferência da cópia

O site preserva os resultados da montagem e oferece o mesmo notebook editável em HTML-WASM.
A demonstração faz inferência com JSON; o editor executa Python no dispositivo do visitante.
No GitHub, a montagem é refeita a cada push, pelo arquivo .github/workflows/publicar_site.yml.

Classes:
  SiteEstatico     -> monta a pasta site/ (roda e exporta o notebook, copia os arquivos, escreve as páginas)
  PaginaDoSite     -> o que as duas páginas têm em comum (estilo, ícones e preenchimento do modelo HTML)
  PaginaInicial    -> escreve o index.html a partir do modelo src/paginas/pagina_inicial.html
  PaginaDeDetalhes -> escreve o detalhes_do_projeto.html a partir do modelo src/paginas/detalhes_do_projeto.html

Uso: pixi run site        (para ver no navegador antes de publicar: pixi run ver-site)
"""
import html
import json
import shutil
import subprocess
import sys
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from string import Template
from typing import NamedTuple

import pandas as pd

import config
from eda import AnaliseExploratoria, DescritorDeDados
from leitura_de_planilhas import LeitorDePlanilhas
from notebook_no_navegador import NotebookNoNavegador
from pastas_do_projeto import PastasDoProjeto
from preparacao_dos_dados import CopiaDosOriginais
from secoes_do_site import ArquivoNoFluxo, EtapaNoFluxo, SecaoDaEtapa8, SecoesDasEtapas5a7
from tabelas import COLUNA_ORIGEM, colunas_de_dados

PASTA_DAS_PAGINAS = Path(__file__).parent / "paginas"


class MediaNoCartao(NamedTuple):
    """Uma média mostrada no cartão de cada planilha."""
    coluna: str
    rotulo: str
    casas_decimais: int = 0
    multiplicador: float = 1  # 100 para mostrar uma proporção como porcentagem
    sufixo: str = ""


class EnderecosDoSite(NamedTuple):
    """Onde fica cada página dentro do site (os links entre elas usam estes nomes)."""
    pagina_inicial: str
    detalhes: str
    notebook: str
    pasta_dos_arquivos: str  # pasta com os arquivos para baixar


# Final de endereço que abre o notebook exportado com o código das células oculto (só textos, tabelas e gráficos).
# É a mesma página: no menu "…" do marimo, a opção "Show code" mostra o código de novo.
CODIGO_OCULTO = "?show-code=false"


class PaginaDoSite:
    """O que as duas páginas do site têm em comum: o mesmo estilo, os mesmos ícones e o preenchimento do
    modelo HTML (string.Template: $campo). Cada página diz qual é o seu modelo e devolve os seus campos."""

    MODELO = None
    ESTILO = PASTA_DAS_PAGINAS / "estilo.css"
    ICONES = PASTA_DAS_PAGINAS / "icones.svg"

    def __init__(self, pastas, config, enderecos):
        self.pastas = pastas
        self.config = config
        self.enderecos = enderecos

    def salvar(self, caminho):
        modelo = Template(self.MODELO.read_text(encoding="utf-8"))
        pagina = modelo.substitute(
            nome_do_projeto=self._escapar(self.config.NOME_DO_PROJETO),
            estilo=self.ESTILO.read_text(encoding="utf-8"),
            icones=self.ICONES.read_text(encoding="utf-8"),
            **self.campos(),
        )
        Path(caminho).write_text(pagina, encoding="utf-8")

    def campos(self):
        """Os campos próprios da página."""
        raise NotImplementedError

    @staticmethod
    def _icone(nome):
        return f'<svg class="icone" aria-hidden="true"><use href="#icone-{nome}"/></svg>'

    @staticmethod
    def _escapar(valor):
        """Texto seguro para pôr dentro do HTML."""
        return html.escape(str(valor))


class PaginaInicial(PaginaDoSite):
    """A página inicial (index.html): a demonstração do CLARO GUIA e "Como fizemos isso?", com os botões para os
    detalhes do projeto e para o notebook (abrem em outra aba)."""

    MODELO = PASTA_DAS_PAGINAS / "pagina_inicial.html"
    SCRIPT_DA_DEMONSTRACAO = PASTA_DAS_PAGINAS / "claro_guia.js"

    def __init__(self, pastas, config, enderecos, demonstracao):
        super().__init__(pastas, config, enderecos)
        self.demonstracao = demonstracao  # dados da demonstração (Cliente Claro, cenários, planos e modelos)

    def campos(self):
        return {
            "titulo_do_projeto": self._titulo_do_projeto(),
            "detalhes": self.enderecos.detalhes,
            "notebook": self.enderecos.notebook,
            "notebook_editavel": NotebookNoNavegador.PAGINA,
            "notebook_com_codigo_oculto": self.enderecos.notebook + CODIGO_OCULTO,
            "tabela_unida": f"{self.enderecos.pasta_dos_arquivos}/{self.pastas.uniao_csv.name}",
            "dados_da_demonstracao": self._json_embutido(self.demonstracao),
            "script_da_demonstracao": self._endereco_da_demonstracao(),
        }

    def _endereco_da_demonstracao(self):
        """Muda o endereço quando o código muda, para evitar uma cópia antiga no cache do navegador."""
        versao = sha256(self.SCRIPT_DA_DEMONSTRACAO.read_bytes()).hexdigest()
        return f"{self.SCRIPT_DA_DEMONSTRACAO.name}?v={versao}"

    def _titulo_do_projeto(self):
        """O nome do projeto em duas partes: a primeira palavra (CLARO) e o resto (GUIA), com estilos diferentes."""
        primeira_palavra, _, resto = self.config.NOME_DO_PROJETO.partition(" ")
        return (f'<span class="titulo-destaque">{self._escapar(primeira_palavra)}</span> '
                f'<span class="titulo-resto">{self._escapar(resto)}</span>')

    @staticmethod
    def _json_embutido(dados):
        """JSON dentro de <script>: sem "</" para o navegador não achar que o script acabou."""
        return json.dumps(dados, ensure_ascii=False).replace("</", "<\\/")


class PaginaDeDetalhes(PaginaDoSite):
    """A página "Detalhes sobre o projeto": os dados, a união e as etapas (com fluxogramas) e as seções das
    etapas 5 a 8. Os números vêm dos dados; os textos fixos ficam no modelo HTML e em config.py."""

    MODELO = PASTA_DAS_PAGINAS / "detalhes_do_projeto.html"
    MEDIAS_NOS_CARTOES = [
        MediaNoCartao("dados_gb", "GB de internet", casas_decimais=1),
        MediaNoCartao("minutos", "minutos de ligação"),
        MediaNoCartao("deslocamento_dias", "dias em deslocamento"),
        MediaNoCartao("risco_cancelamento", "com risco de cancelar", multiplicador=100, sufixo="%"),
    ]

    def __init__(self, pastas, config, enderecos, secoes_do_modelo, secao_da_etapa_8):
        super().__init__(pastas, config, enderecos)
        self.secoes = secoes_do_modelo
        self.etapa_8 = secao_da_etapa_8

    def campos(self):
        return {
            "pagina_inicial": self.enderecos.pagina_inicial,
            "notebook_editavel": NotebookNoNavegador.PAGINA,
            "notebook_com_codigo_oculto": self.enderecos.notebook + CODIGO_OCULTO,
            "planilhas": self.uniao[COLUNA_ORIGEM].nunique(),
            "clientes": len(self.uniao),
            "colunas": len(colunas_de_dados(self.uniao)),
            "cartoes_das_planilhas": self._cartoes_das_planilhas(),
            "planilha_do_exemplo": self._escapar(self._titulo_da_planilha(self._nome_da_planilha_do_exemplo)),
            "linha_de_exemplo": self._linha_de_exemplo(),
            "colunas_explicadas": self._colunas_explicadas(),
            "fluxograma_das_etapas": self._fluxograma_das_etapas(),
            "secao_enriquecimento": self.secoes.secao_enriquecimento(),
            "secao_modelo": self.secoes.secao_modelo(),
            "secao_recomendacoes": self.secoes.secao_recomendacoes(),
            "secao_simulacao": self.secoes.secao_simulacao(),
            "secao_etapa8": self.etapa_8.secao(),
        }

    # ---------------------------------------------------------------- dados usados na página
    @cached_property
    def uniao(self):
        return pd.read_csv(self.pastas.uniao_csv)

    @cached_property
    def medias_por_planilha(self):
        return self.uniao.groupby(COLUNA_ORIGEM, sort=False).mean(numeric_only=True)

    @cached_property
    def _primeiro_original(self):
        return CopiaDosOriginais(self.pastas, self.config).listar_originais()[0]

    @property
    def _nome_da_planilha_do_exemplo(self):
        return CopiaDosOriginais(self.pastas, self.config).nome_curto(self._primeiro_original)

    def _titulo_da_planilha(self, nome):
        descricao = self.config.DESCRICAO_DAS_PLANILHAS.get(nome, {})
        return descricao.get("titulo", nome.replace("_", " "))

    # ---------------------------------------------------------------- seção "Os dados"
    def _cartoes_das_planilhas(self):
        return "".join(self._cartao_da_planilha(nome, planilha)
                       for nome, planilha in self.uniao.groupby(COLUNA_ORIGEM, sort=False))

    def _cartao_da_planilha(self, nome, planilha):
        descricao = self.config.DESCRICAO_DAS_PLANILHAS.get(nome, {}).get("descricao", "")
        grupo = planilha[self.config.COLUNA_GRUPO].iloc[0]
        medias = "".join(
            f"<div><dd>{self._media_formatada(nome, media)}</dd><dt>{self._escapar(media.rotulo)}</dt></div>"
            for media in self.MEDIAS_NOS_CARTOES if media.coluna in self.medias_por_planilha
        )
        return (f'<article class="planilha"><div class="planilha-topo">{self._icone("tabela")}<div>'
                f"<h3>{self._escapar(self._titulo_da_planilha(nome))}</h3>"
                f"<span>Grupo “{self._escapar(grupo)}” · {len(planilha)} clientes</span></div></div>"
                f"<p>{self._escapar(descricao)}</p><dl class=\"medias\">{medias}</dl></article>")

    def _media_formatada(self, nome, media):
        valor = self.medias_por_planilha.loc[nome, media.coluna] * media.multiplicador
        return DescritorDeDados.formatar_numero(valor, media.casas_decimais) + media.sufixo

    def _linha_de_exemplo(self):
        """A primeira linha da primeira planilha original, exatamente como chegou."""
        planilha = LeitorDePlanilhas().ler(self._primeiro_original, self._primeiro_original.name)
        cabecalho = "".join(f"<th>{self._escapar(coluna)}</th>" for coluna in planilha.columns)
        valores = "".join(f"<td>{self._escapar('' if pd.isna(valor) else valor)}</td>"
                          for valor in planilha.iloc[0])
        return (f'<div class="rolagem"><table class="exemplo"><thead><tr>{cabecalho}</tr></thead>'
                f"<tbody><tr>{valores}</tr></tbody></table></div>")

    def _colunas_explicadas(self):
        return "".join(f"<dt><code>{self._escapar(coluna)}</code></dt><dd>{self._escapar(explicacao)}</dd>"
                       for coluna, explicacao in self.config.DESCRICAO_DAS_COLUNAS.items())

    # ---------------------------------------------------------------- seção "Passo a passo"
    def _fluxograma_das_etapas(self):
        etapas = self._etapas() + self.secoes.etapas_no_fluxo() + [self.etapa_8.etapa_no_fluxo()]
        return '<div class="registro-das-etapas">' + "".join(self._passo(etapa) for etapa in etapas) + "</div>"

    def _passo(self, etapa):
        arquivos = "".join(
            f'<li><a href="{arquivo.endereco}">{self._escapar(arquivo.nome)}</a>'
            f"<span>{self._escapar(arquivo.explicacao)}</span></li>"
            for arquivo in etapa.arquivos
        )
        detalhes = f' <a href="#{etapa.ancora}">Ver explicação</a>' if etapa.ancora else ""
        return (f'<details class="detalhe-documentacao"><summary>Etapa {etapa.numero} — '
                f'{self._escapar(etapa.titulo)}</summary><p>{self._escapar(etapa.explicacao)}{detalhes}</p>'
                f'<ul class="arquivos">{arquivos}</ul></details>')

    def _etapas(self):
        planilha_analisada = self.config.PLANILHA_ANALISADA_SOZINHA
        titulo_da_planilha_analisada = self._titulo_da_planilha(planilha_analisada)
        quantidade = self.uniao[COLUNA_ORIGEM].nunique()
        return [
            EtapaNoFluxo(
                0, "Cópia dos originais", "copia",
                "As planilhas recebidas são copiadas sem nenhuma mudança, e todo o trabalho é feito na cópia. "
                "O hash (uma “impressão digital” do arquivo) prova que os originais não foram alterados.",
                [self._arquivo_para_baixar(self.pastas.conferencia_da_copia, "Conferência da cópia",
                                           "o hash de cada planilha original")]),
            EtapaNoFluxo(
                1, "Limpeza de cada planilha", "limpeza",
                "Cada planilha é corrigida separadamente: “24 GB” vira 24, “não informado” vira vazio, "
                "e os vazios são preenchidos com a mediana da própria planilha.",
                [self._arquivo_para_baixar(self.pastas.registro_da_preparacao, "Registro da preparação",
                                           "tudo o que foi corrigido, planilha por planilha (inclui a união)")]),
            EtapaNoFluxo(
                2, f"EDA da planilha {titulo_da_planilha_analisada}", "grafico",
                f"A planilha {titulo_da_planilha_analisada}, já limpa, é analisada sozinha: estatísticas, "
                "gráficos e alertas de qualidade.",
                self._arquivos_da_eda(self.pastas.eda_da_planilha(planilha_analisada))),
            EtapaNoFluxo(
                3, "União das planilhas", "uniao",
                f"As {quantidade} planilhas limpas são empilhadas numa tabela só, de {len(self.uniao)} linhas. "
                "A coluna origem guarda de qual planilha veio cada linha.",
                [self._arquivo_para_baixar(self.pastas.uniao_csv, "Tabela unida (CSV)",
                                           "as planilhas limpas juntas, pronta para análise"),
                 self._arquivo_para_baixar(self.pastas.uniao_excel, "Tabela unida (Excel)",
                                           "a mesma tabela, com uma aba para cada planilha limpa")]),
            EtapaNoFluxo(
                4, "EDA da união", "grafico",
                "A tabela unida é analisada: comparação entre os grupos, estatísticas e gráficos de cada coluna.",
                self._arquivos_da_eda(self.pastas.eda_da_uniao)),
        ]

    def _arquivo_para_baixar(self, arquivo, nome, explicacao):
        return ArquivoNoFluxo(f"{self.enderecos.pasta_dos_arquivos}/{arquivo.name}", nome, explicacao)

    @staticmethod
    def _arquivos_da_eda(pasta_da_eda):
        return [
            ArquivoNoFluxo(f"{pasta_da_eda.name}/{AnaliseExploratoria.ARQUIVO_RELATORIO_DETALHADO}",
                           "Relatório estatístico detalhado", "gráficos e estatísticas de cada coluna"),
            ArquivoNoFluxo(f"{pasta_da_eda.name}/{AnaliseExploratoria.ARQUIVO_DESCRICAO}",
                           "Descrição dos dados (texto)", "frases prontas para a documentação"),
        ]


class SiteEstatico:
    """Monta a pasta site/ a partir dos resultados do notebook completo."""

    ENDERECOS = EnderecosDoSite(pagina_inicial="index.html", detalhes="detalhes_do_projeto.html",
                                notebook="claro_guia_etapas_0_a_8.html", pasta_dos_arquivos="arquivos")
    SCRIPT_DA_DEMONSTRACAO = PaginaInicial.SCRIPT_DA_DEMONSTRACAO

    def __init__(self, pastas, config):
        self.pastas = pastas
        self.config = config
        self.destino = pastas.site

    def montar(self):
        """Roda o notebook, junta os resultados em site/ e escreve as duas páginas."""
        self._recriar_destino()
        self._rodar_notebook(self.pastas.notebook_completo)
        self._exportar_notebook(self.pastas.notebook_completo, self.ENDERECOS.notebook)
        NotebookNoNavegador(self.pastas).executar()
        self._copiar_pastas_de_eda()
        self._copiar_arquivos_para_baixar()
        secoes = SecoesDasEtapas5a7(self.pastas, self.ENDERECOS.pagina_inicial)
        etapa_8 = SecaoDaEtapa8(self.pastas, self.ENDERECOS.pagina_inicial)
        self._copiar_arquivos_das_etapas_5_a_8({**secoes.arquivos_no_site(), **etapa_8.arquivos_no_site()})
        shutil.copy2(self.SCRIPT_DA_DEMONSTRACAO, self.destino / self.SCRIPT_DA_DEMONSTRACAO.name)
        PaginaInicial(self.pastas, self.config, self.ENDERECOS, secoes.demonstracao).salvar(
            self.destino / self.ENDERECOS.pagina_inicial)
        PaginaDeDetalhes(self.pastas, self.config, self.ENDERECOS, secoes, etapa_8).salvar(
            self.destino / self.ENDERECOS.detalhes)
        (self.destino / ".nojekyll").touch()  # o GitHub Pages serve os arquivos como estão, sem processar
        return self.destino

    def _recriar_destino(self):
        if self.destino.exists():
            shutil.rmtree(self.destino)
        self.destino.mkdir(parents=True)

    @staticmethod
    def _rodar_notebook(notebook):
        """Roda o notebook como script: se alguma etapa der erro, a montagem para aqui
        (a exportação abaixo não para sozinha quando uma célula dá erro)."""
        subprocess.run([sys.executable, str(notebook)], check=True)

    def _exportar_notebook(self, notebook, nome_no_site):
        """Salva a página do notebook com todos os resultados (marimo export html)."""
        subprocess.run([sys.executable, "-m", "marimo", "export", "html", str(notebook),
                        "-o", str(self.destino / nome_no_site), "--force"], check=True)

    def _copiar_pastas_de_eda(self):
        pasta_da_eda_da_planilha = self.pastas.eda_da_planilha(self.config.PLANILHA_ANALISADA_SOZINHA)
        for pasta_da_eda in [pasta_da_eda_da_planilha, self.pastas.eda_da_uniao]:
            shutil.copytree(pasta_da_eda, self.destino / pasta_da_eda.name)

    def _copiar_arquivos_para_baixar(self):
        pasta_dos_arquivos = self.destino / self.ENDERECOS.pasta_dos_arquivos
        pasta_dos_arquivos.mkdir()
        for arquivo in [self.pastas.uniao_csv, self.pastas.uniao_excel,
                        self.pastas.registro_da_preparacao, self.pastas.conferencia_da_copia]:
            shutil.copy2(arquivo, pasta_dos_arquivos / arquivo.name)

    def _copiar_arquivos_das_etapas_5_a_8(self, arquivos):
        """Só os arquivos listados pelas seções: os .joblib (pickle) e o consumo diário não vão para o site."""
        for destino, origem in arquivos.items():
            caminho = self.destino / destino
            caminho.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, caminho)


if __name__ == "__main__":
    pastas_do_projeto = PastasDoProjeto(Path(__file__).resolve().parent.parent)
    pasta_do_site = SiteEstatico(pastas_do_projeto, config).montar()
    print(f"\nSite montado em {pasta_do_site}. Para ver no navegador: pixi run ver-site")
