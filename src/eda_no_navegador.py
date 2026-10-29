"""EDA recalculada no navegador com pandas, sem dependências nativas de profiling.

As estatísticas, a descrição e as checagens portáveis usam os mesmos componentes
da análise local. O relatório identifica seu escopo para não sugerir equivalência
com todas as análises do fg-data-profiling.
"""
from html import escape
from pathlib import Path

from eda import AnaliseExploratoria, ResultadoDaEda


class AnaliseExploratoriaNoNavegador(AnaliseExploratoria):
    """Interface da EDA local com relatório portável recalculado a cada execução."""

    ARQUIVO_RELATORIO_DETALHADO = "analise_exploratoria_no_navegador.html"

    def executar(self, pasta_dos_relatorios="."):
        pasta = Path(pasta_dos_relatorios)
        pasta.mkdir(parents=True, exist_ok=True)
        caminho = pasta / self.ARQUIVO_RELATORIO_DETALHADO
        resultado = ResultadoDaEda(
            alertas=self.verificador.verificar_sem_profiling(self.df),
            estatisticas_descritivas=self.estatisticas_descritivas(),
            descricao_dos_dados=self.descritor.descrever(self.df),
            caminho_do_relatorio_detalhado=caminho,
        )
        caminho.write_text(PaginaDaEdaNoNavegador().montar(self.df, resultado), encoding="utf-8")
        self._salvar_descricao_txt(pasta / self.ARQUIVO_DESCRICAO, resultado.descricao_dos_dados)
        return resultado


class PaginaDaEdaNoNavegador:
    """Relatório HTML independente, com tabelas escapadas e escopo explícito."""

    def montar(self, dados, resultado):
        alertas = (self._tabela(resultado.alertas) if len(resultado.alertas)
                   else "<p>Nenhum alerta nas checagens executadas.</p>")
        descricao = "".join(f"<li>{escape(str(frase))}</li>" for frase in resultado.descricao_dos_dados.frase)
        return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análise exploratória recalculada no navegador · CLARO GUIA</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; color: #242424; margin: 24px; line-height: 1.5; }}
h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 28px; }}
.escopo {{ padding: 16px; border: 1px solid #d8d8d8; border-radius: 8px; background: #f8f8f8; }}
.tabela {{ max-width: 100%; overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 14px; }}
th, td {{ border: 1px solid #dedede; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Análise exploratória recalculada no navegador</h1>
<p>{len(dados)} linhas e {len(dados.columns)} colunas na tabela analisada.</p>
<p class="escopo">Estatísticas e alertas foram calculados com os dados desta execução.
As checagens cobrem ausentes, colunas vazias ou constantes, valores únicos, infinitos,
duplicadas e as verificações adicionais de texto do projeto. Este relatório portável
não executa o fg-data-profiling nem suas análises adicionais; o relatório completo
continua disponível na versão original do projeto.</p>
<h2>Alertas de qualidade</h2>
{alertas}
<h2>Estatísticas descritivas</h2>
{self._tabela(resultado.estatisticas_descritivas)}
<h2>Descrição dos dados</h2>
<ul>{descricao}</ul>
</body>
</html>"""

    @staticmethod
    def _tabela(tabela):
        return '<div class="tabela">' + tabela.to_html(index=False, escape=True, border=0) + "</div>"
