"""
Gráficos da etapa 6 em SVG (texto puro, sem biblioteca), para o site e para o notebook.

Regras seguidas: uma série = uma cor (azul), sem legenda (o título diz o que é); marcas finas
(barras de até 24 px com ponta arredondada, pontos com anel da cor do fundo); linhas de grade
sólidas e discretas; textos nas cores de texto da página, nunca na cor do dado; cada marca tem
uma dica (<title>) e os números completos estão nos CSVs da etapa 6 (a "tabela" do gráfico).
As cores vêm de variáveis CSS do site, com valores padrão para o notebook (modo claro).
"""
import html
import math

import numpy as np

from tabelas import formatar_numero_br as _numero

COR_DA_SERIE = "var(--serie, #2a78d6)"
COR_DO_TEXTO = "var(--texto-secundario, #6b6a66)"
COR_DA_GRADE = "var(--borda, #e3e1db)"
COR_DO_FUNDO = "var(--superficie, #ffffff)"
COR_DA_REFERENCIA = "var(--texto-secundario, #6b6a66)"


def _teto_redondo(valor, passo):
    return passo * math.ceil(valor / passo)


class GraficosDoModelo:
    """Três gráficos da avaliação no teste: previsto x observado, distribuição dos erros e importância."""

    # tamanho de desenho perto do tamanho exibido: o texto fica legível no celular e no computador
    LARGURA, ALTURA = 400, 300
    MARGEM_ESQUERDA, MARGEM_DIREITA, MARGEM_SUPERIOR, MARGEM_INFERIOR = 46, 12, 12, 44
    LARGURA_DA_IMPORTANCIA, COLUNA_DOS_NOMES = 420, 168
    TAMANHO_DO_TEXTO = 11

    def previsto_vs_observado(self, previsoes, coluna_observada, coluna_prevista):
        """Cada ponto é um cliente de teste. Na linha cinza, a previsão seria perfeita."""
        observado = previsoes[coluna_observada].to_numpy()
        previsto = previsoes[coluna_prevista].to_numpy()
        maximo = _teto_redondo(max(observado.max(), previsto.max()), 20)
        x, y = self._escala_x(0, maximo), self._escala_y(0, maximo)
        pontos = "".join(
            f'<circle cx="{x(o):.1f}" cy="{y(p):.1f}" r="4" fill="{COR_DA_SERIE}" stroke="{COR_DO_FUNDO}" '
            f'stroke-width="2"><title>Cliente {int(cliente)}: real {_numero(o)} GB, previsto {_numero(p)} GB</title></circle>'
            for cliente, o, p in zip(previsoes["id_cliente"], observado, previsto))
        referencia = (f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(maximo):.1f}" y2="{y(maximo):.1f}" '
                      f'stroke="{COR_DA_REFERENCIA}" stroke-width="1" opacity="0.7"/>'
                      f'<text x="{x(maximo * 0.97):.1f}" y="{y(maximo * 0.97) - 6:.1f}" text-anchor="end" '
                      f'font-size="{self.TAMANHO_DO_TEXTO}" fill="{COR_DO_TEXTO}">previsão perfeita</text>')
        corpo = (self._grade_y(0, maximo, 20, y, "GB") + self._eixo_x(0, maximo, 20, x, "GB")
                 + referencia + pontos)
        return self._svg("Previsto x real no teste",
                         f"{len(observado)} clientes de teste. Quanto mais perto da linha, melhor a previsão.",
                         corpo, "Demanda real até a renovação (GB)", "Demanda prevista (GB)")

    def distribuicao_dos_erros(self, residuos, largura_da_faixa=2):
        """Histograma do erro (real − previsto). Barras à direita do zero: o modelo previu menos."""
        residuos = np.asarray(residuos)
        limite = _teto_redondo(np.abs(residuos).max(), largura_da_faixa)
        bordas = np.arange(-limite, limite + largura_da_faixa, largura_da_faixa)
        contagens, _ = np.histogram(residuos, bins=bordas)
        maximo = _teto_redondo(contagens.max(), 10)
        x, y = self._escala_x(-limite, limite), self._escala_y(0, maximo)
        largura_da_barra = min(24, x(bordas[1]) - x(bordas[0]) - 2)
        barras = "".join(
            self._coluna(x((inicio + fim) / 2) - largura_da_barra / 2, y(contagem), largura_da_barra, y(0),
                         f"De {_numero(inicio, 0)} a {_numero(fim, 0)} GB: {contagem} clientes")
            for inicio, fim, contagem in zip(bordas[:-1], bordas[1:], contagens) if contagem > 0)
        zero = (f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(0):.1f}" y2="{y(maximo):.1f}" '
                f'stroke="{COR_DA_REFERENCIA}" stroke-width="1" opacity="0.7"/>'
                f'<text x="{x(0) + 6:.1f}" y="{y(maximo) + 12:.1f}" font-size="{self.TAMANHO_DO_TEXTO}" fill="{COR_DO_TEXTO}">erro zero</text>')
        passo_x = 5 if limite <= 30 else 10
        corpo = (self._grade_y(0, maximo, max(1, maximo // 5), y, "")
                 + self._eixo_x(-limite, limite, passo_x, x, "GB", inicio_dos_ticks=-_teto_redondo(limite, passo_x))
                 + barras + zero)
        return self._svg("Distribuição dos erros no teste",
                         "Erro = demanda real − prevista. A maioria dos clientes fica perto de zero.",
                         corpo, "Erro (GB)", "Clientes")

    def importancia(self, importancia, nomes_legiveis, maximo_de_variaveis=10):
        """Barras horizontais: quanto o erro (MAE) piora ao embaralhar cada variável no teste."""
        dados = importancia.head(maximo_de_variaveis)
        altura_da_linha, espessura = 24, 12
        margem_esquerda = self.COLUNA_DOS_NOMES
        largura_util = self.LARGURA_DA_IMPORTANCIA - margem_esquerda - 66
        altura = len(dados) * altura_da_linha + 24
        maximo = max(dados.aumento_do_mae_gb.max(), 1e-9)
        linhas = []
        for posicao, linha in enumerate(dados.itertuples()):
            topo = 12 + posicao * altura_da_linha
            comprimento = max(0.0, linha.aumento_do_mae_gb) / maximo * largura_util
            nome = html.escape(nomes_legiveis.get(linha.variavel, linha.variavel))
            linhas.append(
                f'<text x="{margem_esquerda - 10}" y="{topo + espessura - 2}" text-anchor="end" font-size="{self.TAMANHO_DO_TEXTO}" '
                f'fill="{COR_DO_TEXTO}">{nome}</text>'
                + self._barra_horizontal(margem_esquerda, topo, comprimento, espessura,
                                         f"{nome}: o erro aumenta {_numero(linha.aumento_do_mae_gb, 2)} GB")
                + f'<text x="{margem_esquerda + comprimento + 8:.1f}" y="{topo + espessura - 2}" font-size="{self.TAMANHO_DO_TEXTO}" '
                  f'fill="{COR_DO_TEXTO}">+{_numero(linha.aumento_do_mae_gb, 2)} GB</text>')
        base = (f'<line x1="{margem_esquerda}" y1="6" x2="{margem_esquerda}" y2="{altura - 6}" '
                f'stroke="{COR_DA_GRADE}" stroke-width="1"/>')
        return (f'<svg class="grafico" viewBox="0 0 {self.LARGURA_DA_IMPORTANCIA} {altura}" role="img" '
                f'aria-label="Importância das variáveis: quanto o erro aumenta ao embaralhar cada uma" '
                f'style="width:100%;height:auto;font-family:inherit">{base}{"".join(linhas)}</svg>')

    # ---------------------------------------------------------------- peças comuns
    def _escala_x(self, minimo, maximo):
        inicio, fim = self.MARGEM_ESQUERDA, self.LARGURA - self.MARGEM_DIREITA
        return lambda valor: inicio + (valor - minimo) / (maximo - minimo) * (fim - inicio)

    def _escala_y(self, minimo, maximo):
        inicio, fim = self.ALTURA - self.MARGEM_INFERIOR, self.MARGEM_SUPERIOR
        return lambda valor: inicio + (valor - minimo) / (maximo - minimo) * (fim - inicio)

    def _grade_y(self, minimo, maximo, passo, y, unidade):
        partes = []
        for valor in np.arange(minimo, maximo + passo / 2, passo):
            partes.append(f'<line x1="{self.MARGEM_ESQUERDA}" y1="{y(valor):.1f}" x2="{self.LARGURA - self.MARGEM_DIREITA}" '
                          f'y2="{y(valor):.1f}" stroke="{COR_DA_GRADE}" stroke-width="1"/>'
                          f'<text x="{self.MARGEM_ESQUERDA - 6}" y="{y(valor) + 4:.1f}" text-anchor="end" '
                          f'font-size="{self.TAMANHO_DO_TEXTO}" fill="{COR_DO_TEXTO}">{_numero(valor, 0)}</text>')
        return "".join(partes)

    def _eixo_x(self, minimo, maximo, passo, x, unidade, inicio_dos_ticks=None):
        base_y = self.ALTURA - self.MARGEM_INFERIOR
        partes = [f'<line x1="{self.MARGEM_ESQUERDA}" y1="{base_y}" x2="{self.LARGURA - self.MARGEM_DIREITA}" '
                  f'y2="{base_y}" stroke="{COR_DA_GRADE}" stroke-width="1"/>']
        inicio = minimo if inicio_dos_ticks is None else inicio_dos_ticks
        for valor in np.arange(inicio, maximo + passo / 2, passo):
            if minimo <= valor <= maximo:
                partes.append(f'<text x="{x(valor):.1f}" y="{base_y + 16}" text-anchor="middle" font-size="{self.TAMANHO_DO_TEXTO}" '
                              f'fill="{COR_DO_TEXTO}">{_numero(valor, 0)}</text>')
        return "".join(partes)

    def _coluna(self, esquerda, topo, largura, base, dica):
        """Coluna com ponta de cima arredondada (4 px) e base reta."""
        altura = base - topo
        raio = min(4, altura / 2, largura / 2)
        caminho = (f"M{esquerda:.1f},{base:.1f} V{topo + raio:.1f} Q{esquerda:.1f},{topo:.1f} {esquerda + raio:.1f},{topo:.1f} "
                   f"H{esquerda + largura - raio:.1f} Q{esquerda + largura:.1f},{topo:.1f} {esquerda + largura:.1f},{topo + raio:.1f} "
                   f"V{base:.1f} Z")
        return f'<path d="{caminho}" fill="{COR_DA_SERIE}"><title>{html.escape(dica)}</title></path>'

    @staticmethod
    def _barra_horizontal(esquerda, topo, comprimento, espessura, dica):
        """Barra com ponta direita arredondada (4 px) e base reta."""
        raio = min(4, comprimento / 2, espessura / 2)
        fim = esquerda + comprimento
        caminho = (f"M{esquerda:.1f},{topo:.1f} H{fim - raio:.1f} Q{fim:.1f},{topo:.1f} {fim:.1f},{topo + raio:.1f} "
                   f"V{topo + espessura - raio:.1f} Q{fim:.1f},{topo + espessura:.1f} {fim - raio:.1f},{topo + espessura:.1f} "
                   f"H{esquerda:.1f} Z")
        return f'<path d="{caminho}" fill="{COR_DA_SERIE}"><title>{html.escape(dica)}</title></path>'

    def _svg(self, titulo, descricao, corpo, rotulo_x, rotulo_y):
        meio_x = (self.MARGEM_ESQUERDA + self.LARGURA - self.MARGEM_DIREITA) / 2
        meio_y = (self.MARGEM_SUPERIOR + self.ALTURA - self.MARGEM_INFERIOR) / 2
        rotulos = (f'<text x="{meio_x:.1f}" y="{self.ALTURA - 8}" text-anchor="middle" font-size="{self.TAMANHO_DO_TEXTO}" '
                   f'fill="{COR_DO_TEXTO}">{html.escape(rotulo_x)}</text>'
                   f'<text x="12" y="{meio_y:.1f}" text-anchor="middle" font-size="{self.TAMANHO_DO_TEXTO}" fill="{COR_DO_TEXTO}" '
                   f'transform="rotate(-90 12 {meio_y:.1f})">{html.escape(rotulo_y)}</text>')
        return (f'<svg class="grafico" viewBox="0 0 {self.LARGURA} {self.ALTURA}" role="img" '
                f'aria-label="{html.escape(titulo)}. {html.escape(descricao)}" '
                f'style="width:100%;height:auto;font-family:inherit">{corpo}{rotulos}</svg>')
