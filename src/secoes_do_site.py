"""
Seções das etapas 5 a 8 na página "Detalhes sobre o projeto". O HTML é montado a partir dos arquivos que o
notebook claro_guia_etapas_0_a_8.py gera: os números vêm sempre desses arquivos (nada é digitado à mão).

Classes:
  SecoesDasEtapas5a7 -> enriquecimento, modelo de demanda e recomendações
  SecaoDaEtapa8      -> histórico de ofertas simulado e modelos de recomendação
"""
import html
import json
from datetime import date
from typing import NamedTuple

import pandas as pd

import config
from arquivos_da_etapa_8 import ArquivosDaEtapa8
from arquivos_das_etapas_5_a_7 import ArquivosDasEtapas5a7
from referencias_cientificas import BibliografiaDaRecomendacao
from dicionario_de_dados import exemplos_de_calculo
from enriquecimento import colunas_novas
from modelo_de_demanda import FlorestaExportada
from recomendacao import AtributosDaSimulacao
from tabelas import COLUNA_ORIGEM
from tabelas import formatar_numero_br as numero


class ArquivoNoFluxo(NamedTuple):
    """Um arquivo gerado por uma etapa, com link no fluxograma do passo a passo."""
    endereco: str  # caminho dentro do site
    nome: str
    explicacao: str


class EtapaNoFluxo(NamedTuple):
    """Uma etapa do fluxograma do passo a passo."""
    numero: int
    titulo: str
    icone: str  # nome de um ícone do modelo HTML (tabela, grafico, copia, limpeza, uniao, alvo)
    explicacao: str
    arquivos: list
    ancora: str = ""  # seção com os detalhes da etapa, se houver


def _texto(valor):
    return html.escape(str(valor))


class SecoesDasEtapas5a7:
    """Lê os resultados das etapas 5 a 7 e devolve as seções (HTML, na página de detalhes) e a lista de
    arquivos do site."""

    PASTA_ETAPA5, PASTA_ETAPA6, PASTA_ETAPA7 = "etapa5_base_enriquecida", "etapa6_modelo", "etapa7_recomendacoes"
    ACAO_POR_PERFIL = {
        "Economia": "Mostra planos mais baratos que ainda cobrem o uso de ciclos completos, sem obrigar a troca.",
        "Equilibrio_Preco_Adequacao": "Quando o plano cobre a demanda, sugere manter o plano atual.",
        "Excesso_de_Consumo": "Avisa antes de a franquia acabar e lista planos com mais GB, na ordem da pontuação de recomendação.",
        "Mobilidade": "Avalia mudanças no consumo sem inferir destino ou motivo do deslocamento.",
    }

    def __init__(self, pastas, pagina_inicial):
        self.pastas = pastas
        self.pagina_inicial = pagina_inicial  # endereço da página da demonstração, para o link "Voltar"
        avaliacao = pastas.avaliacao_do_modelo
        self.base = pd.read_csv(pastas.base_final)
        self.registro = json.loads(pastas.registro_do_treino.read_text(encoding="utf-8"))
        self.demonstracao = json.loads(pastas.dados_da_demonstracao.read_text(encoding="utf-8"))
        self.metricas = pd.read_csv(avaliacao / ArquivosDasEtapas5a7.METRICAS)
        self.metricas_por_perfil = pd.read_csv(avaliacao / ArquivosDasEtapas5a7.METRICAS_POR_PERFIL)
        self.selecao = pd.read_csv(avaliacao / ArquivosDasEtapas5a7.SELECAO_DE_VARIAVEIS)
        self.graficos = {nome: (avaliacao / arquivo).read_text(encoding="utf-8")
                         for nome, arquivo in ArquivosDasEtapas5a7.GRAFICOS.items()}

    # ---------------------------------------------------------------- arquivos publicados
    def arquivos_no_site(self):
        """{caminho dentro do site: arquivo de origem}: o SiteEstatico copia exatamente estes."""
        avaliacao = self.pastas.avaliacao_do_modelo
        etapa6 = [ArquivosDasEtapas5a7.METRICAS, ArquivosDasEtapas5a7.METRICAS_POR_PERFIL,
                  ArquivosDasEtapas5a7.SELECAO_DE_VARIAVEIS, ArquivosDasEtapas5a7.IMPORTANCIA,
                  ArquivosDasEtapas5a7.PREVISOES_NO_TESTE]
        return {
            f"{self.PASTA_ETAPA5}/{self.pastas.base_final.name}": self.pastas.base_final,
            f"{self.PASTA_ETAPA6}/{self.pastas.modelo_json.name}": self.pastas.modelo_json,
            f"{self.PASTA_ETAPA6}/{self.pastas.registro_do_treino.name}": self.pastas.registro_do_treino,
            **{f"{self.PASTA_ETAPA6}/{nome}": avaliacao / nome for nome in etapa6},
            f"{self.PASTA_ETAPA7}/{self.pastas.dados_da_demonstracao.name}": self.pastas.dados_da_demonstracao,
        }

    # ---------------------------------------------------------------- fluxograma do passo a passo
    def etapas_no_fluxo(self):
        metricas = self._metricas_de("Random Forest")
        return [
            EtapaNoFluxo(
                5, "Enriquecimento e dicionário de dados", "tabela",
                f"A base unida ganhou dados sintéticos de contrato, franquia e consumo diário: "
                f"{self.base.shape[1]} colunas, sem alterar as de entrada.",
                [ArquivoNoFluxo(f"{self.PASTA_ETAPA5}/{self.pastas.base_final.name}", "Base final de modelagem (CSV)",
                                f"{len(self.base)} clientes e {self.base.shape[1]} colunas")], "enriquecimento"),
            EtapaNoFluxo(
                6, "Treinamento e avaliação do modelo", "grafico",
                f"Um Random Forest aprendeu a prever a demanda de internet até a renovação. No teste, errou em média "
                f"{numero(metricas.mae_gb, 2)} GB.",
                [ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{ArquivosDasEtapas5a7.METRICAS}", "Métricas no teste (CSV)",
                                "Random Forest, Ridge e a projeção pela média diária"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{self.pastas.registro_do_treino.name}", "Registro do treino (JSON)",
                                "versão, dados, semente, parâmetros, variáveis e bibliotecas"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{self.pastas.modelo_json.name}", "Modelo para o navegador (JSON)",
                                "as árvores que o site usa para prever"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{ArquivosDasEtapas5a7.METRICAS_POR_PERFIL}", "Erro por perfil (CSV)",
                                "métricas da previsão por grupo"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{ArquivosDasEtapas5a7.SELECAO_DE_VARIAVEIS}", "Seleção de variáveis (CSV)",
                                "comparação dos conjuntos na validação cruzada"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{ArquivosDasEtapas5a7.IMPORTANCIA}", "Importância das variáveis (CSV)",
                                "aumento do erro ao embaralhar cada variável"),
                 ArquivoNoFluxo(f"{self.PASTA_ETAPA6}/{ArquivosDasEtapas5a7.PREVISOES_NO_TESTE}", "Previsões no teste (CSV)",
                                "demanda simulada e previsão por cliente")], "modelo"),
            EtapaNoFluxo(
                7, "Previsões, regras e recomendações", "alvo",
                "A previsão é comparada com o saldo e com os planos das planilhas: alerta de risco e até 3 planos que "
                "cobrem a demanda, na ordem da pontuação de recomendação (modelo da etapa 8).",
                [ArquivoNoFluxo(f"{self.PASTA_ETAPA7}/{self.pastas.dados_da_demonstracao.name}", "Casos e cenários (JSON)",
                                "Cliente Claro, cenários e os números da demonstração")], "recomendacoes"),
        ]

    # ---------------------------------------------------------------- etapa 5
    def secao_enriquecimento(self):
        id_do_exemplo = self.demonstracao["cliente_do_exemplo"]["id_cliente"]
        registro = self.base[self.base.id_cliente == id_do_exemplo].iloc[0]
        exemplos = exemplos_de_calculo(registro)
        linhas_dos_exemplos = "".join(
            f"<tr><td data-rotulo=\"Coluna\"><code>{_texto(linha.coluna)}</code></td>"
            f"<td data-rotulo=\"Fórmula\">{_texto(linha.formula)}</td>"
            f"<td data-rotulo=\"Com os valores\">{_texto(linha.conta_com_os_valores)}</td>"
            f"<td data-rotulo=\"Resultado\"><b>{_texto(linha.resultado)}</b></td></tr>" for linha in exemplos.itertuples())
        novas = len(colunas_novas(self.base))
        return f"""
<section id="enriquecimento" class="secao-tecnica">
  <h2>Dados acrescentados para o experimento</h2>
  <p>Datas, franquias, consumo diário e valores futuros para avaliação são simulados com semente fixa
    {self.registro['semente']}. A base mantém {len(self.base)} clientes e passa a {self.base.shape[1]} colunas,
    das quais {novas} foram acrescentadas.</p>
  <p class="nota"><b>Hipótese:</b> <code>dados_gb</code> representa o consumo típico mensal, embora a periodicidade
    não esteja documentada nas planilhas.</p>
  <details class="detalhe-documentacao">
    <summary>Fórmulas aplicadas ao registro da Cliente Claro</summary>
    <p>Registro nº {id_do_exemplo}; inclui as variáveis simuladas da demonstração.</p>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Coluna</th><th>Fórmula</th><th>Com os valores</th><th>Resultado</th></tr></thead>
      <tbody>{linhas_dos_exemplos}</tbody></table></div>
  </details>
</section>"""

    # ---------------------------------------------------------------- etapa 6
    def _metricas_de(self, modelo):
        return self.metricas[self.metricas.modelo == modelo].iloc[0]

    def secao_modelo(self):
        floresta, baseline = self._metricas_de("Random Forest"), self._metricas_de("Baseline (média diária do ciclo)")
        linhas_das_metricas = "".join(
            f"<tr><td data-rotulo=\"Modelo\">{_texto(linha.modelo)}</td>"
            f"<td data-rotulo=\"Clientes\">{linha.clientes_avaliados}</td>"
            f"<td data-rotulo=\"MAE\">{numero(linha.mae_gb, 2)} GB</td><td data-rotulo=\"RMSE\">{numero(linha.rmse_gb, 2)} GB</td>"
            f"<td data-rotulo=\"R²\">{numero(linha.r2, 3)}</td></tr>" for linha in self.metricas.itertuples())
        linhas_por_perfil = "".join(
            f"<tr><td data-rotulo=\"Perfil\">{_texto(linha.perfil_grupo)}</td>"
            f"<td data-rotulo=\"Clientes de teste\">{linha.clientes_de_teste}</td>"
            f"<td data-rotulo=\"MAE do Random Forest\">{numero(linha.mae_floresta_gb, 2)} GB</td>"
            f"<td data-rotulo=\"MAE da média diária\">{numero(linha.mae_baseline_gb, 2)} GB</td>"
            f"<td data-rotulo=\"Demanda futura simulada\">{numero(linha.demanda_media_real_gb, 1)} GB</td></tr>"
            for linha in self.metricas_por_perfil.itertuples())
        linhas_da_selecao = "".join(
            f"<tr{' class=\"escolhida\"' if linha.escolhido else ''}><td data-rotulo=\"Conjunto\">{_texto(linha.conjunto)}"
            f"{' <b>(escolhido)</b>' if linha.escolhido else ''}</td>"
            f"<td data-rotulo=\"MAE na validação cruzada\">{numero(linha.mae_validacao_cruzada_gb, 2)} GB</td></tr>"
            for linha in self.selecao.itertuples())
        treinado_em = pd.Timestamp(self.registro["treinado_em"])
        parametros = self.registro["parametros"]
        return f"""
<section id="modelo" class="secao-tecnica">
  <h2>Resultado da previsão de consumo</h2>
  <p class="resultado-documentacao">No teste sintético, o Random Forest teve erro médio de
    <b>{numero(floresta.mae_gb, 2)} GB</b>, contra <b>{numero(baseline.mae_gb, 2)} GB</b> da projeção pela média diária.
    A comparação usa os mesmos {int(floresta.clientes_avaliados)} clientes com média diária disponível.</p>
  <details class="detalhe-documentacao">
    <summary>Método e métricas da previsão</summary>
    <p>A regressão estima os GB adicionais até renovar. Usa consumo observado, ritmo recente, hábitos e calendário;
      a franquia e o nome do plano não são entradas desse modelo.</p>
    <p>A divisão por cliente, estratificada pelo perfil, reservou {self.registro['clientes_no_treino']} clientes
      para treino e {self.registro['clientes_no_teste']} para teste. A codificação de categorias e a seleção de
      variáveis por validação cruzada de cinco dobras são ajustadas no treino.</p>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Modelo</th><th>Clientes</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
      <tbody>{linhas_das_metricas}</tbody></table></div>
    <p class="nota">MAE é o erro absoluto médio; RMSE dá mais peso aos erros grandes; R² compara a variação
      explicada com uma previsão constante. Ciclos sem dias decorridos ficam fora da comparação com a média diária.</p>
    <p class="nota">O consumo observado pode esconder demanda após esgotar a franquia. Mudanças futuras de ritmo
      sorteadas pela simulação não são observáveis pelo modelo.</p>
  </details>
  <details class="detalhe-documentacao">
    <summary>Diagnósticos, parâmetros e arquivos da previsão</summary>
    <h3>Seleção de variáveis no treino</h3>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Conjunto</th><th>MAE na validação cruzada</th></tr></thead>
      <tbody>{linhas_da_selecao}</tbody></table></div>
    <h3>Erro por perfil</h3>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Perfil</th><th>Clientes de teste</th><th>MAE do Random Forest</th><th>MAE da média diária</th>
        <th>Demanda futura simulada</th></tr></thead><tbody>{linhas_por_perfil}</tbody></table></div>
    <div class="graficos">
      <figure class="cartao"><h4>Previsão × demanda simulada</h4>{self.graficos['previsto_vs_real']}
        <figcaption>Cada ponto é um cliente de teste; a linha indica previsão igual ao valor futuro simulado.</figcaption></figure>
      <figure class="cartao"><h4>Distribuição dos erros</h4>{self.graficos['distribuicao_dos_erros']}
        <figcaption>Erro = demanda simulada − previsão. Valores positivos indicam subestimação.</figcaption></figure>
    </div>
    <figure class="cartao"><h4>Importância das variáveis</h4>{self.graficos['importancia']}
      <figcaption>Aumento do MAE ao embaralhar uma variável no teste.</figcaption></figure>
    <p class="versao">{parametros['n_estimators']} árvores; mínimo de {parametros['min_samples_leaf']} registros por folha;
      {numero(parametros['max_features'] * 100, 0)}% das variáveis por divisão; semente {parametros['random_state']}.
      Versão <code>{_texto(self.registro['versao_do_modelo'])}</code>, treinada em {treinado_em:%d/%m/%Y %H:%M}.</p>
  </details>
</section>"""

    # ---------------------------------------------------------------- etapa 7
    def secao_recomendacoes(self):
        return f"""
<section id="recomendacoes" class="secao-tecnica">
  <h2 id="titulo-arquitetura">Como o sistema decide</h2>
  <figure class="arquitetura" aria-labelledby="titulo-arquitetura" aria-describedby="legenda-arquitetura">
    <div class="arquitetura-trilhas">
      <div class="arquitetura-trilha">
        <h3>Previsão de consumo</h3>
        <ol class="fluxo-treinamento">
          <li><b>Consumo diário simulado</b><span>Histórico até a data de referência</span></li>
          <li><b>Random Forest de demanda</b><span>Estima os GB até renovar</span></li>
        </ol>
      </div>
      <div class="arquitetura-trilha">
        <h3>Ordenação de planos</h3>
        <ol class="fluxo-treinamento">
          <li><b>Ofertas e notas simuladas</b><span>Respostas e avaliações dos clientes</span></li>
          <li><b>Random Forest de propensão + Pearson</b><span>Calcula a pontuação híbrida</span></li>
        </ol>
      </div>
    </div>
    <p class="arquitetura-convergencia"><span aria-hidden="true">↓</span> Modelos já treinados entram na demonstração</p>
    <ol class="fluxo-simples">
      <li><b>Informar o cenário</b><span>Plano, GB usados e dias decorridos</span></li>
      <li><b>Prever e analisar</b><span>Demanda restante, saldo e risco</span></li>
      <li><b>Filtrar planos</b><span>Cobertura da demanda e custo</span></li>
      <li><b>Ordenar e explicar</b><span>Mensagem, análise e opções</span></li>
    </ol>
    <figcaption id="legenda-arquitetura">O treinamento ocorre na montagem do projeto. No navegador, os ajustes
      recalculam previsões com modelos já treinados; não iniciam um novo treinamento.</figcaption>
  </figure>
  <p>O saldo é a franquia menos o consumo, com mínimo zero. Só entram na comparação os <b>GB principais</b>:
    bônus temporários e benefícios para aplicativos ficam de fora. São exibidas até três opções que cobrem a demanda;
    quando o plano atual basta, ele aparece junto às alternativas mais baratas.</p>
  <details class="detalhe-documentacao">
    <summary>Resultado por perfil de cliente</summary>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Perfil</th><th>Uso típico ÷ franquia (mediana)</th><th>Risco alto ou franquia esgotada</th>
        <th>Ação</th></tr></thead><tbody>{self._linhas_por_perfil()}</tbody></table></div>
    <p class="nota">Diagnóstico simulado dos {len(self.base)} clientes; não representa incidência medida em uma operação real.</p>
  </details>
</section>"""

    def secao_simulacao(self):
        dados = self.demonstracao
        cenario = dados["cenarios"]["mais_internet"]
        plano = next(plano for plano in dados["planos"] if plano["nome"] == cenario["plano"])
        utilizados = plano["gb_principais"]
        dias = cenario["dias_desde_renovacao"]
        atributos = AtributosDaSimulacao(date.fromisoformat(dados["data_de_referencia"]),
                                        dados["dias_de_um_mes"], dados["dias_do_ciclo"]).montar(
            utilizados, dias, cenario["uso_tipico_mes_gb"], cenario["habitos"])
        modelo = json.loads(self.pastas.modelo_json.read_text(encoding="utf-8"))
        previstos = FlorestaExportada(modelo).prever_um(atributos)
        total = utilizados + previstos
        maior_franquia = max(plano["gb_principais"] for plano in dados["planos"])
        conclusao = (f"A demanda supera os {numero(maior_franquia, 0)} GB do maior plano; por isso, nenhuma opção cobre o cenário."
                     if max(total, cenario["uso_tipico_mes_gb"]) > maior_franquia
                     else "A demonstração apresenta os planos que cobrem a demanda estimada.")
        return f"""
<section id="simulacao" class="secao-tecnica">
  <h2>Como usar a simulação</h2>
  <dl class="controles-da-simulacao">
    <div><dt>Plano de referência</dt><dd>Define a franquia disponível e o preço para comparar as opções.</dd></div>
    <div><dt>GB utilizados</dt><dd>Quanto já foi consumido no ciclo, entre zero e a franquia selecionada.</dd></div>
    <div><dt>Dias decorridos</dt><dd>Tempo desde o início de um ciclo simulado de {dados['dias_do_ciclo']} dias.
      Com {dias} dias decorridos, faltam {atributos['dias_ate_renovacao']} para renovar.</dd></div>
  </dl>
  <p><b>Ao trocar o plano, o controle de GB utilizados recebe a franquia dele.</b> Isso representa 100% consumidos;
    ajuste o controle para o consumo que deseja testar. Consumo, análise e opções acompanham os ajustes;
    “Ver simulação do atendimento” abre a aba Análise.</p>
  <div class="exemplo-da-simulacao">
    <p class="etapa">Exemplo: {html.escape(plano['nome'])}, com {dias} dias decorridos</p>
    <p class="conta-da-simulacao"><b>{numero(utilizados, 0)} GB usados</b> +
      <b>{numero(previstos, 1)} GB previstos até renovar</b> = <b>{numero(total, 1)} GB no ciclo inteiro</b>.</p>
    <p>{conclusao}</p>
  </div>
  <p>A demanda para comparar planos é o <b>maior valor entre o total do ciclo e o uso típico mensal</b>.
    O cenário “Estou usando mais internet” usa {numero(cenario['uso_tipico_mes_gb'], 0)} GB/mês como referência;
    “Quero avaliar se posso economizar”, {numero(dados['cenarios']['economia']['uso_tipico_mes_gb'], 0)} GB/mês.
    Os hábitos não informados vêm do grupo correspondente na base.</p>
  <p class="nota">O modelo considera ritmo e hábitos; a previsão não é apenas uma regra de três.
    Mantidos consumo e hábitos, mudar a franquia não muda a previsão. A troca no seletor pode mudá-la porque
    também preenche o consumo.</p>
</section>"""

    def _linhas_por_perfil(self):
        nome_curto_por_perfil = dict(zip(self.base.perfil_grupo, self.base.origem))
        linhas = []
        for perfil in self.demonstracao["resumo_por_perfil"]:
            risco = perfil["percentual_alto"] + perfil["percentual_esgotada"]
            acao = self.ACAO_POR_PERFIL.get(nome_curto_por_perfil.get(perfil["perfil_grupo"]), "")
            linhas.append(f"<tr><td data-rotulo=\"Perfil\">{_texto(perfil['perfil_grupo'])}</td>"
                          f"<td data-rotulo=\"Uso típico ÷ franquia\">{numero(perfil['uso_tipico_sobre_franquia_mediana'] * 100, 0)}%</td>"
                          f"<td data-rotulo=\"Risco alto ou esgotada\">{numero(risco, 0)}% dos clientes</td>"
                          f"<td data-rotulo=\"O que o CLARO GUIA faz\">{_texto(acao)}</td></tr>")
        return "".join(linhas)


class SecaoDaEtapa8:
    """Lê os resultados da etapa 8 (histórico de ofertas simulado e modelos de recomendação) e devolve a
    seção (HTML), o passo do fluxograma e a lista de arquivos do site."""

    PASTA = "etapa8_recomendacao"
    ANCORA = "modelo-de-recomendacao"
    NOME_DO_RANDOM_FOREST = "Random Forest (propensão)"
    NOME_DA_COLABORATIVA = "Filtragem colaborativa (usuário a usuário, Pearson)"
    NOME_DO_HIBRIDO = "Híbrido ponderado (peso escolhido na validação)"

    def __init__(self, pastas, pagina_inicial):
        self.pastas = pastas
        self.pagina_inicial = pagina_inicial  # endereço da página da demonstração, para o link "Voltar"
        avaliacao = pastas.avaliacao_da_recomendacao
        self.registro = json.loads(pastas.registro_da_recomendacao.read_text(encoding="utf-8"))
        self.metricas = pd.read_csv(avaliacao / ArquivosDaEtapa8.METRICAS)
        self.planos = pd.read_csv(avaliacao / ArquivosDaEtapa8.PLANOS)
        self.classificacao = json.loads((avaliacao / ArquivosDaEtapa8.CLASSIFICACAO).read_text(encoding="utf-8"))
        self.graficos = {nome: (avaliacao / arquivo).read_text(encoding="utf-8")
                         for nome, arquivo in ArquivosDaEtapa8.GRAFICOS.items()}

    # ---------------------------------------------------------------- arquivos publicados
    def arquivos_no_site(self):
        """{caminho dentro do site: arquivo de origem}: o .joblib (pickle) não vai para o site."""
        avaliacao = self.pastas.avaliacao_da_recomendacao
        return {
            f"{self.PASTA}/{self.pastas.historico_de_ofertas.name}": self.pastas.historico_de_ofertas,
            f"{self.PASTA}/{self.pastas.modelo_de_propensao_json.name}": self.pastas.modelo_de_propensao_json,
            f"{self.PASTA}/{self.pastas.registro_da_recomendacao.name}": self.pastas.registro_da_recomendacao,
            **{f"{self.PASTA}/{nome}": avaliacao / nome
               for nome in (ArquivosDaEtapa8.METRICAS, ArquivosDaEtapa8.ACEITACAO_POR_GRUPO, ArquivosDaEtapa8.PLANOS,
                            ArquivosDaEtapa8.PESOS, ArquivosDaEtapa8.INCERTEZA, ArquivosDaEtapa8.CALIBRACAO,
                            ArquivosDaEtapa8.REFERENCIAS, ArquivosDaEtapa8.CLASSIFICACAO,
                            ArquivosDaEtapa8.MATRIZ_DE_CONFUSAO, ArquivosDaEtapa8.RELATORIO_DE_CLASSIFICACAO,
                            ArquivosDaEtapa8.CURVAS)},
            **{f"{self.PASTA}/{nome}": avaliacao / nome for nome in ArquivosDaEtapa8.GRAFICOS.values()},
        }

    # ---------------------------------------------------------------- fluxograma do passo a passo
    def _metricas_de(self, modelo):
        return self.metricas[self.metricas.modelo == modelo].iloc[0]

    def etapa_no_fluxo(self):
        hibrido = self._metricas_de(self.NOME_DO_HIBRIDO)
        return EtapaNoFluxo(
            8, "Histórico de ofertas (simulado) e modelo de recomendação", "ajustes",
            "Um histórico de ofertas aceitas e recusadas foi simulado e treinou dois modelos; a demonstração usa a "
            f"combinação ponderada, escolhida na validação. No último ciclo, a 1ª oferta do híbrido foi aceita em "
            f"{numero(hibrido.acerto_da_primeira_oferta * 100, 1)}% dos clientes com aceitação e recusa no ciclo simulado.",
            [ArquivoNoFluxo(f"{self.PASTA}/{self.pastas.historico_de_ofertas.name}", "Histórico de ofertas simulado (CSV)",
                            "uma linha por oferta: ciclo, plano oferecido e resposta"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.METRICAS}", "Comparação dos modelos (CSV)",
                            "AUC e acerto da 1ª oferta, no ciclo de teste"),
             ArquivoNoFluxo(f"{self.PASTA}/{self.pastas.registro_da_recomendacao.name}", "Registro do treino (JSON)",
                            "regra da simulação, parâmetros, variáveis e limitações"),
             ArquivoNoFluxo(f"{self.PASTA}/{self.pastas.modelo_de_propensao_json.name}", "Modelo para o navegador (JSON)",
                            "as árvores do Random Forest, que o site junta com a filtragem colaborativa"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.PESOS}", "Pesos na validação (CSV)",
                            "comparação anterior ao teste, incluindo modelos isolados e média 50/50"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.INCERTEZA}", "Incerteza da diferença (JSON)",
                            "bootstrap pareado por cliente, híbrido selecionado menos Random Forest"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.CALIBRACAO}", "Diagnóstico de calibração do RF (CSV)",
                            "probabilidade média e frequência de aceitação por faixa no teste sintético"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.CLASSIFICACAO}", "Diagnóstico de classificação (JSON)",
                            "ROC-AUC, average precision, matriz de confusão e métricas por classe"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.REFERENCIAS}", "Referências científicas verificadas (JSON)",
                            "artigos originais de revistas, DOI, publicação final, recorte e fonte dos indicadores"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.ACEITACAO_POR_GRUPO}", "Aceitação simulada por grupo (CSV)",
                            "respostas às ofertas por perfil e plano"),
             ArquivoNoFluxo(f"{self.PASTA}/{ArquivosDaEtapa8.PLANOS}", "Catálogo de planos fictícios (CSV)",
                            "franquias, bônus e mensalidades")],
            self.ANCORA)

    # ---------------------------------------------------------------- seção
    def secao(self):
        regra = self.registro["regra_do_historico_simulado"]
        hibrido = self._metricas_de(self.NOME_DO_HIBRIDO)
        total_de_ofertas = (self.registro["ofertas_no_treino"] + self.registro["ofertas_na_validacao"]
                            + self.registro["ofertas_no_teste"])
        peso = self.registro["selecao_do_peso"]["peso_random_forest"]
        comparacao = self.registro["comparacao_pareada_no_teste"]
        diagnostico = self.registro["diagnostico_de_propensao_no_teste"]
        pesos = "; ".join(numero(linha["peso_random_forest"], 2)
                          for linha in self.registro["selecao_do_peso"]["comparacao_na_validacao"])
        ciclo_de_validacao = regra["ciclos"] - 1
        conclusao = ("A comparação estatística indica ganho do híbrido sobre o Random Forest neste teste sintético."
                     if comparacao["evidencia_de_ganho"]
                     else "A comparação estatística não sustenta superioridade do híbrido sobre o Random Forest neste teste sintético.")
        return f"""
<section id="{self.ANCORA}" class="secao-tecnica">
  <h2>Resultado da ordenação de planos</h2>
  <p>O recomendador combina um Random Forest de propensão, que estima a aceitação de uma oferta,
    com filtragem colaborativa de Pearson, que usa notas de clientes semelhantes.
    O peso escolhido na validação foi de <b>{numero(peso * 100, 0)}% e {numero((1 - peso) * 100, 0)}%</b>,
    respectivamente. A pontuação de 0 a 100 serve para ordenar opções; não representa uma probabilidade de aceitação.</p>
  <p class="resultado-documentacao">A oferta colocada em primeiro lugar pelo híbrido foi aceita em
    <b>{numero(hibrido.acerto_primeira_oferta_todos * 100, 1)}%</b> dos
    <b>{int(hibrido.clientes_avaliados)} clientes</b> do ciclo de teste sintético.
    A métrica avalia a ordenação das {regra['ofertas_por_ciclo']} ofertas sorteadas por cliente no ciclo,
    antes dos filtros de cobertura e economia do site.</p>
  <p>{conclusao} Esses resultados não demonstram conversão, economia ou satisfação em uma operação real.</p>
  {self._graficos_da_classificacao()}
  <details id="protocolo-cientifico" class="detalhe-documentacao">
    <summary>Protocolo científico: combinação e validação</summary>
    <h3>Combinação dos modelos</h3>
    <p><code>pontuação = peso × propensão_RF + (1 − peso) × (nota_colaborativa / 10)</code>.
      Normalizar a nota para a faixa de 0 a 1 não a transforma em probabilidade nem torna os componentes equivalentes.
      O resultado em [0, 1] é multiplicado por 100 para exibição. A fórmula e a grade de pesos são escolhas do protótipo.</p>
    <h3>Separação temporal</h3>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Fase</th><th>Ciclos</th><th>Procedimento</th></tr></thead><tbody>
      <tr><td data-rotulo="Fase">Treino inicial</td><td data-rotulo="Ciclos">1–{ciclo_de_validacao - 1}</td>
        <td data-rotulo="Procedimento">Ajustar Random Forest, codificação One-Hot e filtragem colaborativa.</td></tr>
      <tr><td data-rotulo="Fase">Validação</td><td data-rotulo="Ciclos">{ciclo_de_validacao}</td>
        <td data-rotulo="Procedimento">Comparar os pesos do RF: {pesos}. Os extremos usam apenas um componente.</td></tr>
      <tr><td data-rotulo="Fase">Reajuste</td><td data-rotulo="Ciclos">1–{ciclo_de_validacao}</td>
        <td data-rotulo="Procedimento">Reajustar os componentes e o One-Hot, mantendo o peso selecionado.</td></tr>
      <tr><td data-rotulo="Fase">Teste</td><td data-rotulo="Ciclos">{regra['ciclos']}</td>
        <td data-rotulo="Procedimento">Avaliar sem selecionar pesos, hiperparâmetros ou calibração pelas respostas do teste.</td></tr>
      </tbody></table></div>
    <p>A seleção maximiza o acerto da primeira oferta entre clientes com aceitação e recusa na validação.
      Os desempates, definidos antes do teste, são maior AUC, proximidade de 0,5 e menor peso.
      O reajuste pode mudar o desempenho relativo; isso não autoriza escolher novamente pelo teste.</p>
    <p>A comparação final mantém os componentes isolados, a média original 50/50, popularidade, regra de preço
      e sorteio. NDCG@3, MRR e acerto em todos os clientes são diagnósticos; não selecionam o peso.</p>
    <h3>Compatibilidade técnica</h3>
    <p>Os nomes antigos <code>chance_de_aceitar</code>, <code>chance_random_forest</code> e
      <code>chance_filtragem_colaborativa</code> continuam no Python e no JSON como aliases de
      <code>pontuacao_recomendacao</code>, <code>propensao_random_forest</code> e <code>pontuacao_colaborativa</code>.
      Esses nomes não mudam a interpretação da saída. Sem peso explícito, o construtor mantém o padrão 50/50.</p>
  </details>
  <details class="detalhe-documentacao">
    <summary>Métricas, incerteza e diagnóstico da ordenação</summary>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Modelo</th><th>AUC</th><th>Acerto condicional da 1ª oferta</th></tr></thead>
      <tbody>{self._linhas_das_metricas()}</tbody></table></div>
    <p class="nota">AUC mede a discriminação entre aceitação e recusa em todas as ofertas do teste (0,5 equivale ao acaso).
      O acerto condicional usa apenas os {comparacao['clientes_elegiveis']} clientes com ambas as respostas;
      não é a taxa geral de aceitação. O indicador para todos inclui os {int(hibrido.clientes_avaliados)} clientes,
      independentemente de terem aceitado alguma oferta.</p>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Modelo</th><th>Clientes</th><th>Acerto da 1ª oferta: todos</th><th>NDCG@3 observado</th>
        <th>MRR observado</th></tr></thead><tbody>{self._linhas_da_ordenacao()}</tbody></table></div>
    <p class="nota">NDCG@3 usa aceitação como relevância binária, desconta posições e normaliza o ganho das três
      primeiras ofertas pelo ranking ideal dos mesmos candidatos. MRR é o inverso da posição da primeira oferta aceita.
      Ambas são médias por cliente, sem ponderar pelo número de ofertas; quem recusou todas recebe zero.</p>
    <p class="nota">Os candidatos são as mesmas ofertas observadas para cada modelo; planos não ofertados não são
      tratados como recusados. Empates na avaliação usam o nome do plano; na demonstração, preço e nome.
      Para o sorteio, calcula-se a esperança exata sob permutação uniforme das ofertas.</p>
    <p>Diferença de acerto condicional, híbrido menos Random Forest:
      {numero(comparacao['diferenca_de_acerto'] * 100, 2)} pontos percentuais.
      IC de 95%: [{numero(comparacao['ic95_inferior'] * 100, 2)}; {numero(comparacao['ic95_superior'] * 100, 2)}],
      obtido por bootstrap percentil pareado, com {comparacao['reamostragens']} reamostragens de clientes com reposição
      e semente {comparacao['semente']}. Cada reamostragem preserva os dois acertos de cada cliente;
      pares com clientes diferentes ou acertos ausentes são rejeitados.</p>
    <p class="nota">Se o intervalo inclui zero, não sustenta uma afirmação de ganho. Ele é condicionado aos modelos
      ajustados e à regra sintética: não incorpora a variação de retreinar nem comprova superioridade em clientes reais,
      outras sementes ou regras. O bootstrap é uma análise complementar do projeto, não um método atribuído aos artigos.</p>
    <p>Diagnóstico de propensão do RF: Brier {numero(diagnostico['brier'], 4)} e
      log loss {numero(diagnostico['log_loss'], 4)}. São registradas frequências e probabilidades médias em dez
      faixas fixas; interpretar os indicadores junto às frequências e ao tamanho de cada faixa, pois não isolam calibração.
      Não há recalibração pelo teste. Uma calibração futura exige dados separados ou previsões fora da amostra de treino,
      preservando um teste independente.</p>
  </details>
  <details class="detalhe-documentacao">
    <summary>Catálogo e geração do histórico de ofertas</summary>
    <p>O catálogo contém {len(self.planos)} planos fictícios, com mensalidades das planilhas.</p>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Plano</th><th>GB principais</th><th>Bônus de uso livre</th><th>Bônus só para apps</th>
        <th>Mensalidade</th></tr></thead><tbody>{self._linhas_dos_planos()}</tbody></table></div>
    <p>Foram geradas {numero(total_de_ofertas, 0)} ofertas, com semente {regra['semente']}.
      A regra considera cobertura, diferença de preço e preferências por grupo; sorteia aceitação e notas de 0 a 10.
      Notas e aceitações compartilham a mesma preferência latente simulada, com aleatoriedade.
      A correlação de Pearson compara as notas dos mesmos planos e pondera a contribuição de clientes semelhantes.</p>
    <p class="nota">Para visitantes sem histórico individual, a nota colaborativa é a média do grupo.
      O peso foi selecionado para clientes com histórico. Para avaliar novos clientes, é preciso separá-los
      integralmente, impedindo que suas interações entrem no treino, na validação ou nas médias de grupo.</p>
  </details>
</section>
<section id="limites" class="secao-tecnica">
  <h2>Limites e próximos passos</h2>
  <ul class="lista">
    <li>As métricas medem a recuperação das regras sintéticas de consumo e aceitação; não validam preferências reais.</li>
    <li>O teste de recomendação já foi consultado em versões anteriores. A confirmação do resultado exige novas
      observações ainda não examinadas.</li>
    <li>Uma evolução para uso real requer consumo diário e histórico de ofertas reais autorizados,
      avaliação com novos clientes, mais cenários e sementes e validação do fluxo completo antes da operação.</li>
  </ul>
</section>
<section id="referencias" class="secao-tecnica">
  <h2>Fundamentação científica</h2>
  <p>Os artigos selecionados apoiam a separação entre validação e teste e a avaliação da ordenação.
    Não comprovam que a combinação RF + Pearson seja a melhor arquitetura para telefonia.</p>
  <details class="detalhe-documentacao">
    <summary>Artigos e critérios da seleção</summary>
    <p>Somente artigos originais revisados por pares, em versão final de revista, publicados online entre
      29/10/2021 e 29/10/2026; conferência em 29/10/2026. A seleção é dirigida ao problema, não uma revisão sistemática.
      Preprints, revisões, editoriais, anais e documentação de bibliotecas foram excluídos da fundamentação científica.</p>
    {BibliografiaDaRecomendacao().como_html()}
    <p class="nota">Critério de impacto adotado: JIF de 2025 acima de 5, verificado nos editores.
      ACM TOIS: 11,2 (<a href="https://www.acm.org/media-center/2026/july/impact-factors-2026">ACM</a>);
      IEEE TPAMI: 20,4 (<a href="https://www.computer.org/publications">IEEE Computer Society</a>).
      O limiar é operacional, sem pretensão de definir universalmente alto impacto. O indicador pertence à revista
      e não garante a validade de um resultado.</p>
    <p class="nota">O projeto adapta princípios de avaliação; não reproduz o benchmark completo DaisyRec nem
      demonstra que a arquitetura ou o peso 50/50 sejam ótimos para telefonia.</p>
  </details>
</section>"""

    def _graficos_da_classificacao(self):
        classificacao = self.classificacao
        verdadeiros_negativos, falsos_positivos = classificacao["matriz_de_confusao"][0]
        falsos_negativos, verdadeiros_positivos = classificacao["matriz_de_confusao"][1]
        nomes_dos_graficos = {
            "roc": "Curva ROC (SVG)",
            "precisao_revocacao": "Curva precisão–revocação (SVG)",
            "matriz_de_confusao": "Matriz de confusão (SVG)",
            "importancia": "Importância por permutação (SVG)",
            "arvore": "Árvore ilustrativa (SVG)",
        }
        arquivos = [
            (ArquivosDaEtapa8.MATRIZ_DE_CONFUSAO, "Matriz de confusão (CSV)"),
            (ArquivosDaEtapa8.RELATORIO_DE_CLASSIFICACAO, "Métricas por classe (CSV)"),
            (ArquivosDaEtapa8.CURVAS, "Comparação das curvas (CSV)"),
            (ArquivosDaEtapa8.CLASSIFICACAO, "Diagnóstico completo (JSON)"),
            *((arquivo, nomes_dos_graficos[nome]) for nome, arquivo in ArquivosDaEtapa8.GRAFICOS.items()),
        ]
        links = "; ".join(f'<a href="{self.PASTA}/{arquivo}" download>{titulo}</a>'
                          for arquivo, titulo in arquivos)
        return f"""
  <details id="graficos-da-recomendacao" class="detalhe-documentacao">
    <summary>Random Forest: curvas ROC, precisão–revocação e matriz de confusão</summary>
    <p>Diagnósticos de classificação gerados com scikit-learn nas
      <b>{numero(classificacao['ofertas_avaliadas'], 0)} ofertas</b> do último ciclo sintético,
      correspondentes a {numero(self.metricas.iloc[0].clientes_avaliados, 0)} clientes.
      A unidade é a oferta: recusa = 0; aceitação = 1 (classe positiva).
      Esses resultados não medem preferências de clientes reais.</p>
    <h3>Discriminação entre aceitação e recusa</h3>
    <div class="graficos">
      <figure class="cartao"><h4>Curva ROC</h4>{self.graficos['roc']}
        <figcaption>Compara sensibilidade e taxa de falsos positivos ao variar o limiar.
          ROC-AUC de 0,5 é a referência aleatória; valores maiores indicam melhor discriminação.</figcaption></figure>
      <figure class="cartao"><h4>Precisão–revocação</h4>{self.graficos['precisao_revocacao']}
        <figcaption>Precisão: proporção de aceitações entre previsões positivas.
          Revocação: proporção das aceitações observadas recuperada pelo modelo.
          A referência é a taxa de aceitação do teste.</figcaption></figure>
    </div>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Modelo</th><th>ROC-AUC</th><th>Average precision (AP)</th></tr></thead>
      <tbody>{self._linhas_das_curvas()}</tbody></table></div>
    <p class="nota">AP resume precisão–revocação por diferenças de revocação; não é ROC-AUC nem a área
      trapezoidal da curva PR. As curvas usam os mesmos candidatos; não reajustam os modelos nem selecionam limiares pelo teste.</p>
    <h3>Matriz de confusão do Random Forest</h3>
    <p>Limiar diagnóstico fixo de <b>{numero(classificacao['limiar'], 2)}</b>, definido antes desta avaliação:
      propensão igual ou maior prevê aceitação. O limiar não foi escolhido pelas respostas do teste e não altera
      a ordenação dos planos. Linhas = respostas observadas; colunas = previstas.</p>
    <figure class="cartao">{self.graficos['matriz_de_confusao']}
      <figcaption>Contagens de ofertas corretamente classificadas e erros no ciclo de teste.</figcaption></figure>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <caption>Resposta observada × resposta prevista: contagem de ofertas</caption>
      <thead><tr><th>Resposta observada</th><th>Recusa prevista (0)</th><th>Aceitação prevista (1)</th></tr></thead>
      <tbody>
        <tr><td data-rotulo="Resposta observada">Recusa (0)</td>
          <td data-rotulo="Recusa prevista (0)">{verdadeiros_negativos} — verdadeiros negativos</td>
          <td data-rotulo="Aceitação prevista (1)">{falsos_positivos} — falsos positivos</td></tr>
        <tr><td data-rotulo="Resposta observada">Aceitação (1)</td>
          <td data-rotulo="Recusa prevista (0)">{falsos_negativos} — falsos negativos</td>
          <td data-rotulo="Aceitação prevista (1)">{verdadeiros_positivos} — verdadeiros positivos</td></tr>
      </tbody></table></div>
    <div class="rolagem-tabela"><table class="tabela tabela-responsiva">
      <thead><tr><th>Classe</th><th>Precisão</th><th>Sensibilidade</th><th>F1</th><th>Suporte (ofertas)</th></tr></thead>
      <tbody>{self._linhas_da_classificacao()}</tbody></table></div>
    <p class="nota">Acurácia: <b>{numero(classificacao['acuracia'] * 100, 1)}%</b> das ofertas corretamente
      classificadas nesse limiar. F1 combina precisão e sensibilidade; suporte conta as respostas observadas de
      cada classe. Essas métricas diferem do acerto da primeira oferta por cliente.</p>
    <h3>Como interpretar o Random Forest</h3>
    <figure class="cartao"><h4>Importância por permutação</h4>{self.graficos['importancia']}
      <figcaption>Queda de ROC-AUC ao embaralhar cada variável no teste, mantendo o modelo ajustado.
        Barras de erro: um desvio-padrão em {config.REPETICOES_DA_IMPORTANCIA_DA_OFERTA} repetições.
        Variáveis relacionadas podem dividir importância. A análise não demonstra causalidade nem seleciona
        variáveis pelo teste.</figcaption></figure>
    <figure class="cartao"><h4>Uma árvore da floresta</h4>
      <div class="rolagem-da-arvore" tabindex="0" role="region" aria-label="Árvore ilustrativa: role para ver todos os ramos">
        {self.graficos['arvore']}</div>
      <figcaption>Uma das {self.registro['parametros']['n_estimators']} árvores, com visualização cortada na
        profundidade {config.PROFUNDIDADE_DA_ARVORE_ILUSTRATIVA}. É uma ilustração do modelo ajustado;
        a previsão final combina as probabilidades de todas as árvores. Os nós descrevem as amostras de ajuste,
        com proporções na ordem [recusa, aceitação]. Role na horizontal para examinar os ramos.</figcaption></figure>
    <p class="nota">Downloads: {links}.</p>
  </details>"""

    def _linhas_da_classificacao(self):
        return "".join(
            f'<tr><td data-rotulo="Classe">{_texto(linha["classe"])}</td>'
            f'<td data-rotulo="Precisão">{numero(linha["precisao"], 3)}</td>'
            f'<td data-rotulo="Sensibilidade">{numero(linha["sensibilidade"], 3)}</td>'
            f'<td data-rotulo="F1">{numero(linha["f1"], 3)}</td>'
            f'<td data-rotulo="Suporte (ofertas)">{int(linha["suporte"])}</td></tr>'
            for linha in self.classificacao["relatorio_por_classe"])

    def _linhas_das_curvas(self):
        return "".join(
            f'<tr><td data-rotulo="Modelo">{_texto(linha["modelo"])}</td>'
            f'<td data-rotulo="ROC-AUC">{numero(linha["roc_auc"], 3)}</td>'
            f'<td data-rotulo="Average precision (AP)">{numero(linha["average_precision"], 3)}</td></tr>'
            for linha in self.classificacao["comparacao_de_curvas"])

    def _linhas_dos_planos(self):
        return "".join(
            f"<tr><td data-rotulo=\"Plano\">{_texto(plano.nome)}</td>"
            f"<td data-rotulo=\"GB principais\">{numero(plano.gb_principais, 0)} GB</td>"
            f"<td data-rotulo=\"Bônus de uso livre\">{numero(plano.bonus_livre_gb, 0)} GB</td>"
            f"<td data-rotulo=\"Bônus só para apps\">{numero(plano.bonus_para_apps_gb, 0)} GB</td>"
            f"<td data-rotulo=\"Mensalidade\">R$ {numero(plano.mensalidade_brl, 2)}</td></tr>"
            for plano in self.planos.itertuples())

    def _linhas_das_metricas(self):
        return "".join(
            f"<tr><td data-rotulo=\"Modelo\">{_texto(linha.modelo)}</td>"
            f"<td data-rotulo=\"AUC\">{numero(linha.auc, 3)}</td>"
            f"<td data-rotulo=\"Acerto condicional da 1ª oferta\">{numero(linha.acerto_da_primeira_oferta * 100, 1)}%</td></tr>"
            for linha in self.metricas.itertuples())

    def _linhas_da_ordenacao(self):
        return "".join(
            f'<tr><td data-rotulo="Modelo">{_texto(linha.modelo)}</td>'
            f'<td data-rotulo="Clientes">{int(linha.clientes_avaliados)}</td>'
            f'<td data-rotulo="Acerto: todos">{numero(linha.acerto_primeira_oferta_todos * 100, 1)}%</td>'
            f'<td data-rotulo="NDCG@3 observado">{numero(linha.ndcg_3_observado, 3)}</td>'
            f'<td data-rotulo="MRR observado">{numero(linha.mrr_observado, 3)}</td></tr>'
            for linha in self.metricas.itertuples())
