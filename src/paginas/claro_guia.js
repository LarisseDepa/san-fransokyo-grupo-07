/*
 * CLARO GUIA: demonstração interativa (simulação acadêmica; não é um serviço oficial).
 *
 * - O Random Forest roda AQUI, no navegador, a partir do JSON exportado por src/modelo_de_demanda.py.
 *   O visitante nunca treina o modelo: só faz previsões.
 * - As contas repetem src/formulas.py e src/recomendacao.py. Um teste (tests/test_site.py) abre esta
 *   página no Chrome e confere que os resultados são iguais aos do Python.
 * - Nada é enviado a lugar nenhum: não há login, telefone, mensagens, compras nem pagamentos.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ formatação (padrão brasileiro)
  const Formato = {
    numero(valor, casas = 1) {
      return Number(valor).toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
    },
    gb(valor, casas = 1) { return `${Formato.numero(valor, casas)} GB`; },
    gbPorDia(valor) { return `${Formato.numero(valor, 2)} GB/dia`; },
    reais(valor) { return `R$ ${Formato.numero(valor, 2)}`; },
    percentual(proporcao) { return `${Formato.numero(proporcao * 100, 0)}%`; },
    data(data) {
      const dia = String(data.getUTCDate()).padStart(2, "0");
      const mes = String(data.getUTCMonth() + 1).padStart(2, "0");
      return `${dia}/${mes}/${data.getUTCFullYear()}`;
    },
    dataIso(texto) {
      const [ano, mes, dia] = texto.split("-").map(Number);
      return new Date(Date.UTC(ano, mes - 1, dia));
    },
    escapar(texto) {
      return String(texto ?? "").replace(/[&<>"']/g, (caractere) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[caractere]));
    },
  };

  // ------------------------------------------------------------------ fórmulas (iguais a src/formulas.py)
  const Formulas = {
    saldoGeral(franquia, consumo) { return Math.max(0, franquia - consumo); },
    mediaDiaria(consumo, dias) { return dias > 0 ? consumo / dias : NaN; },
    projecaoPeloRitmo(ritmo, dias) { return ritmo * dias; },
    diasAteEsgotamento(saldo, ritmo) {
      if (saldo <= 0) return 0;
      return Number.isFinite(ritmo) && ritmo > 0 ? saldo / ritmo : NaN;
    },
    percentualUtilizado(consumo, franquia) {
      if (!(franquia > 0)) return NaN;
      return Math.min(100, Math.max(0, (100 * consumo) / franquia));
    },
    diasDeAntecipacao(diasAteRenovacao, diasAteEsgotar) {
      return Number.isNaN(diasAteEsgotar) ? 0 : Math.max(0, diasAteRenovacao - diasAteEsgotar);
    },
  };

  // ------------------------------------------------------------------ Random Forest no navegador
  class FlorestaNoNavegador {
    constructor(modelo) { this.modelo = modelo; }

    vetor(atributos) {
      // mesmo pré-processamento do treino: números em float32 (como o scikit-learn) + One-Hot
      const vetor = this.modelo.variaveis_numericas.map((nome) => Math.fround(Number(atributos[nome])));
      for (const [nome, categorias] of Object.entries(this.modelo.variaveis_categoricas)) {
        for (const categoria of categorias) vetor.push(String(atributos[nome]) === categoria ? 1 : 0);
      }
      return vetor;
    }

    prever(atributos) {
      const vetor = this.vetor(atributos);
      let soma = 0;
      for (const arvore of this.modelo.arvores) {
        let no = 0;
        while (arvore.esquerda[no] !== -1) {
          no = vetor[arvore.variavel[no]] <= arvore.valor[no] ? arvore.esquerda[no] : arvore.direita[no];
        }
        soma += arvore.valor[no];
      }
      return soma / this.modelo.arvores.length;
    }
  }

  // ------------------------------------------------------------------ regras (iguais a src/recomendacao.py)
  class AtributosDaSimulacao {
    constructor(dataDeReferencia, diasDeUmMes, diasDoCiclo) {
      this.referencia = Formato.dataIso(dataDeReferencia);
      this.diasDeUmMes = diasDeUmMes;
      this.diasDoCiclo = diasDoCiclo;
    }

    montar(gbUsados, diasDesdeRenovacao, usoTipicoMes, habitos) {
      const dias = Math.trunc(diasDesdeRenovacao);
      const usados = Number(gbUsados);
      const ritmoTipico = Number(usoTipicoMes) / this.diasDeUmMes;
      const restantes = this.diasDoCiclo - dias;
      const ritmoRecente = dias >= 7
        ? Math.max(0, (usados - ritmoTipico * (dias - 7)) / 7)
        : (usados + ritmoTipico * (7 - dias)) / 7;
      const ritmoDoCiclo = Formulas.mediaDiaria(usados, dias);
      return {
        ...habitos,
        dados_gb: Number(usoTipicoMes),
        consumo_ciclo_atual_gb: usados,
        dias_decorridos: dias,
        dias_ate_renovacao: restantes,
        media_ultimos_7_dias_gb: ritmoRecente,
        media_7_dias_anteriores_gb: ritmoTipico,
        tendencia_consumo_gb_dia: ritmoRecente - ritmoTipico,
        consumo_ciclo_anterior_gb: Number(usoTipicoMes),
        dias_fim_de_semana_restantes: this.finsDeSemana(restantes),
        projecao_pelo_ritmo_do_ciclo_gb: Formulas.projecaoPeloRitmo(ritmoDoCiclo, restantes),
        projecao_pelo_ritmo_recente_gb: Formulas.projecaoPeloRitmo(ritmoRecente, restantes),
        projecao_pelo_habito_gb: Formulas.projecaoPeloRitmo(ritmoTipico, restantes),
      };
    }

    finsDeSemana(diasRestantes) {
      let total = 0;
      for (let dia = 0; dia < diasRestantes; dia += 1) {
        const semana = new Date(this.referencia.getTime() + dia * 86400000).getUTCDay();
        if (semana === 0 || semana === 6) total += 1;
      }
      return total;
    }

    dataDaRenovacao(diasRestantes) {
      return new Date(this.referencia.getTime() + diasRestantes * 86400000);
    }
  }

  class AnalisadorDeRisco {
    constructor(proporcaoDeAtencao) { this.proporcaoDeAtencao = proporcaoDeAtencao; }

    analisar(capacidade, consumo, demandaPrevista, diasAteRenovacao) {
      const consumoEfetivo = Math.min(consumo, capacidade);
      const saldo = Formulas.saldoGeral(capacidade, consumoEfetivo);
      const ritmo = demandaPrevista / diasAteRenovacao;
      const diasAteAcabar = Formulas.diasAteEsgotamento(saldo, ritmo);
      const antecipacao = Formulas.diasDeAntecipacao(diasAteRenovacao, diasAteAcabar);
      let nivel = "baixo";
      if (saldo <= 0) nivel = "esgotada";
      else if (antecipacao >= 1) nivel = "alto";
      else if (demandaPrevista > this.proporcaoDeAtencao * saldo) nivel = "atencao";
      return {
        capacidade_gb: capacidade, consumo_informado_gb: consumo, consumo_excede_capacidade: consumo > capacidade,
        consumo_efetivo_gb: consumoEfetivo, percentual_utilizado: Formulas.percentualUtilizado(consumoEfetivo, capacidade),
        saldo_gb: saldo, demanda_prevista_gb: demandaPrevista, dias_ate_renovacao: diasAteRenovacao,
        ritmo_previsto_gb_dia: ritmo, dias_ate_acabar: Number.isNaN(diasAteAcabar) ? null : diasAteAcabar,
        dias_de_antecipacao: Math.floor(antecipacao), nivel,
      };
    }
  }

  // variáveis de um par (cliente, plano) para o modelo de propensão (iguais a AtributosDaOferta, em Python)
  class AtributosDaOferta {
    constructor(planos) { this.planos = planos; }

    porNome(nome) { return this.planos.find((plano) => plano.nome === nome); }

    deUmPar(cliente, nomeDoPlanoOfertado, nomeDoPlanoAtual) {
      const ofertado = this.porNome(nomeDoPlanoOfertado);
      const atual = this.porNome(nomeDoPlanoAtual);
      const diferenca = ofertado.mensalidade_brl - atual.mensalidade_brl;
      return {
        dados_gb: cliente.dados_gb, minutos: cliente.minutos, streaming_horas: cliente.streaming_horas,
        deslocamento_dias: cliente.deslocamento_dias, grupo: cliente.grupo,
        mensalidade_atual: atual.mensalidade_brl, plano_ofertado: ofertado.nome,
        gb_do_plano_ofertado: ofertado.gb_principais, mensalidade_ofertada: ofertado.mensalidade_brl,
        diferenca_de_preco_brl: diferenca, diferenca_de_preco_relativa: diferenca / atual.mensalidade_brl,
        cobertura_gb: ofertado.gb_principais - cliente.dados_gb,
      };
    }
  }

  // filtragem colaborativa usuário a usuário (igual a FiltragemColaborativa, em Python): a nota prevista é a média
  // das notas dos vizinhos parecidos (correlação de Pearson > 0), ponderada pela semelhança
  class FiltragemColaborativaNoNavegador {
    constructor(dados) { this.dados = dados; }

    notaPrevista(idCliente, plano, grupo) {
      const vizinhos = this.dados.clientes[String(idCliente)];
      if (!vizinhos) return this.dados.media_por_grupo[grupo][plano];  // sem histórico: a média do grupo
      let somaPonderada = 0;
      let somaDasSemelhancas = 0;
      for (const vizinho of vizinhos) {
        const nota = vizinho.notas[plano];
        if (nota === null || nota === undefined) continue;
        somaPonderada += vizinho.semelhanca * nota;
        somaDasSemelhancas += vizinho.semelhanca;
      }
      return somaDasSemelhancas > 0 ? somaPonderada / somaDasSemelhancas : this.dados.media_dos_planos[plano];
    }

    percentual(idCliente, plano, grupo) { return this.notaPrevista(idCliente, plano, grupo) / this.dados.nota_maxima; }
  }

  class RecomendadorDePlanos {
    constructor(planos, margem, pesoRandomForest = 0.5) {
      this.planos = planos;
      this.atributosDaOferta = new AtributosDaOferta(planos);
      this.margem = margem;
      this.pesoRandomForest = pesoRandomForest;
      this.maximoDeOpcoes = 3;
    }

    porNome(nome) { return this.atributosDaOferta.porNome(nome); }

    necessidade(consumo, demandaPrevista, usoTipicoMes) { return Math.max(consumo + demandaPrevista, usoTipicoMes); }

    // modeloDePropensao: o Random Forest da etapa 8 (com prever) ou null, se não carregou (aí, ordem de preço);
    // Pontuação ponderada; o peso vem da validação. O padrão preserva a média original.
    recomendar(necessidade, nomeDoPlanoAtual, cliente, modeloDePropensao, colaborativa) {
      const atual = this.porNome(nomeDoPlanoAtual);
      const atualAtende = atual.gb_principais >= necessidade;
      const cobrem = this.planos.filter((plano) => plano.nome !== atual.nome && plano.gb_principais >= necessidade
        && (!atualAtende || plano.mensalidade_brl < atual.mensalidade_brl));
      const comChances = cobrem.map((plano) => ({
        plano, chances: this.chances(plano, atual, cliente, modeloDePropensao, colaborativa),
      }));
      comChances.sort((a, b) => ((b.chances?.hibrida ?? 0) - (a.chances?.hibrida ?? 0))
        || a.plano.mensalidade_brl - b.plano.mensalidade_brl
        || (a.plano.nome < b.plano.nome ? -1 : a.plano.nome > b.plano.nome ? 1 : 0));
      const limite = atualAtende ? this.maximoDeOpcoes - 1 : this.maximoDeOpcoes;
      const opcoes = comChances.slice(0, limite).map(({ plano, chances }) => this.opcao(plano, atual, necessidade, chances));
      if (atualAtende) opcoes.push(this.opcao(atual, atual, necessidade, null));
      return {
        necessidade_gb: necessidade, margem_de_seguranca: this.margem, plano_atual_atende: atualAtende, opcoes,
        nenhum_plano_cobre: opcoes.length === 0,
      };
    }

    chances(plano, atual, cliente, modeloDePropensao, colaborativa) {
      if (!modeloDePropensao) return null;
      const randomForest = modeloDePropensao.prever(this.atributosDaOferta.deUmPar(cliente, plano.nome, atual.nome));
      const filtragemColaborativa = colaborativa.percentual(cliente.id_cliente, plano.nome, cliente.grupo);
      return { hibrida: this.pesoRandomForest * randomForest + (1 - this.pesoRandomForest) * filtragemColaborativa,
        randomForest, filtragemColaborativa };
    }

    opcao(plano, atual, necessidade, chances) {
      let tipo = "mais_gb";
      if (plano.nome === atual.nome) tipo = "manter";
      else if (plano.mensalidade_brl < atual.mensalidade_brl) tipo = "economia";
      const folga = plano.gb_principais - necessidade;
      return {
        nome_do_plano: plano.nome, tipo, gb_do_plano: plano.gb_principais, mensalidade_brl: plano.mensalidade_brl,
        diferenca_de_preco_brl: Math.round((plano.mensalidade_brl - atual.mensalidade_brl) * 100) / 100,
        folga_gb: folga, folga_pequena: folga < this.margem * necessidade,
        chance_de_aceitar: chances ? chances.hibrida : null,
        chance_random_forest: chances ? chances.randomForest : null,
        chance_filtragem_colaborativa: chances ? chances.filtragemColaborativa : null,
        pontuacao_recomendacao: chances ? chances.hibrida : null,
        propensao_random_forest: chances ? chances.randomForest : null,
        pontuacao_colaborativa: chances ? chances.filtragemColaborativa : null,
      };
    }
  }

  // ------------------------------------------------------------------ campos do painel "Quer simular outra situação?" (controles deslizantes)
  const gbInteiros = (valor) => Formato.gb(valor, Number.isInteger(valor) ? 0 : 1);
  const contagem = (singular, plural) => (valor) => `${valor} ${valor === 1 ? singular : plural}`;
  const CAMPOS_MOVEIS = [
    { nome: "gb_usados", rotulo: "GB utilizados", minimo: 0, passo: 0.5, formatar: gbInteiros },
    { nome: "dias_desde_renovacao", rotulo: "Dias decorridos do ciclo", minimo: 1, maximo: 29, passo: 1,
      formatar: contagem("dia", "dias") },
  ];
  const CENARIOS = {
    mais_internet: { titulo: "Estou usando mais internet" },
    economia: { titulo: "Quero avaliar se posso economizar" },
  };
  const NOME_DO_EXEMPLO = "Cliente Claro";  // a cliente fictícia do exemplo (registro real da base enriquecida)
  const TITULO_DO_APP = "Minha Claro APP";  // a tela de análise simula o app (o WhatsApp leva até ela)
  const AVISO_DA_SIMULACAO = "Não acessamos sua conta Claro, não enviamos WhatsApp e este não é um serviço oficial.";
  const AVISO_PERSISTENTE ="Simulação acadêmica, não oficial. Não acessamos sua conta Claro nem enviamos mensagens.";

  // ------------------------------------------------------------------ a jornada (interface)
  class JornadaClaroGuia {
    constructor(raiz, dados) {
      this.raiz = raiz;
      this.dados = dados;
      this.modo = "exemplo";  // "exemplo" (a Cliente Claro) ou "simulacao"
      this.painelAberto = false;  // o painel "Quer simular outra situação?" começa fechado
      this.passo = 1;
      this.cenario = "mais_internet";
      this.entradas = this.entradasPadrao(this.cenario);
      this.floresta = null;  // Random Forest da demanda (etapa 6)
      this.propensao = null;  // Random Forest de propensão (etapa 8): ordena os planos
      this.estadoDoModelo = "carregando";  // carregando, pronto ou indisponivel
      this.atributos = new AtributosDaSimulacao(dados.data_de_referencia, dados.dias_de_um_mes, dados.dias_do_ciclo);
      this.analisador = new AnalisadorDeRisco(dados.proporcao_de_atencao);
      this.recomendador = new RecomendadorDePlanos(dados.planos, dados.margem_de_seguranca, dados.peso_random_forest);
      this.colaborativa = new FiltragemColaborativaNoNavegador(dados.filtragem_colaborativa);
      this.esperaDoControle = null;
    }

    iniciar() {
      this.raiz.innerHTML = `
        <div class="jornada-contexto"></div>
        <div class="jornada">
          <nav aria-label="Etapas da demonstração"><ol class="passos"></ol></nav>
          <div class="jornada-tela"></div>
          <div class="jornada-simulacao"></div>
        </div>
        <p class="jornada-status" role="status" aria-live="polite"></p>
        <p class="aviso-persistente">${Formato.escapar(AVISO_PERSISTENTE)}</p>`;
      this.raiz.addEventListener("click", (evento) => this.aoClicar(evento));
      this.raiz.addEventListener("input", (evento) => this.aoDeslizar(evento));
      this.raiz.addEventListener("change", (evento) => this.aoMudar(evento));
      this.renderizar();
      this.carregarModelo();
    }

    async carregarModelo() {
      try {
        const [demanda, propensao] = await Promise.all([this.dados.modelo, this.dados.modelo_de_propensao]
          .map((endereco) => this.baixarModelo(endereco)));
        this.floresta = new FlorestaNoNavegador(demanda);
        this.propensao = new FlorestaNoNavegador(propensao);
        this.estadoDoModelo = "pronto";
      } catch (erro) {
        this.estadoDoModelo = "indisponivel";
        this.anunciar("Os modelos não carregaram. Usamos a projeção pela média diária e a ordem de preço.");
      }
      this.renderizarTela();
    }

    async baixarModelo(endereco) {
      const resposta = await fetch(endereco);
      if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);
      return resposta.json();
    }

    entradasPadrao(cenario) {
      const padrao = this.dados.cenarios[cenario];
      return {
        plano: padrao.plano, gb_usados: padrao.gb_usados, dias_desde_renovacao: padrao.dias_desde_renovacao,
        uso_tipico_mes_gb: padrao.uso_tipico_mes_gb,
      };
    }

    // ---------------------------------------------------------------- cálculo
    prever(atributos) {
      if (this.estadoDoModelo === "pronto") {
        return { valor: this.floresta.prever(atributos), fonte: "modelo" };
      }
      if (this.estadoDoModelo === "carregando") return null;
      // modelo indisponível: regra transparente (média diária do ciclo)
      const ritmo = Formulas.mediaDiaria(atributos.consumo_ciclo_atual_gb, atributos.dias_decorridos);
      return { valor: Formulas.projecaoPeloRitmo(ritmo, atributos.dias_ate_renovacao), fonte: "regra" };
    }

    calcular() {
      if (this.modo === "exemplo") {
        const exemplo = this.dados.cliente_do_exemplo;
        return this.calcularMovel(exemplo.atributos, exemplo.plano_atual, exemplo.consumo_ciclo_atual_gb,
          exemplo.atributos.dados_gb, Formato.dataIso(exemplo.data_proxima_renovacao), exemplo.cliente);
      }
      const cenario = this.dados.cenarios[this.cenario];
      const atributos = this.atributos.montar(this.entradas.gb_usados, this.entradas.dias_desde_renovacao,
        this.entradas.uso_tipico_mes_gb, cenario.habitos);
      return this.calcularMovel(atributos, this.entradas.plano, this.entradas.gb_usados, this.entradas.uso_tipico_mes_gb,
        this.atributos.dataDaRenovacao(atributos.dias_ate_renovacao), this.clienteDoCenario(cenario));
    }

    // as variáveis do cliente para o modelo de propensão (iguais a cliente_para_recomendacao, em Python)
    clienteDoCenario(cenario) {
      return {
        id_cliente: null, dados_gb: this.entradas.uso_tipico_mes_gb, minutos: cenario.habitos.minutos,
        streaming_horas: cenario.habitos.streaming_horas, deslocamento_dias: cenario.habitos.deslocamento_dias,
        grupo: cenario.grupo_dos_habitos,
      };
    }

    calcularMovel(atributos, nomeDoPlano, consumo, usoTipico, dataDaRenovacao, cliente) {
      const previsao = this.prever(atributos);
      if (previsao === null) return { carregando: true };
      const plano = this.recomendador.porNome(nomeDoPlano);
      const risco = this.analisador.analisar(plano.gb_principais, consumo, previsao.valor, atributos.dias_ate_renovacao);
      const necessidade = this.recomendador.necessidade(consumo, previsao.valor, usoTipico);
      const recomendacao = this.recomendador.recomendar(necessidade, nomeDoPlano, cliente,
        this.estadoDoModelo === "pronto" ? this.propensao : null, this.colaborativa);
      return { atributos, previsao, plano, risco, dataDaRenovacao, recomendacao };
    }

    // ---------------------------------------------------------------- eventos
    aoClicar(evento) {
      const botao = evento.target.closest("[data-acao]");
      if (!botao) return;
      const acao = botao.dataset.acao;
      if (acao === "avancar") this.irPara(this.passo + 1);
      else if (acao === "voltar") this.irPara(this.passo - 1);
      else if (acao === "ir-para") this.irPara(Number(botao.dataset.passo));
      else if (acao === "simular") this.abrirSimulacao();
      else if (acao === "ver-simulacao") this.verSimulacao();
      else if (acao === "voltar-exemplo") this.voltarAoExemplo();
    }

    aoMudar(evento) {
      const alvo = evento.target;
      if (alvo.name === "cenario") this.escolherCenario(alvo.value);
      else if (alvo.dataset.campo) {
        clearTimeout(this.esperaDoControle);
        this.lerCampo(alvo);
        this.atualizarResultado();
        this.anunciarAtualizacao(alvo.dataset.campo);
      }
    }

    aoDeslizar(evento) {
      // enquanto o controle desliza, o valor aparece na hora e a análise acompanha (sem anunciar a cada passo)
      const alvo = evento.target;
      if (alvo.type !== "range") return;
      this.lerCampo(alvo);
      clearTimeout(this.esperaDoControle);
      this.esperaDoControle = setTimeout(() => this.atualizarResultado(), 150);
    }

    lerCampo(alvo) {
      const nome = alvo.dataset.campo;
      if (nome === "plano") {
        this.entradas.plano = alvo.value;
        this.sincronizarConsumoComPlano();
        this.raiz.querySelector("#dica-plano").textContent = this.descricaoDoPlano();
      } else this.entradas[nome] = Number(alvo.value);
      if (alvo.type === "range") this.mostrarValor(alvo);
    }

    mostrarValor(controle) {
      const campo = CAMPOS_MOVEIS.find((item) => item.nome === controle.dataset.campo);
      const texto = campo.formatar(Number(controle.value));
      controle.setAttribute("aria-valuetext", texto);
      this.raiz.querySelector(`#valor-${campo.nome}`).textContent = texto;
    }

    sincronizarConsumoComPlano() {
      const controle = this.raiz.querySelector("#campo-gb_usados");
      const franquia = this.recomendador.porNome(this.entradas.plano).gb_principais;
      this.entradas.gb_usados = franquia;
      controle.max = String(franquia);
      controle.value = String(this.entradas.gb_usados);
      this.mostrarValor(controle);
    }

    atualizarResultado() {
      // O primeiro ajuste já mostra a análise dos valores do painel, sem mover o foco do controle.
      if (this.modo === "exemplo") {
        this.modo = "simulacao";
        this.passo = 2;
        this.renderizarContexto();
        this.renderizarPassos();
      }
      this.renderizarTela();
    }

    anunciarAtualizacao(nome) {
      if (this.modo === "exemplo") return;
      if (nome === "plano") {
        this.anunciar(`Plano ${this.entradas.plano} selecionado. Consumo ajustado para ${gbInteiros(this.entradas.gb_usados)}, `
          + "a franquia do plano. Você pode ajustar os GB utilizados. Análise e opções atualizadas.");
      } else this.anunciar("Análise atualizada.");
    }

    // ---------------------------------------------------------------- navegação
    irPara(passo) {
      this.passo = Math.min(3, Math.max(1, passo));
      this.renderizarContexto();
      this.renderizarPassos();
      this.renderizarTela();
      const titulo = this.raiz.querySelector(".jornada-tela h3");
      if (titulo) {
        titulo.focus({ preventScroll: true });
        const reduzido = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        this.raiz.querySelector(".jornada").scrollIntoView({ behavior: reduzido ? "auto" : "smooth", block: "start" });
      }
    }

    abrirSimulacao() {
      this.painelAberto = true;
      this.renderizarSimulacao();
      this.raiz.querySelector("#titulo-simulacao").focus({ preventScroll: true });
    }

    verSimulacao() {
      clearTimeout(this.esperaDoControle);
      this.modo = "simulacao";
      this.irPara(2);
      this.anunciar(`Análise da simulação pronta: ${CENARIOS[this.cenario].titulo}.`);
    }

    escolherCenario(cenario) {
      clearTimeout(this.esperaDoControle);
      this.cenario = cenario;
      this.entradas = this.entradasPadrao(cenario);
      this.renderizarSimulacao();
      this.raiz.querySelector(`input[name="cenario"][value="${cenario}"]`).focus();
      this.atualizarResultado();
      this.renderizarContexto();
      this.anunciar(`Situação escolhida: ${CENARIOS[cenario].titulo}.`);
    }

    voltarAoExemplo() {
      clearTimeout(this.esperaDoControle);
      this.modo = "exemplo";
      this.painelAberto = false;
      this.cenario = "mais_internet";
      this.entradas = this.entradasPadrao(this.cenario);
      this.passo = 1;
      this.renderizar();
      this.raiz.querySelector("#titulo-do-exemplo").focus({ preventScroll: true });
      this.raiz.querySelector(".jornada-contexto").scrollIntoView({ block: "start" });
    }

    anunciar(texto) {
      const status = this.raiz.querySelector(".jornada-status");
      status.textContent = "";
      window.setTimeout(() => { status.textContent = texto; }, 50);
    }

    // ---------------------------------------------------------------- desenho
    renderizar() {
      this.renderizarContexto();
      this.renderizarPassos();
      this.renderizarTela();
      this.renderizarSimulacao();
    }

    renderizarPassos() {
      // abas clicáveis: levam direto a cada tela (os botões "Voltar" continuam valendo)
      const nomes = ["WhatsApp", "Análise", "Planos"];
      this.raiz.querySelector(".passos").innerHTML = nomes.map((nome, indice) => {
        const passo = indice + 1;
        return `<li><button type="button" class="aba" data-acao="ir-para" data-passo="${passo}"`
          + `${passo === this.passo ? ' aria-current="step"' : ""}>${nome}</button></li>`;
      }).join("");
    }

    renderizarContexto() {
      this.raiz.querySelector(".jornada-contexto").innerHTML = this.modo === "exemplo"
        ? this.contextoDoExemplo() : this.contextoDaSimulacao();
    }

    renderizarSimulacao() {
      this.raiz.querySelector(".jornada-simulacao").innerHTML = this.painelDeSimulacao();
    }

    contextoDaSimulacao() {
      return `<p class="contexto-curto">Sua simulação: <b>${CENARIOS[this.cenario].titulo}</b> · dados fictícios, que você
        ajusta no fim do cartão</p>`;
    }

    contextoDoExemplo() {
      const exemplo = this.dados.cliente_do_exemplo;
      const plano = this.recomendador.porNome(exemplo.plano_atual);
      const linhaDoPlano = `Plano atual: <b>${Formato.escapar(plano.nome)}</b> (plano fictício das planilhas) · `
        + `${Formato.gb(plano.gb_principais, 0)} de uso geral · ${Formato.reais(plano.mensalidade_brl)}/mês`;
      if (this.passo > 1) {
        return `<p class="contexto-curto">Exemplo fictício: <b>${NOME_DO_EXEMPLO}</b> · ${linhaDoPlano}</p>`;
      }
      // um cartão só, numa coluna. Os números vêm do mesmo cálculo do WhatsApp e da análise (nada digitado à mão):
      // a antecipação usa o ritmo PREVISTO pelo modelo, não a média do ciclo até agora
      const dias = (quantidade) => (quantidade === 1 ? "1 dia" : `${quantidade} dias`);
      return `
        <div class="historia">
          <p class="rotulo-pequeno">Exemplo fictício</p>
          <h3 id="titulo-do-exemplo" tabindex="-1">${NOME_DO_EXEMPLO}</h3>
          <p>Imagine uma cliente da Claro que utiliza a internet móvel todos os dias e acredita que seu pacote será
            suficiente até a próxima renovação.</p>
          <p>Mas algo passa despercebido: em apenas ${dias(exemplo.dias_decorridos)}, ela já consumiu
            ${Formato.numero(exemplo.consumo_ciclo_atual_gb, 1)} dos ${Formato.gb(plano.gb_principais, 0)} disponíveis para
            uso geral. Ainda faltam ${dias(exemplo.dias_ate_renovacao)} para receber uma nova franquia.</p>
          <p>Se o consumo seguir o ritmo previsto, poderá ficar sem internet cerca de
            ${dias(exemplo.risco.dias_de_antecipacao)} antes do esperado.</p>
          <p>E se ela pudesse descobrir isso antes de ficar sem internet?</p>
          <p>É nesse momento que entra o CLARO GUIA.</p>
          <p>O sistema analisa o consumo, prevê o risco de a franquia acabar antecipadamente e envia um alerta pelo
            WhatsApp.</p>
          <p>Ao acessar o aviso, a cliente pode entender melhor seu consumo e descobrir se o plano atual continua
            adequado às suas necessidades.</p>
          <p class="plano-de-referencia">${linhaDoPlano}</p>
          <p class="nota">História fictícia inspirada em dificuldades relatadas por consumidores. Os dados apresentados
            são simulados.</p>
          <p class="historia-rodape"><b>Simulação acadêmica.</b> ${Formato.escapar(AVISO_DA_SIMULACAO)}</p>
        </div>`;
    }

    painelDeSimulacao() {
      const conteudo = this.painelAberto ? this.formularioDeSimulacao() : `
        <button type="button" class="botao secundario destaque largo" data-acao="simular">Simular
          <svg class="icone" aria-hidden="true"><use href="#icone-seta"/></svg></button>`;
      return `
        <section class="simulacao" aria-labelledby="titulo-simulacao">
          <div class="simulacao-topo">
            <span class="simulacao-icone" aria-hidden="true"><svg class="icone"><use href="#icone-ajustes"/></svg></span>
            <div>
              <h3 id="titulo-simulacao" tabindex="-1">Quer simular outra situação?</h3>
              <p>Experimente um cenário diferente e veja como a mensagem, a análise e as orientações mudam.</p>
            </div>
          </div>
          ${conteudo}
        </section>`;
    }

    formularioDeSimulacao() {
      const opcoes = Object.entries(CENARIOS).map(([id, cenario]) => `
        <label class="cenario"><input type="radio" name="cenario" value="${id}"${id === this.cenario ? " checked" : ""}>
          ${cenario.titulo}</label>`).join("");
      return `
        <fieldset class="cenarios"><legend>Escolha uma situação</legend>${opcoes}</fieldset>
        <div class="campos">${this.camposMoveis()}</div>
        <button type="button" class="botao principal largo" data-acao="ver-simulacao">Ver simulação do atendimento
          <svg class="icone" aria-hidden="true"><use href="#icone-seta"/></svg></button>
        <button type="button" class="botao-link centralizado" data-acao="voltar-exemplo">Voltar ao exemplo da ${NOME_DO_EXEMPLO}</button>`;
    }

    controleDeslizante(campo) {
      const valor = this.entradas[campo.nome];
      const texto = campo.formatar(valor);
      const maximo = campo.nome === "gb_usados"
        ? this.recomendador.porNome(this.entradas.plano).gb_principais : campo.maximo;
      return `
        <div class="campo">
          <div class="campo-topo">
            <label for="campo-${campo.nome}">${campo.rotulo}</label>
            <output id="valor-${campo.nome}" for="campo-${campo.nome}">${texto}</output>
          </div>
          <input type="range" id="campo-${campo.nome}" data-campo="${campo.nome}" min="${campo.minimo}" max="${maximo}"
            step="${campo.passo}" value="${valor}" aria-valuetext="${texto}">
        </div>`;
    }

    camposMoveis() {
      const planos = this.dados.planos.map((plano) => `
        <option value="${plano.nome}"${plano.nome === this.entradas.plano ? " selected" : ""}>
          ${Formato.escapar(plano.nome)}</option>`).join("");
      return `
        <div class="campo">
          <label for="campo-plano">Plano de referência</label>
          <select id="campo-plano" data-campo="plano" aria-describedby="dica-plano">${planos}</select>
          <p class="dica" id="dica-plano">${this.descricaoDoPlano()}</p>
        </div>
        ${CAMPOS_MOVEIS.map((campo) => this.controleDeslizante(campo)).join("")}`;
    }

    descricaoDoPlano() {
      const plano = this.recomendador.porNome(this.entradas.plano);
      return `${Formato.gb(plano.gb_principais, 0)} de uso geral · ${Formato.reais(plano.mensalidade_brl)}/mês · plano `
        + `fictício das planilhas. GB utilizados: de 0 a ${gbInteiros(plano.gb_principais)}. `
        + "Ao trocar o plano, os GB utilizados são preenchidos com a franquia dele. Depois, ajuste o controle para simular seu consumo.";
    }

    renderizarTela() {
      const tela = this.raiz.querySelector(".jornada-tela");
      const resultado = this.calcular();
      tela.innerHTML = resultado.carregando
        ? `<h3 tabindex="-1">Preparando a análise</h3><p class="estado">Carregando os modelos…</p>`
        : [this.telaMensagem, this.telaAnalise, this.telaOpcoes][this.passo - 1].call(this, resultado);
    }

    botoes(principal, rotulo) {
      const voltar = this.passo > 1 ? '<button type="button" class="botao secundario" data-acao="voltar">Voltar</button>' : "";
      const principalHtml = principal ? `<button type="button" class="botao principal" data-acao="${principal}">${rotulo}</button>` : "";
      return `<div class="acoes">${voltar}${principalHtml}</div>`;
    }

    // ---------------------------------------------------------------- telas: celular
    saudacao() { return this.modo === "exemplo" ? `Olá, ${NOME_DO_EXEMPLO}!` : "Olá!"; }

    textoDaMensagem(resultado) {
      // só dados de consumo (ritmo, saldo, franquia, renovação): nunca o motivo da mudança
      const { risco, recomendacao } = resultado;
      const analise = "Preparamos uma análise para ajudar você a entender seu consumo e avaliar opções mais adequadas.";
      const dias = risco.dias_de_antecipacao === 1 ? "1 dia" : `${risco.dias_de_antecipacao} dias`;
      let corpo;
      if (risco.consumo_excede_capacidade) {
        corpo = `Você informou ${Formato.gb(risco.consumo_informado_gb)} de consumo para um plano com `
          + `${Formato.gb(risco.capacidade_gb, 0)} de uso geral. Com esse plano, a franquia já teria acabado. ${analise}`;
      } else if (risco.nivel === "esgotada") corpo = `Seus GB de uso geral deste ciclo acabaram antes da renovação. ${analise}`;
      else if (risco.nivel === "alto") corpo = `Seu ritmo de consumo indica que a franquia pode acabar cerca de ${dias} antes da renovação. ${analise}`;
      else if (risco.nivel === "atencao") corpo = `Seu ritmo de consumo indica que você deve usar quase todo o saldo até a renovação. ${analise}`;
      else if (recomendacao.opcoes.some((opcao) => opcao.tipo === "economia")) {
        corpo = "Seu consumo tem ficado abaixo do que seu plano oferece. Preparamos uma análise para você conferir se há uma opção que combine mais com o seu uso.";
      } else corpo = "No ritmo atual, sua franquia deve durar até a renovação. Preparamos uma análise do seu consumo neste ciclo.";
      return { saudacao: `${this.saudacao()} Temos uma dica sobre sua internet.`, corpo };
    }

    conversa(mensagem) {
      return `
        <div class="conversa" role="group" aria-label="Prévia de mensagem simulada no WhatsApp">
          <div class="conversa-topo">
            <span class="avatar" aria-hidden="true"><svg class="icone"><use href="#icone-atendimento"/></svg></span>
            <span><b>Claro · Atendimento</b><span class="selo-simulada">Mensagem simulada — não enviada · Não oficial</span></span>
          </div>
          <div class="balao">
            <p class="saudacao">${Formato.escapar(mensagem.saudacao)}</p>
            <p>${Formato.escapar(mensagem.corpo)}</p>
            <button type="button" class="botao-mensagem" data-acao="avancar">Ver minha análise
              <svg class="icone" aria-hidden="true"><use href="#icone-seta"/></svg></button>
            <p class="nota-da-mensagem">Em uma aplicação real, a mensagem seria personalizada com dados autorizados do cliente.</p>
          </div>
        </div>`;
    }

    telaMensagem(resultado) {
      return `<h3 class="visualmente-oculto" tabindex="-1">Mensagem simulada</h3>${this.conversa(this.textoDaMensagem(resultado))}`;
    }

    resumoDoCalculo(resultado) {
      // explicação curta, para quem está testando (executivo ou cliente); os detalhes ficam na página do projeto
      const { risco } = resultado;
      const previsao = resultado.previsao.fonte === "modelo"
        ? "Um modelo de previsão (Random Forest), treinado com dados fictícios, estima quantos GB ainda serão usados até a renovação."
        : "O modelo de previsão não carregou: a estimativa usa a média diária do ciclo.";
      let prazo = `o saldo dura cerca de ${Formato.numero(risco.dias_ate_acabar, 0)} dias`;
      if (risco.saldo_gb <= 0) prazo = "os GB de uso geral deste ciclo já acabaram";
      else if (risco.dias_ate_acabar === null) prazo = "o saldo não acaba nesse ritmo";
      const renovacao = risco.dias_ate_renovacao === 1 ? "1 dia" : `${risco.dias_ate_renovacao} dias`;
      const usoTipico = this.modo === "simulacao"
        ? ` Nesta simulação, o uso típico de um mês é ${Formato.gb(this.entradas.uso_tipico_mes_gb, 0)}.` : "";
      return `
        <p>Comparamos o consumo recente com o histórico. ${previsao}</p>
        <p>Com ${Formato.gb(risco.saldo_gb)} de saldo e um ritmo previsto de ${Formato.gbPorDia(risco.ritmo_previsto_gb_dia)},
          ${prazo}; a renovação é em ${renovacao}.${usoTipico}</p>
        <p class="nota">É uma estimativa com dados fictícios.
          <a href="${this.raiz.dataset.detalhes}#modelo" target="_blank" rel="noopener">Ver como o modelo foi treinado</a></p>`;
    }

    blocoDeRisco(risco) {
      const textos = {
        esgotada: ["icone-alerta", "A franquia acabou", risco.consumo_excede_capacidade
          ? `Você informou ${Formato.gb(risco.consumo_informado_gb)} de consumo: são `
            + `${Formato.gb(risco.consumo_informado_gb - risco.capacidade_gb)} além da franquia de `
            + `${Formato.gb(risco.capacidade_gb, 0)}. Com esse plano, os GB de uso geral já teriam acabado. `
            + "Esse uso exige mais internet (ou um pacote adicional)."
          : "Seus GB de uso geral deste ciclo já acabaram."],
        alto: ["icone-alerta", "Atenção", `Mantendo esse ritmo, sua franquia pode terminar cerca de ${risco.dias_de_antecipacao} ${risco.dias_de_antecipacao === 1 ? "dia" : "dias"} antes da renovação.`],
        atencao: ["icone-info", "Fique de olho", "Mantendo esse ritmo, você deve usar quase todo o saldo até a renovação."],
        baixo: ["icone-ok", "Tudo certo", "Mantendo esse ritmo, sua franquia deve durar até a renovação."],
      };
      const [icone, rotulo, texto] = textos[risco.nivel];
      return `<div class="risco risco-${risco.nivel}"><svg class="icone" aria-hidden="true"><use href="#${icone}"/></svg>
        <p><b>${rotulo}:</b> ${texto} <span class="estimativa">É uma estimativa.</span></p></div>`;
    }

    telaAnalise(resultado) {
      const { risco, atributos, plano } = resultado;
      const percentual = Math.round(risco.percentual_utilizado);
      const consumo = risco.consumo_excede_capacidade
        ? `Você informou <b>${Formato.gb(risco.consumo_informado_gb)}</b> de consumo para um plano com `
          + `<b>${Formato.gb(plano.gb_principais, 0)}</b> de uso geral.`
        : `Você usou <b>${Formato.gb(risco.consumo_informado_gb)}</b> de `
          + `<b>${Formato.gb(plano.gb_principais, 0)}</b> disponíveis para uso geral.`;
      return `
        <p class="titulo-do-app">${TITULO_DO_APP}</p>
        <h3 tabindex="-1">${this.modo === "exemplo" ? `Análise do consumo da ${NOME_DO_EXEMPLO}` : "Sua análise"}</h3>
        <p class="consumo-principal">${consumo}</p>
        <div class="barra" role="img" aria-label="${percentual}% dos GB de uso geral usados"><span style="width:${percentual}%"></span></div>
        <p class="legenda-da-barra">${percentual}% da franquia utilizados · ${Formato.gb(risco.saldo_gb)} restantes · renova em ${Formato.data(resultado.dataDaRenovacao)}</p>
        <dl class="numeros-da-analise">
          <div><dt>Saldo atual</dt><dd>${Formato.gb(risco.saldo_gb)}</dd></div>
          <div><dt>Dias até renovar</dt><dd>${risco.dias_ate_renovacao}</dd></div>
          <div><dt>Média recente (7 dias)</dt><dd>${Formato.gbPorDia(atributos.media_ultimos_7_dias_gb)}</dd></div>
          <div><dt>Demanda prevista até renovar</dt><dd>${Formato.gb(risco.demanda_prevista_gb)}</dd><span>estimativa</span></div>
        </dl>
        ${this.blocoDeRisco(risco)}
        <details>
          <summary>Como calculamos?</summary>
          ${this.resumoDoCalculo(resultado)}
        </details>
        <details>
          <summary>O que são GB de uso geral?</summary>
          <p><b>GB de uso geral</b> é a internet que serve para qualquer aplicativo. O plano ${Formato.escapar(plano.nome)}
            tem ${Formato.gb(plano.gb_principais, 0)} principais, que valem o mês todo;
            ${Formato.gb(plano.bonus_livre_gb, 0)} de bônus de uso livre, que é temporário; e
            ${Formato.gb(plano.bonus_para_apps_gb, 0)} só para redes sociais e apps do benefício. Para comparar planos, o
            CLARO GUIA conta só os GB principais.</p>
        </details>
        ${this.botoes("avancar", "Ver opções para meu consumo")}`;
    }

    cartaoDePlano(opcao) {
      const necessidade = Formato.gb(this.ultimaNecessidade);
      const porQue = {
        manter: `Seu plano tem ${Formato.gb(opcao.gb_do_plano, 0)} de uso geral e cobre a demanda estimada de ${necessidade}.`,
        economia: `Cobre a sua demanda estimada de ${necessidade} e custa ${Formato.reais(-opcao.diferenca_de_preco_brl)} a menos por mês. Compare antes de decidir.`,
        mais_gb: `Cobre a sua demanda estimada de ${necessidade}. Compare antes de decidir.`,
      }[opcao.tipo];
      const rotulos = { manter: "Manter o plano atual", economia: "Mais econômico", mais_gb: "Mais internet" };
      const diferenca = opcao.diferenca_de_preco_brl === 0 ? "Mesmo preço do seu plano"
        : `${opcao.diferenca_de_preco_brl > 0 ? "+" : "−"}${Formato.reais(Math.abs(opcao.diferenca_de_preco_brl))}/mês em relação ao seu plano`;
      const origemDaFiltragem = this.modo === "simulacao" ? " (média do grupo: sem histórico)" : "";
      const chance = opcao.pontuacao_recomendacao === null ? "" : `
          <p class="opcao-chance"><b>Pontuação de recomendação: ${Formato.numero(opcao.pontuacao_recomendacao * 100, 0)} pontos de 100</b></p>
          <p class="opcao-modelos">Estimativa da simulação para ordenar as opções; não é uma probabilidade de aceitação.${origemDaFiltragem}</p>`;
      return `
        <li class="opcao">
          <h4>${Formato.escapar(opcao.nome_do_plano)} <span class="etiqueta">${rotulos[opcao.tipo]}</span></h4>
          <p class="opcao-gb"><b>${Formato.gb(opcao.gb_do_plano, 0)} de uso geral</b></p>
          <p class="opcao-preco"><b>${Formato.reais(opcao.mensalidade_brl)}/mês</b></p>
          <p class="opcao-diferenca">${diferenca}</p>
          ${chance}
          <p class="opcao-porque"><b>Por que esta opção?</b> ${porQue}</p>
          ${opcao.folga_pequena ? `<p class="aviso-folga"><svg class="icone" aria-hidden="true"><use href="#icone-info"/></svg>
            Folga pequena: sobram ${Formato.gb(opcao.folga_gb)}. Se o ritmo subir, pode faltar internet.</p>` : ""}
        </li>`;
    }

    telaOpcoes(resultado) {
      const { recomendacao, risco, plano, atributos } = resultado;
      this.ultimaNecessidade = recomendacao.necessidade_gb;
      const usadoMaisPrevisto = risco.consumo_informado_gb + risco.demanda_prevista_gb;
      const maiorPlano = Math.max(...this.dados.planos.map((plano) => plano.gb_principais));
      const lista = recomendacao.nenhum_plano_cobre
        ? `<div class="sem-opcao"><p>Nenhum plano das planilhas cobre essa demanda: o maior tem ${Formato.gb(maiorPlano, 0)}.</p></div>`
        : `<ol class="opcoes">${recomendacao.opcoes.map((opcao) => this.cartaoDePlano(opcao)).join("")}</ol>`;
      const ordem = this.estadoDoModelo === "pronto"
        ? "na ordem da pontuação de recomendação"
        : "em ordem de preço (o modelo de recomendação não carregou)";
      const economia = recomendacao.opcoes.some((opcao) => opcao.tipo === "economia")
        ? '<p class="nota">Antes de trocar, confira se o seu uso continua baixo por 2 ou 3 ciclos completos.</p>' : "";
      return `
        <h3 tabindex="-1">Seu plano ainda combina com você?</h3>
        <p class="plano-de-referencia">Plano de referência: <b>${Formato.escapar(plano.nome)}</b> ·
          <b>${Formato.gb(plano.gb_principais, 0)} de uso geral</b>.</p>
        <p class="resumo-da-demanda"><b>${Formato.gb(risco.consumo_informado_gb)}</b> usados +
          <b>${Formato.gb(risco.demanda_prevista_gb)}</b> previstos até renovar =
          <b>${Formato.gb(usadoMaisPrevisto)}</b> estimados para este ciclo.
          Seu uso típico mensal é de <b>${Formato.gb(atributos.dados_gb)}</b>.</p>
        <p class="demanda-do-ciclo">Demanda estimada para um ciclo: <b>${Formato.gb(recomendacao.necessidade_gb)}</b>.
          Usamos o maior valor entre o total estimado para este ciclo e seu uso típico mensal. Mostramos os planos
          que cobrem essa demanda, ${ordem}.</p>
        ${lista}
        ${economia}
        <details><summary>Por que recomendamos estes planos?</summary>
          <p>Entram só os planos das planilhas cujos GB principais cobrem a demanda estimada; com folga menor que
            ${Formato.numero(recomendacao.margem_de_seguranca * 100, 0)}%, o plano aparece com o aviso “folga pequena”. A
            ordem vem de um modelo híbrido treinado com um histórico de ofertas <b>simulado</b> (Etapa 8), que combina
            hábitos, preço e avaliações de clientes parecidos. Sem histórico individual, usa a média do grupo.
            A pontuação serve para comparar opções e não representa uma chance comprovada de aceitação.
            Se o seu plano cobre a demanda, ele aparece como
            “Manter o plano atual”, junto com os planos mais baratos que também cobrem. Não sugerimos planos mais caros
            quando o seu já basta.</p>
        </details>
        <p class="nota">Esta é uma simulação: os planos são fictícios e não é possível contratar por aqui.</p>
        ${this.botoes(null)}`;
    }
  }

  // ------------------------------------------------------------------ início
  const ClaroGuia = { Formato, Formulas, FlorestaNoNavegador, AtributosDaSimulacao, AnalisadorDeRisco,
    AtributosDaOferta, FiltragemColaborativaNoNavegador, RecomendadorDePlanos, JornadaClaroGuia };
  window.ClaroGuia = ClaroGuia;

  document.addEventListener("DOMContentLoaded", () => {
    const raiz = document.getElementById("claro-guia");
    const dados = document.getElementById("dados-da-demonstracao");
    if (!raiz || !dados) return;
    try {
      new JornadaClaroGuia(raiz, JSON.parse(dados.textContent)).iniciar();
    } catch (erro) {
      raiz.innerHTML = '<p class="estado">Não foi possível iniciar a demonstração neste navegador. As seções abaixo continuam disponíveis.</p>';
    }
  });
})();
