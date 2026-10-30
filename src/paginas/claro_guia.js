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

  // A apresentação usa a maquete fornecida; as classes e contas acima permanecem as mesmas.
  const gbInteiros = (valor) => Formato.gb(valor, Number.isInteger(valor) ? 0 : 1);
  const CENARIOS = {
    mais_internet: { titulo: "Estou usando mais internet" },
    economia: { titulo: "Quero avaliar se posso economizar" },
  };
  const AVISO_DA_SIMULACAO = "Não acessamos sua conta Claro, não enviamos WhatsApp e este não é um serviço oficial.";

  class JornadaClaroGuia {
    constructor(raiz, dados) {
      this.raiz = raiz;
      this.dados = dados;
      this.modo = "exemplo";
      this.passo = 1;
      this.cenario = "mais_internet";
      this.entradas = this.entradasPadrao(this.cenario);
      this.floresta = null;
      this.propensao = null;
      this.estadoDoModelo = "carregando";
      this.atributos = new AtributosDaSimulacao(dados.data_de_referencia, dados.dias_de_um_mes, dados.dias_do_ciclo);
      this.analisador = new AnalisadorDeRisco(dados.proporcao_de_atencao);
      this.recomendador = new RecomendadorDePlanos(dados.planos, dados.margem_de_seguranca, dados.peso_random_forest);
      this.colaborativa = new FiltragemColaborativaNoNavegador(dados.filtragem_colaborativa);
    }

    iniciar() {
      this.raiz.dataset.modelo = "carregando";
      this.mostrarEtapa();
      this.configurarFormulario();
      this.raiz.addEventListener("click", (evento) => this.aoClicar(evento));
      this.raiz.addEventListener("input", (evento) => {
        if (evento.target.type === "range") this.aoMudar(evento);
      });
      this.raiz.addEventListener("change", (evento) => this.aoMudar(evento));
      this.raiz.querySelector("form").addEventListener("submit", (evento) => {
        evento.preventDefault();
        this.verSimulacao();
      });
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
        const aviso = "Os modelos não carregaram. Usamos a projeção pela média diária e a ordem de preço.";
        this.raiz.querySelector(".aviso-persistente").textContent = `${AVISO_DA_SIMULACAO} ${aviso}`;
        this.anunciar(aviso);
      }
      this.raiz.dataset.modelo = this.estadoDoModelo;
      this.renderizarResultados();
    }

    async baixarModelo(endereco) {
      const resposta = await fetch(endereco);
      if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);
      return resposta.json();
    }

    entradasPadrao(cenario) {
      const padrao = this.dados.cenarios[cenario];
      return { plano: padrao.plano, gb_usados: padrao.gb_usados,
        dias_desde_renovacao: padrao.dias_desde_renovacao, uso_tipico_mes_gb: padrao.uso_tipico_mes_gb };
    }

    prever(atributos) {
      if (this.estadoDoModelo === "pronto") return { valor: this.floresta.prever(atributos), fonte: "modelo" };
      if (this.estadoDoModelo === "carregando") return null;
      const ritmo = Formulas.mediaDiaria(atributos.consumo_ciclo_atual_gb, atributos.dias_decorridos);
      return { valor: Formulas.projecaoPeloRitmo(ritmo, atributos.dias_ate_renovacao), fonte: "regra" };
    }

    calcular(modo = this.modo) {
      if (modo === "exemplo") {
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

    clienteDoCenario(cenario) {
      return { id_cliente: null, dados_gb: this.entradas.uso_tipico_mes_gb, minutos: cenario.habitos.minutos,
        streaming_horas: cenario.habitos.streaming_horas, deslocamento_dias: cenario.habitos.deslocamento_dias,
        grupo: cenario.grupo_dos_habitos };
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

    configurarFormulario() {
      const campo = this.raiz.querySelector("#campo-plano");
      campo.innerHTML = this.dados.planos.map((plano) => `<option value="${Formato.escapar(plano.nome)}">`
        + `${Formato.escapar(plano.nome)} · ${gbInteiros(plano.gb_principais)}</option>`).join("");
      this.raiz.querySelector("#campo-dias_desde_renovacao").max = String(this.dados.dias_do_ciclo - 1);
      this.raiz.querySelector("#limite-dias").textContent = String(this.dados.dias_do_ciclo - 1);
      this.raiz.querySelector("#ciclo-do-plano").textContent = `▦ Ciclo de ${this.dados.dias_do_ciclo} dias`;
      this.preencherFormulario();
    }

    preencherFormulario() {
      this.raiz.querySelector("#campo-plano").value = this.entradas.plano;
      const plano = this.recomendador.porNome(this.entradas.plano);
      this.raiz.querySelector("#campo-gb_usados").max = String(plano.gb_principais);
      this.raiz.querySelector("#limite-consumo").textContent = gbInteiros(plano.gb_principais);
      this.raiz.querySelector("#dica-plano").textContent = `${gbInteiros(plano.gb_principais)} de uso geral · `
        + `${Formato.reais(plano.mensalidade_brl)}/mês. Ao trocar o plano, os GB utilizados passam à franquia dele; depois, ajuste o consumo.`;
      for (const nome of ["gb_usados", "dias_desde_renovacao"]) {
        const controle = this.raiz.querySelector(`#campo-${nome}`);
        controle.value = String(this.entradas[nome]);
        this.mostrarValor(controle);
      }
      this.raiz.querySelectorAll('input[name="cenario"]').forEach((controle) => {
        controle.checked = controle.value === this.cenario;
      });
    }

    mostrarValor(controle) {
      const valor = Number(controle.value);
      const texto = controle.dataset.campo === "gb_usados" ? gbInteiros(valor) : `${valor} ${valor === 1 ? "dia" : "dias"}`;
      this.raiz.querySelector(`#valor-${controle.dataset.campo}`).textContent = texto;
      controle.setAttribute("aria-valuetext", texto);
    }

    aoMudar(evento) {
      const campo = evento.target;
      if (campo.name === "cenario") {
        if (!Object.hasOwn(CENARIOS, campo.value)) return;
        this.cenario = campo.value;
        this.entradas = this.entradasPadrao(this.cenario);
        this.preencherFormulario();
      } else if (campo.dataset.campo === "plano") {
        const plano = this.recomendador.porNome(campo.value);
        if (!plano) return;
        this.entradas.plano = plano.nome;
        this.entradas.gb_usados = plano.gb_principais;
        this.preencherFormulario();
      } else if (["gb_usados", "dias_desde_renovacao"].includes(campo.dataset.campo)) {
        this.entradas[campo.dataset.campo] = Number(campo.value);
        this.mostrarValor(campo);
      } else return;
      this.modo = "simulacao";
      this.renderizarResultados();
      if (evento.type === "change") this.anunciar("Análise e recomendações atualizadas com os valores da simulação.");
    }

    aoClicar(evento) {
      const botao = evento.target.closest("button");
      if (!botao || !this.raiz.contains(botao)) return;
      if (botao.dataset.acao === "voltar-etapa") {
        this.irPara(this.passo - 1);
        return;
      }
      if (botao.dataset.acao === "manter-plano") {
        const confirmacao = this.raiz.querySelector("#confirmacao-plano");
        if (!confirmacao || !this.planoAdequado(this.calcular("simulacao"))) return;
        confirmacao.hidden = false;
        botao.textContent = "Plano mantido nesta simulação";
        botao.disabled = true;
        this.anunciar(confirmacao.textContent);
        return;
      }
      if (botao.dataset.acao === "voltar-exemplo") {
        this.modo = "exemplo";
        this.cenario = "mais_internet";
        this.entradas = this.entradasPadrao(this.cenario);
        this.preencherFormulario();
        this.renderizarResultados();
        this.irPara(1);
        this.anunciar("Exemplo original restaurado.");
        return;
      }
      if (botao.dataset.acao === "comecar-simulacao") {
        this.modo = "simulacao";
        this.renderizarResultados();
      }
      if (botao.dataset.acao === "ver-simulacao") {
        evento.preventDefault();
        this.verSimulacao();
      } else if (botao.dataset.ir) this.irPara(Number(botao.dataset.ir));
      else if (botao.dataset.plano) {
        const conteudo = this.raiz.querySelector(`#${botao.getAttribute("aria-controls")}`);
        const aberto = botao.getAttribute("aria-expanded") === "true";
        botao.setAttribute("aria-expanded", String(!aberto));
        conteudo.hidden = aberto;
      }
    }

    verSimulacao() {
      this.modo = "simulacao";
      this.renderizarResultados();
      this.irPara(7);
      this.anunciar(this.raiz.querySelector("#etapa-7").dataset.resultado === "adequado"
        ? "Resultado da simulação: plano adequado ao consumo previsto."
        : "Panorama completo da simulação: análise e recomendações atualizadas.");
    }

    irPara(numero) {
      this.passo = Math.min(7, Math.max(1, numero));
      this.mostrarEtapa();
      const titulo = this.raiz.querySelector(`#titulo-etapa-${this.passo}`);
      titulo.focus({ preventScroll: true });
      const reduzido = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      titulo.closest(".step").scrollIntoView({ behavior: reduzido ? "auto" : "smooth", block: "nearest", inline: "nearest" });
      this.anunciar(`Etapa ${this.passo} de 7: ${titulo.textContent}.`);
    }

    mostrarEtapa() {
      this.raiz.querySelectorAll(".step").forEach((etapa, indice) => {
        const atual = indice + 1 === this.passo;
        etapa.hidden = !atual;
        etapa.inert = !atual;
        if (atual) etapa.setAttribute("aria-current", "step");
        else etapa.removeAttribute("aria-current");
      });
      this.raiz.dataset.etapa = String(this.passo);
      this.raiz.querySelector('[data-acao="voltar-etapa"]').hidden = this.passo === 1;
      this.raiz.querySelector("#progresso-fluxo").textContent = `Etapa ${this.passo} de 7`;
    }

    anunciar(texto) { this.raiz.querySelector(".jornada-status").textContent = texto; }

    dataCurta(data) { return Formato.data(data).slice(0, 5); }

    prazo(resultado) {
      const { risco, dataDaRenovacao } = resultado;
      if (risco.saldo_gb <= 0) return { rotulo: "Já esgotada", data: null, dias: 0 };
      if (risco.dias_ate_acabar === null || risco.dias_ate_acabar >= risco.dias_ate_renovacao) {
        return { rotulo: "Até a renovação", data: dataDaRenovacao, dias: null };
      }
      const dias = Math.ceil(risco.dias_ate_acabar);
      if (dias >= risco.dias_ate_renovacao) return { rotulo: "Até a renovação", data: dataDaRenovacao, dias: null };
      const data = new Date(this.atributos.referencia.getTime() + dias * 86400000);
      return { rotulo: this.dataCurta(data), data, dias };
    }

    textoDaMensagem(resultado) {
      const { risco } = resultado;
      const prazo = this.prazo(resultado);
      let previsao;
      if (risco.saldo_gb <= 0) previsao = "Sua internet de uso geral já acabou, antes da renovação.";
      else if (prazo.dias !== null) previsao = `Nesse ritmo, sua internet pode acabar <b>em ${prazo.dias} `
        + `${prazo.dias === 1 ? "dia" : "dias"} (${this.dataCurta(prazo.data)})</b>, antes da renovação.`;
      else if (risco.nivel === "atencao") previsao = "Nesse ritmo, você poderá usar quase todo o seu saldo até a renovação.";
      else previsao = "Nesse ritmo, sua internet deve ser suficiente até a renovação.";
      return `Olá, cliente Claro! 👋<br><br>Você já usou <b>${Formato.numero(risco.consumo_informado_gb, Number.isInteger(risco.consumo_informado_gb) ? 0 : 1)} dos `
        + `${gbInteiros(risco.capacidade_gb)}</b> do seu plano. ${previsao}<br><br>`
        + 'Preparamos uma análise rápida com opções para você.<div class="time">10:24</div>';
    }

    avisoDeRisco(resultado) {
      const { risco } = resultado;
      const falta = Math.max(0, risco.demanda_prevista_gb - risco.saldo_gb);
      let classe = "", titulo, descricao;
      if (risco.saldo_gb <= 0) {
        classe = "esgotada";
        titulo = "A franquia acabou";
        descricao = "Os GB de uso geral já acabaram. Com base no seu uso recente, você pode precisar de mais internet até a renovação.";
      } else if (falta > 0) {
        titulo = `Vão faltar cerca de ${gbInteiros(falta)}`;
        descricao = "Com base no seu uso recente, nossa inteligência identificou que a sua internet pode não ser suficiente até a renovação.";
      } else if (risco.nivel === "atencao") {
        titulo = "Fique de olho no seu consumo";
        descricao = "Com base no seu uso recente, você poderá usar quase todo o seu saldo até a renovação.";
      } else {
        classe = "ok";
        titulo = "Sua internet deve ser suficiente";
        descricao = "Com base no seu uso recente, a sua internet deve durar até a renovação.";
      }
      return `<div class="warn ${classe}" data-risco="${risco.nivel}"><div class="ic" aria-hidden="true">${classe === "ok" ? "✓" : "!"}</div>`
        + `<div><div class="tt">${titulo}</div><div class="dd">${descricao}</div></div></div>`;
    }

    telaAnalise(resultado) {
      const { risco, dataDaRenovacao } = resultado;
      const total = risco.consumo_informado_gb + risco.demanda_prevista_gb;
      const escala = Math.max(total, risco.capacidade_gb, 1);
      const usado = 100 * risco.consumo_informado_gb / escala;
      const previsto = 100 * risco.demanda_prevista_gb / escala;
      const limite = 100 * risco.capacidade_gb / escala;
      const prazo = this.prazo(resultado);
      return `<div class="h2">Sua análise de consumo</div>${this.avisoDeRisco(resultado)}`
        + `<div class="meter"><div class="top"><span>Usado até hoje</span><span>Plano: ${gbInteiros(risco.capacidade_gb)}</span></div>`
        + `<div class="big">${gbInteiros(risco.consumo_informado_gb)}</div>`
        + `<div class="bar" role="img" aria-label="${gbInteiros(risco.consumo_informado_gb)} usados; ${gbInteiros(risco.demanda_prevista_gb)} previstos até renovar; franquia de ${gbInteiros(risco.capacidade_gb)}">`
        + `<div class="used" style="width:${usado}%"></div><div class="fc" style="left:${usado}%;width:${previsto}%"></div><div class="lim" style="left:${limite}%"></div></div>`
        + `<div class="scale" aria-hidden="true"><span style="left:2%">0</span><span style="left:${Math.min(96, Math.max(10, usado))}%">${Formato.numero(risco.consumo_informado_gb, 1)}</span>`
        + `<span style="left:${Math.min(96, Math.max(10, limite))}%">${Formato.numero(risco.capacidade_gb, 0)}</span></div>`
        + `<div class="legend"><span><i style="background:var(--ink)"></i>Usado</span><span><i style="background:#e8a317"></i>Previsão até ${this.dataCurta(dataDaRenovacao)}</span></div></div>`
        + `<div class="kv"><div>Previsão de término dos GB<b>${prazo.rotulo}</b></div><div>Renova em<b>${this.dataCurta(dataDaRenovacao)}</b></div></div>`
        + '<div class="aino"><span aria-hidden="true">💡</span><span>Essa previsão é feita pela nossa inteligência, com base no seu histórico de uso e no seu perfil de consumo.</span></div>'
        + '<div class="mt"><button type="button" class="cta" data-ir="4">Ver planos recomendados</button></div>';
    }

    cartaoDePlano(opcao, indice, resultado, prefixo) {
      const nome = Formato.escapar(opcao.nome_do_plano);
      const rotulo = opcao.tipo === "manter" ? "Manter o plano atual" : opcao.tipo === "economia"
        ? (indice === 0 ? "Mais recomendado" : "Mais econômico") : ["Mais recomendado", "Mais internet para o dia a dia", "Para maior tranquilidade"][indice];
      const detalheId = `${prefixo}-plano-${indice}`;
      const pontuacao = opcao.tipo === "manter" ? "Seu plano atual cobre a demanda estimada; não é necessária uma nova oferta."
        : this.estadoDoModelo !== "pronto" ? "Ordem de preço: o modelo de recomendação não carregou."
        : `Pontuação de recomendação: ${Formato.numero(opcao.pontuacao_recomendacao * 100, 0)} pontos de 100. `
          + "É uma nota para ordenar opções; não representa uma chance comprovada de aceitação.";
      const preco = opcao.diferenca_de_preco_brl < 0 ? `${Formato.reais(-opcao.diferenca_de_preco_brl)} a menos por mês.`
        : opcao.diferenca_de_preco_brl > 0 ? `${Formato.reais(opcao.diferenca_de_preco_brl)} a mais por mês.` : "Mesmo preço do plano de referência.";
      const custoBeneficio = opcao.tipo === "economia" ? "Boa relação custo-benefício" : "Mais tranquilidade até a renovação";
      const folga = opcao.folga_pequena ? `Folga pequena: sobram ${gbInteiros(opcao.folga_gb)}; se o ritmo subir, pode faltar internet.`
        : `Folga estimada de ${gbInteiros(opcao.folga_gb)}.`;
      return `<button type="button" class="plan${indice === 0 ? " rec" : ""}" data-plano="${nome}" aria-expanded="false" aria-controls="${detalheId}">`
        + `<div><span class="tag${indice === 0 ? "" : " alt"}">${rotulo}</span><div class="gb">${gbInteiros(opcao.gb_do_plano)}</div>`
        + `<div class="pr">${Formato.reais(opcao.mensalidade_brl)}/mês</div><div class="ds">${nome}</div>`
        + '<div class="ds"><span class="c" aria-hidden="true">✓</span>Atende ao seu uso previsto</div>'
        + `<div class="ds"><span class="c" aria-hidden="true">✓</span>${opcao.folga_pequena ? "Compare a folga disponível" : custoBeneficio}</div></div><span class="arr" aria-hidden="true">›</span></button>`
        + `<div class="detalhe-do-plano" id="${detalheId}" hidden>${pontuacao}<br>${preco}<br>Cobre a demanda estimada de `
        + `${gbInteiros(resultado.recomendacao.necessidade_gb)}. ${folga}</div>`;
    }

    listaDePlanos(resultado, prefixo) {
      if (resultado.recomendacao.nenhum_plano_cobre) {
        const maior = Math.max(...this.dados.planos.map((plano) => plano.gb_principais));
        return `<div class="sem-opcao">Nenhum plano das planilhas cobre essa demanda: o maior tem ${gbInteiros(maior)}.</div>`;
      }
      return resultado.recomendacao.opcoes.map((opcao, indice) => this.cartaoDePlano(opcao, indice, resultado, prefixo)).join("");
    }

    resumoDaDemanda(resultado) {
      const { risco, atributos, recomendacao } = resultado;
      return `<p class="resumo-da-demanda">${Formato.gb(risco.consumo_informado_gb)} usados + ${Formato.gb(risco.demanda_prevista_gb)} `
        + `previstos até renovar = ${Formato.gb(risco.consumo_informado_gb + risco.demanda_prevista_gb)} estimados para este ciclo. `
        + `Seu uso típico mensal é de ${Formato.gb(atributos.dados_gb)}.</p>`
        + `<p class="demanda-do-ciclo">Demanda estimada para um ciclo: ${Formato.gb(recomendacao.necessidade_gb)}. `
        + "Usamos o maior valor entre o total estimado para este ciclo e seu uso típico mensal.</p>";
    }

    planoAdequado(resultado) {
      return this.estadoDoModelo === "pronto" && !resultado.carregando
        && resultado.risco.nivel === "baixo" && resultado.recomendacao.plano_atual_atende;
    }

    telaPlanoAdequado(resultado) {
      const { risco, plano, recomendacao } = resultado;
      const percentual = Math.round(risco.percentual_utilizado);
      const previsto = 100 * risco.demanda_prevista_gb / plano.gb_principais;
      const sobra = Math.max(0, risco.saldo_gb - risco.demanda_prevista_gb);
      const economia = recomendacao.opcoes.some((opcao) => opcao.tipo === "economia")
        ? '<button type="button" class="btn ghost" data-ir="4">Ver opções para economizar</button>' : "";
      return '<div class="h2">Resultado da sua simulação</div>'
        + '<div class="ok-hero"><img class="mascot" src="mascote-claro-guia.svg" alt="Mascote do Claro Guia" width="62" height="58">'
        + '<div><div class="tt">Boa notícia! Seu plano é adequado para você.</div>'
        + '<div class="dd">Pela previsão, você terá internet de sobra até a renovação.</div></div></div>'
        + '<div class="sec-t">▮▮ Sua análise de consumo</div><div class="grid">'
        + `<div>Plano atual<b>${gbInteiros(plano.gb_principais)}</b><span>${Formato.escapar(plano.nome)}</span></div>`
        + `<div class="r">Usado até agora<b>${gbInteiros(risco.consumo_informado_gb)} (${percentual}%)</b></div>`
        + `<div class="full"><div class="gbar" role="img" aria-label="${gbInteiros(risco.consumo_informado_gb)} usados; ${gbInteiros(risco.demanda_prevista_gb)} previstos até renovar; ${gbInteiros(sobra)} de sobra prevista">`
        + `<span class="u" style="width:${risco.percentual_utilizado}%"></span><span class="f" style="left:${risco.percentual_utilizado}%;width:${previsto}%"></span></div>`
        + '<div class="glg"><span><i style="background:var(--ok)"></i>Usado</span><span><i style="background:#7cc79a"></i>Previsão até a renovação</span></div></div>'
        + `<div class="t">Previsão de uso até a renovação<b>${gbInteiros(risco.demanda_prevista_gb)}</b></div>`
        + `<div class="r t sobra">Sobra prevista<b>~${gbInteiros(sobra)}</b></div></div>`
        + '<div class="hint"><span class="d" aria-hidden="true">i</span><span>Essa previsão é feita pela nossa inteligência, com base no seu histórico de uso e no seu perfil de consumo.</span></div>'
        + '<div class="sec-t">Mantendo seu plano, você tem</div><div class="ben">'
        + '<div><span class="c" aria-hidden="true">✓</span>Internet suficiente até a renovação</div>'
        + '<div><span class="c" aria-hidden="true">✓</span>Sem mudança no valor da sua fatura</div></div>'
        + economia + '<div class="mt"><button type="button" class="btn pri manter-plano" data-acao="manter-plano">Continuar com meu plano</button>'
        + '<p class="confirmacao-plano" id="confirmacao-plano" role="status" hidden>Você escolheu manter o plano atual nesta simulação.</p>'
        + '<button type="button" class="btn ghost" data-ir="6">Simular outro cenário</button></div>';
    }

    telaPanorama(resultado) {
      const { risco, plano } = resultado;
      const percentual = Math.round(risco.percentual_utilizado);
      return '<div class="sec-t">▮▮ Sua análise de consumo</div><div class="grid">'
        + `<div>Plano atual<b>${gbInteiros(plano.gb_principais)}</b><span>${Formato.escapar(plano.nome)}</span></div>`
        + `<div class="r">Usado até agora<b>${gbInteiros(risco.consumo_informado_gb)} (${percentual}%)</b></div>`
        + `<div class="full"><div class="bar" role="img" aria-label="${percentual}% da franquia usados"><i style="width:${percentual}%"></i></div></div>`
        + `<div class="t">Disponível<b>${gbInteiros(risco.saldo_gb)}</b></div><div class="r t">Previsão de uso até a renovação<b>${gbInteiros(risco.demanda_prevista_gb)}</b></div></div>`
        + `<div class="hint"><span class="d" aria-hidden="true">i</span><span>${risco.demanda_prevista_gb > risco.saldo_gb
          ? "Se o seu uso continuar nesse ritmo, a sua internet pode não ser suficiente." : "Se o seu uso continuar nesse ritmo, a sua internet deve ser suficiente até a renovação."}</span></div>`
        + this.resumoDaDemanda(resultado) + '<div class="sec-t">★ Nossas recomendações</div>'
        + this.listaDePlanos(resultado, "panorama")
        + '<button type="button" class="btn sec" data-ir="6">Ajustar simulação</button>';
    }

    renderizarResultados() {
      const resultado = this.calcular();
      if (resultado.carregando) return;
      const simulacao = this.calcular("simulacao");
      this.raiz.dataset.modo = this.modo;
      for (const numero of [6]) {
        const nota = this.raiz.querySelector(`#etapa-${numero} .note p`);
        let contexto = nota.querySelector(".contexto-cenario");
        if (!contexto) {
          contexto = document.createElement("small");
          contexto.className = "contexto-cenario";
          nota.append(contexto);
        }
        contexto.textContent = this.modo === "exemplo" ? "Outro cenário: ajuste os valores no simulador." : "";
      }
      this.raiz.querySelector("#mensagem-dinamica").innerHTML = this.textoDaMensagem(resultado);
      this.raiz.querySelector("#previa-mensagem").textContent = resultado.risco.nivel === "baixo" ? "Seu consumo está em dia..." : "Atenção ao seu consumo...";
      this.raiz.querySelector("#analise-dinamica").innerHTML = this.telaAnalise(resultado);
      this.raiz.querySelector("#planos-dinamicos").innerHTML = this.listaDePlanos(resultado, "sugestoes");
      const adequado = this.planoAdequado(simulacao);
      const etapa = this.raiz.querySelector("#etapa-7");
      etapa.classList.toggle("resultado-positivo", adequado);
      etapa.dataset.resultado = adequado ? "adequado" : "panorama";
      etapa.querySelector(".note h2").textContent = adequado ? "Resultado: plano adequado" : "Panorama completo";
      etapa.querySelector(".note p").textContent = adequado
        ? "Após a simulação, a inteligência mostra que o plano atual atende bem ao consumo e que vale continuar com ele."
        : "Mostra a análise de consumo, as nossas recomendações e outras informações, como no site.";
      this.raiz.querySelector("#panorama-dinamico").innerHTML = adequado
        ? this.telaPlanoAdequado(simulacao) : this.telaPanorama(simulacao);
    }
  }

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
      raiz.dataset.modelo = "erro";
      raiz.querySelector(".jornada-status").textContent = "Não foi possível iniciar a simulação neste navegador.";
      raiz.querySelector(".aviso-persistente").textContent = "Não foi possível iniciar a simulação. Os detalhes do projeto continuam disponíveis no botão abaixo.";
    }
  });
})();
