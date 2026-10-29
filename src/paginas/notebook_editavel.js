/* O controle de restauração fica fora do notebook para continuar disponível mesmo se uma célula falhar. */
(() => {
  "use strict";

  class LaboratorioDoProjeto {
    constructor(elemento) {
      this.elemento = elemento;
      this.quadro = elemento.querySelector("#quadro-do-editor");
      this.status = elemento.querySelector("#status-do-editor");
      this.confirmacao = elemento.querySelector("#confirmacao-da-restauracao");
      this.botaoRestaurar = elemento.querySelector("#restaurar-notebook");
      this.tamanhoDoCodigo = elemento.querySelector("#tamanho-do-codigo");
      this.botaoAmpliar = elemento.querySelector("#ampliar-editor");
      this.contadorDeRestauracoes = 0;
    }

    iniciar() {
      this.tamanhoDoCodigo.addEventListener("change", () => this.ajustarFonte());
      this.botaoAmpliar.addEventListener("click", () => this.ampliarEditor());
      this.botaoRestaurar.addEventListener("click", () => this.confirmacao.showModal());
      this.elemento.querySelector("#cancelar-restauracao").addEventListener("click", () => {
        this.confirmacao.close();
        this.botaoRestaurar.focus();
      });
      this.elemento.querySelector("#confirmar-restauracao").addEventListener("click", () => {
        this.confirmacao.close();
        this.restaurar();
        this.botaoRestaurar.focus();
      });
      this.abrirEditor();
    }

    enderecoOriginal() {
      // A URL publicada é a única fonte: não reutilizar hash, filename ou código da sessão editada.
      const endereco = new URL(this.elemento.dataset.editor, document.baseURI);
      endereco.search = "";
      endereco.hash = "";
      endereco.searchParams.set("v", this.elemento.dataset.versao);
      if (this.contadorDeRestauracoes > 0) {
        endereco.searchParams.set("restauracao", String(this.contadorDeRestauracoes));
      }
      return endereco.href;
    }

    abrirEditor() {
      const editor = document.createElement("iframe");
      editor.id = "editor-python";
      editor.title = "Editor Python do CLARO GUIA";
      editor.src = this.enderecoOriginal();
      editor.addEventListener("load", () => {
        this.ajustarFonte();
        this.status.textContent = this.contadorDeRestauracoes > 0
          ? "Versão original reaberta. Aguarde a inicialização do Python no notebook."
          : "Editor aberto. Aguarde a inicialização do Python no notebook.";
      }, { once: true });
      // Substituir o documento encerra a sessão anterior, inclusive seu ambiente Python.
      this.quadro.replaceChildren(editor);
    }

    ajustarFonte() {
      const documento = this.quadro.querySelector("iframe")?.contentDocument;
      if (!documento?.head) return;
      const tamanho = Number(this.tamanhoDoCodigo.value);
      if (![16, 18, 20, 22].includes(tamanho)) return;
      let estilo = documento.getElementById("legibilidade-do-codigo");
      if (!estilo) {
        estilo = documento.createElement("style");
        estilo.id = "legibilidade-do-codigo";
        documento.head.appendChild(estilo);
      }
      estilo.textContent = `.cm-editor, .cm-editor .cm-content, .cm-editor .cm-gutters {
        font-size: ${tamanho}px !important; line-height: 1.5 !important;
      }`;
    }

    ampliarEditor() {
      const ampliado = this.elemento.classList.toggle("laboratorio--ampliado");
      this.botaoAmpliar.setAttribute("aria-pressed", String(ampliado));
      this.botaoAmpliar.textContent = ampliado ? "Voltar à visão normal" : "Ampliar editor";
    }

    restaurar() {
      this.contadorDeRestauracoes += 1;
      this.status.textContent = "Restaurando a versão original e reiniciando o ambiente Python…";
      this.abrirEditor();
    }
  }

  const elemento = document.getElementById("laboratorio-do-projeto");
  if (elemento) new LaboratorioDoProjeto(elemento).iniciar();
})();
