"""Publica o mesmo notebook em modo editável, com arquivos no ambiente do visitante."""
import html
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from string import Template
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class NotebookNoNavegador:
    """Exportação WASM e controles de recuperação independentes do kernel Python."""

    PAGINA = "notebook_editavel.html"
    EDITOR = "editor/index.html"
    PACOTE = "public/projeto.zip"

    def __init__(self, pastas):
        self.pastas = pastas
        self.paginas = Path(__file__).parent / "paginas"

    def executar(self):
        destino_do_editor = self.pastas.site / self.EDITOR
        subprocess.run([
            sys.executable, "-m", "marimo", "export", "html-wasm", str(self.pastas.notebook_completo),
            "-o", str(destino_do_editor), "--mode", "edit", "--no-sandbox", "--no-execute", "--force",
        ], check=True)
        pacote = destino_do_editor.parent / self.PACOTE
        self._empacotar_arquivos(pacote)
        self._salvar_pagina(pacote)

    def _empacotar_arquivos(self, destino):
        """Inclui código fonte e as quatro entradas; nenhum pickle ou saída de treino."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        arquivos = [*sorted((self.pastas.raiz / "src").glob("*.py")),
                    *sorted(self.pastas.originais.glob("*.csv"))]
        with ZipFile(destino, "w", compression=ZIP_DEFLATED) as pacote:
            for arquivo in arquivos:
                nome = arquivo.relative_to(self.pastas.raiz).as_posix()
                entrada = ZipInfo(nome, date_time=(2026, 1, 1, 0, 0, 0))
                entrada.compress_type = ZIP_DEFLATED
                pacote.writestr(entrada, arquivo.read_bytes())

    def _salvar_pagina(self, pacote):
        script = self.paginas / "notebook_editavel.js"
        modelo = self.paginas / self.PAGINA
        versao = sha256(self.pastas.notebook_completo.read_bytes() + pacote.read_bytes()
                        + script.read_bytes() + modelo.read_bytes()).hexdigest()
        pagina = Template(modelo.read_text(encoding="utf-8")).substitute(
            estilo=(self.paginas / "estilo.css").read_text(encoding="utf-8"),
            editor=html.escape(self.EDITOR, quote=True), versao_do_editor=versao,
        )
        (self.pastas.site / self.PAGINA).write_text(pagina, encoding="utf-8")
        (self.pastas.site / script.name).write_bytes(script.read_bytes())
