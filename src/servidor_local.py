"""Prévia local dos arquivos estáticos, incluindo os assets do editor marimo."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pastas_do_projeto import PastasDoProjeto


class ServidorLocal(ThreadingHTTPServer):
    """Aceita o carregamento simultâneo dos módulos JavaScript do notebook."""

    request_queue_size = 128

    def __init__(self, pastas, porta=8000):
        super().__init__(("127.0.0.1", porta), partial(SimpleHTTPRequestHandler, directory=str(pastas.site)))


if __name__ == "__main__":
    pastas = PastasDoProjeto(Path(__file__).resolve().parent.parent)
    with ServidorLocal(pastas) as servidor:
        print("Site disponível em http://127.0.0.1:8000. Para fechar, pressione Ctrl + C.", flush=True)
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass
