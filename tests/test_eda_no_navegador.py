"""A EDA portável recalcula os dados sem importar dependências nativas de profiling."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eda_no_navegador import AnaliseExploratoriaNoNavegador


def test_eda_recalcula_estatisticas_e_alertas_apos_alteracao(tmp_path):
    dados = pd.DataFrame({"consumo": [1.0, 3.0, None], "grupo": ["A", "A", "A"]})
    inicial = AnaliseExploratoriaNoNavegador(dados).executar(tmp_path / "inicial")
    assert "MISSING" in set(inicial.alertas.tipo_de_alerta)
    assert "CONSTANT" in set(inicial.alertas.tipo_de_alerta)
    assert inicial.estatisticas_descritivas.set_index("coluna").loc["consumo", "mean"] == 2.0

    dados["consumo"] = [10.0, 20.0, 30.0]
    alterado = AnaliseExploratoriaNoNavegador(dados).executar(tmp_path / "alterado")
    assert "MISSING" not in set(alterado.alertas.tipo_de_alerta)
    assert alterado.estatisticas_descritivas.set_index("coluna").loc["consumo", "mean"] == 20.0
    assert alterado.caminho_do_relatorio_detalhado.exists()
    assert (tmp_path / "alterado" / "descricao_dos_dados.txt").exists()


def test_relatorio_identifica_limites_e_escapa_texto_inserido(tmp_path):
    dados = pd.DataFrame({"<script>alert(1)</script>": ["<img src=x onerror=alert(1)>", "outro"]})
    resultado = AnaliseExploratoriaNoNavegador(dados).executar(tmp_path)
    html = resultado.caminho_do_relatorio_detalhado.read_text(encoding="utf-8")
    assert "não executa o fg-data-profiling" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html
    assert "<img src=x" not in html


def test_importacao_da_eda_portavel_nao_carrega_profiling_nativo():
    raiz = Path(__file__).resolve().parent.parent
    codigo = (
        "import sys; "
        f"sys.path.insert(0, {str(raiz / 'src')!r}); "
        "from eda_no_navegador import AnaliseExploratoriaNoNavegador; "
        "assert 'data_profiling' not in sys.modules; "
        "assert 'numba' not in sys.modules; "
        "assert 'phik' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", codigo], check=True, capture_output=True, text=True)
