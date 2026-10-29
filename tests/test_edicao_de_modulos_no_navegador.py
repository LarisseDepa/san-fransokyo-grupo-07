"""Edição dos módulos do projeto no sistema de arquivos virtual do visitante."""
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from edicao_de_modulos_no_navegador import EdicaoDeModulosNoNavegador


@pytest.fixture
def ambiente_do_editor(tmp_path, monkeypatch):
    raiz = tmp_path / "projeto"
    (raiz / "src" / "pacote").mkdir(parents=True)
    editor = tmp_path / "editor"
    editor.mkdir()
    monkeypatch.chdir(editor)
    return raiz, editor


def test_expoe_codigo_e_reutiliza_atalho(ambiente_do_editor):
    raiz, editor = ambiente_do_editor
    (raiz / "src" / "modelo.py").write_text("VALOR = 1\n", encoding="utf-8")
    edicao = EdicaoDeModulosNoNavegador(raiz)

    assert edicao.executar() == ()
    atalho = editor / edicao.NOME_DO_ATALHO
    assert atalho.is_symlink()
    assert atalho.resolve() == raiz
    assert (atalho / "src" / "modelo.py").read_text(encoding="utf-8") == "VALOR = 1\n"
    assert edicao.executar() == ()
    assert atalho.resolve() == raiz


def test_remove_apenas_modulos_do_proprio_codigo(ambiente_do_editor, monkeypatch):
    raiz, editor = ambiente_do_editor
    arquivos = {
        "teste_editor_modelo": raiz / "src" / "modelo.py",
        "teste_editor_transitivo": raiz / "src" / "pacote" / "auxiliar.py",
        "teste_editor_biblioteca": editor / "site-packages" / "biblioteca.py",
        "teste_editor_outro_projeto": raiz.parent / "outro" / "src" / "modelo.py",
        "teste_editor_pasta_semelhante": raiz / "src_alheio" / "modelo.py",
        "teste_editor_fora_do_codigo": raiz / "outro.py",
    }
    modulos = {}
    for nome, arquivo in arquivos.items():
        modulo = ModuleType(nome)
        modulo.__file__ = str(arquivo)
        modulos[nome] = modulo
        monkeypatch.setitem(sys.modules, nome, modulo)
    sem_arquivo = ModuleType("teste_editor_sem_arquivo")
    monkeypatch.setitem(sys.modules, sem_arquivo.__name__, sem_arquivo)
    modulo_do_sys = sys.modules["sys"]
    modulo_do_importlib = sys.modules["importlib"]
    invalidacoes = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidacoes.append(True))

    removidos = EdicaoDeModulosNoNavegador(raiz).executar()

    assert set(removidos) == {"teste_editor_modelo", "teste_editor_transitivo"}
    assert all(nome not in sys.modules for nome in removidos)
    for nome, modulo in modulos.items():
        if nome not in removidos:
            assert sys.modules[nome] is modulo
    assert sys.modules[sem_arquivo.__name__] is sem_arquivo
    assert sys.modules["sys"] is modulo_do_sys
    assert sys.modules["importlib"] is modulo_do_importlib
    assert invalidacoes == [True]


@pytest.mark.parametrize("tipo_de_conflito", ["arquivo", "pasta", "atalho", "atalho_quebrado"])
def test_conflito_preserva_entrada_e_modulos(ambiente_do_editor, monkeypatch, tipo_de_conflito):
    raiz, editor = ambiente_do_editor
    atalho = editor / EdicaoDeModulosNoNavegador.NOME_DO_ATALHO
    destino_alheio = editor / "outro_projeto"
    if tipo_de_conflito == "arquivo":
        atalho.write_text("conteúdo alheio", encoding="utf-8")
    elif tipo_de_conflito == "pasta":
        atalho.mkdir()
        (atalho / "preservar.txt").write_text("conteúdo alheio", encoding="utf-8")
    else:
        if tipo_de_conflito == "atalho":
            destino_alheio.mkdir()
        atalho.symlink_to(destino_alheio, target_is_directory=True)
    modulo = ModuleType("teste_editor_preservado_em_conflito")
    modulo.__file__ = str(raiz / "src" / "modelo.py")
    monkeypatch.setitem(sys.modules, modulo.__name__, modulo)

    with pytest.raises(FileExistsError, match="arquivo existente foi preservado"):
        EdicaoDeModulosNoNavegador(raiz).executar()

    assert sys.modules[modulo.__name__] is modulo
    if tipo_de_conflito == "arquivo":
        assert atalho.read_text(encoding="utf-8") == "conteúdo alheio"
    elif tipo_de_conflito == "pasta":
        assert (atalho / "preservar.txt").read_text(encoding="utf-8") == "conteúdo alheio"
    else:
        assert atalho.is_symlink()
        assert atalho.readlink() == destino_alheio


def test_symlink_para_modulo_alheio_nao_remove_biblioteca(ambiente_do_editor, monkeypatch):
    raiz, editor = ambiente_do_editor
    arquivo_alheio = editor / "modulo_externo.py"
    arquivo_alheio.write_text("VALOR = 1\n", encoding="utf-8")
    link = raiz / "src" / "aparente_modulo_do_projeto.py"
    link.symlink_to(arquivo_alheio)
    modulo = ModuleType("teste_editor_link_alheio")
    modulo.__file__ = str(link)
    monkeypatch.setitem(sys.modules, modulo.__name__, modulo)

    assert EdicaoDeModulosNoNavegador(raiz).executar() == ()
    assert sys.modules[modulo.__name__] is modulo


def test_remove_somente_bytecode_do_proprio_codigo(ambiente_do_editor, monkeypatch):
    raiz, editor = ambiente_do_editor
    cache_proprio = raiz / "src" / "__pycache__" / "modelo.cpython.pyc"
    cache_proprio.parent.mkdir()
    cache_alheio = editor / "biblioteca.pyc"
    arquivo_fonte = raiz / "src" / "preservar.py"
    for arquivo in (cache_proprio, cache_alheio, arquivo_fonte):
        arquivo.write_bytes(b"preservar apenas arquivos fora do cache proprio")
    casos = [
        ("teste_editor_cache_proprio", raiz / "src" / "modelo.py", cache_proprio),
        ("teste_editor_cache_alheio", raiz / "src" / "outro.py", cache_alheio),
        ("teste_editor_cache_fonte", raiz / "src" / "terceiro.py", arquivo_fonte),
    ]
    for nome, arquivo, cache in casos:
        modulo = ModuleType(nome)
        modulo.__file__ = str(arquivo)
        modulo.__cached__ = str(cache)
        monkeypatch.setitem(sys.modules, nome, modulo)

    removidos = EdicaoDeModulosNoNavegador(raiz).executar()

    assert set(removidos) == {nome for nome, _, _ in casos}
    assert not cache_proprio.exists()
    assert cache_alheio.read_bytes() == b"preservar apenas arquivos fora do cache proprio"
    assert arquivo_fonte.read_bytes() == b"preservar apenas arquivos fora do cache proprio"


def test_novo_import_aplica_edicao_com_mesmo_tamanho_e_data(ambiente_do_editor, monkeypatch):
    raiz, _ = ambiente_do_editor
    nome = "teste_editor_import_real"
    arquivo = raiz / "src" / f"{nome}.py"
    arquivo.write_text("VALOR = 1\n", encoding="utf-8")
    data_original = arquivo.stat()
    monkeypatch.syspath_prepend(str(raiz / "src"))
    try:
        modulo = importlib.import_module(nome)
        assert modulo.VALOR == 1
        assert Path(modulo.__cached__).is_file()
        arquivo.write_text("VALOR = 2\n", encoding="utf-8")
        os.utime(arquivo, ns=(data_original.st_atime_ns, data_original.st_mtime_ns))

        removidos = EdicaoDeModulosNoNavegador(raiz).executar()

        assert nome in removidos
        assert importlib.import_module(nome).VALOR == 2
    finally:
        sys.modules.pop(nome, None)
