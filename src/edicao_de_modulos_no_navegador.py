"""Expõe o código auxiliar e renova seus imports no ambiente do navegador."""
import importlib
import sys
from pathlib import Path


class EdicaoDeModulosNoNavegador:
    """Aplica edições em módulos próprios sem reiniciar o ambiente Python."""

    NOME_DO_ATALHO = "codigo_do_projeto"

    def __init__(self, raiz):
        self.raiz = Path(raiz).resolve()
        self.pasta_do_codigo = self.raiz / "src"

    def executar(self):
        self._expor_codigo_no_explorador()
        nomes_removidos = self._remover_modulos_do_projeto()
        importlib.invalidate_caches()
        return nomes_removidos

    def _expor_codigo_no_explorador(self):
        atalho = Path.cwd() / self.NOME_DO_ATALHO
        if atalho.is_symlink() and atalho.resolve() == self.raiz:
            return
        if atalho.exists() or atalho.is_symlink():
            raise FileExistsError(
                f"Não foi possível mostrar o código: {atalho} já existe e não é "
                "o atalho deste projeto. O arquivo existente foi preservado."
            )
        atalho.symlink_to(self.raiz, target_is_directory=True)

    def _remover_modulos_do_projeto(self):
        nomes_removidos = []
        for nome, modulo in tuple(sys.modules.items()):
            if self._arquivo_pertence_ao_codigo(getattr(modulo, "__file__", None)):
                self._remover_bytecode_do_modulo(modulo)
                sys.modules.pop(nome, None)
                nomes_removidos.append(nome)
        return tuple(nomes_removidos)

    def _remover_bytecode_do_modulo(self, modulo):
        arquivo_de_cache = getattr(modulo, "__cached__", None)
        if self._arquivo_pertence_ao_codigo(arquivo_de_cache):
            caminho_do_cache = Path(arquivo_de_cache)
            if caminho_do_cache.suffix == ".pyc":
                caminho_do_cache.unlink(missing_ok=True)

    def _arquivo_pertence_ao_codigo(self, arquivo):
        if not arquivo:
            return False
        try:
            return Path(arquivo).resolve().is_relative_to(self.pasta_do_codigo)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
