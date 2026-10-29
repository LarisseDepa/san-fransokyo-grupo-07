"""
Leitura de planilhas (CSV ou Excel) e junção de várias planilhas numa só, com a coluna 'origem'.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tabelas import COLUNA_ORIGEM, tipo_simplificado


@dataclass
class JuncaoDePlanilhas:
    """Resultado de LeitorDePlanilhas.juntar()."""
    tabela: pd.DataFrame
    resumo: str
    avisos: list = field(default_factory=list)  # diferenças de colunas e de tipos entre as planilhas


class LeitorDePlanilhas:
    """Lê planilhas CSV ou Excel, lista as planilhas de uma pasta e junta várias numa só."""

    EXTENSOES = (".csv", ".xlsx", ".xls")
    EXTENSOES_DO_EXCEL = (".xlsx", ".xls")

    def listar(self, caminhos):
        """Aceita arquivos e pastas; uma pasta vira todas as planilhas dentro dela, em ordem."""
        planilhas = []
        for caminho in map(Path, caminhos):
            if caminho.is_dir():
                planilhas += sorted(str(arquivo) for arquivo in caminho.iterdir() if self.eh_planilha(arquivo))
            else:
                planilhas.append(str(caminho))
        return planilhas

    def eh_planilha(self, caminho):
        return Path(caminho).suffix.lower() in self.EXTENSOES

    def ler(self, arquivo, nome_do_arquivo):
        """Lê CSV ou Excel. `arquivo` pode ser um caminho ou um arquivo em memória (upload)."""
        if str(nome_do_arquivo).lower().endswith(self.EXTENSOES_DO_EXCEL):
            return pd.read_excel(arquivo)
        return pd.read_csv(arquivo, encoding="utf-8-sig")  # utf-8-sig remove o BOM do início do arquivo

    def juntar(self, arquivos):
        """Junta várias planilhas numa só (sem limpar nada) e cria a coluna 'origem'.
        arquivos: lista de (nome do arquivo, arquivo)."""
        planilhas = [self._ler_com_origem(arquivo, nome_do_arquivo) for nome_do_arquivo, arquivo in arquivos]
        tabela = pd.concat([planilha for _, planilha in planilhas], ignore_index=True)
        resumo = (f"{len(planilhas)} planilhas unificadas: {len(tabela)} linhas no total ("
                  + ", ".join(f"{nome}: {len(planilha)}" for nome, planilha in planilhas) + ")")
        avisos = self._diferencas_de_colunas(planilhas) + self._diferencas_de_tipos(planilhas)
        return JuncaoDePlanilhas(tabela, resumo, avisos)

    def _ler_com_origem(self, arquivo, nome_do_arquivo):
        """Devolve (nome curto, planilha com a coluna 'origem' na frente)."""
        planilha = self.ler(arquivo, nome_do_arquivo)
        planilha.columns = [str(coluna).strip() for coluna in planilha.columns]
        nome = Path(str(nome_do_arquivo)).stem
        planilha.insert(0, COLUNA_ORIGEM, nome)
        return nome, planilha

    @staticmethod
    def _diferencas_de_colunas(planilhas):
        """Colunas que faltam ou sobram em cada planilha, comparando com a primeira."""
        nome_da_referencia, referencia = planilhas[0]
        colunas_da_referencia = set(referencia.columns)
        avisos = []
        for nome, planilha in planilhas[1:]:
            for coluna in sorted(colunas_da_referencia - set(planilha.columns)):
                parecidas = [existente for existente in planilha.columns if existente.lower() == coluna.lower()]
                dica = f" (existe '{parecidas[0]}': renomeie para ficar igual)" if parecidas else ""
                avisos.append(f"{nome}: não tem a coluna '{coluna}' que existe em {nome_da_referencia}{dica}")
            for coluna in sorted(set(planilha.columns) - colunas_da_referencia):
                if not any(existente.lower() == coluna.lower() for existente in colunas_da_referencia):
                    avisos.append(f"{nome}: tem a coluna '{coluna}', que não existe em {nome_da_referencia}")
        return avisos

    @staticmethod
    def _diferencas_de_tipos(planilhas):
        """Colunas que são número numa planilha e texto em outra, por exemplo."""
        _, referencia = planilhas[0]
        avisos = []
        for coluna in referencia.columns:
            tipos = {nome: tipo_simplificado(planilha[coluna]) for nome, planilha in planilhas
                     if coluna in planilha and planilha[coluna].notna().any()}
            if len(set(tipos.values())) > 1:
                detalhe = ", ".join(f"{nome}: {tipo}" for nome, tipo in tipos.items())
                avisos.append(f"Coluna '{coluna}' tem tipos diferentes entre as planilhas ({detalhe})")
        return avisos
