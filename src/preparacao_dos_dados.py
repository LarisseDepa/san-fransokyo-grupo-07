"""
Preparação dos dados (fase do CRISP-DM): cópia dos originais, limpeza de cada planilha e união.
As regras ficam em config.py; aqui só há a lógica.

Classes, na ordem de uso (em notebooks/claro_guia_etapas_0_a_8.py):
  CopiaDosOriginais      -> etapa 0: copia os originais e anota o hash de cada um
  LimpezaDePlanilha      -> etapa 1: limpa UMA planilha, um método por passo
  IndicadoresDeQualidade -> compara cada planilha antes x depois da limpeza
  UniaoDasPlanilhas      -> etapa 3: empilha as planilhas limpas e corrige o que depende de tudo junto
  ArquivosDaPreparacao   -> salva os resultados nas pastas do projeto
  RegistroDeAcoes        -> anota cada ação feita (vira relatorios/registro_da_preparacao_dos_dados.csv)
"""
import hashlib
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from skrub import Cleaner, deduplicate

from leitura_de_planilhas import LeitorDePlanilhas
from tabelas import (COLUNA_ORIGEM, COLUNAS_DE_CONTROLE, LIMITE_DE_CATEGORIAS, PROPORCAO_MINIMA_NUMERICA,
                     colunas_de_dados, colunas_de_texto, converter_texto_para_numero, linhas_duplicadas,
                     normalizar_texto, tipo_simplificado)


class RegistroDeAcoes:
    """Anota as ações feitas em uma planilha (ou na união), para o registro da preparação."""

    def __init__(self, planilha):
        self.planilha = planilha
        self.acoes = []

    def anotar(self, tipo_de_acao, acao):
        self.acoes.append({"planilha": self.planilha, "tipo_de_acao": tipo_de_acao, "acao": acao})

    @staticmethod
    def exemplos(valores, quantidade=3):
        """Até `quantidade` valores diferentes, entre aspas, para mostrar no registro."""
        return ", ".join(repr(valor) for valor in valores.unique()[:quantidade])

    @staticmethod
    def exemplos_de_troca(troca, quantidade=4):
        """Algumas trocas feitas, no formato 'antigo'->'novo', para mostrar no registro."""
        return "; ".join(f"{antigo!r}->{novo!r}" for antigo, novo in list(troca.items())[:quantidade])


# ------------------------------------------------------------------ etapa 0
@dataclass
class ResultadoDaCopia:
    planilhas_brutas: dict  # {nome curto: tabela lida da cópia, ainda sem limpeza}
    conferencia: pd.DataFrame  # uma linha por original: planilha, arquivo, linhas, sha256


class CopiaDosOriginais:
    """Etapa 0: copia os originais sem alterar nada, lê as cópias e anota o hash
    (a "impressão digital") de cada original, para provar que ele não mudou."""

    CARACTERES_DO_HASH = 16

    def __init__(self, pastas, config, leitor=None):
        self.pastas = pastas
        self.config = config
        self.leitor = leitor or LeitorDePlanilhas()

    def listar_originais(self):
        return sorted(arquivo for arquivo in self.pastas.originais.glob("*") if self.leitor.eh_planilha(arquivo))

    def nome_curto(self, caminho):
        """'Projeto_1_Recomendador_de_Planos_Economia.csv' -> 'Economia'."""
        return Path(caminho).stem.replace(self.config.PREFIXO_DOS_ARQUIVOS, "")

    def executar(self):
        self._recriar_pasta_da_copia()
        planilhas_brutas, linhas_da_conferencia = {}, []
        for original in self.listar_originais():
            copia = self.pastas.copia_dos_originais / original.name
            shutil.copy2(original, copia)  # cópia idêntica; os originais nunca são alterados
            nome = self.nome_curto(original)
            planilhas_brutas[nome] = self.leitor.ler(copia, original.name)
            linhas_da_conferencia.append({"planilha": nome, "arquivo": original.name,
                                          "linhas": len(planilhas_brutas[nome]), "sha256": self._hash(original)})
        return ResultadoDaCopia(planilhas_brutas, pd.DataFrame(linhas_da_conferencia))

    def _recriar_pasta_da_copia(self):
        if self.pastas.copia_dos_originais.exists():
            shutil.rmtree(self.pastas.copia_dos_originais)
        self.pastas.copia_dos_originais.mkdir(parents=True)

    def _hash(self, arquivo):
        return hashlib.sha256(arquivo.read_bytes()).hexdigest()[:self.CARACTERES_DO_HASH]


# ------------------------------------------------------------------ etapa 1
class LimpezaDePlanilha:
    """Etapa 1: padroniza e limpa UMA planilha, passo a passo (um método por passo).

    Uso:  limpeza = LimpezaDePlanilha(config, "Economia")
          planilha_limpa = limpeza.executar(planilha_bruta)
          limpeza.registro.acoes   # o que foi feito
    """

    IDS_MOSTRADOS_NO_REGISTRO = 10

    def __init__(self, config, nome_da_planilha):
        self.config = config
        self.nome_da_planilha = nome_da_planilha
        self.regras_da_planilha = config.REGRAS_POR_PLANILHA.get(nome_da_planilha, {})
        self.registro = RegistroDeAcoes(nome_da_planilha)

    def executar(self, df):
        """Devolve a planilha limpa. As ações feitas ficam em self.registro."""
        self.registro = RegistroDeAcoes(self.nome_da_planilha)  # começa do zero a cada execução
        passos_em_ordem = [
            self._padronizar_nomes_das_colunas,
            self._remover_colunas,
            self._remover_espacos_extras,
            self._trocar_nulos_disfarcados_por_vazio,
            self._trocar_codigos,
            self._retirar_unidade_escrita_no_numero,
            self._converter_datas,
            self._converter_textos_numericos,
            self._converter_unidades,
            self._anular_valores_fora_das_regras,
            self._preencher_vazios,
            self._remover_linhas_duplicadas,
        ]
        df = df.copy()
        for passo in passos_em_ordem:
            df = passo(df)
        self._avisar_ids_repetidos(df)
        self.registro.anotar("resultado", f"{len(df)} linhas, {len(colunas_de_dados(df))} colunas")
        return df.reset_index(drop=True)

    def _padronizar_nomes_das_colunas(self, df):
        """Tira dos nomes das colunas o caractere invisível BOM e os espaços; renomeia as configuradas."""
        df.columns = [str(coluna).replace("﻿", "").strip() for coluna in df.columns]
        renomear = {**self.config.RENOMEAR_COLUNAS, **self.regras_da_planilha.get("renomear", {})}
        renomear = {antigo: novo for antigo, novo in renomear.items() if antigo in df.columns}
        if renomear:
            df = df.rename(columns=renomear)
            self.registro.anotar("padronização", "colunas renomeadas: "
                                 + ", ".join(f"{antigo} -> {novo}" for antigo, novo in renomear.items()))
        return df

    def _remover_colunas(self, df):
        colunas_presentes = [coluna for coluna in self.config.COLUNAS_PARA_REMOVER if coluna in df]
        if colunas_presentes:
            df = df.drop(columns=colunas_presentes)
            self.registro.anotar("padronização", "colunas removidas: " + ", ".join(colunas_presentes))
        return df

    def _remover_espacos_extras(self, df):
        """Tira espaços do início e do fim dos textos e troca espaços repetidos por um só."""
        for coluna in colunas_de_texto(df):
            antes = df[coluna]
            df[coluna] = df[coluna].str.strip().str.replace(r"\s+", " ", regex=True)
            corrigidos = int((antes.fillna("") != df[coluna].fillna("")).sum())
            if corrigidos:
                self.registro.anotar("limpeza", f"{coluna}: {corrigidos} valores com espaços extras corrigidos")
        return df

    def _trocar_nulos_disfarcados_por_vazio(self, df):
        """Textos como 'não informado' ou '-' viram vazio de verdade."""
        nulos_em_minusculas = {texto.lower() for texto in self.config.NULOS_DISFARCADOS}
        for coluna in colunas_de_texto(df):
            disfarcados = df[coluna].str.lower().isin(nulos_em_minusculas).fillna(False)
            if disfarcados.any():
                exemplos = RegistroDeAcoes.exemplos(df.loc[disfarcados, coluna])
                df.loc[disfarcados, coluna] = None
                self.registro.anotar("limpeza", f"{coluna}: {int(disfarcados.sum())} vazios disfarçados "
                                                f"({exemplos}) viraram vazio")
        return df

    def _trocar_codigos(self, df):
        """Códigos diferentes para a mesma coisa: {'coluna': {'S': 'sim'}}."""
        for coluna, troca in self.regras_da_planilha.get("trocar_valores", {}).items():
            if coluna in df:
                trocados = int(df[coluna].isin(list(troca)).sum())
                df[coluna] = df[coluna].replace(troca)
                self.registro.anotar("padronização", f"{coluna}: {trocados} códigos trocados "
                                     f"({', '.join(f'{antigo}->{novo}' for antigo, novo in troca.items())})")
        return df

    def _retirar_unidade_escrita_no_numero(self, df):
        """'24 GB' -> '24' (a conversão para número vem depois)."""
        for coluna, unidade in self.config.SUFIXOS_DE_UNIDADE.items():
            if coluna not in df or coluna not in colunas_de_texto(df):
                continue
            padrao = rf"\s*{re.escape(unidade)}\s*$"
            com_unidade = df[coluna].str.contains(padrao, case=False, regex=True).fillna(False)
            if com_unidade.any():
                exemplos = RegistroDeAcoes.exemplos(df.loc[com_unidade, coluna])
                df[coluna] = df[coluna].str.replace(padrao, "", case=False, regex=True)
                self.registro.anotar("padronização", f"{coluna}: unidade '{unidade}' retirada de "
                                                     f"{int(com_unidade.sum())} valores ({exemplos})")
        return df

    def _converter_datas(self, df):
        formatos_de_data = {**self.config.FORMATOS_DE_DATA, **self.regras_da_planilha.get("formatos_de_data", {})}
        for coluna, formato in formatos_de_data.items():
            if coluna not in df or pd.api.types.is_datetime64_any_dtype(df[coluna]):
                continue
            preenchidas = df[coluna].notna()
            df[coluna] = pd.to_datetime(df[coluna].astype("string"), format=formato, errors="coerce")
            invalidas = int((preenchidas & df[coluna].isna()).sum())
            self.registro.anotar("padronização", f"{coluna}: datas lidas no formato {formato}"
                                 + (f"; {invalidas} datas inválidas viraram vazio" if invalidas else ""))
        return df

    def _converter_textos_numericos(self, df):
        """Converte para número as colunas de texto em que quase todos os valores são números."""
        separador_decimal = self.regras_da_planilha.get("separador_decimal", "auto")
        for coluna in colunas_de_texto(df):
            preenchidos = df[coluna].dropna()
            if preenchidos.empty:
                continue
            convertida = converter_texto_para_numero(df[coluna], separador_decimal)
            if convertida.notna().sum() / len(preenchidos) < PROPORCAO_MINIMA_NUMERICA:
                continue
            nao_numericos = df[coluna][df[coluna].notna() & convertida.isna()]
            df[coluna] = convertida
            acao = f"{coluna}: texto convertido para número"
            if len(nao_numericos):
                acao += (f"; {len(nao_numericos)} não numéricos viraram vazio "
                         f"({RegistroDeAcoes.exemplos(nao_numericos, 5)})")
            self.registro.anotar("padronização", acao)
        return df

    def _converter_unidades(self, df):
        """Unidades diferentes: multiplica a coluna pelo fator configurado (ex.: MB -> GB)."""
        for coluna, fator in self.regras_da_planilha.get("multiplicar_por", {}).items():
            if coluna in df:
                df[coluna] = pd.to_numeric(df[coluna], errors="coerce") * fator
                self.registro.anotar("padronização", f"{coluna}: valores multiplicados por {fator} "
                                                     "(unidade padronizada)")
        return df

    def _anular_valores_fora_das_regras(self, df):
        """Valores que quebram uma regra de negócio viram vazio (e depois são preenchidos)."""
        for coluna, eh_valido in self.config.REGRAS_DE_NEGOCIO.items():
            if coluna not in df:
                continue
            invalidos = ~(eh_valido(df[coluna]) | df[coluna].isna())
            if invalidos.any():
                df.loc[invalidos, coluna] = None
                self.registro.anotar("limpeza", f"{coluna}: {int(invalidos.sum())} valores fora da regra "
                                                "de negócio viraram vazio")
        return df

    def _preencher_vazios(self, df):
        """Preenche os vazios com a estatística (mediana, média ou moda) desta planilha."""
        coluna_id = self.config.COLUNA_ID
        for coluna, estatistica in self.config.IMPUTACAO_POR_COLUNA.items():
            if coluna not in df or not df[coluna].isna().any():
                continue
            vazios = df[coluna].isna()
            valor = self._calcular_estatistica(df[coluna], estatistica)
            quais = ""
            if coluna_id in df:
                ids_preenchidos = df.loc[vazios, coluna_id].tolist()[:self.IDS_MOSTRADOS_NO_REGISTRO]
                quais = f" (ids: {', '.join(map(str, ids_preenchidos))})"
            df.loc[vazios, coluna] = valor
            self.registro.anotar("limpeza", f"{coluna}: {int(vazios.sum())} vazios preenchidos com a "
                                            f"{estatistica} da planilha ({valor:.2f}){quais}")
        return df

    @staticmethod
    def _calcular_estatistica(serie, estatistica):
        if estatistica == "mediana":
            return serie.median()
        if estatistica == "media":
            return serie.mean()
        if estatistica == "moda":
            return serie.mode().iloc[0]
        raise ValueError(f"Estatística desconhecida em IMPUTACAO_POR_COLUNA: {estatistica!r}")

    def _remover_linhas_duplicadas(self, df):
        """Remove linhas repetidas dentro da planilha (compara todas as colunas, menos o id)."""
        colunas_comparadas = [coluna for coluna in colunas_de_dados(df) if coluna != self.config.COLUNA_ID]
        duplicadas = df.duplicated(subset=colunas_comparadas)
        if duplicadas.any():
            df = df[~duplicadas]
            self.registro.anotar("limpeza", f"{int(duplicadas.sum())} linhas duplicadas dentro da planilha removidas")
        return df

    def _avisar_ids_repetidos(self, df):
        coluna_id = self.config.COLUNA_ID
        if coluna_id in df and df[coluna_id].duplicated().any():
            self.registro.anotar("atenção", f"{coluna_id}: {int(df[coluna_id].duplicated().sum())} ids "
                                            "repetidos com dados diferentes")


class IndicadoresDeQualidade:
    """Indicadores de qualidade de uma planilha, para comparar antes x depois da limpeza."""

    PROPORCAO_PARA_PARECER_NUMERICA = 0.5  # metade dos valores preenchidos vira número

    def __init__(self, config):
        self.config = config
        self.nulos_disfarcados = {texto.lower() for texto in config.NULOS_DISFARCADOS}

    def calcular(self, df):
        coluna_id = self.config.COLUNA_ID
        sem_origem_e_id = df.drop(columns=[coluna for coluna in [COLUNA_ORIGEM, coluna_id] if coluna in df])
        return {
            "linhas": len(df),
            "valores ausentes": int(df.isna().sum().sum()),
            "vazios disfarçados (ex.: 'não informado')": self._contar_nulos_disfarcados(df),
            "linhas duplicadas": int(linhas_duplicadas(sem_origem_e_id).sum()),
            "ids repetidos": int(df[coluna_id].duplicated().sum()) if coluna_id in df else 0,
            "colunas numéricas guardadas como texto": self._contar_colunas_numericas_como_texto(df),
        }

    def comparar(self, antes, depois):
        """Tabela com os indicadores antes e depois da limpeza."""
        indicadores_antes, indicadores_depois = self.calcular(antes), self.calcular(depois)
        return pd.DataFrame({"indicador": list(indicadores_antes),
                             "antes": list(indicadores_antes.values()),
                             "depois": list(indicadores_depois.values())})

    def _contar_nulos_disfarcados(self, df):
        return sum(int(df[coluna].astype("string").str.strip().str.lower().isin(self.nulos_disfarcados).sum())
                   for coluna in colunas_de_texto(df))

    def _contar_colunas_numericas_como_texto(self, df):
        return sum(1 for coluna in colunas_de_texto(df)
                   if coluna != self.config.COLUNA_ID and self._parece_numerica(df[coluna]))

    def _parece_numerica(self, serie):
        preenchidos = serie.notna().sum()
        convertidos = converter_texto_para_numero(serie).notna().sum()
        return preenchidos > 0 and convertidos / preenchidos >= self.PROPORCAO_PARA_PARECER_NUMERICA


# ------------------------------------------------------------------ etapa 3
class UniaoDasPlanilhas:
    """Etapa 3: empilha as planilhas limpas, cria a coluna 'origem' e corrige só o que depende de tudo junto.

    Uso:  uniao_das_planilhas = UniaoDasPlanilhas(config)
          uniao = uniao_das_planilhas.executar(planilhas_limpas)
          uniao_das_planilhas.diferencas       # colunas ou tipos diferentes entre as planilhas
          uniao_das_planilhas.registro.acoes   # o que foi corrigido
    """

    SIMILARIDADE_MINIMA = 0.8  # trava do skrub deduplicate: só junta textos pelo menos 80% parecidos

    def __init__(self, config):
        self.config = config
        self.registro = RegistroDeAcoes("(todas)")
        self.diferencas = []

    def executar(self, planilhas):
        """planilhas: {nome: tabela limpa}. Devolve a união corrigida."""
        self.registro = RegistroDeAcoes("(todas)")  # começa do zero a cada execução
        self.diferencas = self._diferencas_de_colunas(planilhas) + self._diferencas_de_tipos(planilhas)
        passos_em_ordem = [
            self._padronizar_tipos_e_remover_colunas_constantes,
            self._unificar_variacoes_de_escrita,
            self._corrigir_erros_de_digitacao,
            self._remover_duplicadas_entre_planilhas,  # por último: depois de padronizar, ficam iguais
        ]
        df = self._empilhar(planilhas)
        for passo in passos_em_ordem:
            df = passo(df)
        self._avisar_ids_repetidos_entre_planilhas(df)
        if not self.registro.acoes:
            self.registro.anotar("união", "nada a corrigir: as planilhas já chegaram limpas e compatíveis")
        return df

    @staticmethod
    def _empilhar(planilhas):
        com_origem = []
        for nome, planilha in planilhas.items():
            planilha = planilha.copy()
            planilha.insert(0, COLUNA_ORIGEM, nome)
            com_origem.append(planilha)
        return pd.concat(com_origem, ignore_index=True)

    @staticmethod
    def _diferencas_de_colunas(planilhas):
        """Colunas que faltam ou sobram em cada planilha, comparando com a primeira."""
        nome_da_referencia, referencia = next(iter(planilhas.items()))
        diferencas = []
        for nome, planilha in list(planilhas.items())[1:]:
            for coluna in sorted(set(colunas_de_dados(referencia)) - set(planilha.columns)):
                diferencas.append(f"{nome}: não tem a coluna '{coluna}' (configure RENOMEAR_COLUNAS em config.py)")
            for coluna in sorted(set(colunas_de_dados(planilha)) - set(referencia.columns)):
                diferencas.append(f"{nome}: tem a coluna '{coluna}', que não existe em {nome_da_referencia}")
        return diferencas

    @staticmethod
    def _diferencas_de_tipos(planilhas):
        """Colunas que são número numa planilha e texto em outra, por exemplo."""
        referencia = next(iter(planilhas.values()))
        diferencas = []
        for coluna in colunas_de_dados(referencia):
            tipos = {nome: tipo_simplificado(planilha[coluna]) for nome, planilha in planilhas.items()
                     if coluna in planilha and planilha[coluna].notna().any()}
            if len(set(tipos.values())) > 1:
                diferencas.append(f"Coluna '{coluna}' com tipos diferentes: "
                                  + ", ".join(f"{nome}: {tipo}" for nome, tipo in tipos.items()))
        return diferencas

    def _padronizar_tipos_e_remover_colunas_constantes(self, df):
        """skrub Cleaner: acerta tipos e datas e remove colunas constantes ou vazias."""
        controle = df[[coluna for coluna in COLUNAS_DE_CONTROLE if coluna in df]]
        dados = df[colunas_de_dados(df)]
        colunas_antes = set(dados.columns)
        dados = Cleaner(drop_if_constant=True).fit_transform(dados)
        removidas = colunas_antes - set(dados.columns)
        if removidas:
            self.registro.anotar("união", "colunas constantes ou vazias removidas (skrub): "
                                 + ", ".join(sorted(removidas)))
        return pd.concat([controle.reset_index(drop=True), dados.reset_index(drop=True)], axis=1)

    def _unificar_variacoes_de_escrita(self, df):
        """Mesma categoria escrita diferente ('SP', 'sp ', 'Sp') -> a forma mais frequente."""
        for coluna in colunas_de_texto(df):
            contagem = df[coluna].value_counts()
            if contagem.empty or len(contagem) > LIMITE_DE_CATEGORIAS:
                continue
            troca = {}
            for _, variacoes in contagem.groupby(contagem.index.map(normalizar_texto)):
                if len(variacoes) > 1:
                    forma_mais_comum = variacoes.idxmax()
                    troca.update({variacao: forma_mais_comum for variacao in variacoes.index
                                  if variacao != forma_mais_comum})
            if troca:
                df[coluna] = df[coluna].replace(troca)
                self.registro.anotar("união", f"{coluna}: {len(troca)} variações de escrita unificadas "
                                              f"({RegistroDeAcoes.exemplos_de_troca(troca)})")
        return df

    def _corrigir_erros_de_digitacao(self, df):
        """skrub deduplicate, com trava: só aceita a sugestão se os textos forem parecidos."""
        for coluna in self.config.COLUNAS_PARA_CORRIGIR_DIGITACAO:
            if coluna not in df:
                continue
            troca, recusadas = self._trocas_sugeridas_pelo_skrub(df[coluna])
            if troca:
                corrigidos = int(df[coluna].isin(list(troca)).sum())
                df[coluna] = df[coluna].replace(troca)
                self.registro.anotar("união", f"{coluna}: {corrigidos} erros de digitação corrigidos (skrub) "
                                              f"({RegistroDeAcoes.exemplos_de_troca(troca)})")
            if recusadas:
                self.registro.anotar("união", f"{coluna}: {recusadas} sugestões do skrub recusadas "
                                              "(palavras diferentes)")
        return df

    def _trocas_sugeridas_pelo_skrub(self, serie):
        """Devolve ({texto errado: texto certo}, quantidade de sugestões recusadas pela trava)."""
        valores_unicos = pd.Series(serie.dropna().astype(str).unique())
        if len(valores_unicos) < 2:
            return {}, 0
        grupo_sugerido = pd.Series(deduplicate(valores_unicos).values, index=valores_unicos.values)
        contagem = serie.astype(str).value_counts()
        troca, recusadas = {}, 0
        for _, grupo in grupo_sugerido.groupby(grupo_sugerido):
            textos = list(grupo.index)
            forma_mais_comum = max(textos, key=lambda texto: contagem.get(texto, 0))
            for texto in textos:
                if texto == forma_mais_comum:
                    continue
                if self._similaridade(texto, forma_mais_comum) >= self.SIMILARIDADE_MINIMA:
                    troca[texto] = forma_mais_comum
                else:
                    recusadas += 1
        return troca, recusadas

    @staticmethod
    def _similaridade(texto, outro_texto):
        """De 0 (nada parecidos) a 1 (iguais), ignorando maiúsculas, acentos e espaços."""
        return SequenceMatcher(None, normalizar_texto(texto), normalizar_texto(outro_texto)).ratio()

    def _remover_duplicadas_entre_planilhas(self, df):
        colunas = colunas_de_dados(df)
        duplicadas = df.duplicated(subset=colunas)
        if duplicadas.any():
            planilha_da_primeira = df.groupby(colunas, dropna=False, sort=False)[COLUNA_ORIGEM].transform("first")
            entre_planilhas = int((duplicadas & (df[COLUNA_ORIGEM] != planilha_da_primeira)).sum())
            df = df[~duplicadas].reset_index(drop=True)
            self.registro.anotar("união", f"{int(duplicadas.sum())} linhas duplicadas removidas "
                                          f"({entre_planilhas} entre planilhas diferentes)")
        return df

    def _avisar_ids_repetidos_entre_planilhas(self, df):
        coluna_id = self.config.COLUNA_ID
        if coluna_id in df and df[coluna_id].duplicated().any():
            self.registro.anotar("união", f"ATENÇÃO: {int(df[coluna_id].duplicated().sum())} {coluna_id} "
                                          "repetidos entre planilhas")


# ------------------------------------------------------------------ gravação dos resultados
class ArquivosDaPreparacao:
    """Salva os resultados da preparação nas pastas do projeto (ver PastasDoProjeto)."""

    LIMITE_DO_NOME_DA_ABA = 31  # o Excel aceita até 31 caracteres no nome de uma aba

    def __init__(self, pastas):
        self.pastas = pastas

    def salvar_conferencia_da_copia(self, conferencia):
        # fora da pasta da cópia, para ela ter só as planilhas (o app de EDA lê a pasta inteira)
        self.pastas.relatorios.mkdir(exist_ok=True)
        conferencia.to_csv(self.pastas.conferencia_da_copia, index=False)

    def salvar_planilha_limpa(self, nome_da_planilha, planilha):
        self.pastas.planilhas_limpas.mkdir(parents=True, exist_ok=True)
        planilha.to_csv(self.pastas.planilha_limpa(nome_da_planilha), index=False)

    def salvar_uniao(self, uniao, planilhas_limpas):
        """A união em CSV e em Excel (no Excel, também uma aba para cada planilha limpa)."""
        self.pastas.pasta_da_uniao.mkdir(parents=True, exist_ok=True)
        uniao.to_csv(self.pastas.uniao_csv, index=False)
        with pd.ExcelWriter(self.pastas.uniao_excel) as arquivo_excel:
            for nome, planilha in planilhas_limpas.items():
                planilha.to_excel(arquivo_excel, sheet_name=nome[:self.LIMITE_DO_NOME_DA_ABA], index=False)
            uniao.to_excel(arquivo_excel, sheet_name="Uniao", index=False)

    def salvar_registro(self, acoes):
        """Tudo o que a preparação fez (limpeza + união), para a documentação."""
        self.pastas.relatorios.mkdir(exist_ok=True)
        pd.DataFrame(acoes).to_csv(self.pastas.registro_da_preparacao, index=False)
