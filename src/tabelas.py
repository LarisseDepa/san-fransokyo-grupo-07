"""
Funções de apoio sobre tabelas (DataFrames do pandas), usadas pelas classes dos outros módulos.
São funções pequenas e sem estado (não guardam nada entre uma chamada e outra);
por isso ficam soltas, e não dentro de uma classe.
"""
import unicodedata

import pandas as pd

COLUNA_ORIGEM = "origem"  # nome da planilha de cada linha, criada na união
COLUNAS_DE_CONTROLE = [COLUNA_ORIGEM]  # rastreio: não contam para duplicadas e não entram no modelo
PROPORCAO_MINIMA_NUMERICA = 0.8  # coluna de texto vira número se pelo menos 80% dos valores forem números
LIMITE_DE_CATEGORIAS = 5000  # acima disso, não procura variações de escrita (ficaria lento)


def formatar_numero_br(numero, casas_decimais=1):
    """Número no formato brasileiro: 1234.5 -> '1.234,5'."""
    texto = f"{numero:,.{casas_decimais}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def normalizar_texto(texto):
    """Minúsculas, sem acentos e sem espaços repetidos: '  São  Paulo ' -> 'sao paulo'."""
    sem_acentos = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sem_acentos.lower().split())


def tipo_simplificado(serie):
    """'número', 'data' ou 'texto'."""
    if pd.api.types.is_numeric_dtype(serie):
        return "número"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "data"
    return "texto"


def colunas_de_dados(df):
    """Todas as colunas, menos as de controle."""
    return [coluna for coluna in df.columns if coluna not in COLUNAS_DE_CONTROLE]


def colunas_de_texto(df):
    """Colunas de texto, menos as de controle."""
    return [coluna for coluna in df.select_dtypes(include=["object", "string"]).columns
            if coluna not in COLUNAS_DE_CONTROLE]


def converter_texto_para_numero(serie, separador_decimal="auto"):
    """Texto -> número. separador_decimal: ',' (1.234,56), '.' (1,234.56) ou 'auto' (valor por valor)."""
    texto = serie.astype("string").str.strip()
    formato_brasileiro = texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    formato_americano = texto.str.replace(",", "", regex=False)
    if separador_decimal == ",":
        texto = formato_brasileiro
    elif separador_decimal == ".":
        texto = formato_americano
    else:
        tem_virgula = texto.str.contains(",", regex=False).fillna(False)
        texto = texto.where(~tem_virgula, formato_brasileiro)
    return pd.to_numeric(texto, errors="coerce")


def linhas_duplicadas(df):
    """Marca as linhas repetidas, ignorando a coluna 'origem':
    o mesmo registro em planilhas diferentes também conta como duplicado."""
    return df.duplicated(subset=colunas_de_dados(df))


def contar_duplicadas_entre_planilhas(df):
    """Quantas linhas duplicadas têm a primeira ocorrência em outra planilha."""
    if COLUNA_ORIGEM not in df:
        return 0
    colunas = colunas_de_dados(df)
    planilha_da_primeira = df.groupby(colunas, dropna=False, sort=False)[COLUNA_ORIGEM].transform("first")
    return int((linhas_duplicadas(df) & (df[COLUNA_ORIGEM] != planilha_da_primeira)).sum())
