"""
EDA (análise exploratória de dados) automática de uma tabela.

AnaliseExploratoria(df).executar(pasta) calcula a EDA e salva 2 arquivos na pasta:
  fg-data-profiling-relatorio-detalhado.html -> estatísticas e gráficos de cada coluna (fg-data-profiling)
  descricao_dos_dados.txt                    -> frases sobre os dados, prontas para a documentação

Classes:
  AnaliseExploratoria    -> faz a EDA completa, usando as três abaixo
  VerificadorDeQualidade -> gera os alertas de qualidade
  DescritorDeDados       -> escreve as frases da descrição dos dados

Uso no terminal (sempre pelo pixi):
    pixi run python src/eda.py dados.csv
    pixi run python src/eda.py planilha1.xlsx planilha2.xlsx     (junta as planilhas antes)
    pixi run python src/eda.py dados/etapa0_copia_dos_originais   (uma pasta = todas as planilhas dela)
    pixi run python src/eda.py dados.csv --saida minha_pasta      (onde salvar; padrão: pasta atual)
"""
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

# Os filtros vêm antes dos imports abaixo para esconder os avisos que as bibliotecas dão ao serem importadas.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="data_profiling")
warnings.filterwarnings("ignore", message=".*deprecated.*")

import pandas as pd

from config import NULOS_DISFARCADOS
from leitura_de_planilhas import LeitorDePlanilhas
from tabelas import (COLUNA_ORIGEM, LIMITE_DE_CATEGORIAS, PROPORCAO_MINIMA_NUMERICA,
                     contar_duplicadas_entre_planilhas, linhas_duplicadas, normalizar_texto)

if not pd.__version__.startswith("3."):
    print(f"Aviso: pandas {pd.__version__} instalado; o projeto usa o 3.1.0. "
          "Rode pelo pixi (pixi run ...) para usar o ambiente certo.")


class Alerta(NamedTuple):
    """Um alerta de qualidade dos dados: uma linha da tabela de alertas."""
    coluna: str
    tipo_de_alerta: str
    detalhe: str
    gravidade: str  # "alta", "media" ou "baixa"
    acao_sugerida: str


class ResultadoDaEda(NamedTuple):
    """O que AnaliseExploratoria.executar() devolve (além dos arquivos que salva)."""
    alertas: pd.DataFrame
    estatisticas_descritivas: pd.DataFrame
    descricao_dos_dados: pd.DataFrame  # colunas: tema, frase
    caminho_do_relatorio_detalhado: Path


# ------------------------------------------------------------------ alertas de qualidade
class VerificadorDeQualidade:
    """Gera a tabela de alertas de qualidade: os do fg-data-profiling e os das checagens extras."""

    ORDEM_DE_GRAVIDADE = {"alta": 0, "media": 1, "baixa": 2}

    # tipo de alerta do fg-data-profiling -> (gravidade, ação sugerida)
    GRAVIDADE_E_ACAO_POR_ALERTA = {
        "CONSTANT": ("alta", "Remover a coluna (skrub DropUninformative)"),
        "EMPTY": ("alta", "Coluna ou dataset vazio: remover ou verificar a origem"),
        "INFINITE": ("alta", "Trocar infinitos por nulo: df[col].replace([np.inf, -np.inf], None)"),
        "NEAR_DUPLICATES": ("alta", "Linhas quase iguais: verificar se são o mesmo registro"),
        "MISSING": ("media", "Imputar no pipeline (SimpleImputer) ou remover se > 50%"),
        "HIGH_CARDINALITY": ("media", "Verificar se é ID; se for categoria, usar skrub deduplicate/encoders"),
        "UNSUPPORTED": ("media", "Tipo não reconhecido: converter (skrub Cleaner)"),
        "TYPE_DATE": ("media", "Data guardada como texto: converter (skrub to_datetime)"),
        "DIRTY_CATEGORY": ("media", "Categoria com variações de escrita: skrub deduplicate"),
        "REJECTED": ("media", "Coluna rejeitada pelo profiling: verificar o tipo"),
        "HIGH_CORRELATION": ("baixa", "Avaliar remoção de uma das colunas (skrub DropSimilar)"),
        "UNIQUE": ("baixa", "Provável ID: não usar como feature"),
        "IMBALANCE": ("baixa", "Categoria dominante: avaliar se a coluna é útil"),
        "SKEWED": ("baixa", "Assimetria: avaliar escala robusta (skrub SquashingScaler)"),
        "ZEROS": ("baixa", "Confirmar se zero é valor real ou nulo disfarçado"),
        "UNIFORM": ("baixa", "Distribuição uniforme: pode ser ID ou valor gerado"),
        "CONSTANT_LENGTH": ("baixa", "Texto de tamanho fixo: provável código (CPF, CEP, ID)"),
        "NON_STATIONARY": ("baixa", "Série temporal não estacionária: avaliar tendência"),
        "SEASONAL": ("baixa", "Série temporal com sazonalidade: considerar na modelagem"),
    }

    def __init__(self, nulos_disfarcados=NULOS_DISFARCADOS):
        self.nulos_disfarcados = {texto.lower() for texto in nulos_disfarcados}

    def verificar(self, df, relatorio_detalhado):
        """Tabela de alertas, da gravidade alta para a baixa."""
        alertas = self._alertas_do_profiling(relatorio_detalhado) + self._alertas_extras(df)
        return self._organizar(alertas)

    def verificar_sem_profiling(self, df):
        """Checagens portáveis reais, sem depender do pacote nativo de profiling."""
        alertas = self._alertas_basicos(df) + self._alertas_extras(df)
        return self._organizar(alertas)

    def _organizar(self, alertas):
        tabela = pd.DataFrame(alertas, columns=Alerta._fields)
        tabela = tabela.sort_values(["gravidade", "coluna"], key=self._chave_de_ordenacao)
        return tabela.reset_index(drop=True)

    @staticmethod
    def _alertas_basicos(df):
        alertas = []
        for coluna in df.columns:
            serie = df[coluna]
            ausentes = int(serie.isna().sum())
            distintos = int(serie.nunique())
            if serie.empty or ausentes == len(serie):
                alertas.append(Alerta(coluna, "EMPTY", "Coluna sem valores observados.", "alta",
                                      "Verificar a origem dos dados antes de usar a coluna."))
            elif distintos == 1:
                alertas.append(Alerta(coluna, "CONSTANT", "A coluna tem um único valor observado.", "alta",
                                      "Avaliar se a coluna acrescenta informação à análise."))
            elif distintos == len(serie):
                alertas.append(Alerta(coluna, "UNIQUE", "Todos os valores são distintos.", "baixa",
                                      "Verificar se a coluna é um identificador."))
            if ausentes and ausentes < len(serie):
                alertas.append(Alerta(coluna, "MISSING", f"{ausentes} valores ausentes.", "media",
                                      "Investigar a origem e tratar os ausentes no pipeline de modelagem."))
            if pd.api.types.is_numeric_dtype(serie):
                infinitos = int(serie.isin([float("inf"), float("-inf")]).sum())
                if infinitos:
                    alertas.append(Alerta(coluna, "INFINITE", f"{infinitos} valores infinitos.", "alta",
                                          "Investigar o cálculo que produziu os valores infinitos."))
        return alertas

    def _chave_de_ordenacao(self, serie):
        """Gravidade de alta para baixa; as outras colunas em ordem alfabética."""
        return serie.map(self.ORDEM_DE_GRAVIDADE) if serie.name == "gravidade" else serie

    def _alertas_do_profiling(self, relatorio_detalhado):
        alertas = []
        for alerta in relatorio_detalhado.get_description().alerts:
            tipo = alerta.alert_type.name
            if tipo == "DUPLICATES":  # já verificado em _alerta_de_linhas_duplicadas, evita repetição
                continue
            gravidade, acao_sugerida = self.GRAVIDADE_E_ACAO_POR_ALERTA.get(tipo, ("baixa", "Verificar"))
            alertas.append(Alerta(alerta.column_name or "(dataset)", tipo, str(alerta), gravidade, acao_sugerida))
        return alertas

    def _alertas_extras(self, df):
        """Alertas que o fg-data-profiling não detecta bem: duplicadas e problemas em colunas de texto."""
        alertas = self._alerta_de_linhas_duplicadas(df)
        for coluna in df.select_dtypes(include=["object", "string"]).columns:
            valores = df[coluna].dropna().astype(str)
            if valores.empty:
                continue
            alertas_da_coluna = [
                self._alerta_de_nulos_disfarcados(coluna, valores),
                self._alerta_de_espacos_extras(coluna, valores),
                self._alerta_de_numero_como_texto(coluna, valores),
                self._alerta_de_categoria_escrita_diferente(coluna, valores),
            ]
            alertas += [alerta for alerta in alertas_da_coluna if alerta is not None]
        return alertas

    @staticmethod
    def _alerta_de_linhas_duplicadas(df):
        quantidade = int(linhas_duplicadas(df).sum())
        if not quantidade:
            return []
        entre_planilhas = contar_duplicadas_entre_planilhas(df)
        detalhe = f"{quantidade} linhas repetidas" + (f" ({entre_planilhas} entre planilhas diferentes)"
                                                      if entre_planilhas else "")
        return [Alerta("(dataset)", "Linhas duplicadas", detalhe, "alta", "df = df.drop_duplicates()")]

    def _alerta_de_nulos_disfarcados(self, coluna, valores):
        quantidade = valores.str.strip().str.lower().isin(self.nulos_disfarcados).sum()
        if not quantidade:
            return None
        return Alerta(coluna, "Nulos disfarçados", f"{quantidade} valores como 'N/A', '-', '' etc.",
                      "alta", "skrub Cleaner (converte para nulo)")

    @staticmethod
    def _alerta_de_espacos_extras(coluna, valores):
        quantidade = (valores != valores.str.strip()).sum()
        if not quantidade:
            return None
        return Alerta(coluna, "Espaços extras", f"{quantidade} valores com espaço no início/fim",
                      "media", "df[col] = df[col].str.strip()")

    @staticmethod
    def _alerta_de_numero_como_texto(coluna, valores):
        proporcao_numerica = pd.to_numeric(valores.str.replace(",", ".", regex=False), errors="coerce").notna().mean()
        if PROPORCAO_MINIMA_NUMERICA <= proporcao_numerica < 1:
            return Alerta(coluna, "Número como texto (com sujeira)",
                          f"{proporcao_numerica:.0%} dos valores são numéricos",
                          "alta", "Ver os não numéricos e converter (skrub ToFloat)")
        if proporcao_numerica == 1:
            return Alerta(coluna, "Número como texto", "todos os valores são numéricos",
                          "media", "skrub Cleaner / ToFloat")
        return None

    @staticmethod
    def _alerta_de_categoria_escrita_diferente(coluna, valores):
        """Mesma categoria escrita de jeitos diferentes ('SP', 'sp', 'Sp ')."""
        valores_unicos = pd.Series(valores.unique())
        if not 1 < len(valores_unicos) <= LIMITE_DE_CATEGORIAS or len(valores_unicos) == len(valores):
            return None
        formas_por_texto_normalizado = valores_unicos.groupby(valores_unicos.map(normalizar_texto)).size()
        com_variacoes = formas_por_texto_normalizado[formas_por_texto_normalizado > 1]
        if com_variacoes.empty:
            return None
        exemplos = ", ".join(list(com_variacoes.index[:3]))
        return Alerta(coluna, "Mesma categoria escrita diferente",
                      f"{len(com_variacoes)} grupos (maiúsc./acento/espaço). Ex.: {exemplos}",
                      "alta", "Normalizar texto ou skrub deduplicate")


# ------------------------------------------------------------------ descrição dos dados
class DescritorDeDados:
    """Escreve frases que descrevem os dados, para a documentação (colunas: tema, frase).
    Não repete o que já está nos alertas de qualidade (ausentes, duplicadas, colunas constantes)."""

    MULTIPLICADOR_IQR = 1.5  # valor atípico (outlier): abaixo de Q1 - 1,5·IQR ou acima de Q3 + 1,5·IQR

    def descrever(self, df):
        frases = [("Geral", self._frase_geral(df))]
        if COLUNA_ORIGEM in df:
            frases.append(("Origem", self._frase_por_origem(df)))
        colunas_numericas = df.select_dtypes(include="number").columns
        colunas_de_texto = df.select_dtypes(include=["object", "string"]).columns
        for coluna in df.columns:
            valores = df[coluna].dropna()
            if coluna == COLUNA_ORIGEM or valores.nunique() <= 1:  # vazia ou constante: já aparece nos alertas
                continue
            if coluna in colunas_numericas:
                frases.append(("Coluna", self._frase_coluna_numerica(coluna, valores)))
            elif coluna in colunas_de_texto:
                frases.append(("Coluna", self._frase_coluna_de_texto(coluna, valores)))
        return pd.DataFrame(frases, columns=["tema", "frase"])

    @staticmethod
    def formatar_numero(numero, casas_decimais=1):
        """Número no formato brasileiro: 1234.5 -> '1.234,5'."""
        texto = f"{numero:,.{casas_decimais}f}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")

    def _frase_geral(self, df):
        linhas, colunas = df.shape
        numericas = len(df.select_dtypes(include="number").columns)
        de_texto = len(df.select_dtypes(include=["object", "string"]).columns)
        return (f"O dataset tem {self.formatar_numero(linhas, 0)} linhas e {colunas} colunas "
                f"({numericas} numéricas, {de_texto} de texto, {colunas - numericas - de_texto} de outros tipos).")

    def _frase_por_origem(self, df):
        linhas_por_planilha = df[COLUNA_ORIGEM].value_counts(sort=False)
        return ("Registros por planilha de origem: "
                + ", ".join(f"{planilha}: {self.formatar_numero(quantidade, 0)}"
                            for planilha, quantidade in linhas_por_planilha.items()) + ".")

    def _frase_coluna_numerica(self, coluna, valores):
        q1, q3 = valores.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inferior, limite_superior = q1 - self.MULTIPLICADOR_IQR * iqr, q3 + self.MULTIPLICADOR_IQR * iqr
        atipicos = int(((valores < limite_inferior) | (valores > limite_superior)).sum())
        negativos = int((valores < 0).sum())
        frase = (f"A coluna {coluna} varia de {self.formatar_numero(valores.min(), 2)} a "
                 f"{self.formatar_numero(valores.max(), 2)}, com média {self.formatar_numero(valores.mean(), 2)} "
                 f"e mediana {self.formatar_numero(valores.median(), 2)}.")
        if atipicos:
            frase += f" Tem {self.formatar_numero(atipicos, 0)} valores atípicos (outliers pelo critério IQR)."
        if negativos:
            frase += f" Tem {self.formatar_numero(negativos, 0)} valores negativos."
        if valores.nunique() == len(valores):
            frase += " Todos os valores são diferentes (pode ser um ID)."
        return frase

    def _frase_coluna_de_texto(self, coluna, valores):
        contagem = valores.value_counts()
        mais_comum, vezes = contagem.index[0], contagem.iloc[0]
        frase = (f"A coluna {coluna} tem {self.formatar_numero(valores.nunique(), 0)} valores distintos; "
                 f"o mais comum é \"{mais_comum}\" ({self.formatar_numero(vezes, 0)} vezes, "
                 f"{self.formatar_numero(vezes / len(valores) * 100)}%).")
        if valores.nunique() == len(valores):
            frase += " Todos os valores são diferentes (pode ser um ID)."
        return frase


# ------------------------------------------------------------------ EDA completa
class AnaliseExploratoria:
    """EDA de uma tabela: calcula tudo e salva os 2 arquivos na pasta indicada.

    Uso:  resultado = AnaliseExploratoria(df).executar("relatorios/minha_eda")
          resultado.alertas, resultado.estatisticas_descritivas, resultado.descricao_dos_dados
    """

    ARQUIVO_RELATORIO_DETALHADO = "fg-data-profiling-relatorio-detalhado.html"
    ARQUIVO_DESCRICAO = "descricao_dos_dados.txt"

    def __init__(self, df, verificador=None, descritor=None):
        self.df = df
        self.verificador = verificador or VerificadorDeQualidade()
        self.descritor = descritor or DescritorDeDados()

    def executar(self, pasta_dos_relatorios="."):
        # A biblioteca tem dependências nativas. Importá-la só na execução local
        # permite reutilizar estatísticas e alertas portáveis no navegador.
        from data_profiling import ProfileReport  # pacote fg-data-profiling

        pasta = Path(pasta_dos_relatorios)
        pasta.mkdir(parents=True, exist_ok=True)
        caminho_do_relatorio_detalhado = pasta / self.ARQUIVO_RELATORIO_DETALHADO

        relatorio_detalhado = ProfileReport(self.df, title="Relatório estatístico detalhado (fg-data-profiling)",
                                            progress_bar=False)
        relatorio_detalhado.to_file(caminho_do_relatorio_detalhado)
        resultado = ResultadoDaEda(
            alertas=self.verificador.verificar(self.df, relatorio_detalhado),
            estatisticas_descritivas=self.estatisticas_descritivas(),
            descricao_dos_dados=self.descritor.descrever(self.df),
            caminho_do_relatorio_detalhado=caminho_do_relatorio_detalhado,
        )
        self._salvar_descricao_txt(pasta / self.ARQUIVO_DESCRICAO, resultado.descricao_dos_dados)
        return resultado

    def estatisticas_descritivas(self):
        """Uma linha por coluna: tipo, nulos, valores únicos e todas as medidas do describe() do pandas."""
        medidas = self.df.describe(include="all").T
        for medida in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if medida in medidas:
                medidas[medida] = pd.to_numeric(medidas[medida], errors="coerce").round(2)
        informacoes = pd.DataFrame({
            "tipo": self.df.dtypes.astype(str),
            "nulos": self.df.isna().sum(),
            "% nulos": (self.df.isna().mean() * 100).round(1),
            "unicos": self.df.nunique(),
        })
        tabela = informacoes.join(medidas)
        tabela.index.name = "coluna"
        return tabela.reset_index()

    @staticmethod
    def _salvar_descricao_txt(caminho, descricao_dos_dados):
        caminho.write_text("\n".join(f"- {frase}" for frase in descricao_dos_dados.frase), encoding="utf-8")


# ------------------------------------------------------------------ uso no terminal
def main(argumentos):
    """Faz a EDA dos arquivos passados no terminal (ver o início deste arquivo).
    Com várias planilhas, a junção delas também é salva na pasta dos relatórios."""
    pasta_dos_relatorios = Path(".")
    if "--saida" in argumentos:
        posicao = argumentos.index("--saida")
        pasta_dos_relatorios = Path(argumentos[posicao + 1]).resolve()
        del argumentos[posicao:posicao + 2]

    leitor = LeitorDePlanilhas()
    planilhas = [str(Path(caminho).resolve()) for caminho in leitor.listar(argumentos)]
    if len(planilhas) > 1:
        juncao = leitor.juntar([(planilha, planilha) for planilha in planilhas])
        print(juncao.resumo)
        for aviso in juncao.avisos:
            print("  ATENÇÃO:", aviso)
        pasta_dos_relatorios.mkdir(parents=True, exist_ok=True)
        caminho_da_juncao = pasta_dos_relatorios / "uniao_das_planilhas_exploradas.csv"
        juncao.tabela.to_csv(caminho_da_juncao, index=False)
        print(f"Salvo em {caminho_da_juncao}\n")
        df = juncao.tabela
    else:
        df = leitor.ler(planilhas[0], planilhas[0])

    alertas = AnaliseExploratoria(df).executar(pasta_dos_relatorios).alertas
    print(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas | "
          f"linhas duplicadas: {int(linhas_duplicadas(df).sum())}\n")
    for alerta in alertas.itertuples():
        print(f"[{alerta.gravidade.upper():5}] {alerta.coluna}: {alerta.tipo_de_alerta}")
        print(f"         {alerta.detalhe}")
        print(f"         -> {alerta.acao_sugerida}\n")
    print(f"{len(alertas)} alertas | alta: {(alertas.gravidade == 'alta').sum()} | relatórios salvos em "
          f"{pasta_dos_relatorios.resolve()} ({AnaliseExploratoria.ARQUIVO_RELATORIO_DETALHADO}, "
          f"{AnaliseExploratoria.ARQUIVO_DESCRICAO})")


if __name__ == "__main__":
    main(sys.argv[1:])
