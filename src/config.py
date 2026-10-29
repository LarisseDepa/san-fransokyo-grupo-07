"""
Configuração da preparação dos dados e da EDA (notebooks/claro_guia_etapas_0_a_8.py).
Tudo o que muda o resultado fica aqui, em código, para o processo ser reproduzível.
No fim do arquivo ficam o nome do projeto e as descrições mostradas no site.

Etapas: 0. cópia dos originais -> 1. limpeza de cada planilha -> 2. EDA de uma planilha
        -> 3. união das planilhas limpas -> 4. EDA da união
"""
from datetime import date, timedelta, timezone

# Horário de Brasília (sem horário de verão desde 2019), para as datas no notebook e no site.
# O site é montado num servidor do GitHub, que usa outro fuso.
FUSO_HORARIO = timezone(timedelta(hours=-3), "Brasília")

# ---------------------------------------------------------------------------
# ARQUIVOS E COLUNAS
# ---------------------------------------------------------------------------

# Parte comum do nome dos arquivos, retirada para formar o nome curto (coluna "origem")
PREFIXO_DOS_ARQUIVOS = "Projeto_1_Recomendador_de_Planos_"

# Planilha que passa pela EDA sozinha na etapa 2 (nome curto, sem o prefixo)
PLANILHA_ANALISADA_SOZINHA = "Excesso_de_Consumo"

# Coluna que identifica o cliente (não pode repetir)
COLUNA_ID = "id_cliente"

# Coluna que já identifica o grupo de cada planilha (é constante dentro de cada uma)
COLUNA_GRUPO = "perfil_grupo"

# Plano que o cliente tem hoje (usado na contagem de planos por grupo)
COLUNA_PLANO_ATUAL = "plano_atual"

# ---------------------------------------------------------------------------
# 1. LIMPEZA DE CADA PLANILHA (tudo é feito antes da união)
# ---------------------------------------------------------------------------

# Renomear colunas em todas as planilhas: {"nome_errado": "nome_certo"}
RENOMEAR_COLUNAS = {}

# Regras que valem só para uma planilha (chave = nome curto). Opções:
#   "renomear" ({"nome_errado": "nome_certo"}), "separador_decimal" ("," ou "."),
#   "formatos_de_data" ({"coluna": "%d/%m/%Y"}), "trocar_valores" ({"coluna": {"S": "sim"}}),
#   "multiplicar_por" ({"coluna": 1000}, para padronizar a unidade)
REGRAS_POR_PLANILHA = {}

# Formato das colunas de data em todas as planilhas: {"coluna": "%d/%m/%Y"}
FORMATOS_DE_DATA = {}

# Textos que significam "vazio"
NULOS_DISFARCADOS = ["", "-", "--", "?", "na", "n/a", "nan", "null", "none", "nd",
                     "s/i", "sem informacao", "sem informação", "#n/d",
                     "não informado", "nao informado"]

# Unidade escrita junto do número: "24 GB" -> 24
SUFIXOS_DE_UNIDADE = {
    "dados_gb": "GB",
}

# Regras do negócio: função que devolve True para valores VÁLIDOS (inválidos viram vazio)
REGRAS_DE_NEGOCIO = {
    "dados_gb": lambda valores: valores >= 0,
    "minutos": lambda valores: valores >= 0,
    "streaming_horas": lambda valores: valores.between(0, 24 * 31),
    "deslocamento_dias": lambda valores: valores.between(0, 31),
    "risco_cancelamento": lambda valores: valores.isin([0, 1]),
}

# Como preencher os vazios de cada coluna, com a estatística da PRÓPRIA planilha
# ("mediana", "media" ou "moda"). Os grupos têm consumos muito diferentes,
# por isso cada planilha usa a sua mediana.
IMPUTACAO_POR_COLUNA = {
    "dados_gb": "mediana",
}

# Colunas para remover de todas as planilhas
COLUNAS_PARA_REMOVER = []

# ---------------------------------------------------------------------------
# 3. UNIÃO (o que só dá para conferir com tudo junto)
# ---------------------------------------------------------------------------

# Colunas de categoria com erros de digitação, corrigidos pelo skrub deduplicate
COLUNAS_PARA_CORRIGIR_DIGITACAO = []

# ---------------------------------------------------------------------------
# 5. ENRIQUECIMENTO (dados sintéticos; ver src/enriquecimento.py)
# ---------------------------------------------------------------------------

# Data em que a "operadora" olha os dados. Tudo antes dela é histórico; depois, futuro (targets).
DATA_DE_REFERENCIA = date(2026, 10, 28)

# Semente fixa: a mesma semente gera exatamente os mesmos dados sintéticos.
SEMENTE = 2026

# Hipótese da simulação: dados_gb é o consumo TÍPICO DE UM MÊS (a periodicidade não está
# documentada nas planilhas). Ele é usado como o hábito de consumo de cada cliente.
DIAS_DE_UM_MES_TIPICO = 30

# Franquia dos planos ACADÊMICOS da base (nomes fictícios; não são ofertas da Claro):
# {plano: (GB principais, GB de bônus de uso livre, GB de bônus só para apps)}.
# Os GB para apps nunca entram nos GB de uso geral.
FRANQUIAS_DOS_PLANOS_ACADEMICOS = {
    "Essencial": (15, 5, 3),
    "Conecta": (20, 5, 5),
    "Flex": (30, 5, 5),
    "Digital": (35, 10, 5),
    "Streaming": (40, 10, 10),
    "Max": (50, 10, 5),
    "Viagem": (50, 15, 5),
    "Total": (70, 20, 5),
}

# O bônus de uso livre vale por este número de meses a partir do início do plano atual.
MESES_DE_VALIDADE_DO_BONUS = 12

# ---------------------------------------------------------------------------
# 6 e 7. MODELO E RECOMENDAÇÃO (ver src/modelo_de_demanda.py e src/recomendacao.py)
# ---------------------------------------------------------------------------

PROPORCAO_DE_TESTE = 0.2  # 20% dos clientes ficam fora do treino, para a avaliação

# Folga recomendada (10%): uma oferta que atende à demanda estimada com folga menor que esta
# aparece com o aviso "folga pequena" (se o ritmo subir, pode faltar internet)
MARGEM_DE_SEGURANCA = 0.10


# ---------------------------------------------------------------------------
# 8. HISTÓRICO DE OFERTAS (SIMULADO) E MODELO DE RECOMENDAÇÃO
#    (ver src/historico_de_ofertas.py e src/modelo_de_recomendacao.py)
# ---------------------------------------------------------------------------
# As planilhas não têm histórico de ofertas aceitas ou recusadas: ele é SIMULADO por esta regra (o
# "gosto" de cada grupo, aprovado pela pessoa autora). Os modelos da etapa 8 aprendem esta regra; as
# métricas mostram o que eles aprenderam da simulação, não a preferência de clientes reais.

CICLOS_DO_HISTORICO = 6  # penúltimo: validação; último: teste; anteriores: treino inicial
OFERTAS_POR_CICLO = 3  # planos sorteados entre os 7 que o cliente não tem
PESOS_DO_RANDOM_FOREST = (0.0, 0.25, 0.5, 0.75, 1.0)  # inclui os modelos isolados e a média original
REAMOSTRAGENS_DA_RECOMENDACAO = 2000  # bootstrap pareado por cliente, com semente fixa
LIMIAR_DA_CLASSIFICACAO_DA_OFERTA = 0.5  # diagnóstico de aceitação; fixo, sem seleção pelo teste
REPETICOES_DA_IMPORTANCIA_DA_OFERTA = 5  # permutações por variável, com a semente do experimento
PROFUNDIDADE_DA_ARVORE_ILUSTRATIVA = 2  # corte de exibição; não altera o modelo treinado

# Chance de aceitar = logística(utilidade), com
#   utilidade = INTERCEPTO + VALOR_DOS_GB[grupo] × cobertura − SENSIBILIDADE_AO_PRECO[grupo] × diferença de preço
#               + preferência pelo plano
#   cobertura = (GB principais do plano − uso típico) ÷ uso típico, limitada entre −1 e 1
#   diferença de preço = (mensalidade ofertada − mensalidade atual) ÷ mensalidade atual
INTERCEPTO_DA_ACEITACAO = -1.0
VALOR_DOS_GB_POR_GRUPO = {  # quanto cada grupo valoriza ter GB de sobra
    "Economia": 0.5,
    "Equilibrio_Preco_Adequacao": 1.5,
    "Excesso_de_Consumo": 3.0,
    "Mobilidade": 2.5,
}
SENSIBILIDADE_AO_PRECO_POR_GRUPO = {  # quanto um aumento de preço afasta cada grupo
    "Economia": 4.0,
    "Equilibrio_Preco_Adequacao": 2.0,
    "Excesso_de_Consumo": 1.0,
    "Mobilidade": 1.5,
}
# {plano: (hábito, valor de referência, peso)}: preferência = peso × mín(1, hábito ÷ referência)
PREFERENCIA_PELO_PLANO = {
    "Streaming": ("streaming_horas", 20.0, 2.0),  # quem usa muito streaming prefere o Streaming
    "Viagem": ("deslocamento_dias", 20.0, 2.0),  # quem se desloca muito prefere o Viagem
}

# Nota de 0 a 10 (também SIMULADA): em cada oferta, a pessoa "responde" quanto o plano combina com ela.
# nota = 10 × chance de aceitar + variação aleatória (desvio-padrão abaixo), arredondada e limitada a 0–10.
# É a "nota" da filtragem colaborativa usuário a usuário (correlação de Pearson), como no vídeo de referência.
NOTA_MAXIMA = 10
DESVIO_PADRAO_DA_NOTA = 1.5

# ---------------------------------------------------------------------------
# NOME E DESCRIÇÕES (mostrados no site e no notebook; não mudam os dados)
# ---------------------------------------------------------------------------

NOME_DO_PROJETO = "CLARO GUIA"

# O que é cada planilha (chave = nome curto). As médias dos cartões são calculadas dos dados.
DESCRICAO_DAS_PLANILHAS = {
    "Economia": {
        "titulo": "Economia",
        "descricao": "Clientes que usam pouco o plano: bem menos internet e ligações que os outros "
                     "grupos, mas pagam uma mensalidade parecida.",
    },
    "Equilibrio_Preco_Adequacao": {
        "titulo": "Equilíbrio Preço e Adequação",
        "descricao": "Clientes com consumo intermediário, em que o preço e o plano combinam com o uso.",
    },
    "Excesso_de_Consumo": {
        "titulo": "Excesso de Consumo",
        "descricao": "Clientes que usam mais do que a franquia do plano oferece: o maior consumo de "
                     "internet e de ligações, e o maior risco de cancelamento.",
    },
    "Mobilidade": {
        "titulo": "Mobilidade",
        "descricao": "Clientes que se deslocam muito: passam bem mais dias em deslocamento que os outros grupos.",
    },
}

# O que significa cada coluna das planilhas
DESCRICAO_DAS_COLUNAS = {
    "id_cliente": "Número que identifica o cliente. Cada planilha usa uma faixa (10001+, 20001+, 30001+, 40001+).",
    "idade_faixa": "Faixa de idade: 18-25, 26-40, 41-60 ou 60+.",
    "regiao": "Onde o cliente mora: urbana, periférica ou rural.",
    "dados_gb": "Consumo de internet móvel, em GB.",
    "minutos": "Minutos de ligação.",
    "streaming_horas": "Horas de uso de streaming.",
    "deslocamento_dias": "Dias em deslocamento (viagens e trajetos).",
    "plano_atual": "Plano que o cliente tem hoje: Conecta, Digital, Essencial, Flex, Max, Streaming, Total ou Viagem.",
    "mensalidade_atual": "Quanto o cliente paga por mês, em reais.",
    "risco_cancelamento": "Risco de cancelar o plano: 1 = sim, 0 = não.",
    "perfil_grupo": "O grupo do cliente. É o que diferencia as 4 planilhas.",
}
