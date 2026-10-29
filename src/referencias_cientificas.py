"""Referências conferidas em 29/10/2026; a montagem local não faz consultas à internet.

Somente artigos originais em versão final de revista entram nesta bibliografia.
O JIF é um indicador da revista, não uma garantia de qualidade de cada resultado.
"""
from dataclasses import asdict, dataclass
from datetime import date
from html import escape


@dataclass(frozen=True)
class ArtigoCientifico:
    autores: str
    titulo: str
    revista: str
    volume_e_paginas: str
    ano_da_edicao: int
    publicacao_final_online: date
    doi: str
    fonte_da_versao_final: str
    aplicacao_no_projeto: str

    @property
    def endereco(self):
        return f"https://doi.org/{self.doi}"


class BibliografiaDaRecomendacao:
    """Recorte fixo, auditável: 29/10/2021–29/10/2026; revistas com JIF verificado acima de 5.

    Este é o critério operacional desta seleção, não uma definição universal de alto impacto.
    Os artigos sustentam o protocolo de avaliação, não a superioridade da arquitetura RF + Pearson.
    """

    INICIO = date(2021, 10, 29)
    CONFERIDO_EM = date(2026, 10, 29)
    REVISTAS = {
        "ACM Transactions on Information Systems": {
            "jif": 11.2, "ano_do_jif": 2025,
            "fonte": "https://www.acm.org/media-center/2026/july/impact-factors-2026"},
        "IEEE Transactions on Pattern Analysis and Machine Intelligence": {
            "jif": 20.4, "ano_do_jif": 2025,
            "fonte": "https://www.computer.org/publications"},
    }
    ARTIGOS = (
        ArtigoCientifico(
            autores="Wayne Xin Zhao; Zihan Lin; Zhichao Feng; Pengfei Wang; Ji-Rong Wen",
            titulo="A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation Algorithms",
            revista="ACM Transactions on Information Systems",
            volume_e_paginas="41(2), artigo 32, p. 1–41", ano_da_edicao=2023,
            publicacao_final_online=date(2022, 12, 21), doi="10.1145/3545796",
            fonte_da_versao_final="https://wangpengfeii.github.io/publication/zhao-revisiting-2023/"
                                  "Zhao%20%E7%AD%89%20-%202022%20-%20A%20Revisiting%20Study%20of%20Appropriate%20Offline%20Evaluati.pdf",
            aplicacao_no_projeto="Validação separada para selecionar pesos e métricas de ordenação complementares "
                                "à AUC. Seções 4 e 6.2; não equiparar listas observadas ao catálogo completo."),
        ArtigoCientifico(
            autores="Zhu Sun; Hui Fang; Jie Yang; Xinghua Qu; Hongyang Liu; Di Yu; Yew-Soon Ong; Jie Zhang",
            titulo="DaisyRec 2.0: Benchmarking Recommendation for Rigorous Evaluation",
            revista="IEEE Transactions on Pattern Analysis and Machine Intelligence",
            volume_e_paginas="45(7), p. 8206–8226", ano_da_edicao=2023,
            publicacao_final_online=date(2022, 12, 26), doi="10.1109/TPAMI.2022.3231891",
            fonte_da_versao_final="https://research.tudelft.nl/files/154515120/"
                                  "DaisyRec_2.0_Benchmarking_Recommendation_for_Rigorous_Evaluation.pdf",
            aplicacao_no_projeto="Protocolo reprodutível com validação temporal e reajuste antes do teste; "
                                "documentação dos candidatos, métricas e referências de comparação."),
    )

    def como_registro(self):
        artigos = []
        for artigo in self.ARTIGOS:
            if not self.INICIO <= artigo.publicacao_final_online <= self.CONFERIDO_EM:
                raise ValueError("Referência fora do recorte de cinco anos.")
            artigos.append({**asdict(artigo), "publicacao_final_online": artigo.publicacao_final_online.isoformat(),
                            "tipo": "artigo original de pesquisa, revisado por pares, versão final de revista"})
        return {"inicio_do_recorte": self.INICIO.isoformat(), "conferido_em": self.CONFERIDO_EM.isoformat(),
                "criterio_de_impacto": "JIF de 2025 acima de 5, conferido em fontes dos editores",
                "revistas": self.REVISTAS, "artigos": artigos,
                "alcance": "Evidência para a avaliação; não valida a combinação específica nem os pesos deste projeto."}

    def como_markdown(self):
        return "\n".join(
            f"- {artigo.autores}. **{artigo.titulo}**. *{artigo.revista}*, {artigo.volume_e_paginas}, "
            f"{artigo.ano_da_edicao}. [DOI]({artigo.endereco}). "
            f"Publicação final online: {artigo.publicacao_final_online:%d/%m/%Y}."
            for artigo in self.ARTIGOS)

    def como_html(self):
        return '<ul class="lista">' + "".join(
            f'<li>{escape(artigo.autores)}. <a href="{artigo.endereco}">{escape(artigo.titulo)}</a>. '
            f'<i>{escape(artigo.revista)}</i>, {escape(artigo.volume_e_paginas)}, {artigo.ano_da_edicao}. '
            f'Publicação final online: {artigo.publicacao_final_online:%d/%m/%Y}. '
            f'<a href="{escape(artigo.fonte_da_versao_final, quote=True)}">Versão final do artigo</a>.'
            f'<p class="nota">Aplicação no projeto: {escape(artigo.aplicacao_no_projeto)}</p></li>'
            for artigo in self.ARTIGOS) + "</ul>"
