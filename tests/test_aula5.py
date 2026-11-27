# tests/test_aula5.py
"""Aula 5 — RAG: quebrar, indexar, buscar e usar como ferramenta."""
# Os documentos de teste ficam na pasta documentos/; a fixture falso usa um MongoDB em memória.
from app.rag.busca import buscar
from app.rag.indexador import ler_documentos, quebrar_em_chunks
from app.tools.conhecimento import buscar_documentos


# Nenhum pedaço passa muito do tamanho (a sobreposição pode somar um pouco).
def test_chunks_respeitam_tamanho():
    texto = "\n\n".join(f"Parágrafo {i} " + "x" * 200 for i in range(10))
    chunks = quebrar_em_chunks(texto, tamanho=700)
    assert len(chunks) > 1 and all(len(c) <= 920 for c in chunks)


# Os arquivos da pasta documentos/ são lidos.
def test_le_documentos():
    assert "tesouro_direto.md" in {d["fonte"] for d in ler_documentos()}


# Cada pergunta encontra, em 1º lugar, o documento que trata do assunto.
def test_busca_acha_o_documento_certo(falso):
    assert buscar("tabela regressiva imposto de renda 22,5%")[0]["fonte"] == "impostos_investimentos.md"
    assert buscar("P/VP dividend yield fundo imobiliário")[0]["fonte"] == "fundos_imobiliarios.md"
    assert buscar("FGC 250 mil garantia")[0]["fonte"] == "renda_fixa_cdi.md"


# A ferramenta do agente devolve trechos com a fonte.
def test_ferramenta_de_busca(falso):
    trechos = buscar_documentos.invoke({"consulta": "marcação a mercado"})
    assert trechos[0]["fonte"] == "tesouro_direto.md"