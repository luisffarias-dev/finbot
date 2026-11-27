# app/rag/indexador.py
"""Aula 5 — Indexação: documentos -> pedaços (chunks) -> vetores -> MongoDB.

Roda uma vez (ou quando os documentos mudarem):  python -m scripts.indexar
"""
from pathlib import Path

from app.config.settings import settings
from app.rag import vetores
from app.repository import mongo

# Pasta documentos/ na raiz do projeto (parents[2] sobe de app/rag/ até a raiz).
PASTA = Path(__file__).resolve().parents[2] / "documentos"


def colecao():
    # Coleção do MongoDB onde ficam os pedaços (chunks) indexados.
    return mongo.banco()["chunks"]


# Lê todos os arquivos .md, .txt e .pdf da pasta e devolve [{fonte, texto}, ...].
def ler_documentos(pasta: Path = PASTA) -> list[dict]:
    documentos = []
    for arquivo in sorted(pasta.glob("*")):
        if arquivo.suffix in (".md", ".txt"):
            texto = arquivo.read_text(encoding="utf-8")
        elif arquivo.suffix == ".pdf":
            from pypdf import PdfReader
            # O texto tirado de um PDF vem cheio de quebras de linha no meio das frases.
            # " ".join(...split()) junta cada página numa linha só ("or ''" evita erro em página sem texto).
            paginas = [" ".join((pagina.extract_text() or "").split()) for pagina in PdfReader(arquivo).pages]
            # Uma linha em branco entre as páginas: cada página vira pelo menos um parágrafo.
            texto = "\n\n".join(p for p in paginas if p)
        else:
            continue  # outros tipos de arquivo são ignorados
        documentos.append({"fonte": arquivo.name, "texto": texto})
    return documentos


def dividir_paragrafo(paragrafo: str, tamanho: int) -> list[str]:
    """Corta um parágrafo grande demais em partes de até 'tamanho' caracteres, sem cortar palavras.
    Nos .md da aula quase nunca acontece; numa página de PDF, sempre."""
    if len(paragrafo) <= tamanho:
        return [paragrafo]
    partes, atual = [], ""
    for palavra in paragrafo.split():
        # Se a próxima palavra passar do tamanho, fecha a parte atual e começa outra.
        if atual and len(atual) + 1 + len(palavra) > tamanho:
            partes.append(atual)
            atual = palavra
        else:
            atual = f"{atual} {palavra}".strip()
    if atual:
        partes.append(atual)
    return partes


def quebrar_em_chunks(texto: str, tamanho: int = 700, sobreposicao: int = 1) -> list[str]:
    """Junta parágrafos até ~'tamanho' caracteres. O último parágrafo de um pedaço
    se repete no começo do próximo (sobreposição) para não cortar uma ideia ao meio."""
    # Separa o texto em parágrafos (blocos separados por linha em branco);
    # parágrafo maior que 'tamanho' é cortado em partes por dividir_paragrafo.
    paragrafos = [parte for p in texto.split("\n\n") if p.strip()
                  for parte in dividir_paragrafo(p.strip(), tamanho)]
    chunks, atual = [], []
    for paragrafo in paragrafos:
        # Se o próximo parágrafo passar do tamanho, fecha o pedaço atual...
        if atual and len("\n\n".join(atual + [paragrafo])) > tamanho:
            chunks.append("\n\n".join(atual))
            # ...e começa o próximo repetindo o último parágrafo (sobreposição).
            atual = atual[-sobreposicao:] if sobreposicao else []
        atual.append(paragrafo)
    # O que sobrou no final vira o último pedaço.
    if atual:
        chunks.append("\n\n".join(atual))
    return chunks


def indexar(pasta: Path = PASTA) -> int:
    # Um registro por pedaço: de qual arquivo veio, a posição dele e o texto.
    registros = [
        {"fonte": doc["fonte"], "ordem": i, "texto": chunk}
        for doc in ler_documentos(pasta)
        for i, chunk in enumerate(quebrar_em_chunks(doc["texto"]))
    ]
    # No modo embeddings, o vetor de cada pedaço é calculado agora e salvo junto.
    # (No modo tfidf, os vetores são calculados na hora da busca.)
    if settings.rag_modo == "embeddings":
        for registro, vetor in zip(registros, vetores.embeddings([r["texto"] for r in registros])):
            registro["vetor"] = vetor.tolist()

    # Apaga o índice antigo e grava o novo (assim não ficam pedaços repetidos).
    colecao().delete_many({})
    if registros:
        colecao().insert_many(registros)
    return len(registros)  # quantos pedaços foram indexados