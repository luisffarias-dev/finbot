# app/rag/busca.py
"""Aula 5 — Busca (o "R" de RAG): achar os pedaços mais parecidos com a pergunta."""
import numpy as np

from app.config.settings import settings
from app.rag import vetores
from app.rag.indexador import colecao


# Devolve os k pedaços mais parecidos com a consulta, com a nota de semelhança.
def buscar(consulta: str, k: int = 3) -> list[dict]:
    # Carrega todos os pedaços do MongoDB (para poucos documentos, isso é rápido).
    chunks = list(colecao().find({}, {"_id": 0}))
    if not chunks:
        return []

    # Monta a matriz dos pedaços e o vetor da consulta, conforme o modo escolhido.
    if settings.rag_modo == "embeddings":
        matriz = np.array([c["vetor"] for c in chunks])
        vetor = vetores.embeddings([consulta])[0]
    else:
        matriz, vetor = vetores.tfidf([c["texto"] for c in chunks], consulta)

    # Nota de 0 a 1 para cada pedaço: quanto maior, mais parecido com a consulta.
    notas = vetores.similaridade_cosseno(vetor, matriz)
    # argsort ordena do menor para o maior; [::-1] inverte; [:k] pega os k melhores.
    melhores = np.argsort(notas)[::-1][:k]
    return [
        {"fonte": chunks[i]["fonte"], "trecho": chunks[i]["texto"], "nota": round(float(notas[i]), 3)}
        for i in melhores
        if notas[i] > 0  # descarta pedaços sem nada em comum com a consulta
    ]