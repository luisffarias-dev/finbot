# scripts/indexar.py
"""Aula 5 — Indexa a pasta documentos/ no MongoDB e testa uma busca.

    python -m scripts.indexar

Rode de novo sempre que adicionar ou mudar documentos (inclusive PDFs da B3).
Com MONGO_URL apontando para o Atlas, o índice já fica pronto para o app hospedado.
"""
from app.config.settings import settings
from app.rag.busca import buscar
from app.rag.indexador import indexar

# Indexa os documentos e mostra quantos pedaços foram gravados.
print(f"Modo: {settings.rag_modo} | pedaços indexados: {indexar()}")
# Faz uma busca de teste e mostra o começo de cada trecho encontrado.
for r in buscar("Quanto de imposto pago no CDB depois de 2 anos?"):
    print(f"\n[{r['fonte']}] nota={r['nota']}\n{r['trecho'][:200]}...")