# app/tools/conhecimento.py
"""Aula 5 — A busca em documentos vira uma FERRAMENTA do agente (RAG agêntico).

Por quê ferramenta? O próprio agente decide O QUE buscar (pode reformular a consulta
ou buscar duas vezes), em vez de o código sempre buscar a pergunta crua.
"""
from langchain_core.tools import tool

from app.rag.busca import buscar


# @tool: o agente vê o nome, a docstring e o argumento "consulta" e decide quando buscar.
@tool
def buscar_documentos(consulta: str) -> list[dict]:
    """Busca trechos nos documentos educativos do FinBot sobre renda fixa, CDI, Selic,
    Tesouro Direto, fundos imobiliários (FIIs), ações, B3 e impostos.
    consulta: palavras-chave em português, ex.: "tabela regressiva imposto renda CDB".
    Retorna até 3 trechos com o nome do arquivo de origem no campo "fonte"."""
    resultados = buscar(consulta, k=3)
    # Se nada for encontrado, avisa o agente (em vez de devolver lista vazia).
    return resultados or [{"aviso": "nenhum trecho encontrado"}]