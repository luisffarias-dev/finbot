# app/rag/vetores.py
"""Aula 5 — Transformar texto em vetores (números) e medir semelhança.

tfidf      -> vetor = peso de cada PALAVRA no texto. Simples, offline, não entende sinônimos.
embeddings -> vetor = SIGNIFICADO do texto (rede neural). "rende" fica perto de "rendimento".
"""
import numpy as np

# Modelo de embeddings multilíngue (entende português).
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Stopwords: palavras muito comuns que não ajudam a diferenciar um texto do outro.
STOPWORDS = """a o as os um uma de da do das dos e em no na nos nas para por com que se ao aos
ou mais como mas seu sua seus suas ele ela isso esse essa este esta qual quais quando onde
quem foi ser sao estao ter tem ha""".split()

# O modelo de embeddings é pesado: carregamos só uma vez e guardamos aqui.
_modelo = None


# Transforma cada texto em um vetor de SIGNIFICADO (uma linha da matriz por texto).
def embeddings(textos: list[str]) -> np.ndarray:
    global _modelo  # "global" permite alterar a variável _modelo de fora da função
    if _modelo is None:
        from fastembed import TextEmbedding  # baixa o modelo (~220 MB) na primeira vez
        _modelo = TextEmbedding(MODELO_EMBEDDINGS)
    return np.array(list(_modelo.embed(textos)))


# Transforma os pedaços E a pergunta em vetores de palavras, com o MESMO vocabulário.
def tfidf(textos: list[str], pergunta: str) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    # strip_accents: "ações" = "acoes"; lowercase: tudo minúsculo; stop_words: ignora palavras comuns.
    vetorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=STOPWORDS)
    # fit_transform aprende o vocabulário dos pedaços e gera a matriz (uma linha por pedaço).
    matriz = vetorizador.fit_transform(textos).toarray()
    # transform usa o mesmo vocabulário para gerar o vetor da pergunta.
    return matriz, vetorizador.transform([pergunta]).toarray()[0]


def similaridade_cosseno(vetor: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """1 = mesmo sentido; 0 = nada em comum. Mede o ÂNGULO entre os vetores, não o tamanho."""
    # Tamanho (norma) de cada vetor; o produto das normas é o divisor da fórmula do cosseno.
    normas = np.linalg.norm(matriz, axis=1) * np.linalg.norm(vetor)
    # matriz @ vetor = produto escalar com cada pedaço; np.where evita divisão por zero.
    return (matriz @ vetor) / np.where(normas == 0, 1, normas)