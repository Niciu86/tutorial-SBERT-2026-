"""Avaliacao didatica de STS e recuperacao; nao e um benchmark cientifico.

Uso: python codigo/avaliacao.py
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CORPUS = [
    "Diagnostico de falhas em bombas hidraulicas com sensores inteligentes.",
    "Manutencao preditiva em motores industriais por vibracao.",
    "Classificacao automatica de imagens em exames medicos.",
    "Deteccao de anomalias em maquinas usando redes neurais.",
    "Sistema de comunicacao de cargas em terminais portuarios.",
    "Navegacao autonoma de veiculos roboticos em armazens.",
    "Monitoramento de equipamentos fabris com analise de dados de sensores.",
    "Deteccao de fraudes em transacoes financeiras com aprendizagem profunda.",
]
QUERIES = [
    "Como diagnosticar defeitos em maquinario industrial?",
    "Como usar inteligencia artificial para exames clinicos?",
    "Como melhorar operacoes de um porto?",
]
# Julgamentos pedagogicos de relevancia 0/1/2 (nao derivados de especialistas).
RELEVANCIA = np.array([
    [2, 2, 0, 2, 0, 0, 2, 0],
    [0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0],
], dtype=int)
STS = [
    ("Deteccao de falhas em motores.", "Identificacao de defeitos em equipamentos eletricos.", 0.9),
    ("Busca semantica em patentes.", "Recuperacao por sentido em documentos de patente.", 0.9),
    ("Robos em armazens.", "Sistemas autonomos em centros de distribuicao.", 0.7),
    ("Controle de qualidade por sensores.", "Analise sensorial da industria.", 0.6),
    ("Seguranca de redes sem fio.", "Uso de motores eletricos.", 0.1),
    ("Algoritmos de classificacao.", "Cultivo de frutas tropicais.", 0.1),
    ("Portos inteligentes.", "Automatizacao de terminais maritimos.", 0.9),
]


def ndcg_at_k(ranking: np.ndarray, relevance: np.ndarray, k: int = 5) -> float:
    """nDCG definido pelo ganho 2^rel-1 e desconto log2(rank+1)."""
    observed = relevance[np.asarray(ranking)[:k]]
    discounts = np.log2(np.arange(2, len(observed) + 2))
    dcg = np.sum((2. ** observed - 1) / discounts)
    ideal = np.sort(relevance)[::-1][:k]
    idcg = np.sum((2. ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg else 0.0


def recall_at_k(ranking: np.ndarray, relevance: np.ndarray, k: int = 5) -> float:
    good = relevance > 0
    return float(good[ranking[:k]].sum() / good.sum()) if good.sum() else 0.0


def evaluate_retrieval(matrix: np.ndarray, k: int = 5) -> dict:
    return {
        "ndcg": float(np.mean([ndcg_at_k(np.argsort(-scores), rel, k)
                               for scores, rel in zip(matrix, RELEVANCIA)])),
        "recall": float(np.mean([recall_at_k(np.argsort(-scores), rel, k)
                                 for scores, rel in zip(matrix, RELEVANCIA)])),
    }


def main():
    Path("resultados").mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL_ID)
    enc_q = model.encode(QUERIES, convert_to_numpy=True, normalize_embeddings=True)
    enc_d = model.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=True)
    sim_sbert = enc_q @ enc_d.T
    tfidf = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
    corpus_tfidf = tfidf.fit_transform(CORPUS)
    query_tfidf = tfidf.transform(QUERIES)
    sim_tfidf = cosine_similarity(query_tfidf, corpus_tfidf)
    left = model.encode([x[0] for x in STS], normalize_embeddings=True)
    right = model.encode([x[1] for x in STS], normalize_embeddings=True)
    predicted = np.sum(left * right, axis=1)
    gold = np.array([x[2] for x in STS], dtype=float)
    rho = float(spearmanr(predicted, gold).statistic)
    results = {"disclaimer": "Exemplos sinteticos, demonstracao sem validade externa.",
               "checkpoint": MODEL_ID, "spearman_sts": rho,
               "sbert": evaluate_retrieval(sim_sbert),
               "tfidf": evaluate_retrieval(sim_tfidf)}
    print(json.dumps(results, indent=2, ensure_ascii=False))
    Path("resultados/avaliacao.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
