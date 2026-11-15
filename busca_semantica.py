"""Busca semantica, pares semelhantes e agrupamento usando SBERT multilíngue.

Uso: python codigo/busca_semantica.py
Baixa o checkpoint na primeira execucao; requer internet nesse primeiro acesso.
"""
from pathlib import Path
import csv
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENTOS = [
    "Sistema de monitoramento de falhas em motores eletricos por sensores.",
    "Metodo para detectar anomalias em maquinas industriais com aprendizado de maquina.",
    "Dispositivo para irrigacao automatizada com sensores de umidade do solo.",
    "Plataforma de comunicacao sem fio entre navios e terminais portuarios.",
    "Processo de classificacao de imagens medicas por redes neurais.",
    "Controle inteligente da temperatura de baterias recarregaveis.",
    "Sistema de diagnostico de defeitos em equipamentos de fabricacao.",
    "Tecnologia de navegacao de robos autonomos em armazens.",
    "Analise de documentos de propriedade intelectual com processamento de linguagem natural.",
    "Tecnica de busca semantica de patentes usando representacoes vetoriais.",
]
CONSULTA = "deteccao de defeitos em equipamentos industriais"


def main():
    pasta = Path("resultados")
    pasta.mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL_ID)
    docs_emb = model.encode(DOCUMENTOS, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
    query_emb = model.encode([CONSULTA], convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)[0]
    notas = docs_emb @ query_emb
    ordem = np.argsort(-notas)
    print("Consulta:", CONSULTA)
    for indice in ordem[:5]:
        print(f"{notas[indice]:.4f} | {DOCUMENTOS[indice]}")
    # Limiar puramente ilustrativo; validar em dados reais antes de usar.
    clusters = AgglomerativeClustering(n_clusters=4, metric="cosine", linkage="average")
    rotulos = clusters.fit_predict(docs_emb)
    with (pasta / "busca_semantica.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "documento", "similaridade", "cluster"])
        for i, doc in enumerate(DOCUMENTOS):
            wr.writerow([i, doc, f"{notas[i]:.6f}", int(rotulos[i])])
    (pasta / "busca_metadados.json").write_text(
        json.dumps({"checkpoint": MODEL_ID, "consulta": CONSULTA,
                    "n_documentos": len(DOCUMENTOS)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
