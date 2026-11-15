"""Recuperacao por trechos e agregacao por patente usando SBERT.

Uso: python codigo/indexacao_patentes.py
Patentes inteiramente sinteticas para fins pedagogicos.
Primeira execucao exige download de checkpoint da internet.
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PATENTES = [
    {"id": "P-001", "titulo": "Controle de maquinas industriais",
     "resumo": "Sensores medem sinais de vibracao para diagnosticar falhas precoces.",
     "reivindicacoes": "Um metodo de monitoramento com deteccao automatizada de anomalias."},
    {"id": "P-002", "titulo": "Analise de imagens clinicas",
     "resumo": "Rede neural identifica padroes de imagem em exames de saude.",
     "reivindicacoes": "Um sistema de segmentacao por classificacao computacional."},
    {"id": "P-003", "titulo": "Sistema digital em terminal portuario",
     "resumo": "Rastreamento de containers e coordenacao de horarios de atracacao.",
     "reivindicacoes": "Um sistema de comunicacao para otimizar a logistica portuaria."},
]
CONSULTA = "Como detectar defeitos em maquinas com sensores?"


def segmentar(texto, model, sobreposicao=16):
    """Divide em trechos respeitando a janela de tokens e tokens especiais."""
    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(texto, add_special_tokens=False)
    max_len = model.max_seq_length or 128
    capacidade = int(max_len - tokenizer.num_special_tokens_to_add(pair=False))
    if capacidade < 2:
        raise ValueError("Janela de tokens insuficiente para a segmentacao")
    sobreposicao = min(max(0, int(sobreposicao)), capacidade - 1)
    passo = capacidade - sobreposicao
    return [tokenizer.decode(token_ids[i:i + capacidade], skip_special_tokens=True)
            for i in range(0, len(token_ids), passo)] if token_ids else []


def executar(model_id=MODEL_ID):
    model = SentenceTransformer(model_id)
    textos, donos = [], []
    for patente in PATENTES:
        fonte = " ".join([patente["titulo"], patente["resumo"], patente["reivindicacoes"]])
        for trecho in segmentar(fonte, model):
            textos.append(trecho)
            donos.append(patente["id"])
    if not textos:
        raise RuntimeError("Nenhum trecho disponivel")
    emb = model.encode(textos, normalize_embeddings=True, convert_to_numpy=True)
    enc_q = model.encode([CONSULTA], normalize_embeddings=True, convert_to_numpy=True)[0]
    sim = np.asarray(emb @ enc_q).reshape(-1)
    por_patente = {}
    for pid, valor in zip(donos, sim):
        por_patente[pid] = max(por_patente.get(pid, -np.inf), float(valor))
    ranking = sorted(por_patente.items(), key=lambda x: x[1], reverse=True)
    resultado = {"nota": "Patentes artificiais; escores sem validade para analise juridica.",
                 "modelo": model_id, "consulta": CONSULTA,
                 "n_trechos": len(textos), "ranking": ranking}
    Path("resultados").mkdir(exist_ok=True)
    Path("resultados/indexacao_patentes.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    executar()
