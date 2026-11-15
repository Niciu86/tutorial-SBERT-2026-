"""Fine-tuning pedagogico com MultipleNegativesRankingLoss / Trainer (Sentence-Transformers >=3).

Uso: python codigo/treinamento.py
Os poucos exemplos abaixo sao sinteticos: demonstram API, nao produzem modelo validado.
"""
import json
from pathlib import Path
import random
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses
)
from sentence_transformers.training_args import BatchSamplers

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRAIN_PAIRS = [
    ("Identificar falhas de motores por vibracao", "Detectar defeitos de motores por sinais de vibracao"),
    ("Patentes de sensores industriais", "Documentos patentarios sobre sensores para industrias"),
    ("Navegacao de robos autonomos", "Planejamento de trajetorias para sistemas roboticos autonomos"),
    ("Busca semantica de documentos", "Recuperacao de textos a partir de similaridade de significado"),
    ("Diagnostico de anomalias em fabricas", "Deteccao de condicoes anormais em linhas de producao"),
    ("Previsao de falhas de equipamentos", "Estimativa antecipada de defeitos em maquinas"),
    ("Classificacao de textos tecnicos", "Organizacao automatica de documentos de engenharia"),
    ("Automatizacao portuaria", "Otimizacao de operacoes em terminais maritimos"),
    ("Monitoramento remoto de maquinas", "Acompanhamento de dispositivos a distancia"),
    ("Mineração de patentes", "Extracao de conhecimento em documentos de patentes"),
    ("Comunicação de redes IoT", "Transmissao de dados entre dispositivos conectados"),
    ("Reconhecimento de imagens medicas", "Classificacao visual assistida de exames clinicos"),
]
TEST_PAIRS = [
    ("Manutencao baseada em condicao", "Monitoramento preventivo do estado de maquinas"),
    ("Consulta semantica em propriedade intelectual", "Busca pelo significado de inventos patenteados"),
    ("Sistema automatizado de carga portuaria", "Controle digital de movimentacao em terminais"),
]


def mean_pair_cosine(model, pairs):
    a = model.encode([p[0] for p in pairs], normalize_embeddings=True)
    b = model.encode([p[1] for p in pairs], normalize_embeddings=True)
    return float(np.mean(np.sum(a * b, axis=1)))


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    out = Path("resultados/modelo_sbert_ajustado")
    out.parent.mkdir(exist_ok=True, parents=True)
    model = SentenceTransformer(MODEL_ID)
    baseline = mean_pair_cosine(model, TEST_PAIRS)
    data = Dataset.from_dict({"anchor": [p[0] for p in TRAIN_PAIRS],
                              "positive": [p[1] for p in TRAIN_PAIRS]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=1,
        per_device_train_batch_size=4, learning_rate=2e-5,
        warmup_ratio=0.1, fp16=False, bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", report_to="none", seed=42
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=data,
        loss=losses.MultipleNegativesRankingLoss(model)
    )
    trainer.train()
    model.save_pretrained(str(out / "final"))
    post = mean_pair_cosine(model, TEST_PAIRS)
    resultado = {"checkpoint_inicial": MODEL_ID, "sementes": 42,
                 "train_pairs": len(TRAIN_PAIRS), "test_pairs": len(TEST_PAIRS),
                 "cosine_positives_before": baseline,
                 "cosine_positives_after": post,
                 "aviso": "Media de similaridade positiva nao mede retrieval e pode aumentar com colapso."}
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    (out / "diagnostico.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
