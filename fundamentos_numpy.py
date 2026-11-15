"""Fundamentos matematicos do SBERT com exemplos NUMPY sem pesos de Transformer.

Executar: python codigo/fundamentos_numpy.py
"""
import numpy as np


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """token_embeddings: (batch, seq, dim); attention_mask: (batch, seq)."""
    mask = np.asarray(attention_mask, dtype=np.float64)[..., None]
    tokens = np.asarray(token_embeddings, dtype=np.float64)
    assert tokens.ndim == 3 and mask.shape[:2] == tokens.shape[:2]
    return (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))


def cosine_regression_loss(u: np.ndarray, v: np.ndarray, target: float) -> float:
    """Erro quadratico, com alvo normalizado em [0,1]."""
    return (cosine_similarity(u, v) - float(target)) ** 2


def triplet_loss(anchor, positive, negative, margin: float = 0.5) -> float:
    """Triplet loss com distancia euclidiana, interpretacao didatica."""
    dap = np.linalg.norm(np.asarray(anchor) - np.asarray(positive))
    dan = np.linalg.norm(np.asarray(anchor) - np.asarray(negative))
    return float(max(0.0, dap - dan + margin))


def multiple_negatives_loss(anchors: np.ndarray, positives: np.ndarray,
                            temperature: float = 0.05) -> float:
    """InfoNCE in-batch: a diagonal define os pares positivos."""
    a = np.asarray(anchors, dtype=float)
    p = np.asarray(positives, dtype=float)
    assert a.shape == p.shape and a.ndim == 2
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    logits = a @ p.T / temperature
    logits -= logits.max(axis=1, keepdims=True)  # estabilidade numerica
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -float(np.diag(log_probs).mean())


if __name__ == "__main__":
    hidden_states = np.array([[[1., 0.], [0., 1.], [80., 80.]]])
    mask = np.array([[1, 1, 0]])  # terceiro token = padding
    pooled = mean_pooling(hidden_states, mask)
    assert np.allclose(pooled, [[0.5, 0.5]])
    u, v, w = np.array([1., 0.]), np.array([0.8, 0.6]), np.array([0., 1.])
    print("Pooling com mascara:", pooled)
    print("cos(u,v):", cosine_similarity(u, v))
    print("cos(u,w):", cosine_similarity(u, w))
    print("MSE STS:", cosine_regression_loss(u, v, 0.9))
    print("Triplet loss:", triplet_loss(u, v, w))
    print("InfoNCE:", multiple_negatives_loss(np.stack([u, w]), np.stack([v, w])))
