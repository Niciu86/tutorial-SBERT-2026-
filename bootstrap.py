"""IC percentil da diferenca pareada de uma metrica por consulta.

Uso: python codigo/bootstrap.py
Valores de entrada fabricados para ilustrar estatistica; nao sao benchmarks.
"""

import numpy as np


def ic_pareado(a, b, B=2000, semente=42):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Series 1D com mesmo tamanho e ao menos duas consultas")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Metricas devem ser numeros finitos")
    rng = np.random.default_rng(semente)
    dif = a - b
    pos = rng.integers(len(dif), size=(B, len(dif)))
    medias = dif[pos].mean(axis=1)
    return tuple(float(x) for x in np.quantile(medias, [0.025, 0.975]))


if __name__ == "__main__":
    # nDCG@10 ficticio obtido nas mesmas cinco consultas com dois sistemas.
    modelo_a = [0.88, 0.79, 0.74, 0.91, 0.69]
    modelo_b = [0.80, 0.75, 0.71, 0.89, 0.68]
    media = np.mean(np.subtract(modelo_a, modelo_b))
    lo, hi = ic_pareado(modelo_a, modelo_b)
    print(f"Delta medio (A-B) = {media:.4f}")
    print(f"IC bootstrap pareado 95%: [{lo:.4f}, {hi:.4f}]")
    print("Exemplo sintetico, sem inferencia sobre desempenho real.")
