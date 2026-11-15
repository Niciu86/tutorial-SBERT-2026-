"""Demonstra e testa a derivada da InfoNCE em relacao aos logits.

Uso: python codigo/gradientes_torch.py
Sem necessidade de baixar modelos. Dados numericos sao sinteticos.
"""

import torch
import torch.nn.functional as F


def verificar_gradiente():
    logits = torch.tensor(
        [[1.2, 0.1, -0.3], [0.2, 1.5, 0.6], [-0.4, 0.8, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    alvos = torch.arange(logits.size(0))
    loss = F.cross_entropy(logits, alvos)
    loss.backward()

    probs = torch.softmax(logits.detach(), dim=1)
    grad_analitico = (probs - F.one_hot(alvos, num_classes=3)) / len(alvos)
    assert torch.allclose(logits.grad, grad_analitico, atol=1e-12)

    # Checagem independente de uma coordenada por diferencas finitas.
    eps = 1e-6
    z = logits.detach().clone()
    idx = (0, 1)
    z_mais, z_menos = z.clone(), z.clone()
    z_mais[idx] += eps
    z_menos[idx] -= eps
    fd = (F.cross_entropy(z_mais, alvos) - F.cross_entropy(z_menos, alvos)) / (2 * eps)
    assert torch.isclose(fd, grad_analitico[idx], atol=1e-8)
    print(f"InfoNCE = {float(loss.detach().item()):.6f}")
    print(f"Gradiente analitico z[0,1]: {float(grad_analitico[idx]):.6f}")
    print(f"Gradiente diferencas finitas: {float(fd):.6f}")
    print("Teste de gradientes: OK")


if __name__ == "__main__":
    verificar_gradiente()
