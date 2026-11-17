"""Statevector primitives extracted unchanged from the executed study."""
import numpy as np
S2 = np.sqrt(0.5)
EPS = 1e-12

def estado_inicial(N, n):
    psi = np.zeros((N, 2 ** n), dtype=np.complex128); psi[:, 0] = 1.0; return psi


def aplica(psi, n, tipo, qs, ang=None):
    """ψ ← U ψ, U = exp(−i·ang·A/2) para A ∈ {X, Y, XX, YY}, ou H. ang: escalar ou (N,)."""
    N = psi.shape[0]; t = psi.reshape((N,) + (2,) * n)
    if tipo == 'H':
        t = np.moveaxis(t, 1 + qs[0], 1); a0, a1 = t[:, 0], t[:, 1]
        out = np.stack(((a0 + a1) * S2, (a0 - a1) * S2), 1); return np.moveaxis(out, 1, 1 + qs[0]).reshape(N, -1)
    ang = np.broadcast_to(np.asarray(ang, dtype=float), (N,))
    if tipo in ('RX', 'RY'):
        k = (N,) + (1,) * (n - 1); c = np.cos(ang / 2).reshape(k); s = np.sin(ang / 2).reshape(k)
        t = np.moveaxis(t, 1 + qs[0], 1); a0, a1 = t[:, 0], t[:, 1]
        if tipo == 'RX': out = np.stack((c * a0 - 1j * s * a1, -1j * s * a0 + c * a1), 1)
        else: out = np.stack((c * a0 - s * a1, s * a0 + c * a1), 1)
        return np.moveaxis(out, 1, 1 + qs[0]).reshape(N, -1)
    k = (N,) + (1,) * (n - 2); c = np.cos(ang / 2).reshape(k); s = np.sin(ang / 2).reshape(k)
    t = np.moveaxis(t, (1 + qs[0], 1 + qs[1]), (1, 2))
    flip = t[:, ::-1, ::-1]                                              # (A ψ)[a,b] ∝ ψ[1−a, 1−b]
    if tipo == 'RYY':                                                     # YY|ab⟩: sinal −1 se a = b, +1 se a ≠ b
        sg = np.array([[-1.0, 1.0], [1.0, -1.0]]).reshape((1, 2, 2) + (1,) * (n - 2)); flip = flip * sg
    out = c[:, None, None] * t - 1j * s[:, None, None] * flip
    return np.moveaxis(out, (1, 2), (1 + qs[0], 1 + qs[1])).reshape(N, -1)


def p0(psi): return np.abs(psi[:, 0]) ** 2


def logloss(y, p): p = np.clip(p, EPS, 1 - EPS); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def angulo(cat, k, th, X):
    if cat == 'dados': return th * X[:, k]
    if cat == 'fixas': return th * np.pi
    return None


class Escala:
    """Min-max por feature para [0, π] ajustado só no treino; recorte fora do intervalo (como o código original)."""
    def fit(self, X):
        self.lo = X.min(0); rg = X.max(0) - self.lo; self.rg = np.where(rg > 0, rg, 1.0); self.const = rg == 0; return self
    def transform(self, X):
        Z = np.clip((X - self.lo) / self.rg, 0, 1) * np.pi; Z[:, self.const] = 0.0; return Z


# Compatibility names for the audited search. Only single Pauli rotations occur here.
import sys
Q = sys.modules[__name__]
def aplica_bloco(*args):
    raise ValueError("Composite legacy gates are outside the ESANN library")

def aplica_escolha(psi, n, e, X):
    if e[0] == 'unica':
        _, tipo, qs, cat, k, th = e; return Q.aplica(psi, n, tipo, qs, Q.angulo(cat, k, th, X))
    _, tipo, qs, (k1, k2), g, t1, t2 = e; return aplica_bloco(psi, n, tipo, qs, t1 * X[:, k1], t2 * X[:, k2], g)

def prediz_B(portas, X, n):
    psi = Q.estado_inicial(len(X), n); out = [Q.p0(psi)]
    for e in portas: psi = aplica_escolha(psi, n, e, X); out.append(Q.p0(psi))
    return out
