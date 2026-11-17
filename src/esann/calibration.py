"""Slope-penalized logistic calibration, unchanged executed functions."""
import numpy as np
from scipy.special import expit

def _J(w, s, y, C): z = w[0] * s + w[1]; return float(np.mean(np.logaddexp(0, z) - y * z) + w[0] ** 2 / (2 * C * len(s)))


def _G(w, s, y, C): r = expit(w[0] * s + w[1]) - y; return np.array([np.mean(r * s) + w[0] / (C * len(s)), np.mean(r)])


def _H(w, s, y, C): p = expit(w[0] * s + w[1]); q = p * (1 - p); return np.array([[np.mean(q * s * s) + 1 / (C * len(s)), np.mean(q * s)], [np.mean(q * s), np.mean(q) + 1e-300]])


def calibra(s, y, C):
    m = np.clip(y.mean(), 1e-6, 1 - 1e-6); w = np.array([0.0, np.log(m / (1 - m))])
    for _ in range(200):
        g = _G(w, s, y, C)
        if np.abs(g).max() <= 1e-10: break
        d = np.linalg.solve(_H(w, s, y, C), g); t = 1.0; j0 = _J(w, s, y, C)
        while _J(w - t * d, s, y, C) > j0 - 1e-4 * t * g @ d and t > 1e-12: t *= 0.5
        w = w - t * d
    return float(w[0]), float(w[1])


def certifica(s, y, C, a, b):
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    sk = LogisticRegression(C=C, tol=1e-12, max_iter=100000).fit(s[:, None], y); wsk = np.r_[sk.coef_[0][0], sk.intercept_[0]]
    sc = minimize(_J, np.zeros(2), args=(s, y, C), jac=_G, hess=_H, method='trust-exact', options=dict(gtol=1e-12, maxiter=10000)); w = np.r_[a, b]; e = max(1.0, np.abs(w).max())
    return dict(grad=float(np.abs(_G(w, s, y, C)).max()), dif_sklearn=float(np.abs(w - wsk).max() / e), dif_scipy=float(np.abs(w - sc.x).max() / e), dJ_min=min(_J(wsk, s, y, C), float(sc.fun)) - _J(w, s, y, C))


def cert_ok(c): return c['grad'] <= 1e-8 and c['dif_sklearn'] <= 1e-6 and c['dif_scipy'] <= 1e-6 and c['dJ_min'] >= -1e-10


def bce_p(y, p): p = np.clip(p, 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
