"""Matched prefix selection and metrics, unchanged executed functions."""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from . import calibration as BA
C_FIX=1.0

def bce(y, p): p = np.clip(p, 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p): r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sel_prefixo_P0(Ptr, Pva, Pte, ytr, yva, ks):
    """Mesma regra de final_c1 (busca_blocos_objetivo_perfilado) sobre arrays de P0 por prefixo (linha k = prefixo de k operações)."""
    if not ks: return np.full(Pte.shape[1], ytr.mean()), dict(k=0, const=True), None
    tab = []
    for k in ks: s = 2 * Ptr[k] - 1; a, b = BA.calibra(s, ytr, C_FIX); tab.append(dict(k=k, C=C_FIX, a=a, b=b, bce_val=BA.bce_p(yva, expit(a * (2 * Pva[k] - 1) + b))))
    const = BA.bce_p(yva, np.full(len(yva), ytr.mean())); best = min(tab, key=lambda d: (d['bce_val'], d['k'], d['C']))
    if best['bce_val'] >= const: return np.full(Pte.shape[1], ytr.mean()), dict(k=0, const=True, bce_val=const), None
    return expit(best['a'] * (2 * Pte[best['k']] - 1) + best['b']), dict(best, const=False), BA.certifica(2 * Ptr[best['k']] - 1, ytr, C_FIX, best['a'], best['b'])

from sklearn.metrics import roc_curve

def limiar_roc_treino(s_tr, y_tr):
    fpr, tpr, thr = roc_curve(np.asarray(y_tr), np.asarray(s_tr)); accs = (tpr + 1 - fpr) * 0.5; i = int(np.argmax(accs)); return float(thr[i]), float(accs[i])

def bacc(s, y, t):
    yh = np.asarray(s) >= t; y = np.asarray(y); return float(0.5 * (np.mean(yh[y == 1]) + np.mean(~yh[y == 0])))
