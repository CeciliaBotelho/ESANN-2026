"""busca_aprendida_temporal (PLANO.md): builder guloso por etapas sobre o universo do Q-FLAIR adaptado (4 qubits, pool qflair39, pontuação RAW =
log-loss do P0 bruto, grade 101 + 21). Braços: FULL (= QFLAIR_RAW, todas as 242 unidades), SURR (ExtraTrees prevê o ganho a partir de um descritor
que NÃO usa a avaliação; avalia M), ALEAT (M aleatórias). Só treino no builder; validação só na regra final; teste uma vez."""
import sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit
EXP = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(EXP / 'blocos_inversos_busca'))
import objetivo_estavel as E                                                  # noqa: E402
S = E.S; Q = S.Q; B = S.B; N_Q = 4; D = 16; G = 12; M = 8; N_EXPL = 2; COLD = 2; TIPOS = ('RX', 'RY', 'RXX', 'RYY'); CS = (0.1, 1.0, 10.0, 100.0)
CONTA = dict(avaliacoes=0, reconstrucoes=0)


# ---------------- dados sintéticos ----------------
def professor(t): return [('unica', 'RY', (q,), 'dados', q, 0.6) for q in range(4)] + [('unica', 'RX', (q,), 'dados', 4 + q, 0.8) for q in range(4)] if t == 0 else None
_B0 = {}
def B0(ns):
    if ns not in _B0:
        X = np.random.default_rng([ns, 998]).uniform(0, np.pi, (100000, D)); s = 2 * B.prediz_B(professor(0), X, N_Q)[-1] - 1; lo, hi = -30.0, 30.0
        for _ in range(80): m = 0.5 * (lo + hi); lo, hi = (m, hi) if expit(8 * s + m).mean() < 0.3 else (lo, m)
        _B0[ns] = 0.5 * (lo + hi)
    return _B0[ns]
def gera(t, g, p, ns, tam=(205, 103, 2000)):
    rng = np.random.default_rng([ns, t, g, p]); n = tam[p]; X = rng.uniform(0, np.pi, (n, D))
    if p == 0:
        for j in range(D): X[np.argmin(X[:, j]), j] = 0.0; X[np.argmax(X[:, j]), j] = np.pi
    pr = np.full(n, 0.3) if t == 1 else expit(8 * (2 * B.prediz_B(professor(0), X, N_Q)[-1] - 1) + B0(ns)); return X, (rng.random(n) < pr).astype(float)


# ---------------- unidades, descritor (sem avaliar) e avaliação exata ----------------
def unidades(Pq):
    return [(0, i, k) for i in range(len(Pq['dados'])) for k in range(D)] + [(1, i, -1) for i in range(len(Pq['fixas']))] + [(2, i, -1) for i in range(len(Pq['H']))]


def descritor(u, Pq, ctx):
    """Só contexto barato do circuito e do treino; NÃO chama reconstroi/melhor_* nem usa ganhos de unidades não avaliadas."""
    cat, i, k = u; tipo, qs = (Pq['dados'], Pq['fixas'], Pq['H'])[cat][i]; v = np.zeros(34)
    v[cat] = 1
    if tipo in TIPOS: v[3 + TIPOS.index(tipo)] = 1
    for q in qs: v[7 + q] = 1
    if cat == 0: v[11 + k] = 1; v[27] = ctx['corr'][k]; v[28] = ctx['uso'][k]; v[29] = (ctx['g'] - ctx['ultimo'][k]) / G if ctx['ultimo'][k] >= 0 else 1.0
    v[30] = max(ctx['prof'][q] for q in qs) / G; v[31] = float(any(ctx['ult_tipo'][q] == tipo for q in qs)); v[32] = ctx['g'] / G; v[33] = ctx['L']; return v


def avalia(u, Pq, psi, f0, y, Z, cache):
    cat, i, k = u; CONTA['avaliacoes'] += 1
    if cat == 2: tipo, qs = Pq['H'][i]; return Q.logloss(y, Q.p0(Q.aplica(psi, N_Q, 'H', qs))), ('unica', 'H', qs, 'H', None, None)
    tipo, qs = (Pq['dados'], Pq['fixas'])[cat][i]
    if (cat, i) not in cache: cache[(cat, i)] = Q.reconstroi(psi, N_Q, tipo, qs, f0); CONTA['reconstrucoes'] += 1
    if cat == 0: L, k2, th = Q.melhor_dados(y, Z, *cache[(cat, i)], np.array([k])); return L, ('unica', tipo, qs, 'dados', k2, th)
    L, th = Q.melhor_fixa(y, *cache[(cat, i)]); return L, ('unica', tipo, qs, 'fixas', None, th)


def constroi(Ztr, ytr, braco, semente, M_=M):
    """Builder guloso; recebe SÓ o treino. braco ∈ {FULL, SURR, ALEAT}."""
    from sklearn.ensemble import ExtraTreesRegressor
    N = len(Ztr); Pq = Q.pool(N_Q); U = unidades(Pq); psi = Q.estado_inicial(N, N_Q); f0 = Q.p0(psi); L_cur = Q.logloss(ytr, f0); rng = np.random.default_rng(semente)
    zs = (Ztr - Ztr.mean(0)) / np.where(Ztr.std(0) > 0, Ztr.std(0), 1); ctx = dict(uso=np.zeros(D), ultimo=np.full(D, -1), prof=np.zeros(N_Q), ult_tipo=[None] * N_Q)
    Xh, yh, portas, log = [], [], [], []; cpu_surr = 0.0; a0 = CONTA['avaliacoes']; r0 = CONTA['reconstrucoes']
    for g in range(G):
        r = ytr - f0; rc = r - r.mean(); ctx.update(g=g, L=L_cur, corr=np.abs(zs.T @ rc) / (N * max(rc.std(), 1e-12)))
        if braco == 'FULL': sel = list(range(len(U)))
        elif braco == 'ALEAT' or g < COLD: sel = sorted(rng.choice(len(U), M_, replace=False).tolist())
        else:
            t0 = time.process_time(); DX = np.array([descritor(u, Pq, ctx) for u in U]); mdl = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=int(rng.integers(2 ** 31))).fit(np.array(Xh), np.array(yh))
            pred = mdl.predict(DX); cpu_surr += time.process_time() - t0; expl = rng.choice(len(U), N_EXPL, replace=False).tolist()
            top = [j for j in np.lexsort((np.arange(len(U)), -pred)) if j not in expl][:M_ - N_EXPL]; sel = sorted(set(expl) | set(int(j) for j in top))
        cache = {}; aval = []
        for j in sel:
            L, esc = avalia(U[j], Pq, psi, f0, ytr, Ztr, cache); aval.append((L_cur - L, U[j], esc))
            if braco == 'SURR': Xh.append(descritor(U[j], Pq, ctx)); yh.append(L_cur - L)
        ganho, u, esc = sorted(aval, key=lambda a: (-a[0], a[1]))[0]
        if ganho < 0:                                                                     # só possível se todas as avaliadas forem H: incumbente comum = identidade (nada inserido)
            log.append(dict(etapa=g, avaliadas=len(sel), unidade=None, ganho=0.0, L=float(L_cur), identidade=True)); continue
        psi = B.aplica_escolha(psi, N_Q, esc, Ztr); f0 = Q.p0(psi); L_new = Q.logloss(ytr, f0); assert L_new <= L_cur + 1e-10, (L_new, L_cur)
        L_cur = L_new; portas.append(esc); log.append(dict(etapa=g, avaliadas=len(sel), unidade=list(u), ganho=float(ganho), L=float(L_cur), identidade=False))
        if esc[3] == 'dados': ctx['uso'][esc[4]] += 1; ctx['ultimo'][esc[4]] = g
        for q in esc[2]: ctx['prof'][q] += 1; ctx['ult_tipo'][q] = esc[1]
    return portas, dict(log=log, etapas_identidade=sum(x.get('identidade', False) for x in log), avaliacoes=CONTA['avaliacoes'] - a0, reconstrucoes=CONTA['reconstrucoes'] - r0, cpu_surrogate_s=cpu_surr,
                        features_distintas=sorted({int(p[4]) for p in portas if p[3] == 'dados'}), recursos=S.recursos(portas))


# ---------------- calibração final comum (cópia de inbreast_crescimento_regularizado) ----------------
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


def final(portas, Ztr, ytr, Zva, yva, Zte):
    """Prefixo k ∈ 1..G × C pela validação, com CONST no conjunto; teste uma vez."""
    sc = lambda Z: [2 * p - 1 for p in B.prediz_B(portas, Z, N_Q)]; Str, Sva = sc(Ztr), sc(Zva); tab = []                       # teste: só o prefixo já selecionado
    for k in range(1, len(portas) + 1):
        for C in CS: a, b = calibra(Str[k], ytr, C); tab.append(dict(k=k, C=C, a=a, b=b, bce_val=bce_p(yva, expit(a * Sva[k] + b))))
    const = bce_p(yva, np.full(len(yva), ytr.mean())); best = min(tab, key=lambda d: (d['bce_val'], d['k'], d['C']))
    if best['bce_val'] >= const: return np.full(len(Zte), ytr.mean()), dict(k=0, C=None, bce_val=const, const=True), None
    s_te = 2 * B.prediz_B(portas[:best['k']], Zte, N_Q)[-1] - 1
    return expit(best['a'] * s_te + best['b']), dict(best, const=False), certifica(Str[best['k']], ytr, best['C'], best['a'], best['b'])


def features_prefixo(portas, k):
    """Features com peso ≠ 0 nas portas de dados do prefixo escolhido (k = 0 ⇒ CONST)."""
    return sorted({int(p[4]) for p in portas[:k] if p[3] == 'dados' and p[5] != 0.0})
