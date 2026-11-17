"""QF-C: adaptação própria do Q-FLAIR (Jäger, Elsässer, Torabian, PRR 8, 023247, 2026; código phels23/Q-FLAIR @4f331783 só LIDO).
NÃO é reprodução fiel (ver PLANO.md: n fios fixo e todos ativos, cadeia linear, G ≤ 12, grade 101 + refino 21, sorteio de posições).
Vetor de estado por amostra (N, 2**n), portas aplicadas localmente por eixo (sem matriz densa 2**n × 2**n).
Convenção de fios: o fio 0 é o bit mais significativo (igual ao PennyLane)."""
import time
import numpy as np

S2 = np.sqrt(0.5)
TIPOS_DADOS = ('RX', 'RY', 'RXX', 'RYY')
EPS = 1e-12                                   # recorte de p na log loss (declarado)
GRADE = np.linspace(-1.0, 1.0, 101)           # θ ∈ [−1, 1] (101 pontos; o original usa 1001) + refino local
PASSO = GRADE[1] - GRADE[0]
REFINO = np.linspace(-PASSO, PASSO, 21)


# ---------------- pool (cadeia linear; ordem canônica = ordem de desempate) ----------------
def pool(n):
    dados = [('RX', (q,)) for q in range(n)] + [('RY', (q,)) for q in range(n)] + [('RXX', (q, q + 1)) for q in range(n - 1)] + [('RYY', (q, q + 1)) for q in range(n - 1)]
    return dict(dados=dados, fixas=list(dados), H=[('H', (q,)) for q in range(n)])


# ---------------- simulação local ----------------
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


def reconstroi(psi, n, tipo, qs, f0=None):
    """Coeficientes por amostra de P0(α) = a·cos(α − b) + c, de 3 avaliações (α = 0 reaproveitada; ±π/2 novas)."""
    f0 = p0(psi) if f0 is None else f0
    fp = p0(aplica(psi, n, tipo, qs, np.pi / 2)); fm = p0(aplica(psi, n, tipo, qs, -np.pi / 2))
    A = 0.5 * (fp + fm); B = f0 - A; C = 0.5 * (fp - fm)
    return np.sqrt(B ** 2 + C ** 2), np.arctan2(C, B), A


# ---------------- varredura clássica de candidatos ----------------
def _perda_grade(y, a, b, c, ang):
    """ang: (T, N, K) → perda média por (T, K)."""
    p = np.clip(a[None, :, None] * np.cos(ang - b[None, :, None]) + c[None, :, None], EPS, 1 - EPS)
    yy = y[None, :, None]; return -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean(1)


def melhor_dados(y, X, a, b, c, feats, chunk=256):
    """Para uma posição com dados: melhor (k, θ) entre feats (ordem crescente), grade + refino. Desempate: menor k, menor θ."""
    best = (np.inf, None, None)
    for i0 in range(0, len(feats), chunk):
        ks = feats[i0:i0 + chunk]; Xk = X[:, ks]
        L = _perda_grade(y, a, b, c, GRADE[:, None, None] * Xk[None])                       # (101, K)
        j = L.argmin(0); th0 = GRADE[j]                                                     # primeiro mínimo por feature
        thl = np.clip(th0[None, :] + REFINO[:, None], -1, 1)                                 # (21, K), contém θ0 (REFINO[10] = 0)
        Ll = _perda_grade(y, a, b, c, thl[:, None, :] * Xk[None])
        jl = Ll.argmin(0); Lk = Ll[jl, np.arange(len(ks))]; thk = thl[jl, np.arange(len(ks))]
        m = int(Lk.argmin())
        if Lk[m] < best[0]: best = (float(Lk[m]), int(ks[m]), float(thk[m]))
    return best


def melhor_fixa(y, a, b, c):
    L = _perda_grade(y, a, b, c, (np.pi * GRADE)[:, None, None])[:, 0]; th0 = GRADE[int(L.argmin())]
    thl = np.clip(th0 + REFINO, -1, 1); Ll = _perda_grade(y, a, b, c, (np.pi * thl)[:, None, None])[:, 0]; j = int(Ll.argmin())
    return float(Ll[j]), float(thl[j])


# ---------------- crescimento ----------------
def sorteia_posicoes(n, rng, modo, M=(12, 12, 4)):
    P = pool(n)
    if modo == 'cheio': return P['dados'], P['fixas'], P['H']
    pick = lambda L, m: [L[i] for i in sorted(rng.choice(len(L), size=min(m, len(L)), replace=False))]   # ordem canônica preservada
    return pick(P['dados'], M[0]), pick(P['fixas'], M[1]), pick(P['H'], M[2])


def cresce(X, y, n, G_max=12, modo='igual', semente=0, controle=None, tempos=None, ties=None):
    """X: (N, d) já escalado em [0, π]. controle ∈ {None, 'feature_aleatoria', 'porta_aleatoria'}.
    Retorna lista de portas (tipo, qs, categoria, k, θ) e curva de perda de treino."""
    N, d = X.shape; psi = estado_inicial(N, n); f0 = p0(psi); L_cur = logloss(y, f0)
    rng = np.random.default_rng([39, n, semente, 0]); rngc = np.random.default_rng([39, n, semente, 1])
    portas, curva, orc = [], [L_cur], []
    feats_all = np.arange(d)
    for g in range(G_max):
        Pd, Pf, Ph = sorteia_posicoes(n, rng, modo); cands = []                                 # (perda, ordem, porta)
        if controle == 'porta_aleatoria':
            todas = [('dados',) + p for p in Pd] + [('fixas',) + p for p in Pf] + [('H',) + p for p in Ph]
            esc = todas[int(rngc.integers(len(todas)))]; Pd = [esc[1:]] if esc[0] == 'dados' else []; Pf = [esc[1:]] if esc[0] == 'fixas' else []; Ph = [esc[1:]] if esc[0] == 'H' else []
        t_sim = t_dados = t_fix = 0.0; n_cand = 0
        for i, (tipo, qs) in enumerate(Pd):
            t0 = time.perf_counter(); a, b, c = reconstroi(psi, n, tipo, qs, f0); t_sim += time.perf_counter() - t0
            feats = np.array([int(rngc.integers(d))]) if controle == 'feature_aleatoria' else feats_all
            t0 = time.perf_counter(); L, k, th = melhor_dados(y, X, a, b, c, feats); t_dados += time.perf_counter() - t0
            n_cand += len(feats) * (len(GRADE) + len(REFINO)); cands.append((L, (0, i), (tipo, qs, 'dados', k, th)))
        for i, (tipo, qs) in enumerate(Pf):
            t0 = time.perf_counter(); a, b, c = reconstroi(psi, n, tipo, qs, f0); t_sim += time.perf_counter() - t0
            t0 = time.perf_counter(); L, th = melhor_fixa(y, a, b, c); t_fix += time.perf_counter() - t0
            n_cand += len(GRADE) + len(REFINO); cands.append((L, (1, i), (tipo, qs, 'fixas', None, th)))
        for i, (tipo, qs) in enumerate(Ph):
            t0 = time.perf_counter(); L = logloss(y, p0(aplica(psi, n, 'H', qs))); t_sim += time.perf_counter() - t0
            n_cand += 1; cands.append((L, (2, i), ('H', qs, 'H', None, None)))
        Lmin = min(cd[0] for cd in cands); esc = next(cd for cd in cands if cd[0] == Lmin)       # primeira na ordem canônica
        if ties is not None: ties.append(sum(abs(cd[0] - Lmin) <= 1e-13 for cd in cands))
        tipo, qs, cat, k, th = esc[2]
        t0 = time.perf_counter(); psi = aplica(psi, n, tipo, qs, angulo(cat, k, th, X)); f0 = p0(psi); t_sim += time.perf_counter() - t0
        assert np.allclose(np.linalg.norm(psi, axis=1), 1, atol=1e-10) and f0.min() >= -1e-12 and f0.max() <= 1 + 1e-12
        L_new = logloss(y, f0)
        if controle is None: assert L_new <= L_cur + 1e-10, (L_new, L_cur)      # θ = 0 (identidade) é candidato: perda não crescente
        L_cur = L_new; portas.append(esc[2]); curva.append(L_cur)
        orc.append(dict(n_pos_dados=len(Pd), n_pos_fixas=len(Pf), n_pos_H=len(Ph), n_candidatos=n_cand, tipos_dados=sorted(set(p[0] for p in Pd)),
                        qubits_dados=sorted(set(q for p in Pd for q in p[1]))))
        if tempos is not None: tempos.append(dict(sim=t_sim, dados=t_dados, fixas=t_fix))
    return portas, curva, orc


def angulo(cat, k, th, X):
    if cat == 'dados': return th * X[:, k]
    if cat == 'fixas': return th * np.pi
    return None


def prediz(portas, X, n):
    psi = estado_inicial(len(X), n); out = [p0(psi)]
    for tipo, qs, cat, k, th in portas: psi = aplica(psi, n, tipo, qs, angulo(cat, k, th, X)); out.append(p0(psi))
    return out                                                          # P0 depois de 0, 1, …, G portas


class Escala:
    """Min-max por feature para [0, π] ajustado só no treino; recorte fora do intervalo (como o código original)."""
    def fit(self, X):
        self.lo = X.min(0); rg = X.max(0) - self.lo; self.rg = np.where(rg > 0, rg, 1.0); self.const = rg == 0; return self
    def transform(self, X):
        Z = np.clip((X - self.lo) / self.rg, 0, 1) * np.pi; Z[:, self.const] = 0.0; return Z


# ---------------- descritores do circuito ----------------
def descreve(portas, n):
    toc = sorted(set(q for p in portas for q in p[1])); arestas = [p[1] for p in portas if len(p[1]) == 2]
    pai = list(range(n))
    def acha(i):
        while pai[i] != i: i = pai[i]
        return i
    for a_, b_ in arestas: pai[acha(a_)] = acha(b_)
    comp = len(set(acha(q) for q in toc)) if toc else 0
    red = 0; ult = {}
    for i, (tipo, qs, cat, k, th) in enumerate(portas):                  # pares consecutivos fundíveis nos mesmos fios
        chave = (tipo, qs); prev = [ult.get(q) for q in qs]
        if len(set(prev)) == 1 and prev[0] is not None and portas[prev[0]][:2] == chave and (cat == 'fixas' and portas[prev[0]][2] == 'fixas' or cat == 'dados' and portas[prev[0]][2] == 'dados' and portas[prev[0]][3] == k):
            red += 1
        for q in qs: ult[q] = i
    feats = [p[3] for p in portas if p[2] == 'dados']
    return dict(qubits_ocupados=len(toc), componentes_conexas=comp, portas_2q=len(arestas), pares_fundiveis=red,
                features_distintas=len(set(feats)), portas_dados=len(feats), portas_fixas=sum(p[2] == 'fixas' for p in portas), portas_H=sum(p[2] == 'H' for p in portas))
