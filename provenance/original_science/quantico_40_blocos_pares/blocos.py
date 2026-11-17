"""QF-B: extensão inspirada no Q-FLAIR com BLOCOS de duas features (não é o original nem reprodução fiel).
Bloco em fios vizinhos (q, q+1): T1(θ1·x_k1) em q → T2(θ2·x_k2) em q+1 → E(γ) em (q, q+1), com T ∈ {RX, RY} e E ∈ {RXX, RYY} fixa (γ treinável).
P0 é trigonométrico de grau 1 em cada ângulo → reconstrução tensorial exata com 3×3×3 = 27 avaliações por amostra.
Usa o simulador local e as funções de porta única da rodada 39 (qflair.py, só importado, sha registrado)."""
import importlib.util, hashlib, time
from pathlib import Path
import numpy as np
P39 = Path(__file__).resolve().parent.parent / 'quantico_39_qflair_sintetico' / 'qflair.py'
_s = importlib.util.spec_from_file_location('qflair39', P39); Q = importlib.util.module_from_spec(_s); _s.loader.exec_module(Q)
SHA_Q39 = hashlib.sha256(P39.read_bytes()).hexdigest()

TIPOS_BLOCO = (('RY', 'RY', 'RYY'), ('RY', 'RY', 'RXX'), ('RX', 'RX', 'RXX'), ('RX', 'RX', 'RYY'))
PONTOS = (0.0, np.pi / 2, -np.pi / 2)
MINV = np.linalg.inv(np.array([[1, 1, 0], [1, 0, 1], [1, 0, -1]], float))   # valores nos 3 pontos → coeficientes de [1, cos α, sin α]
TH = np.linspace(-1, 1, 21)                                                  # grade grossa de θ1, θ2 (passo 0,1)
GAM = np.arange(-3, 5) * np.pi / 4                                           # γ ∈ {−3π/4, …, π}: inclui 0 e ±π/2
REF = np.linspace(-0.1, 0.1, 9)                                              # refino local 9×9 de (θ1, θ2) só no melhor par de cada posição


def posicoes_bloco(n): return [(t, (q, q + 1)) for q in range(n - 1) for t in TIPOS_BLOCO]


def base(a): return np.stack((np.ones_like(a), np.cos(a), np.sin(a)), -1)


def aplica_bloco(psi, n, tipo, qs, a1, a2, g):
    psi = Q.aplica(psi, n, tipo[0], (qs[0],), a1); psi = Q.aplica(psi, n, tipo[1], (qs[1],), a2); return Q.aplica(psi, n, tipo[2], qs, g)


def reconstroi_bloco(psi, n, tipo, qs):
    """C[N, u, v, w] com P0(α1, α2, γ) = Σ C·φu(α1)·φv(α2)·φw(γ), φ = [1, cos, sin]. 27 avaliações (estados parciais reaproveitados)."""
    N = psi.shape[0]; F = np.empty((N, 3, 3, 3))
    for i, a in enumerate(PONTOS):
        pa = Q.aplica(psi, n, tipo[0], (qs[0],), a)
        for j, b in enumerate(PONTOS):
            pab = Q.aplica(pa, n, tipo[1], (qs[1],), b)
            for k, g in enumerate(PONTOS): F[:, i, j, k] = Q.p0(Q.aplica(pab, n, tipo[2], qs, g))
    return np.einsum('Nijk,ui,vj,wk->Nuvw', F, MINV, MINV, MINV)


def avalia_coef(C, a1, a2, g):
    """a1, a2: (N,) ou (..., N); g escalar."""
    return np.einsum('Nuvw,...Nu,...Nv,w->...N', C, base(a1), base(a2), base(np.array(g)))


def _perda(y, f):
    p = np.clip(f, Q.EPS, 1 - Q.EPS); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(-1)


def melhor_bloco(y, X, C, pares, chunk=8):
    """Varre pares ordenados (k1, k2) × θ1 × θ2 (grade 21×21) × γ (8); refina (θ1, θ2) 9×9 no melhor par.
    Desempate: ordem dada dos pares, depois a menor (γ, θ1, θ2) na ordem de varredura."""
    Cg = np.einsum('Nuvw,gw->gNuv', C, base(GAM)); best = (np.inf, None)
    for p0 in range(0, len(pares), chunk):
        P = pares[p0:p0 + chunk]; k1 = np.array([p[0] for p in P]); k2 = np.array([p[1] for p in P])
        B1 = base(TH[None, :, None] * X[:, k1].T[:, None, :]); B2 = base(TH[None, :, None] * X[:, k2].T[:, None, :])   # (P, T, N, 3)
        f = np.einsum('ptNu,gNuv,psNv->pgtsN', B1, Cg, B2, optimize=True)                                              # (P, G, T, T, N)
        L = _perda(y, f).reshape(len(P), -1); j = L.argmin(1); m = int(L[np.arange(len(P)), j].argmin())
        if L[m, j[m]] < best[0]:
            g_, t1, t2 = np.unravel_index(j[m], (len(GAM), len(TH), len(TH))); best = (float(L[m, j[m]]), (P[m], float(GAM[g_]), float(TH[t1]), float(TH[t2])))
    (k1, k2), g, t1, t2 = best[1]
    r1 = np.clip(t1 + REF, -1, 1); r2 = np.clip(t2 + REF, -1, 1)
    f = avalia_coef(C, r1[:, None, None] * X[:, k1][None, None], r2[None, :, None] * X[:, k2][None, None], g)
    L = _perda(y, f); i1, i2 = np.unravel_index(int(L.argmin()), L.shape)
    if L[i1, i2] < best[0]: best = (float(L[i1, i2]), ((k1, k2), g, float(r1[i1]), float(r2[i2])))
    return best


def triagem_pares(Z, y, K):
    """Triagem de 2ª ordem SÓ no treino: S_jk = |Σ_i (y_i − ȳ)·z_ij·z_ik| / N, com z padronizado no treino; j < k; top-K com desempate (−S, j, k).
    Não depende de efeito marginal (detecta interação pura); NÃO usa índices verdadeiros."""
    zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0); r = y - y.mean()
    S = np.abs((zs * r[:, None]).T @ zs) / len(y); iu = np.triu_indices(S.shape[0], 1); s = S[iu]
    ordem = np.lexsort((iu[1], iu[0], -s))[:K]
    return [(int(iu[0][o]), int(iu[1][o])) for o in ordem], S


def pares_aleatorios(d, K, rng):
    """K pares não ordenados uniformes sem reposição entre d(d−1)/2 (sem enumerar)."""
    K = min(K, d * (d - 1) // 2); vistos = set(); out = []
    while len(out) < K:                                                  # rejeição: par distinto uniforme → (min, max) uniforme entre os não ordenados
        j, k = rng.choice(d, size=2, replace=False); p = (int(min(j, k)), int(max(j, k)))
        if p not in vistos: vistos.add(p); out.append(p)
    return sorted(out)


def ordenados(pares): return [p for a, b in pares for p in ((a, b), (b, a))]


# ---------------- crescimento QF-B ----------------
def cresce_B(X, y, n, modo_pares, G_ops=12, K=None, semente=0, verdade=(), Ztri=None, tempos=None, so_blocos=False):
    """modo_pares ∈ {'exaustivo', 'triagem', 'aleatorio', 'oraculo', 'nenhum'} ('nenhum' = só portas únicas, pool cheio da 39).
    Em cada passo: candidatos de porta única (pool cheio da 39) e, se restarem ≥ 3 operações, blocos. Critério: maior queda de perda de
    treino POR OPERAÇÃO (bloco = 3), com desempate pela ordem canônica (portas únicas antes de blocos). 'oraculo' é SÓ diagnóstico.
    so_blocos=True (SÓ diagnóstico de representação): nenhuma porta única é candidata."""
    N, d = X.shape; psi = Q.estado_inicial(N, n); f0 = Q.p0(psi); L_cur = Q.logloss(y, f0); ops = 0
    rng = np.random.default_rng([40, n, semente, 7]); portas, curva, info = [], [(0, L_cur)], []
    pares_tri = None
    if modo_pares == 'triagem': pares_tri, _ = triagem_pares(Ztri if Ztri is not None else X, y, K)
    Pq = Q.pool(n); vpar = set(tuple(sorted(p)) for i, p in enumerate([(verdade[a], verdade[b]) for a in range(len(verdade)) for b in range(a + 1, len(verdade))]))
    while ops < G_ops:
        cands = []; t_sim = t_cl = 0.0; n_av = 0
        for i, (tipo, qs) in enumerate([] if so_blocos else Pq['dados']):
            t0 = time.perf_counter(); a, b, c = Q.reconstroi(psi, n, tipo, qs, f0); t_sim += time.perf_counter() - t0
            t0 = time.perf_counter(); L, k, th = Q.melhor_dados(y, X, a, b, c, np.arange(d)); t_cl += time.perf_counter() - t0; n_av += d * 122
            cands.append(((L_cur - L) / 1, (0, i), ('unica', tipo, qs, 'dados', k, th), 1))
        for i, (tipo, qs) in enumerate([] if so_blocos else Pq['fixas']):
            a, b, c = Q.reconstroi(psi, n, tipo, qs, f0); L, th = Q.melhor_fixa(y, a, b, c); n_av += 122
            cands.append(((L_cur - L) / 1, (1, i), ('unica', tipo, qs, 'fixas', None, th), 1))
        for i, (tipo, qs) in enumerate([] if so_blocos else Pq['H']):
            L = Q.logloss(y, Q.p0(Q.aplica(psi, n, 'H', qs))); n_av += 1; cands.append(((L_cur - L) / 1, (2, i), ('unica', 'H', qs, 'H', None, None), 1))
        cobre = None
        if modo_pares != 'nenhum' and G_ops - ops >= 3:
            if modo_pares == 'exaustivo': base_p = [(j, k) for j in range(d) for k in range(j + 1, d)]
            elif modo_pares == 'triagem': base_p = pares_tri
            elif modo_pares == 'aleatorio': base_p = pares_aleatorios(d, K, rng)
            elif modo_pares == 'oraculo': base_p = sorted(vpar)
            pares = ordenados(base_p); cobre = bool(vpar & set(tuple(sorted(p)) for p in base_p)) if vpar else None
            for i, (tipo, qs) in enumerate(posicoes_bloco(n)):
                t0 = time.perf_counter(); C = reconstroi_bloco(psi, n, tipo, qs); t_sim += time.perf_counter() - t0
                t0 = time.perf_counter(); L, (par, g, t1, t2) = melhor_bloco(y, X, C, pares); t_cl += time.perf_counter() - t0
                n_av += len(pares) * len(TH) ** 2 * len(GAM) + len(REF) ** 2
                cands.append(((L_cur - L) / 3, (3, i), ('bloco', tipo, qs, par, g, t1, t2), 3))
        melhor = max(c[0] for c in cands); esc = next(c for c in cands if c[0] == melhor)
        psi = aplica_escolha(psi, n, esc[2], X); f0 = Q.p0(psi)
        assert np.allclose(np.linalg.norm(psi, axis=1), 1, atol=1e-10)
        L_new = Q.logloss(y, f0); assert L_new <= L_cur + 1e-10, (L_new, L_cur)
        L_cur = L_new; ops += esc[3]; portas.append(esc[2]); curva.append((ops, L_cur))
        info.append(dict(ops=ops, tipo=esc[2][0], par=(esc[2][3] if esc[2][0] == 'bloco' else None), cobre_par_verdadeiro=cobre, avaliacoes_amostra_candidato=n_av * N))
        if tempos is not None: tempos.append(dict(sim=t_sim, classico=t_cl))
    return portas, curva, info


def aplica_escolha(psi, n, e, X):
    if e[0] == 'unica':
        _, tipo, qs, cat, k, th = e; return Q.aplica(psi, n, tipo, qs, Q.angulo(cat, k, th, X))
    _, tipo, qs, (k1, k2), g, t1, t2 = e; return aplica_bloco(psi, n, tipo, qs, t1 * X[:, k1], t2 * X[:, k2], g)


def prediz_B(portas, X, n):
    psi = Q.estado_inicial(len(X), n); out = [Q.p0(psi)]
    for e in portas: psi = aplica_escolha(psi, n, e, X); out.append(Q.p0(psi))
    return out
