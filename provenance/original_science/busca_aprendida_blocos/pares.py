"""busca_aprendida_blocos (PROTOCOLO.md): avaliação de PARES ordenados (u1, u2) por reconstrução tensorial 3×3 (9 pontos por par de posições) e
grade conjunta 21×21 + refino 11×11 dos pesos (log-loss do P0 bruto, no treino). Braços PARES_IA (ExtraTrees no descritor do par) e PARES_ALEAT, com
o MESMO pool amostrado S_e por etapa. Importa busca_aprendida (anterior) sem modificação."""
import sys, time, hashlib
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(EXP / 'busca_aprendida_temporal'))
import busca_aprendida as BA                                                  # noqa: E402
Q, B, D, N_Q = BA.Q, BA.B, BA.D, BA.N_Q
ETAPAS = 6; M_PAR = 32; N_EXPL = 8; N_POOL = 4096; GRADE = np.round(np.linspace(-1, 1, 21), 10); REF = np.linspace(-0.1, 0.1, 11)
NOS = np.array([0.0, np.pi / 2, -np.pi / 2]); MPHI = np.stack([np.ones(3), np.cos(NOS), np.sin(NOS)], 1); MINV = np.linalg.inv(MPHI)
CONTA = dict(pares_avaliados=0, pares_posicao_reconstruidos=0, simulacoes=0)


def unidades(Pq): return [(0, i, k) for i in range(len(Pq['dados'])) for k in range(D)] + [(1, i, -1) for i in range(len(Pq['fixas']))]
def posicao(u, Pq): return (Pq['dados'], Pq['fixas'])[u[0]][u[1]]


def reconstroi_par(psi, p1, p2):
    """C (N, 3, 3) tal que P0(α, β) = φ(α)ᵀ C φ(β); 8 simulações novas."""
    est = [psi] + [Q.aplica(psi, N_Q, p1[0], p1[1], a) for a in NOS[1:]]; CONTA['simulacoes'] += 2; P = np.zeros((psi.shape[0], 3, 3))
    for i, s in enumerate(est):
        for j, b in enumerate(NOS):
            P[:, i, j] = Q.p0(s) if j == 0 else Q.p0(Q.aplica(s, N_Q, p2[0], p2[1], b))
            CONTA['simulacoes'] += int(j != 0)
    return np.einsum('ai,nij,bj->nab', MINV, P, MINV)


def p0_de(C, A, Bt):
    """A: (N, T1), Bt: (N, T2) ângulos por amostra → P0 (T1, T2, N)."""
    fa = np.stack([np.ones_like(A), np.cos(A), np.sin(A)], -1); fb = np.stack([np.ones_like(Bt), np.cos(Bt), np.sin(Bt)], -1)
    return np.einsum('nsa,nab,ntb->stn', fa, C, fb)


def _perda(y, P): P = np.clip(P, Q.EPS, 1 - Q.EPS); return -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean(-1)


def ang(u, Z, th):
    th = np.asarray(th, float); return (th[None, :] * Z[:, u[2]][:, None]) if u[0] == 0 else np.broadcast_to(th[None, :] * np.pi, (len(Z), len(th)))


def avalia_par(u1, u2, C, Z, y):
    """Melhor (L, θ1, θ2) na grade 21×21 (com (0,0)) + refino 11×11 — ótimo global NA GRADE, não contínuo."""
    CONTA['pares_avaliados'] += 1; L = _perda(y, p0_de(C, ang(u1, Z, GRADE), ang(u2, Z, GRADE))); i, j = np.unravel_index(np.argmin(L), L.shape); t1, t2 = GRADE[i], GRADE[j]
    r1, r2 = np.clip(t1 + REF, -1, 1), np.clip(t2 + REF, -1, 1); L2 = _perda(y, p0_de(C, ang(u1, Z, r1), ang(u2, Z, r2))); i2, j2 = np.unravel_index(np.argmin(L2), L2.shape)
    if L2[i2, j2] < L[i, j]: return float(L2[i2, j2]), float(r1[i2]), float(r2[j2])
    return float(L[i, j]), float(t1), float(t2)


def porta(u, th, Pq):
    tipo, qs = posicao(u, Pq); return ('unica', tipo, qs, 'dados', int(u[2]), float(th)) if u[0] == 0 else ('unica', tipo, qs, 'fixas', None, float(th))


def descritor_par(u1, u2, Pq, ctx, zs, rc):
    d1, d2 = BA.descritor(u1, Pq, ctx), BA.descritor(u2, Pq, ctx); p1, p2 = posicao(u1, Pq), posicao(u2, Pq)
    mesmo_fio = float(bool(set(p1[1]) & set(p2[1]))); mesma_f = float(u1[0] == 0 and u2[0] == 0 and u1[2] == u2[2]); mesmo_eixo = float(p1 == p2)
    prod = float(abs(np.mean(zs[:, u1[2]] * zs[:, u2[2]] * rc)) / max(rc.std(), 1e-12)) if (u1[0] == 0 and u2[0] == 0) else 0.0
    return np.r_[d1, d2, mesmo_fio, mesma_f, mesmo_eixo, prod]


def constroi_pares(Ztr, ytr, braco, semente, semente_pool):
    """Recebe SÓ o treino. braco ∈ {PARES_IA, PARES_ALEAT}. semente_pool = [ns, t, g] (pool S_e idêntico entre braços)."""
    from sklearn.ensemble import ExtraTreesRegressor
    N = len(Ztr); Pq = Q.pool(N_Q); U = unidades(Pq); nU = len(U); psi = Q.estado_inicial(N, N_Q); f0 = Q.p0(psi); L_cur = Q.logloss(ytr, f0); rng = np.random.default_rng(semente)
    zs = (Ztr - Ztr.mean(0)) / np.where(Ztr.std(0) > 0, Ztr.std(0), 1); ctx = dict(uso=np.zeros(D), ultimo=np.full(D, -1), prof=np.zeros(N_Q), ult_tipo=[None] * N_Q)
    Xh, yh, portas, log = [], [], [], []; cpu_ia = 0.0; c0 = dict(CONTA)
    for e in range(ETAPAS):
        rp = np.random.default_rng(list(semente_pool) + [31, e]); flat = rp.choice(nU * nU, N_POOL * 2, replace=False); flat = flat[flat // nU != flat % nU][:N_POOL]; S = [(int(a // nU), int(a % nU)) for a in flat]
        S_sha = hashlib.sha256(np.asarray(S, np.int64).tobytes()).hexdigest()[:16]
        r = ytr - f0; rc = r - r.mean(); ctx.update(g=2 * e, L=L_cur, corr=np.abs(zs.T @ rc) / (N * max(rc.std(), 1e-12)))
        if braco == 'PARES_ALEAT' or e == 0: idx = sorted(rng.choice(N_POOL, M_PAR, replace=False).tolist())
        else:
            t0 = time.process_time(); DX = np.array([descritor_par(U[a], U[b], Pq, ctx, zs, rc) for a, b in S])
            mdl = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=int(rng.integers(2 ** 31))).fit(np.array(Xh), np.array(yh)); pred = mdl.predict(DX); cpu_ia += time.process_time() - t0
            expl = rng.choice(N_POOL, N_EXPL, replace=False).tolist(); top = [j for j in np.lexsort((np.arange(N_POOL), -pred)) if j not in expl][:M_PAR - N_EXPL]; idx = sorted(set(expl) | set(int(j) for j in top))
        cache = {}; aval = []
        for j in idx:
            a, b = S[j]; u1, u2 = U[a], U[b]; key = (posicao(u1, Pq), posicao(u2, Pq))
            if key not in cache: cache[key] = reconstroi_par(psi, *key); CONTA['pares_posicao_reconstruidos'] += 1
            L, t1, t2 = avalia_par(u1, u2, cache[key], Ztr, ytr); aval.append((L_cur - L, (a, b), t1, t2))
            if braco == 'PARES_IA': Xh.append(descritor_par(u1, u2, Pq, ctx, zs, rc)); yh.append(L_cur - L)
        ganho, (a, b), t1, t2 = sorted(aval, key=lambda x: (-x[0], x[1]))[0]
        if ganho <= 0: log.append(dict(etapa=e, S_sha=S_sha, sel=idx, avaliados=len(idx), par=None, ganho=0.0, L=float(L_cur), identidade=True)); continue
        e1, e2 = porta(U[a], t1, Pq), porta(U[b], t2, Pq); psi = B.aplica_escolha(B.aplica_escolha(psi, N_Q, e1, Ztr), N_Q, e2, Ztr); CONTA['simulacoes'] += 2
        f0 = Q.p0(psi); L_new = Q.logloss(ytr, f0); assert L_new <= L_cur - ganho + 1e-9, (L_new, L_cur, ganho)
        L_cur = L_new; portas += [e1, e2]; log.append(dict(etapa=e, S_sha=S_sha, sel=idx, avaliados=len(idx), par=[list(U[a]), list(U[b])], th=[t1, t2], ganho=float(ganho), L=float(L_cur), identidade=False, eixos_iguais_adjacentes=bool(posicao(U[a], Pq) == posicao(U[b], Pq))))
        for x in (e1, e2):
            if x[3] == 'dados': ctx['uso'][x[4]] += 1; ctx['ultimo'][x[4]] = 2 * e
            for q in x[2]: ctx['prof'][q] += 1; ctx['ult_tipo'][q] = x[1]
    return portas, dict(log=log, etapas_identidade=sum(x['identidade'] for x in log), pares_avaliados=CONTA['pares_avaliados'] - c0['pares_avaliados'],
                        pares_posicao_reconstruidos=CONTA['pares_posicao_reconstruidos'] - c0['pares_posicao_reconstruidos'], simulacoes=CONTA['simulacoes'] - c0['simulacoes'], cpu_ia_s=cpu_ia,
                        fusoes_eixo_adjacente=sum(x.get('eixos_iguais_adjacentes', False) for x in log), recursos=BA.S.recursos(portas))
