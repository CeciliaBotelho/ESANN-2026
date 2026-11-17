"""Ablação da decisão conjunta (PLANO.md §2). BLOCO = objetivo_blocos.constroi_bloco CONGELADO (só importado). Mesma admissibilidade nos braços:
cada feature nova ocupa 2 slots consecutivos (1ª porta RY_q, 2ª porta da MESMA posição da LIB original: RX_q ou RXX adjacente com q), sem
reuso de feature, até 16 etapas / 32 operações, mesmo alvo (BLOCO_CAL: J_C1 perfilado, C = 1), mesmas sementes, 8 finalistas, L-BFGS-B, caixa [-1,1].
  SEQ     : sub-etapa 1 escolhe (k, RY_q) e w1 pelo ganho da porta SOZINHA (1D); sub-etapa 2, com k e q fixos, escolhe a 2ª porta entre as
            posições da LIB com 1ª porta RY_q e w2 (1D), w1 congelado. Para se o par não reduzir o alvo (> TOL), como o BLOCO.
  SEQ_REF : as MESMAS escolhas (k, posição) do SEQ, na mesma ordem; em cada etapa (w1, w2) reotimizados JUNTOS (2D) a partir dos valores do SEQ,
            escolhendo o melhor de {refino 2D, ponto de partida, (0,0)} como no BLOCO.
Reconstrução de 1 ângulo: P0(α) = c0 + c1 cos α + c2 sin α (3 pontos, os mesmos nós do par)."""
import sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
EXP = Path(__file__).resolve().parent.parent; sys.dont_write_bytecode = True
sys.path.insert(0, str(EXP / 'busca_blocos_objetivo_perfilado')); import objetivo_blocos as OB      # noqa: E402
Q, PR, B = OB.Q, OB.PR, OB.B; N_Q, D, ETAPAS, TOL = OB.N_Q, OB.D, OB.ETAPAS, OB.TOL
LIB = OB.LIB; SEM, N_FINAL, OPC = OB.SEMENTES, OB.N_FINAL, OB.OPC
S1, S2 = SEM[:, 0], SEM[:, 1]                                                                       # coordenadas das sementes do par (1D)


def recon1(psi, g):
    """c (N, 3) com P0(α) = c·φ(α); 2 simulações novas."""
    P = np.stack([Q.p0(psi)] + [Q.p0(Q.aplica(psi, N_Q, g[0], g[1], a)) for a in PR.NOS[1:]], 1); return P @ PR.MINV.T


def p0g_1(c, x, w):
    a = w * x; P = np.einsum('na,na->n', c, OB.fi(a)); return P, (x * np.einsum('na,na->n', c, OB.dfi(a)))[None]


def _escolhe_1d(al, f_cur, cands, fase):
    """cands: lista de (c, x, sementes, rotulo). Pré-triagem por sementes → 8 finalistas → L-BFGS-B 1D → melhor de {refino, semente, 0}."""
    pre = []
    for i, (c, x, ss, rot) in enumerate(cands):
        P = np.einsum('na,sna->sn', c, OB.fi(np.outer(ss, x)))
        fs = [al.valor(P[s], 'pretriagem') for s in range(len(ss))]; sb = int(np.argmin(fs)); pre.append((fs[sb], i, sb))
    pre.sort(key=lambda u: (u[0], u[1])); out = []; cont = dict(pretriagem=len(pre) * len(cands[0][2]) if cands else 0, refinos=0, nfev=0)
    for f0, i, sb in pre[:N_FINAL]:
        c, x, ss, rot = cands[i]
        def fun(w, c=c, x=x):
            P, dP = p0g_1(c, x, w[0]); v, g = al.valor_grad(P, dP, 'refino'); return v, np.atleast_1d(g)
        r = minimize(fun, np.array([ss[sb]]), jac=True, method='L-BFGS-B', bounds=[(-1.0, 1.0)], options=OPC); cont['refinos'] += 1; cont['nfev'] += int(r.nfev)
        f, _, w = min([(float(r.fun), 0, float(r.x[0])), (f0, 1, float(ss[sb])), (f_cur, 2, 0.0)], key=lambda u: (u[0], u[1])); out.append((f, i, w))
    f, i, w = min(out, key=lambda u: (u[0], u[1])); return f, i, w, cont


def constroi_seq(Z, y, etapas=ETAPAS):
    al = OB.Alvo('BLOCO_CAL', y); psi = Q.estado_inicial(len(Z), N_Q); portas, usadas, escolhas = [], set(), []
    c = dict(consultas=0, avaliacoes_pretriagem=0, refinos=0, refino_nfev=0, etapas=0, parou_sem_ganho=False, cpu_s=0.0); t0 = time.process_time()
    for e in range(etapas):
        livres = [k for k in range(D) if k not in usadas]
        if not livres: break
        f_cur = al.valor(Q.p0(psi), 'base')
        c1 = {q: recon1(psi, ('RY', (q,))) for q in range(N_Q)}; c['consultas'] += 2 * N_Q
        cands = [(c1[q], Z[:, k], S1, (q, k)) for q in range(N_Q) for k in livres]
        f1, i1, w1, ct = _escolhe_1d(al, f_cur, cands, 'sub1'); q, k = cands[i1][3]
        for kk in ('pretriagem', 'refinos', 'nfev'): c[{'pretriagem': 'avaliacoes_pretriagem', 'refinos': 'refinos', 'nfev': 'refino_nfev'}[kk]] += ct[kk]
        psi1 = Q.aplica(psi, N_Q, 'RY', (q,), w1 * Z[:, k]); f1d = al.valor(Q.p0(psi1), 'direto')
        js = [j for j, (p1, p2, _) in enumerate(LIB) if p1 == ('RY', (q,))]
        c2 = {j: recon1(psi1, LIB[j][1]) for j in js}; c['consultas'] += 2 * len(js)
        f2, i2, w2, ct = _escolhe_1d(al, f1d, [(c2[j], Z[:, k], S2, j) for j in js], 'sub2'); j = js[i2]
        for kk in ('pretriagem', 'refinos', 'nfev'): c[{'pretriagem': 'avaliacoes_pretriagem', 'refinos': 'refinos', 'nfev': 'refino_nfev'}[kk]] += ct[kk]
        if f_cur - f2 <= TOL: c['parou_sem_ganho'] = True; break
        (t1, q1), (t2, q2), _ = LIB[j]; bl = [('unica', t1, tuple(q1), 'dados', int(k), float(w1)), ('unica', t2, tuple(q2), 'dados', int(k), float(w2))]
        for x_ in bl: psi = B.aplica_escolha(psi, N_Q, x_, Z)
        fd = al.valor(Q.p0(psi), 'direto'); assert abs(fd - f2) < 1e-9 and fd <= f_cur + 1e-12, ('SEQ', e, fd, f2, f_cur)
        portas += bl; usadas.add(int(k)); escolhas.append((int(k), int(j), float(w1), float(w2))); c['etapas'] += 1
    c['cpu_s'] = time.process_time() - t0; return portas, c, escolhas


def constroi_ref(Z, y, escolhas):
    """Mesmas (k, j) do SEQ; (w1, w2) reotimizados juntos em 2D a partir dos valores do SEQ, no estado corrente."""
    al = OB.Alvo('BLOCO_CAL', y); psi = Q.estado_inicial(len(Z), N_Q); portas = []
    c = dict(consultas=0, refinos=0, refino_nfev=0, escolha_por=dict(refino=0, partida=0, zero=0), etapas=0, cpu_s=0.0); t0 = time.process_time()
    for k, j, w1, w2 in escolhas:
        f_cur = al.valor(Q.p0(psi), 'base'); p1, p2, _ = LIB[j]; C = PR.reconstroi_par(psi, p1, p2); c['consultas'] += 8; x = Z[:, k]
        fun = lambda w, C=C, x=x: al.valor_grad(*OB.p0_e_grad_tensor(C, x, w), 'refino'); w0 = np.array([w1, w2])
        r = minimize(fun, w0.copy(), jac=True, method='L-BFGS-B', bounds=OB.CAIXA, options=OPC); c['refinos'] += 1; c['refino_nfev'] += int(r.nfev)
        f0 = al.valor(OB.p0_tensor(C, (x * w1)[:, None], (x * w2)[:, None])[0], 'refino')
        f, _, por, w = min([(float(r.fun), 0, 'refino', np.array(r.x, float)), (f0, 1, 'partida', w0), (f_cur, 2, 'zero', np.zeros(2))], key=lambda u: (u[0], u[1]))
        (t1, q1), (t2, q2), _ = LIB[j]; bl = [('unica', t1, tuple(q1), 'dados', int(k), float(w[0])), ('unica', t2, tuple(q2), 'dados', int(k), float(w[1]))]
        for x_ in bl: psi = B.aplica_escolha(psi, N_Q, x_, Z)
        fd = al.valor(Q.p0(psi), 'direto'); assert abs(fd - f) < 1e-9, ('REF', fd, f)
        portas += bl; c['escolha_por'][por] += 1; c['etapas'] += 1
    c['cpu_s'] = time.process_time() - t0; return portas, c
