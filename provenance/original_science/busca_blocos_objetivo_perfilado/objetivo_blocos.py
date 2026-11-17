"""busca_blocos_objetivo_perfilado (PROTOCOLO.md). Busca gulosa por blocos SEM estrutura oracle: a cada etapa, feature AINDA NÃO USADA × posição
da biblioteca LOC (RY→RX, 4 fios) ∪ INT (RY→RXX, 6 orientações da linha); dois pesos conjuntos; até 16 blocos / 32 operações.
Braços primários IDÊNTICOS exceto o ALVO: BLOCO_RAW (BCE do P0 bruto) e BLOCO_CAL (J_C1 perfilado: BA.calibra C=1 + BA._J).
Receita por etapa: tensor de 9 pontos por posição (8 consultas, compartilhado entre features) → pré-triagem de TODAS feature×posição com 4 sementes
fixas não oracle → 8 finalistas refinados por L-BFGS-B (2 pesos, [-1,1]², gradiente analítico pelo tensor) → por finalista, melhor de {refino,
semente, (0,0)} → melhor finalista (desempate posição, feature). Nenhuma reotimização de blocos anteriores. O construtor NÃO acessa professor,
features informativas, validação nem teste. Prefixo: checkpoints pares + CONST pela validação com C = 1 fixo (todos os braços).
RAW_FULL32: adaptado congelado (descritivo secundário), mesmo C = 1 e prefixos pares."""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
H = Path(__file__).resolve().parent; EXP = H.parent; XX = EXP / 'blocos_interacao_xx'
sys.path.insert(0, str(XX)); import interacao as IN      # noqa: E402  (congelado; só importado)
T, PR, BA, Q, B, BL = IN.T, IN.PR, IN.BA, IN.Q, IN.B, IN.BL
N_Q = 4; D = 16; ETAPAS = 16; TOL = 1e-12; TAM = (205, 103, 2000); C_FIX = 1.0; EPS = Q.EPS
LIB = IN.LIB                                                                                           # LOC (4) primeiro, depois INT (6)
SEMENTES = np.array([(0.3, 0.2), (-0.3, 0.2), (0.6, -0.4), (-0.6, -0.4)]); N_FINAL = 8
OPC = dict(maxiter=200, maxfun=400, ftol=1e-15, gtol=1e-9, maxcor=10); CAIXA = [(-1.0, 1.0)] * 2
BRACOS = ('BLOCO_RAW', 'BLOCO_CAL', 'RAW_FULL32'); FASES = ('base', 'pretriagem', 'refino', 'direto', 'diagnostico')


def fi(A): return np.stack([np.ones_like(A), np.cos(A), np.sin(A)], -1)
def dfi(A): return np.stack([np.zeros_like(A), -np.sin(A), np.cos(A)], -1)
def p0_tensor(C, A, Bt):
    """C (N,3,3); A, Bt (N,S) ângulos PAREADOS por coluna ⇒ P0 (S, N)."""
    return np.einsum('nsa,nab,nsb->sn', fi(A), C, fi(Bt))
def p0_e_grad_tensor(C, x, w):
    """P0 (N,) e dP0/dw (2, N) para ângulos (w1·x, w2·x) — gradiente analítico do tensor."""
    a, b = w[0] * x, w[1] * x; fa, fb = fi(a), fi(b); Cb = np.einsum('nab,nb->na', C, fb); aC = np.einsum('na,nab->nb', fa, C)
    return np.einsum('na,na->n', fa, Cb), np.stack([x * np.einsum('na,na->n', dfi(a), Cb), x * np.einsum('nb,nb->n', aC, dfi(b))])


class Alvo:
    """RAW: Q.logloss(y, P0) (corte EPS; derivada 0 nas amostras cortadas). CAL: J_C1(P0) = min_{a,b} BA._J([a,b], 2P0−1, y, 1) via BA.calibra
    (rotina congelada), gradiente pelo teorema do envelope. Conta calibrações, CPU e o maior ‖∇_{a,b}‖∞ interno por fase."""
    def __init__(self, braco, y):
        self.cal = braco == 'BLOCO_CAL'; self.y = y; self.n = len(y)
        self.n_calib = {f: 0 for f in FASES}; self.cpu_calib = {f: 0.0 for f in FASES}; self.grad_interno_max = 0.0

    def _calib(self, s, fase):
        c0 = time.process_time(); a, b = BA.calibra(s, self.y, C_FIX); self.cpu_calib[fase] += time.process_time() - c0; self.n_calib[fase] += 1
        self.grad_interno_max = max(self.grad_interno_max, float(np.abs(BA._G(np.r_[a, b], s, self.y, C_FIX)).max())); return a, b

    def valor(self, P, fase):
        if not self.cal: return Q.logloss(self.y, P)
        s = 2 * P - 1; a, b = self._calib(s, fase); return BA._J(np.r_[a, b], s, self.y, C_FIX)

    def valor_grad(self, P, dP, fase):
        y = self.y
        if not self.cal:
            m = (P > EPS) & (P < 1 - EPS); g = np.where(m, -(y / np.clip(P, EPS, 1) - (1 - y) / np.clip(1 - P, EPS, 1)) / self.n, 0.0); return Q.logloss(y, P), dP @ g
        s = 2 * P - 1; a, b = self._calib(s, fase); return BA._J(np.r_[a, b], s, y, C_FIX), dP @ (2 * a * (expit(a * s + b) - y) / self.n)


def j_c1(P, y):
    s = 2 * P - 1; a, b = BA.calibra(s, y, C_FIX); return BA._J(np.r_[a, b], s, y, C_FIX)


def constroi_bloco(braco, Z, y, etapas=ETAPAS, log=None):
    """Construtor dos braços primários (só treino). Mesma receita nos dois; só o Alvo difere."""
    al = Alvo(braco, y); psi = Q.estado_inicial(len(Z), N_Q); portas, usadas = [], set()
    c = dict(consultas=0, tensores=0, aplic_commit=0, avaliacoes_pretriagem=0, refinos=0, refino_nfev=0, refino_nit=0, refino_status_nao_zero=0, refino_limite_atingido=0,
             refino_piorou_semente=0, escolha_por=dict(refino=0, semente=0, zero=0), rank_pretriagem_do_escolhido=[], grad_projetado_escolhido_max=0.0,
             dif_direta_prevista_max=0.0, etapas=0, parou_sem_ganho=False, parou_sem_feature=False, alvo_por_etapa=[], perda_P0_bruta_por_etapa=[], J_C1_por_etapa=[])
    for e in range(etapas):
        livres = [k for k in range(D) if k not in usadas]
        if not livres: c['parou_sem_feature'] = True; break
        f_cur = al.valor(Q.p0(psi), 'base'); pre = []; Cs = {}
        for j, (p1, p2, _) in enumerate(LIB):
            Cs[j] = PR.reconstroi_par(psi, p1, p2); c['consultas'] += 8; c['tensores'] += 1
            for k in livres:
                x = Z[:, k]; P = p0_tensor(Cs[j], x[:, None] * SEMENTES[:, 0], x[:, None] * SEMENTES[:, 1])
                fs = [al.valor(P[s], 'pretriagem') for s in range(len(SEMENTES))]; c['avaliacoes_pretriagem'] += len(SEMENTES)
                sb = int(np.argmin(fs)); pre.append((fs[sb], j, k, sb))
        pre.sort(key=lambda u: (u[0], u[1], u[2])); cands = []
        for rank, (f0, j, k, sb) in enumerate(pre[:N_FINAL]):
            x = Z[:, k]; fun = lambda w, C=Cs[j], x=x: al.valor_grad(*p0_e_grad_tensor(C, x, w), 'refino')
            r = minimize(fun, SEMENTES[sb].copy(), jac=True, method='L-BFGS-B', bounds=CAIXA, options=OPC)
            c['refinos'] += 1; c['refino_nfev'] += int(r.nfev); c['refino_nit'] += int(r.nit); c['refino_status_nao_zero'] += int(r.status != 0); c['refino_limite_atingido'] += int(r.status == 1)
            fr = float(r.fun); c['refino_piorou_semente'] += int(fr > f0)
            opcoes = [(fr, 0, 'refino', np.array(r.x, float)), (f0, 1, 'semente', SEMENTES[sb].copy()), (f_cur, 2, 'zero', np.zeros(2))]
            f, _, por, w = min(opcoes, key=lambda u: (u[0], u[1])); cands.append((f, j, k, w, por, rank, r))
        f, j, k, w, por, rank, r = min(cands, key=lambda u: (u[0], u[1], u[2])); g = f_cur - f
        if g <= TOL: c['parou_sem_ganho'] = True; break
        if por == 'refino': gp = np.asarray(r.jac); c['grad_projetado_escolhido_max'] = max(c['grad_projetado_escolhido_max'], float(np.abs(w - np.clip(w - gp, -1, 1)).max()))
        (t1, q1), (t2, q2), tp = LIB[j]; bl = [('unica', t1, tuple(q1), 'dados', int(k), float(w[0])), ('unica', t2, tuple(q2), 'dados', int(k), float(w[1]))]
        for x_ in bl: psi = B.aplica_escolha(psi, N_Q, x_, Z); c['aplic_commit'] += 1
        P = Q.p0(psi); fd = al.valor(P, 'direto'); c['dif_direta_prevista_max'] = max(c['dif_direta_prevista_max'], abs(fd - f))
        assert fd <= f_cur + 1e-12 and abs(fd - f) < 1e-9, (braco, e, fd, f, f_cur)
        c['escolha_por'][por] += 1; c['rank_pretriagem_do_escolhido'].append(rank); c['alvo_por_etapa'].append(fd)
        c['perda_P0_bruta_por_etapa'].append(Q.logloss(y, P)); t0 = time.process_time(); c['J_C1_por_etapa'].append(j_c1(P, y)); al.cpu_calib['diagnostico'] += time.process_time() - t0; al.n_calib['diagnostico'] += 1
        if log is not None: log.append(dict(etapa=e, f_antes=f_cur, f_previsto=f, f_direto=fd, bloco=bl, tipo=tp, por=por, rank=rank, livres=len(livres)))
        portas += bl; usadas.add(int(k)); c['etapas'] += 1
    c.update(n_calibracoes=al.n_calib, cpu_calibracoes_s=al.cpu_calib, grad_interno_max=al.grad_interno_max); return portas, c


def constroi(braco, Ztr, ytr):
    if braco in ('BLOCO_RAW', 'BLOCO_CAL'): return constroi_bloco(braco, Ztr, ytr)
    if braco == 'RAW_FULL32': return BL.constroi('RAW_FULL32', Ztr, ytr)                             # congelado (descritivo)
    raise ValueError(braco)


def final_c1(portas, ks, Ztr, ytr, Zva, yva, Zte):
    """Regra de BL.final_ck com C = 1 fixo (conjunto {1}): prefixo k ∈ ks e CONST pela BCE de validação; desempate (bce_val, k, C); certificação.
    ks vazio ⇒ CONST (R4)."""
    if not ks: return np.full(len(Zte), ytr.mean()), dict(k=0, C=None, bce_val=BA.bce_p(yva, np.full(len(yva), ytr.mean())), const=True, n_calib=0), None
    sc = lambda Z: [2 * p - 1 for p in B.prediz_B(portas, Z, N_Q)]; Str, Sva = sc(Ztr), sc(Zva); tab = []
    for k in ks: a, b = BA.calibra(Str[k], ytr, C_FIX); tab.append(dict(k=k, C=C_FIX, a=a, b=b, bce_val=BA.bce_p(yva, expit(a * Sva[k] + b))))
    const = BA.bce_p(yva, np.full(len(yva), ytr.mean())); best = min(tab, key=lambda d: (d['bce_val'], d['k'], d['C']))
    if best['bce_val'] >= const: return np.full(len(Zte), ytr.mean()), dict(k=0, C=None, bce_val=const, const=True, n_calib=len(ks)), None
    s_te = 2 * B.prediz_B(portas[:best['k']], Zte, N_Q)[-1] - 1
    return expit(best['a'] * s_te + best['b']), dict(best, const=False, n_calib=len(ks)), BA.certifica(Str[best['k']], ytr, C_FIX, best['a'], best['b'])


def profundidade(portas):
    nv = [0] * N_Q
    for x in portas: qs = tuple(x[2]); d = max(nv[q] for q in qs) + 1; [nv.__setitem__(q, d) for q in qs]
    return max(nv)


def registro(b, portas, conta, e, cert, Ztr, ytr, inf, cpu_c, cpu_p, rel):
    k = e['k']; dd = [x for x in portas if x[3] == 'dados']; P = B.prediz_B(portas, Ztr, N_Q)[-1] if portas else np.ones(len(ytr))
    nz = lambda Ps: sorted({int(x[4]) for x in Ps if x[3] == 'dados' and x[5] != 0.0}); fc = sorted({int(x[4]) for x in dd}); fp = sorted({int(x[4]) for x in portas[:k] if x[3] == 'dados'})
    tb = conta.get('rank_pretriagem_do_escolhido') is not None
    return dict(portas=json.loads(json.dumps(portas, default=float)), escolha={x: e.get(x) for x in ('k', 'C', 'a', 'b', 'bce_val', 'const')}, n_calib_prefixo=e.get('n_calib'),
                certificacao_ok=(cert is None) or BA.cert_ok(cert), cpu_construcao_s=cpu_c, cpu_prefixo_s=cpu_p, relogio_braco_s=rel, n_operacoes=len(portas), n_operacoes_prefixo=k,
                profundidade_circuito=profundidade(portas), profundidade_prefixo=profundidade(portas[:k]),
                perda_treino_P0_circuito=Q.logloss(ytr, P), J_C1_treino_circuito=j_c1(P, ytr) if portas else None,
                features_distintas_circuito=fc, features_distintas_prefixo=fp, features_peso_nao_nulo_circuito=nz(portas), features_peso_nao_nulo_prefixo=nz(portas[:k]),
                informativas_usadas_circuito_DESCRITIVO=sorted(set(fc) & set(inf)), informativas_usadas_prefixo_DESCRITIVO=sorted(set(fp) & set(inf)),
                pesos_zero_circuito=sum(x[5] == 0.0 for x in dd), tipos_porta={tp: sum(x[1] == tp for x in portas) for tp in ('RX', 'RY', 'RXX', 'RYY', 'H')},
                portas_2q=sum(len(x[2]) == 2 for x in portas), portas_2q_prefixo=sum(len(x[2]) == 2 for x in portas[:k]), blocos_INT_circuito=sum(x[1] == 'RXX' for x in portas) if tb else None,
                **{c_: v for c_, v in conta.items()})


def trabalho(t, g, ns, tam=TAM, perturba=None, bracos=BRACOS):
    Ztr, ytr, Zva, yva, Zte, yte = T.dados(t, g, ns, tam)
    if perturba == 'teste': Zte = np.clip(Zte * 0.5 + 0.3, 0, np.pi)
    if perturba == 'val': Zva = np.clip(Zva * 0.5 + 0.3, 0, np.pi)
    reg = dict(tarefa=T.TAREFAS[t], t=t, g=g, ns=ns, tam=list(tam)); arr = {}
    for b in bracos:
        c0 = time.process_time(); w0 = time.time(); portas, conta = constroi(b, Ztr, ytr); cpu_c = time.process_time() - c0
        c1 = time.process_time(); ks = list(range(2, len(portas) + 1, 2)); p, e, cert = final_c1(portas, ks, Ztr, ytr, Zva, yva, Zte); cpu_p = time.process_time() - c1
        reg[b] = (b, portas, conta, e, cert, cpu_c, cpu_p, time.time() - w0); arr[f'{b}_p_te'] = p
    inf = sorted({int(x[4]) for x in (T.professor(t) or []) if x[3] == 'dados'}); reg['informativas_professor_DESCRITIVO'] = inf      # só depois de construir/selecionar
    for b in bracos: b_, portas, conta, e, cert, cpu_c, cpu_p, rel = reg[b]; reg[b] = registro(b, portas, conta, e, cert, Ztr, ytr, inf, cpu_c, cpu_p, rel)
    arr['BAYES_p_te'] = IN.bayes(t, T.gera(t, g, 2, ns, tam)[0], ns); arr['CONST_p_te'] = np.full(len(yte), ytr.mean()); arr['y_te'] = yte; return arr, reg
