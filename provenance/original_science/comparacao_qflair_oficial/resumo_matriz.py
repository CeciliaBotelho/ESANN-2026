"""Resumo da matriz (PROTOCOLO_FINAL.md; critérios PREFIXADOS). Uso: python resumo_matriz.py [PASTA=matriz]   (venv do projeto)
Por trabalho concluído (CONCLUIDO.sha válido):
  QF_ENTREGUE : P0 bruto do circuito oficial final (qiskit, replay_A) como probabilidade; acurácia balanceada OFICIAL = última linha de ACC_test.txt.
  QF+CAL      : mesmas portas oficiais; prefixo k ∈ 1..n e CONST pela BCE de validação, C = 1, desempate (bce_val, k), certificação — sel_prefixo_P0
                (mesma regra de final_c1), sobre P0 por prefixo do nosso simulador (paridade exigida com o qiskit em replay_B).
  BLOCO_CAL   : previsões do bloco_job (receita congelada); BLOCO_RAW descritivo; CONST (prevalência do treino); BAYES (oracle, descritivo).
Critérios (INTER, BCE de teste): piso CONST − braço ≥ 0,020 (BLOCO_CAL e QF+CAL); PRIMÁRIO Δ = BCE(BLOCO_CAL) − BCE(QF+CAL) média ≤ −0,010
e Δ < 0 em ≥ 8/10; NULO: cada braço média ≥ CONST − 0,010 e cada Δ_g > −0,020. IC t pareado só descritivo. Comparação de MÉTODOS."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import t as tdist, rankdata
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent / 'blocos_interacao_xx')); import interacao as IN      # noqa: E402
BA = IN.BA; C_FIX = 1.0
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
ic = lambda d: [float(d.mean() - tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))), float(d.mean() + tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))] if len(d) > 1 else None
if __name__ == '__main__':
    O = H / (sys.argv[1] if len(sys.argv) > 1 else 'matriz'); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); J = {}
    for D in sorted(p.parent for p in O.glob('*/CONCLUIDO.sha')):
        cs = json.loads((D / 'CONCLUIDO.sha').read_text()); assert all(sha(D / k) == v for k, v in cs.items()), D
        A = np.load(D / 'replay_A.npz'); Bz = np.load(D / 'replay_B.npz'); bl = np.load(D / 'bloco.npz'); v = json.loads((D / 'vigia.json').read_text())
        ytr, yva, yte = A['ytr'], A['yva'], A['yte']; assert np.array_equal(yte, bl['y_te']), 'rótulos de teste divergem entre oficial e BLOCO'
        pq, eq, cert = sel_prefixo_P0(Bz['P0tr_prefixos'], Bz['P0va_prefixos'], Bz['P0te_prefixos'], ytr, yva, list(range(1, Bz['P0tr_prefixos'].shape[0])))
        acc = [l.split('\t') for l in (D / 'ACC_test.txt').read_text().splitlines()[1:]]
        lin = dict(QF_ENTREGUE=A['P0te_final'], QF_CAL=pq, BLOCO_CAL=bl['BLOCO_CAL_p_te'], BLOCO_RAW=bl['BLOCO_RAW_p_te'], CONST=bl['CONST_p_te'], BAYES=bl['BAYES_p_te'])
        J[D.name] = dict(tarefa=D.name.split('_')[0], m={k: dict(bce=bce(yte, p), auc=auc(yte, p) if np.std(p) > 0 else None) for k, p in lin.items()},
                         acc_balanceada_oficial_QF=float(acc[-1][2]), n_portas_oficial=len(v['portas']), motivo_treino=v['motivo'], QF_CAL_escolha=eq, QF_CAL_cert_ok=(cert is None) or BA.cert_ok(cert))
    S = dict(trabalhos=sorted(J), completo=sum(x['tarefa'] == 'INTER' for x in J.values()) == 10 and sum(x['tarefa'] == 'NULO' for x in J.values()) == 5)
    for tf in ('INTER', 'NULO'):
        G = [J[k] for k in sorted(J) if J[k]['tarefa'] == tf]
        if not G: continue
        V = lambda l: np.array([x['m'][l]['bce'] for x in G])
        S[tf] = dict(medias_bce={l: float(V(l).mean()) for l in ('QF_ENTREGUE', 'QF_CAL', 'BLOCO_CAL', 'BLOCO_RAW', 'CONST', 'BAYES')}, por_trabalho={k: J[k] for k in sorted(J) if J[k]['tarefa'] == tf})
        d = V('BLOCO_CAL') - V('QF_CAL'); S[tf]['delta_BLOCO_CAL_menos_QF_CAL'] = dict(media=float(d.mean()), BLOCO_melhor=int((d < 0).sum()), QF_melhor=int((d > 0).sum()), ic95_t=ic(d))
        S[tf]['descritivo_BLOCO_CAL_menos_QF_ENTREGUE'] = dict(media=float((V('BLOCO_CAL') - V('QF_ENTREGUE')).mean()), ic95_t=ic(V('BLOCO_CAL') - V('QF_ENTREGUE')))
    if 'INTER' in S:
        GI = [J[k] for k in J if J[k]['tarefa'] == 'INTER']; piso = {b: float(np.mean([x['m']['CONST']['bce'] - x['m'][b]['bce'] for x in GI])) for b in ('BLOCO_CAL', 'QF_CAL')}
        S['piso'] = dict(CONST_menos_braco=piso, ok=all(x >= 0.020 for x in piso.values())); dI = S['INTER']['delta_BLOCO_CAL_menos_QF_CAL']
        S['PRIMARIO'] = dict(dI, criterio='INCOMPLETO' if not S['completo'] else ('NAO_INTERPRETAVEL_PISO' if not S['piso']['ok'] else ('ATINGIDO' if dI['media'] <= -0.010 and dI['BLOCO_melhor'] >= 8 else 'NAO_ATINGIDO')))
    if 'NULO' in S:
        GN = [J[k] for k in J if J[k]['tarefa'] == 'NULO']
        S['NULO_guarda'] = {b: dict(media_menos_CONST=float(np.mean([x['m'][b]['bce'] - x['m']['CONST']['bce'] for x in GN])), ok=bool(np.mean([x['m'][b]['bce'] - x['m']['CONST']['bce'] for x in GN]) >= -0.010
                                    and all(x['m'][b]['bce'] - x['m']['CONST']['bce'] > -0.020 for x in GN))) for b in ('BLOCO_CAL', 'QF_CAL', 'QF_ENTREGUE')}
    (O / 'RESUMO_MATRIZ.json').write_text(json.dumps(S, indent=1, default=float)); print(json.dumps({k: S.get(k) for k in ('completo', 'piso', 'PRIMARIO', 'NULO_guarda')}, default=float))
