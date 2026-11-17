#!/usr/bin/env python3
"""Portable commands for the ESANN experiments. Run --help for the modes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import ttest_rel
from esann import data, metrics as M, simulator as Q, search as S, controls as C
from esann.reference import replay

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')

def measure(ytr,ptr,yte,pte):
    threshold,_=M.limiar_roc_treino(ptr,ytr)
    return dict(balanced_accuracy=M.bacc(pte,yte,threshold),BCE=M.bce(yte,pte),AUC=M.auc(yte,pte),threshold=threshold)

def evaluate_seed(seed):
    base=data.artifact(seed); A=np.load(base/'qf/replay_A.npz'); B=np.load(base/'qf/replay_B.npz')
    ytr,yva,yte=A['ytr'],A['yva'],A['yte']
    p,e,_=M.sel_prefixo_P0(B['P0tr_prefixos'],B['P0va_prefixos'],B['P0te_prefixos'],ytr,yva,list(range(1,len(B['P0tr_prefixos']))))
    ptr=np.full(len(ytr),ytr.mean()) if e['const'] else expit(e['a']*(2*B['P0tr_prefixos'][e['k']]-1)+e['b'])
    out={'QF+CAL':measure(ytr,ptr,yte,p),'Q-FLAIR final':measure(ytr,2*A['P0tr_prefixos'][-1]-1,yte,2*A['P0te_final']-1)}
    out['Q-FLAIR final'].update(BCE=M.bce(yte,A['P0te_final']),AUC=M.auc(yte,A['P0te_final']))
    for sub in ('bloco','ablacao'):
        jf=base/sub/(sub+'.json'); nf=base/sub/(sub+'.npz')
        if not nf.exists(): continue
        J=json.loads(jf.read_text()); N=np.load(nf)
        for name,record in J.items():
            if not isinstance(record,dict) or 'escolha' not in record: continue
            e=record['escolha']; p=N[name+'_p_te']
            ptr=np.full(len(ytr),ytr.mean()) if e['const'] else expit(e['a']*(2*N[name+'_P0_tr'][e['k']]-1)+e['b'])
            out[name]=measure(ytr,ptr,yte,p)
    out['CONST']=measure(ytr,np.full(len(ytr),ytr.mean()),yte,np.full(len(yte),ytr.mean()))
    return out

def summarize(out):
    out.mkdir(parents=True,exist_ok=True); records=[]; result={}
    for stage,seeds in [('released',(42,)),('initial',data.INITIAL),('confirmation',data.CONFIRMATION)]:
        rows=[]
        for seed in seeds:
            r=evaluate_seed(seed); rows.append(r)
            records.extend(dict(stage=stage,seed=seed,method=k,**v) for k,v in r.items())
        common=set.intersection(*(set(r) for r in rows))
        result[stage]={'means':{m:{metric:float(np.mean([r[m][metric] for r in rows])) for metric in ('balanced_accuracy','BCE','AUC')} for m in sorted(common)}}
        if stage=='confirmation':
            contrasts={}; signs=np.array(list(itertools.product([-1,1],repeat=5)))
            for metric in ('balanced_accuracy','BCE','AUC'):
                a=np.array([r['BLOCO_RAW'][metric] for r in rows]); b=np.array([r['QF+CAL'][metric] for r in rows]); d=a-b
                contrasts[metric]={'differences_RAW_minus_QF_CAL':d.tolist(),'mean':float(d.mean()),
                    'paired_t_p_unadjusted':float(ttest_rel(a,b).pvalue),
                    'exact_sign_permutation_p':float(np.mean(np.abs((signs*d).mean(1))>=abs(d.mean())-1e-14))}
            result[stage]['posthoc_statistics']=contrasts
            da=np.array(contrasts['balanced_accuracy']['differences_RAW_minus_QF_CAL']); db=np.array(contrasts['BCE']['differences_RAW_minus_QF_CAL'])
            result[stage]['prespecified_criterion']={'per_dataset':((da>=.01)&(db<=0)).tolist(),'count':int(((da>=.01)&(db<=0)).sum()),'mean_pass':bool(da.mean()>=.01 and db.mean()<=0)}
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    save(out/'summary.json',result)
    lines=['# Recomputed results','','Statistics are post-evaluation, descriptive, unadjusted; five datasets.','']
    for stage,r in result.items():
        lines += ['## '+stage,'','| Method | Balanced accuracy | BCE | AUC |','|---|---:|---:|---:|']
        lines += [f'| {m} | {v["balanced_accuracy"]:.5f} | {v["BCE"]:.5f} | {v["AUC"]:.5f} |' for m,v in r['means'].items()]
        lines += ['']
    (out/'TABLES.md').write_text('\n'.join(lines))
    return result

def verify(out):
    checks=[]; manifest=json.loads((ROOT/'provenance/sources.json').read_text())
    for e in manifest:
        if 'sha256' in e:
            p=ROOT/e['destination']; assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],p
    # Independently verify exported scientific definitions against preserved original sources.
    import ast
    for e in manifest:
        if 'unchanged_definitions' not in e: continue
        orig=ROOT/'provenance/original_science'/Path(e['source']).parent.name/Path(e['source']).name
        def funcs(p): return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=funcs(orig),funcs(ROOT/e['destination'])
        assert all(a[n]==b[n] for n in e['unchanged_definitions']),e['destination']
    for seed in data.SEEDS:
        gen=data.generate(seed); old=data.archived_data(seed)
        diff=max(float(np.max(np.abs(gen[k]-old[k]))) for k in old)
        assert diff<=5e-15,(seed,diff)
        for part in ('tr','va','te'): assert np.array_equal(gen['y'+part],old['y'+part])
        base=data.artifact(seed); A=np.load(base/'qf/replay_A.npz'); B=np.load(base/'qf/replay_B.npz')
        replaydiff=max(float(np.max(np.abs(replay(base/'qf/Gate_collect.txt',A['X'+part])-B['P0'+part+'_prefixos']))) for part in ('tr','va','te'))
        assert replaydiff<1e-10,(seed,replaydiff)
        for sub in ('bloco','ablacao'):
            jf=base/sub/(sub+'.json'); nf=base/sub/(sub+'.npz')
            if not nf.exists(): continue
            J=json.loads(jf.read_text()); N=np.load(nf)
            for name,v in J.items():
                if not isinstance(v,dict) or 'portas' not in v: continue
                for part in ('tr','va','te'):
                    d=float(np.max(np.abs(np.asarray(Q.prediz_B(v['portas'],old['X'+part],4))-N[name+'_P0_'+part])))
                    assert d<1e-10,(seed,name,part,d)
        checks.append(dict(seed=seed,data_max_difference=diff,qflair_replay_max_difference=replaydiff))
    r=summarize(out)
    expected={'BLOCO_RAW':(.8381,.36916,.90791),'QF+CAL':(.7828,.43861,.84230),'Q-FLAIR final':(.7832,.45623,.84216),'CONST':(.5,.69331,.5)}
    for method,vals in expected.items():
        got=r['confirmation']['means'][method]
        for metric,value in zip(('balanced_accuracy','BCE','AUC'),vals): assert abs(got[metric]-value)<5.1e-5,(method,metric,got[metric],value)
    assert r['confirmation']['prespecified_criterion']['count']==4
    save(out/'verification.json',dict(ok=True,datasets=checks,scientific_definitions_unchanged=True,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    print('Verified generator, archived circuits, reference replays and paper table for all 11 datasets.')

def rerun(seed,mode,out):
    folder=out/str(seed)
    if folder.exists(): raise FileExistsError(f'Refusing overwrite: {folder}')
    D=data.generate(seed); old=data.archived_data(seed)
    for k in old: assert np.max(np.abs(D[k]-old[k]))<=5e-15
    stage=data.stage(seed)
    modes=['BLOCO_RAW'] if mode=='raw' else ['BLOCO_RAW','BLOCO_CAL']
    if mode=='all' and stage=='confirmation': raise ValueError('CAL/SEQ were not confirmation arms. Use --mode raw.')
    folder.mkdir(parents=True); report={}; arrays={}; seq=None
    for name in modes + (['SEQ','SEQ_REF'] if mode=='all' and stage=='initial' else []):
        cpu=time.process_time(); wall=time.monotonic()
        if name=='SEQ': gates,count,seq=C.constroi_seq(D['Xtr'],D['ytr'])
        elif name=='SEQ_REF': gates,count=C.constroi_ref(D['Xtr'],D['ytr'],seq)
        else: gates,count=S.constroi_bloco(name,D['Xtr'],D['ytr'])
        pred,e,cert=S.final_c1(gates,list(range(2,len(gates)+1,2)),D['Xtr'],D['ytr'],D['Xva'],D['yva'],D['Xte'])
        assert cert is None or S.BA.cert_ok(cert)
        saved=data.artifact(seed)/('ablacao' if name.startswith('SEQ') else 'bloco')
        j=json.loads((saved/(saved.name+'.json')).read_text()); n=np.load(saved/(saved.name+'.npz'))
        p0=np.array(Q.prediz_B(gates,D['Xtr'],4)); ptr=np.full(len(D['ytr']),D['ytr'].mean()) if e['const'] else expit(e['a']*(2*p0[e['k']]-1)+e['b'])
        diff=float(np.max(np.abs(pred-n[name+'_p_te'])))
        # Mac/Linux optimizer rounding can differ; report the actual difference, never overwrite historical evidence.
        historical=j[name]['portas']; same_structure=[(g[1],list(g[2]),g[4]) for g in gates]==[(g[1],list(g[2]),g[4]) for g in historical]
        report[name]=dict(gates=gates,choice=e,counters=count,metrics=measure(D['ytr'],ptr,D['yte'],pred),
            cpu_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,prediction_max_difference=diff,same_structure=same_structure,
            selected_prefix_matches=e['k']==j[name]['escolha']['k'])
        arrays[name+'_p_te']=pred
        print(seed,name,'prediction difference',diff,'same structure',same_structure,flush=True)
    save(folder/'rerun.json',report); np.savez(folder/'predictions.npz',**arrays)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('verify','tables'):
        a=sub.add_parser(name); a.add_argument('--output',type=Path,default=ROOT/'runs'/name)
    a=sub.add_parser('rerun'); a.add_argument('--seed',type=int,choices=data.SEEDS); a.add_argument('--stage',choices=['released','initial','confirmation'])
    a.add_argument('--mode',choices=['raw','cal','all'],default='raw'); a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='verify': verify(args.output)
    elif args.command=='tables': summarize(args.output); print(args.output/'TABLES.md')
    else:
        seeds=([args.seed] if args.seed is not None else ([42] if args.stage=='released' else list(data.INITIAL if args.stage=='initial' else data.CONFIRMATION)))
        if args.seed is None and args.stage is None: p.error('rerun requires --seed or --stage')
        # cal means RAW+CAL, matching the initial evaluation rather than changing its primary label.
        for seed in seeds: rerun(seed,args.mode,args.output)
if __name__=='__main__': main()
