#!/usr/bin/env python3
"""Run Q-FLAIR locally: archived-circuit replay or a fresh capped reference fit.

Use the separate Python 3.11 environment in requirements-qflair.txt.
No provider is instantiated and no network/hardware execution is requested.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import difflib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from esann import data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=data.SEEDS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--saved-circuit',action='store_true',help='Replay the archived trained reference; does not retrain it.')
    args=parser.parse_args(); dest=args.output.resolve()
    if dest.exists(): parser.error('Output directory exists; use a fresh directory.')
    # Fail before creating an incomplete job if the legacy environment is unavailable.
    import qiskit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    import sklearn
    assert sklearn.__version__=='1.6.0','Use requirements-qflair.txt (scikit-learn 1.6.0).'
    src=ROOT/'third_party/qflair/executed/QNN'
    dest.mkdir(parents=True); t0=time.monotonic(); cpu=time.process_time()
    D=data.archived_data(args.seed)
    for short,long in [('tr','train'),('va','val'),('te','test')]:
        np.savetxt(dest/f'X{long}.csv',np.column_stack([D['X'+short],np.zeros(len(D['X'+short]))]),delimiter=',',fmt='%.17g')
        np.savetxt(dest/f'Y{long}.csv',D['y'+short],delimiter=',',fmt='%.17g')
    params='\n'.join([f'{key}   {key}.csv' for key in ['Xtest','Xtrain','Xval','Ytest','Ytrain','Yval']]+[
        'seed    12345','gateList   Rxx_x,Ryy_x,U2_x,U3_x,H_y','kernelFile  Kernel_collect.txt',
        'gateFile  Gate_collect.txt','costFile  Cost_collect.txt','parallel  2','accOut  ACC_train.txt'])+'\n'
    (dest/'params').write_text(params)
    if args.saved_circuit:
        for name in ('Gate_collect.txt','Cost_collect.txt'):
            shutil.copyfile(data.artifact(args.seed)/'qf'/name,dest/name)
    else:
        # Unlike the historical external watcher, stop synchronously after the same
        # 32nd complete gate/cost record. The proposal and optimization loop is unchanged.
        runtime=dest/'runtime_QNN'; shutil.copytree(src,runtime)
        f=runtime/'QNN_logLoss.py'; old=f.read_text()
        marker='\t## Check if convergence (i.e., no substantial improvement in cost)'
        assert old.count(marker)==1
        new=old.replace(marker,'\t# Packaging-only synchronous equivalent of the historical external cap.\n\tif count >= 32:\n\t\tbreak\n\n'+marker)
        f.write_text(new)
        (dest/'runtime_cap.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='executed/QNN_logLoss.py',tofile='runtime_QNN/QNN_logLoss.py')))
        with (dest/'training.log').open('w') as log:
            subprocess.run([sys.executable,'-B',str(f),'-p','params'],cwd=dest,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Original preprocessing includes a second min-max pass and the constant pi channel.
    sys.path.insert(0,str(src)); import QNN_in_out, QNN_circuit
    os.chdir(dest); sys.argv=['replay','-p','params']
    tt=QNN_in_out.read_data()[0]
    X={'te':np.array(tt[0],float),'tr':np.array(tt[1],float),'va':np.array(tt[2],float)}
    Y={'te':np.array(tt[3],float),'tr':np.array(tt[4],float),'va':np.array(tt[5],float)}
    for part in X: X[part][:,-1]=np.pi; Y[part][Y[part]==-1]=0
    param=ParameterVector('x',X['tr'].shape[1]); qc=QNN_circuit.initializeMainCirc(4)
    def p0(xs):
        values=[]
        for row in xs:
            circuit=qc.copy()
            for j in range(xs.shape[1]):
                try: circuit.assign_parameters({param[j]:row[j]},inplace=True)
                except Exception: pass
            values.append(Statevector(circuit).probabilities()[0])
        return np.array(values)
    Ptr=[np.ones(len(X['tr']))]
    lines=(dest/'Gate_collect.txt').read_text().splitlines()
    for line in lines:
        e=line.split('\t'); gs=e[:-2]; weight=float(e[-2]); feature=int(e[-1])
        mod=['111111' if gate=='111111' else f'{gate.split("_")[0]}_{feature}' for gate in gs]
        qc.compose(QNN_circuit.circ_convertSinge(mod,1,[weight]*4,4,param),qubits=list(range(4)),inplace=True)
        Ptr.append(p0(X['tr']))
    from esann.reference import replay
    B={part:replay(dest/'Gate_collect.txt',X[part]) for part in X}
    maxdiff=max(float(np.max(np.abs(B['tr']-np.array(Ptr)))),float(np.max(np.abs(B['va'][-1]-p0(X['va'])))),float(np.max(np.abs(B['te'][-1]-p0(X['te'])))))
    assert maxdiff<=1e-10,maxdiff
    from esann import metrics as M
    from scipy.special import expit
    pred,choice,cert=M.sel_prefixo_P0(B['tr'],B['va'],B['te'],Y['tr'],Y['va'],list(range(1,len(B['tr']))))
    ptr=np.full(len(Y['tr']),Y['tr'].mean()) if choice['const'] else expit(choice['a']*(2*B['tr'][choice['k']]-1)+choice['b'])
    threshold,_=M.limiar_roc_treino(ptr,Y['tr'])
    np.savez(dest/'replay.npz',P0tr_prefixos=B['tr'],P0va_prefixos=B['va'],P0te_prefixos=B['te'],p_te=pred,y_te=Y['te'])
    r=dict(seed=args.seed,mode='saved_circuit_replay' if args.saved_circuit else 'fresh_reference_training',
           quantum_evaluations='Exact Qiskit statevector, no hardware/shots',operations=len(lines),qiskit_numpy_max_difference=maxdiff,
           choice=choice,balanced_accuracy=M.bacc(pred,Y['te'],threshold),BCE=M.bce(Y['te'],pred),AUC=M.auc(Y['te'],pred),
           cpu_parent_s=time.process_time()-cpu,wall_s=time.monotonic()-t0,qiskit=qiskit.__version__)
    (dest/'result.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r))
if __name__=='__main__': main()
