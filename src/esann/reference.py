"""Replay the executed Q-FLAIR pool without retraining or importing Qiskit."""
import numpy as np
from . import simulator as Q

def replay(gate_file, X):
    psi=Q.estado_inicial(len(X),4); out=[Q.p0(psi)]
    mapping={'U2':'RX','U3':'RY','Rxx':'RXX','Ryy':'RYY'}
    for line in gate_file.read_text().splitlines():
        cols=line.split('\t'); names=cols[:-2]; weight=float(cols[-2]); feature=int(cols[-1])
        for j,name in enumerate(names):
            if name=='111111': continue
            kind=name.split('_')[0]
            if kind=='H': psi=Q.aplica(psi,4,'H',(j,),np.zeros(len(X)))
            else:
                wires=((j-1)%4,j) if kind in ('Rxx','Ryy') else (j,)
                psi=Q.aplica(psi,4,mapping[kind],wires,weight*X[:,feature])
        out.append(Q.p0(psi))
    return np.asarray(out)
