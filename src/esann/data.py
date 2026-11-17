"""Generate the exact published inputs; no model selection or test-driven tuning."""
import importlib.util
from pathlib import Path
import numpy as np
from .simulator import Escala

ROOT = Path(__file__).resolve().parents[2]
INITIAL = tuple(range(20261201, 20261206))
CONFIRMATION = tuple(range(20263001, 20263006))
SEEDS = (42,) + INITIAL + CONFIRMATION

def stage(seed):
    if seed == 42: return 'released'
    if seed in INITIAL: return 'initial'
    if seed in CONFIRMATION: return 'confirmation'
    raise ValueError('Use an archived study seed; new datasets are a separate experiment.')

def generate(seed):
    """Seed 42 uses released CSVs, preserving their original rounding."""
    folder = ROOT / 'third_party/qflair/released_data'
    if seed == 42:
        X = np.loadtxt(folder/'Xbars-and-stripes_4d_1000.csv',delimiter=',')[:,:16]
        y = np.loadtxt(folder/'Ybars-and-stripes_4d_1000.csv',delimiter=',')
        Xt = np.loadtxt(folder/'Xbars-and-stripes_4d_200.csv',delimiter=',')[:,:16]
        yt = np.loadtxt(folder/'Ybars-and-stripes_4d_200.csv',delimiter=',')
    else:
        spec = importlib.util.spec_from_file_location('bs_generator', ROOT/'third_party/qml_benchmarks/bars_and_stripes.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            X,y = module.generate_bars_and_stripes(1000,4,4,0.5)
            Xt,yt = module.generate_bars_and_stripes(1000,4,4,0.5)
        finally:
            np.random.set_state(state)
        X,Xt=X.reshape(len(X),16),Xt.reshape(len(Xt),16)
        y,yt=(y==1).astype(float),(yt==1).astype(float)
    split_seed = 20263101 + seed - 20263001 if seed in CONFIRMATION else 20261109
    rng=np.random.RandomState(split_seed); iv=[]
    for c in (0,1):
        ix=np.flatnonzero(y==c); iv.extend(rng.permutation(ix)[:int(round(0.2*len(ix)))])
    iv=np.sort(np.array(iv)); it=np.setdiff1d(np.arange(1000),iv)
    assert len(iv)==200
    scale=Escala().fit(X[it])
    return {'Xtr':scale.transform(X[it]),'ytr':y[it], 'Xva':scale.transform(X[iv]),'yva':y[iv],
            'Xte':scale.transform(Xt),'yte':yt,'generation_seed':seed,'partition_seed':split_seed}

def artifact(seed): return ROOT/'artifacts'/stage(seed)/str(seed)

def archived_data(seed):
    p=artifact(seed)/'data'
    out={}
    for short,long in [('tr','train'),('va','val'),('te','test')]:
        out['X'+short]=np.loadtxt(p/f'X{long}.csv',delimiter=',')[:,:16]
        out['y'+short]=np.loadtxt(p/f'Y{long}.csv',delimiter=',')
    return out
