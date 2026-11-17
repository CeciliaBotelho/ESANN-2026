"""Nine-point response reconstruction, unchanged executed function."""
import numpy as np
from . import simulator as Q
N_Q=4
NOS=np.array([0.0,np.pi/2,-np.pi/2])
MPHI=np.stack([np.ones(3),np.cos(NOS),np.sin(NOS)],1)
MINV=np.linalg.inv(MPHI)
CONTA=dict(simulacoes=0)

def reconstroi_par(psi, p1, p2):
    """C (N, 3, 3) tal que P0(α, β) = φ(α)ᵀ C φ(β); 8 simulações novas."""
    est = [psi] + [Q.aplica(psi, N_Q, p1[0], p1[1], a) for a in NOS[1:]]; CONTA['simulacoes'] += 2; P = np.zeros((psi.shape[0], 3, 3))
    for i, s in enumerate(est):
        for j, b in enumerate(NOS):
            P[:, i, j] = Q.p0(s) if j == 0 else Q.p0(Q.aplica(s, N_Q, p2[0], p2[1], b))
            CONTA['simulacoes'] += int(j != 0)
    return np.einsum('ai,nij,bj->nab', MINV, P, MINV)
