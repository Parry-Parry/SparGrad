import torch as t 
from torch import tensor

from typing import Tuple

nn = t.nn
ag = t.autograd
la = t.linalg

def grad_q_d(q : tensor, d: tensor) -> Tuple[tensor, tensor]:
    assert q.requires_grad and d.requires_grad

    sim = nn.cosine_similarity(q, d)
    dd = ag.grad(sim, d)
    dq = ag.grad(sim, q)

    I_d = la.norm(dd, dim=1)
    I_q = la.norm(dq, dim=1)

    return I_q, I_d
