"""Weighted shared-neighbor evidence on observed candidate edges only.

This is an extra statistic, not graph completion. Positive support uses the
fixed inherited posterior floor sigmoid(-2), shared across all cities; there is no top-k,
degree target, identity count, or cluster capacity in this operation.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit


def neighbor_agreement(n,pairs,source_log_odds,logit_floor=-2.):
    p=np.asarray(pairs,dtype=np.int64);u=np.asarray(source_log_odds,dtype=float)
    if len(p)!=len(u) or not np.all(np.isfinite(u)):raise ValueError('Invalid observed scores')
    weight=np.maximum(expit(u)-expit(logit_floor),0.)
    keep=weight>0;edges=p[keep];w=weight[keep]
    a=csr_matrix((np.r_[w,w],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
    norm=np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    shared=(a@a.T).tocsr()
    # Extract ONLY original candidate positions. A two-hop path never adds an edge.
    value=np.asarray(shared[p[:,0],p[:,1]]).ravel()
    denom=norm[p[:,0]]*norm[p[:,1]]
    result=np.divide(value,denom,out=np.zeros(len(p)),where=denom>0)
    result=np.clip(result,0,1)
    return result,dict(observed_pairs=len(p),positive_support_edges=len(edges),
        support_threshold=float(expit(logit_floor)),support_logit_floor=float(logit_floor),intermediate_sparse_nonzeros=int(shared.nnz),
        output_edges=len(result),additional_pairs=0,
        positive_degree_second_moment=int(np.square(np.diff(a.indptr)).sum()),
        zero_support_nodes=int(np.sum(norm==0)),
        note='Common-neighbor cosine computed only as a feature on observed edges; zero means no positive shared support, not an observed negative pair')
