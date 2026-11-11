"""Frozen score features and exported linear mapping; no fit/selection here."""
import numpy as np
from .local_scale import locally_standardized_logits

def endpoint_features(n, pairs, values):
    _, diag = locally_standardized_logits(n, pairs, values)
    mu = np.asarray(diag['node_mean'])
    sd = np.asarray(diag['node_std'])
    i, j = pairs.T
    a = np.divide(values-mu[i], sd[i], out=np.zeros(len(pairs)), where=sd[i]>0)
    b = np.divide(values-mu[j], sd[j], out=np.zeros(len(pairs)), where=sd[j]>0)
    return np.minimum(a,b), np.maximum(a,b)


def edge_features(n, pairs, logits, embeddings):
    pairs = np.asarray(pairs)
    logits = np.asarray(logits, dtype=np.float64)
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or len(emb) != n or not np.all(np.isfinite(emb)):
        raise ValueError('Embeddings must be finite and match vertices')
    norm = np.linalg.norm(emb, axis=1)
    if np.any(norm == 0):
        raise ValueError('Zero embedding has no cosine')
    emb = emb / norm[:,None]
    cosine = np.empty(len(pairs), dtype=np.float64)
    for start in range(0,len(pairs),65536):
        p = pairs[start:start+65536]
        cosine[start:start+len(p)] = np.einsum('ij,ij->i',emb[p[:,0]],emb[p[:,1]])
    cosine = np.clip(cosine,-1,1)
    lo, hi = endpoint_features(n,pairs,logits)
    clo, chi = endpoint_features(n,pairs,cosine)
    return np.column_stack((logits,lo,hi,cosine,clo,chi))


def calibrated_log_odds(features, model):
    """Evaluate exported linear coefficients, without sklearn/model retraining."""
    idx = np.asarray(model['feature_indices'],dtype=int)
    x = np.asarray(features,dtype=np.float64)[:,idx]
    values = x @ np.asarray(model['raw_coefficients']) + model['raw_intercept']
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite calibrated edge evidence')
    return values
