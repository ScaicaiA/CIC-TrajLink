"""Label-free information/POI statistics on existing candidate pairs only.

Reuses the same IDF-weighted POI set representation as the retrieval path.
All raw visits contribute; original neural max-point preprocessing is unchanged.
Indices are batch-local POI codes, not user IDs. No trajectory/user capacity.
"""
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix

NAMES=['poi_idf_cosine','poi_energy_coverage_min','poi_energy_coverage_max',
       'log_visit_count_min','log_visit_count_max','log_unique_poi_min','log_unique_poi_max']


def encode_nodes(trajectories):
    vocab={};offset=[0];codes=[];visits=[]
    for traj in trajectories:
        pois=sorted({str(p.poi_id) for p in traj.points if p.poi_id})
        for poi in pois:codes.append(vocab.setdefault(poi,len(vocab)))
        offset.append(len(codes));visits.append(len(traj.points))
    return dict(offsets=np.asarray(offset,np.int64),poi_codes=np.asarray(codes,np.int64),
                visit_counts=np.asarray(visits,np.int64))


def subset_nodes(nodes,indices):
    ids=[];offset=[0]
    for i in indices:
        row=nodes['poi_codes'][nodes['offsets'][i]:nodes['offsets'][i+1]]
        ids.extend(row.tolist());offset.append(len(ids))
    return dict(offsets=np.asarray(offset,np.int64),poi_codes=np.asarray(ids,np.int64),
                visit_counts=nodes['visit_counts'][indices])


def evidence_on_pairs(nodes,pairs):
    start=perf_counter();offset=nodes['offsets'];n=len(offset)-1
    counts=np.diff(offset);codes=nodes['poi_codes'];pairs=np.asarray(pairs,dtype=np.int64)
    if len(nodes['visit_counts'])!=n or np.any(pairs<0) or np.any(pairs>=n):raise ValueError('Node/pair mismatch')
    if len(codes):
        _,columns=np.unique(codes,return_inverse=True);m=int(columns.max())+1
        rows=np.repeat(np.arange(n),counts)
        binary=csr_matrix((np.ones(len(columns)),(rows,columns)),shape=(n,m))
        if np.any(binary.data!=1):raise ValueError('POI set must contain unique codes per trajectory')
        df=np.asarray(binary.sum(axis=0)).ravel();idf=np.log((n+1)/(df+1))+1
        x=binary.multiply(idf).tocsr();norm2=np.asarray(x.multiply(x).sum(axis=1)).ravel()
        dots=(x@x.T).tocsr();intersection=np.asarray(dots[pairs[:,0],pairs[:,1]]).ravel()
        temporary_nonzeros=int(dots.nnz)
    else:
        norm2=np.zeros(n);intersection=np.zeros(len(pairs));temporary_nonzeros=0
    a,b=pairs.T
    left=np.divide(intersection,norm2[a],out=np.zeros(len(pairs)),where=norm2[a]>0)
    right=np.divide(intersection,norm2[b],out=np.zeros(len(pairs)),where=norm2[b]>0)
    cosine=np.sqrt(np.maximum(left*right,0))
    length=np.log1p(nodes['visit_counts']);unique=np.log1p(counts)
    feature=np.column_stack((np.clip(cosine,0,1),np.minimum(left,right),np.maximum(left,right),
        np.minimum(length[a],length[b]),np.maximum(length[a],length[b]),np.minimum(unique[a],unique[b]),np.maximum(unique[a],unique[b])))
    if not np.all(np.isfinite(feature)):raise ValueError('Nonfinite POI features')
    return feature,dict(n=n,observed_edges=len(pairs),feature_names=NAMES,additional_pairs=0,
        scope='current anonymous batch, raw visits; document frequency counts each trajectory once per POI',
        idf='log((N+1)/(df+1))+1; squared-IDF cosine and squared-energy containment',
        temporary_nonzeros=temporary_nonzeros,elapsed_seconds=perf_counter()-start,
        no_identity_labels=True,no_neural_point_reprocessing=True)
