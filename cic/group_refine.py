"""Unbalanced best-prefix two-set moves on existing signed edges only.

No labels, target number of groups, capacities, or scoring callbacks. This is
a KL-style heuristic, not the balanced original KL algorithm or an exact solver.
All nonempty clusters can split. Two nonempty clusters are visited only when
they have an observed positive edge; this limits search, not the objective.
Missing entries in dense accumulators have no cost term, never a negative label.
"""
from __future__ import annotations
import math
from time import perf_counter
import numpy as np
from .signed_refine import refine_signed_partition


def best_prefix_move(weights, side):
    """Return a strictly improving prefix or original sides on one induced graph.

The caller has already accumulated costs of observed edges. Numerical zeros
can be either no cost term or explicit zero costs. Neither is negative evidence.
Labels 0/1 are the two candidate sides, NOT true identities.
"""
    w=np.asarray(weights,dtype=np.float64)
    original=np.asarray(side,dtype=np.int64)
    n=len(original)
    if w.shape!=(n,n) or not np.all(np.isfinite(w)) or not np.allclose(w,w.T):
        raise ValueError('Need a finite symmetric cost accumulator')
    if np.any(np.diag(w)!=0) or not np.all((original==0)|(original==1)):
        raise ValueError('Invalid diagonal or two-way partition')
    s=original.copy();locked=np.zeros(n,dtype=bool)
    same=s[:,None]==s[None,:]
    gains=np.where(same,-w,w).sum(axis=1)
    cumulative=0.;best_gain=0.;best_at=0;order=[];steps=[]
    guard=128*np.finfo(float).eps*max(1.,float(np.abs(w).sum()))
    for at in range(n):
        available=np.flatnonzero(~locked)
        v=int(available[np.argmax(gains[available])])
        gain=float(gains[v]);cumulative+=gain
        order.append(v);steps.append(gain)
        if cumulative>best_gain+guard:
            best_gain=cumulative;best_at=at+1
        old=s[v]
        # A previously same-side neighbor acquires +2*w in its flip gain.
        gains+=np.where(s==old,2*w[:,v],-2*w[:,v])
        s[v]=1-old;locked[v]=True
    candidate=original.copy()
    if best_at:
        selected=np.asarray(order[:best_at]);candidate[selected]=1-candidate[selected]
    actual=float(np.sum(w[(original[:,None]!=original[None,:])])-np.sum(w[(candidate[:,None]!=candidate[None,:])]))/2
    if abs(actual-best_gain)>max(guard,1e-8*max(1,abs(actual))):
        raise AssertionError('Best-prefix objective accounting failed')
    if actual<=guard:return original.copy(),0.,[]
    return candidate,actual,steps[:best_at]


def refine_groups(n,pairs,costs,pred,max_passes=100):
    start=perf_counter();pairs=np.asarray(pairs,dtype=np.int64);costs=np.asarray(costs,dtype=float)
    if len(pairs)!=len(costs) or not np.all(np.isfinite(costs)):
        raise ValueError('Mismatched finite observed costs')
    if np.any(pairs[:,0]>=pairs[:,1]) or np.any(pairs<0) or np.any(pairs>=n):
        raise ValueError('Canonical observed edges required')
    if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Duplicate edges')
    _,labels=np.unique(pred,return_inverse=True)
    if labels.shape!=(n,):raise ValueError('Incomplete partition')
    w=np.zeros((n,n),float);i,j=pairs.T;w[i,j]=w[j,i]=costs
    positive=pairs[costs>0]
    def energy(p):return math.fsum(costs[p[i]!=p[j]])
    initial=energy(labels);trace=[];converged=False;scanned=0
    for iteration in range(1,max_passes+1):
        before=energy(labels);accepted=0
        # Split any cluster, including ones without attractive external links.
        for cid in np.unique(labels):
            nodes=np.flatnonzero(labels==cid)
            if len(nodes)<2:continue
            block=w[np.ix_(nodes,nodes)]
            if not np.any(block<0):continue  # A nonnegative internal graph has no beneficial split.
            side,gain,steps=best_prefix_move(block,np.zeros(len(nodes),np.int64));scanned+=1
            if gain>0:
                fresh=int(labels.max())+1;labels[nodes[side==1]]=fresh;accepted+=1
                trace.append(dict(pass_index=iteration,kind='split',gain=gain,moved=len(steps),
                    first_step_gain=steps[0],contains_negative_step=any(v<0 for v in steps)))
        # Positive observed boundaries provide search candidates, not invented edges.
        a=labels[positive[:,0]];b=labels[positive[:,1]];cross=a!=b
        uv=np.sort(np.column_stack((a[cross],b[cross])),axis=1)
        if len(uv):
            uv=np.unique(uv,axis=0)
            priorities=[]
            for aa,bb in uv:
                an=np.flatnonzero(labels==aa);bn=np.flatnonzero(labels==bb)
                priorities.append(float(w[np.ix_(an,bn)].sum()))
            ordering=np.lexsort((uv[:,1],uv[:,0],-np.asarray(priorities)))
            for aa,bb in uv[ordering]:
                an=np.flatnonzero(labels==aa);bn=np.flatnonzero(labels==bb)
                if not len(an) or not len(bn):continue
                nodes=np.r_[an,bn];side=np.r_[np.zeros(len(an),np.int64),np.ones(len(bn),np.int64)]
                block=w[np.ix_(nodes,nodes)]
                alternative,gain,steps=best_prefix_move(block,side);scanned+=1
                whole=float(w[np.ix_(an,bn)].sum())
                if whole>gain and whole>0:
                    labels[bn]=aa;accepted+=1
                    trace.append(dict(pass_index=iteration,kind='join',gain=whole,moved=len(bn)))
                elif gain>0:
                    labels[nodes[alternative==0]]=aa;labels[nodes[alternative==1]]=bb;accepted+=1
                    trace.append(dict(pass_index=iteration,kind='two_set_move',gain=gain,moved=len(steps),
                        first_step_gain=steps[0],contains_negative_step=any(v<0 for v in steps)))
        # Enforce observed connectivity and finish any newly available one-node moves.
        refined=refine_signed_partition(n,pairs,costs,labels)
        if not refined['diagnostics']['converged']:raise RuntimeError('Single-node refinement safeguard hit')
        labels=refined['pred'];after=energy(labels)
        guard=256*np.finfo(float).eps*max(1,abs(before))
        if after>before+guard:raise AssertionError('Group descent increased objective')
        if after>=before-guard:
            converged=True;break
    return dict(pred=labels,diagnostics=dict(converged=converged,passes=iteration,
        initial_energy=initial,final_energy=energy(labels),energy_improvement=initial-energy(labels),
        accepted_group_operations=len(trace),scanned_two_set_problems=scanned,
        negative_intermediate_steps=sum(t.get('contains_negative_step',False) for t in trace),
        final_clusters=len(np.unique(labels)),additional_pairs=0,uses_M=False,uses_K=False,
        uses_labels=False,heuristic=True,global_optimality=False,
        search_scope='all cluster splits and observed-positive-connected cluster pairs; restricted KL-style neighborhood',
        elapsed_seconds=perf_counter()-start),trace=trace)
