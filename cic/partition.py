"""Equal-cost expected pair-error minimization on observed candidate edges.

For fixed estimated same-user probabilities p and a partition g:
  R_E(g) = sum_E [(1-p) I(same) + p I(different)]
         = sum_E (1-p) + sum_E (2p-1) I(different).
There is no searched edge offset or cluster-count/capacity input. This is an
estimated-risk objective, not a claim of calibrated target probabilities or
global optimality of the existing heuristic solver.
"""
from time import perf_counter
import numpy as np
from .base_solver import base_decode
from .group_refine import refine_groups


def partition_probabilities(n,pairs,p):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(len(pairs),) or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Finite observed-edge probabilities in [0,1] required')
    if len(pairs)==0:
        return np.arange(n,dtype=np.int64),dict(additional_pairs=0,expected_observed_pair_errors=0.,cut_objective=0.,
            objective_constant=0.,cost_rule='2*p-1',searched_offset=False,uses_M=False,uses_K=False)
    cost=2*p-1;seed=base_decode(n,pairs,cost)
    solved=refine_groups(n,pairs,cost,seed)
    if not solved['diagnostics']['converged']:raise RuntimeError('Grouping unfinished')
    pred=solved['pred'];different=pred[pairs[:,0]]!=pred[pairs[:,1]]
    risk=float(np.where(different,p,1-p).sum());constant=float((1-p).sum());cut=float(cost[different].sum())
    if not np.isclose(risk,constant+cut,rtol=1e-12,atol=1e-8):raise AssertionError('Risk/cut identity violated')
    return pred,dict(cost_rule='2*p-1',searched_offset=False,mismerge_cost=1,missplit_cost=1,
        expected_observed_pair_errors=risk,objective_constant=constant,cut_objective=cut,
        observed_edges=len(pairs),refinement=solved['diagnostics'],
        additional_pairs=0,uses_M=False,uses_K=False,target_probability_calibration_guaranteed=False)

