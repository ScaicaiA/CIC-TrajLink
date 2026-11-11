from .signed_gaec import signed_gaec
from .signed_refine import refine_signed_partition

def base_decode(n,pairs,c):
    a=signed_gaec(n,pairs,c);b=refine_signed_partition(n,pairs,c,a['pred'])
    if not b['diagnostics']['converged']:raise RuntimeError('Single-node solver unfinished')
    return b['pred']

