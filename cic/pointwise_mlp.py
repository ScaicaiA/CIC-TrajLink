"""Shared MLP on original point-pair relations, followed by mean/max pooling.

Every valid original ten-dimensional relation is retained. There is no relation
self-attention, evidence query, superpoint, or learned pooling operation.
"""
from time import perf_counter

import numpy as np
import torch
from torch import nn


class LightPairMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(10, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, rows, mask):
        if rows.ndim != 3 or rows.shape[-1] != 10:
            raise ValueError('rows must have shape [batch, relations, 10]')
        if mask.shape != rows.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('mask must be boolean [batch, relations]')
        # Mask before the MLP as well: padded values cannot affect its outputs
        # even when a caller fills padding with arbitrary finite values.
        h = self.relation(rows.masked_fill(~mask[..., None], 0.))
        count = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
        mean = h.masked_fill(~mask[..., None], 0.).sum(1) / count
        maximum = h.masked_fill(~mask[..., None], torch.finfo(h.dtype).min).amax(1)
        maximum = torch.where(mask.any(1, keepdim=True), maximum, 0.)
        return self.head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)


def pack_rows(rows, ids, device='cpu'):
    """Pack variable-length relation arrays without discarding any row."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if not len(ids):
        raise ValueError('pack_rows requires at least one pair')
    lengths = [len(rows[int(i)]) for i in ids]
    if min(lengths) <= 0:
        raise ValueError('each trajectory pair must have a nonempty relation set')
    values = np.zeros((len(ids), max(lengths), 10), dtype=np.float32)
    valid = np.zeros(values.shape[:2], dtype=bool)
    for k, (idx, length) in enumerate(zip(ids, lengths)):
        arr = np.asarray(rows[int(idx)], dtype=np.float32)
        if arr.shape != (length, 10):
            raise ValueError('each relation array must have shape [relations, 10]')
        values[k, :length] = arr
        valid[k, :length] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(valid).to(device)


def _sync(device):
    kind = torch.device(device).type
    if kind == 'mps':
        torch.mps.synchronize()
    elif kind == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def score_rows(rowslist, model, device='cpu', batch_size=128):
    """Return logits in caller order and complete packing/transfer/model time.

    Input relation construction is the caller's responsibility and is timed
    separately by the experiment. Sorting only reduces padding; it does not
    change pair membership or the set of valid relations supplied to the MLP.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    _sync(device)
    begin = perf_counter()
    logits = np.empty(len(rowslist), dtype=np.float32)
    order = np.argsort([len(r) for r in rowslist], kind='stable')
    slots = 0
    batches = 0
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        values, mask = pack_rows(rowslist, ids, device)
        logits[ids] = model(values, mask).cpu().numpy()
        slots += int(mask.numel())
        batches += 1
    _sync(device)
    return logits, dict(score_seconds=perf_counter() - begin,
                        executed_slots=slots,
                        valid_relation_rows=sum(len(r) for r in rowslist),
                        network_batches=batches,
                        attention_cells=0)
