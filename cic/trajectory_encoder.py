"""Unchanged trajectory encoder and supervised contrastive components.
Supervised contrastive loss learns embeddings; offline_select.py separately
chooses preset retrieval budgets. No identity fields enter model inputs.
"""
import hashlib,math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
CAT_DIM=16
INPUT_DIM=22

def make_features(prepared):
    sequences = []

    for p in prepared:
        n = len(p)
        x = np.zeros((n, INPUT_DIM), dtype=np.float32)

        # Activity category.
        for t, category in enumerate(p.category):
            key = str(category).strip().lower().encode("utf-8")
            bucket = int.from_bytes(
                hashlib.blake2b(key, digest_size=4).digest(),
                "little"
            ) % CAT_DIM
            x[t, bucket] = 1.0

        # Local hour cycle and weekday/weekend.
        hour = np.asarray(p.hour, dtype=np.float32)
        x[:, 16] = np.sin(2 * np.pi * hour / 24)
        x[:, 17] = np.cos(2 * np.pi * hour / 24)
        x[:, 18] = (
            np.asarray(p.day_type).astype(str) == "weekend"
        ).astype(np.float32)

        # Relative movement between consecutive visits.
        if n > 1:
            lon = np.deg2rad(
                np.asarray(p.longitude, dtype=np.float64)
            )
            lat = np.deg2rad(
                np.asarray(p.latitude, dtype=np.float64)
            )

            dlon = np.diff(lon)
            dlat = np.diff(lat)
            lat_mid = (lat[:-1] + lat[1:]) / 2

            east = 6371000 * dlon * np.cos(lat_mid)
            north = 6371000 * dlat
            distance = np.sqrt(east ** 2 + north ** 2)

            x[1:, 19] = (
                np.log1p(distance) / math.log1p(10000)
            ).astype(np.float32)

            bearing = np.arctan2(north, east)
            x[1:, 20] = np.sin(bearing)
            x[1:, 21] = np.cos(bearing)

        sequences.append(x)

    return sequences


def pad_batch(sequences, device):
    lengths = [len(x) for x in sequences]
    max_len = max(lengths)

    x = np.zeros(
        (len(sequences), max_len, INPUT_DIM),
        dtype=np.float32
    )
    mask = np.zeros(
        (len(sequences), max_len),
        dtype=bool
    )

    for i, seq in enumerate(sequences):
        x[i, :len(seq)] = seq
        mask[i, :len(seq)] = True

    return (
        torch.as_tensor(x, device=device),
        torch.as_tensor(mask, device=device)
    )


class TrajectoryEncoder(nn.Module):

    def __init__(self, d=64):
        super().__init__()

        self.input_proj = nn.Linear(INPUT_DIM, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1
        )

        self.norm = nn.LayerNorm(d)

        self.projection = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 32)
        )

    def forward(self, x, mask):
        length = x.shape[1]

        # Positional encoding must match Transformer hidden dimension.
        d_model = self.input_proj.out_features

        position = torch.arange(
            length, device=x.device, dtype=x.dtype
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, d_model, 2,
                device=x.device, dtype=x.dtype
            ) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            length, d_model,
            device=x.device, dtype=x.dtype
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        h = self.input_proj(x) + pe.unsqueeze(0)

        h = self.transformer(
            h,
            src_key_padding_mask=~mask
        )

        h = self.norm(h)

        # Masked mean pooling.
        h = (
            (h * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1).clamp_min(1).unsqueeze(-1)
        )

        return F.normalize(self.projection(h), dim=-1)


def supcon_loss(z, labels, temperature=0.1):
    logits = z @ z.T / temperature

    n = len(labels)
    eye = torch.eye(
        n, dtype=torch.bool, device=z.device
    )

    positive = (
        (labels[:, None] == labels[None, :])
        & ~eye
    )

    logits = logits.masked_fill(eye, -1e4)

    log_prob = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )

    count = positive.sum(1)
    valid = count > 0

    return -(
        log_prob.masked_fill(~positive, 0).sum(1)[valid]
        / count[valid]
    ).mean()


def train_one_epoch(
    model, optimizer, sequences, groups,
    users, device, rng, batch_users=16, temperature=0.1
):
    model.train()

    steps = max(1, math.ceil(len(users) / batch_users))
    losses = []

    user_to_int = {
        str(user): idx
        for idx, user in enumerate(users)
    }

    for _ in range(steps):
        selected = rng.choice(
            users,
            size=min(batch_users, len(users)),
            replace=False
        )

        ids = []
        batch_labels = []

        for user in selected:
            candidates = groups[str(user)]
            chosen = rng.choice(
                candidates, size=2, replace=False
            )

            ids.extend(chosen.tolist())
            batch_labels.extend(
                [user_to_int[str(user)]] * 2
            )

        xb, mask = pad_batch(
            [sequences[i] for i in ids], device
        )

        yb = torch.as_tensor(
            batch_labels,
            dtype=torch.long,
            device=device
        )

        z = model(xb, mask)
        loss = supcon_loss(z, yb, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0
        )

        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


@torch.no_grad()
def encode_all(model, sequences, device):
    model.eval()
    outputs = []

    for start in range(0, len(sequences), 128):
        xb, mask = pad_batch(
            sequences[start:start + 128], device
        )
        outputs.append(model(xb, mask).cpu().numpy())

    return np.concatenate(outputs, axis=0)
