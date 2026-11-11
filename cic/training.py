"""Fresh supervised training, with no historical checkpoint or pair-cache input.

Neural checkpoint selection uses validation losses, never M/K-based clustering.
The offline encoder is frozen before k selection. The main program initializes
the Pointwise-MLP anew, generates its own training pairs, and saves a new model.
"""
from pathlib import Path
from time import perf_counter
import math
import numpy as np
import torch
import torch.nn.functional as F

from .trajectory_encoder import TrajectoryEncoder, make_features, pad_batch, supcon_loss, train_one_epoch
from .pointwise_mlp import LightPairMLP, pack_rows
from .preprocessing import compute_relation_split_stats
from .point_relations import prepare_points
from .scoring import make_rows
from .device import sync
from .io import log, write_json, seed_everything, sha256, hash_strings
from .protocol import neural_provenance


def groups_from_labels(labels):
    result = {str(v): np.flatnonzero(labels == v) for v in np.unique(labels)}
    if len(result) < 2 or any(len(v) < 2 for v in result.values()):
        raise ValueError('Training requires >=2 users and >=2 distinct weekly trajectories per user')
    return result


def encoder_validation_batches(sequences, labels, batch_users, seed):
    groups = groups_from_labels(labels)
    users = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    users = users[rng.permutation(len(users))]
    parts = [users[i:i+batch_users] for i in range(0,len(users),batch_users)]
    if len(parts) > 1 and len(parts[-1]) == 1:
        last = parts.pop()
        parts[-1] = np.r_[parts[-1], last]
    batches = []
    for batch in parts:
        ids = np.concatenate([rng.choice(groups[u], 2, replace=False) for u in batch])
        batches.append(([sequences[i] for i in ids], labels[ids]))
    return batches


@torch.no_grad()
def encoder_val_loss(model, batches, device, temperature):
    model.eval()
    numerator, denominator = 0., 0
    for sequences, labels in batches:
        x, mask = pad_batch(sequences, device)
        z = model(x, mask)
        value = supcon_loss(z, torch.as_tensor(labels, device=device), temperature)
        numerator += len(labels)*float(value.cpu())
        denominator += len(labels)
    return numerator/denominator


def train_encoder(train, validation, cfg, dest, device):
    splits = dict(train=train, val=validation)
    provenance = neural_provenance(train, validation, cfg)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    c = cfg['encoder']
    seed_everything(cfg['seed'], cfg['threads'])
    prepared = {name: prepare_points(splits[name].trajectories, {}, cfg['max_points'])
                for name in ('train','val')}
    sequences = {name: make_features(p) for name,p in prepared.items()}
    del prepared
    labels = splits['train'].labels
    groups = groups_from_labels(labels)
    users = np.asarray(list(groups))
    val_batches = encoder_validation_batches(sequences['val'], splits['val'].labels,
                                             c['batch_users'], cfg['validation_sampling_seed'])
    model = TrajectoryEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    rng = np.random.default_rng(cfg['seed'])
    best, trace = math.inf, []
    for epoch in range(1, c['epochs']+1):
        start = perf_counter()
        loss = train_one_epoch(model, opt, sequences['train'], groups, users, device, rng,
                               batch_users=c['batch_users'], temperature=c['temperature'])
        vloss = encoder_val_loss(model, val_batches, device, c['temperature'])
        if not math.isfinite(loss+vloss):
            raise RuntimeError('Nonfinite encoder loss')
        if vloss < best:
            best = vloss
            selected = epoch
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch, seed=cfg['seed'], selection='minimum fixed validation SupCon loss; earliest tie',
                validation_loss=vloss, input_dim=22, output_dim=32, provenance=provenance,
                train_keys_sha256=hash_strings(t.key for t in splits['train'].trajectories)), dest/'encoder.pt')
        row = dict(epoch=epoch, training_loss=loss, validation_supcon_loss=vloss,
                   selected_epoch=selected, seconds=perf_counter()-start)
        trace.append(row)
        log('encoder_training', **row)
        write_json(dest/'training.json', dict(initialization='fresh random weights',
            selection='validation SupCon loss, not clustering or candidate recall',
            trace=trace, selected_epoch=selected, parameters=c, seed=cfg['seed'], provenance=provenance))
    model.load_state_dict(torch.load(dest/'encoder.pt', map_location='cpu', weights_only=True)['model'])
    model.eval()
    return model


def sample_pairs(labels, count, positive_fraction, seed):
    """New deterministic source samples; unique unordered pairs, no old files."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    positive_parts = []
    for y in np.unique(labels):
        ids = np.flatnonzero(labels == y)
        i,j = np.triu_indices(len(ids),1)
        positive_parts.append(np.column_stack((ids[i], ids[j])))
    positives = np.concatenate(positive_parts)
    want_p = min(int(round(count*positive_fraction)), len(positives))
    possible_negative = len(labels)*(len(labels)-1)//2-len(positives)
    want_n = min(count-want_p, possible_negative)
    if not want_p or not want_n:
        raise ValueError('Need both positive and negative supervised pairs')
    positives = positives[rng.choice(len(positives), want_p, replace=False)]
    # In small synthetic tests enumerate to avoid rejection-sampling near saturation.
    if len(labels) <= 500:
        i,j = np.triu_indices(len(labels),1)
        negatives = np.column_stack((i,j))[labels[i] != labels[j]]
        negatives = negatives[rng.choice(len(negatives), want_n, replace=False)]
    else:
        codes = set()
        n = len(labels)
        while len(codes) < want_n:
            draws = rng.integers(0,n,size=(max(256,2*(want_n-len(codes))),2))
            draws.sort(axis=1)
            draws = draws[labels[draws[:,0]] != labels[draws[:,1]]]
            # Iteration order is RNG order, not hash set order.
            for i,j in draws:
                codes.add(int(i)*n+int(j))
                if len(codes) == want_n:
                    break
        codes = np.asarray(sorted(codes),np.int64)
        negatives = np.column_stack((codes//n,codes%n))
    pairs = np.vstack((positives,negatives))
    rng.shuffle(pairs)
    y = (labels[pairs[:,0]] == labels[pairs[:,1]]).astype(np.float32)
    return pairs,y


@torch.no_grad()
def mlp_val_loss(model, rows, y, batch_size, device):
    model.eval()
    loss = 0.
    for start in range(0,len(rows),batch_size):
        ids = np.arange(start,min(start+batch_size,len(rows)))
        x,mask = pack_rows(rows,ids,device)
        value = F.binary_cross_entropy_with_logits(model(x,mask),torch.as_tensor(y[ids],device=device),reduction='sum')
        loss += float(value.cpu())
    return loss/len(rows)


def train_mlp(train, validation, cfg, dest, device):
    splits = dict(train=train, val=validation)
    provenance = neural_provenance(train, validation, cfg)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    c = cfg['mlp']
    mlp_seed = cfg['seed']
    seed_everything(mlp_seed,cfg['threads'])
    rows,y,keys = {},{},{}
    for name,count in [('train',c['training_pairs']),('val',c['validation_pairs'])]:
        split = splits[name]
        pair_seed = mlp_seed if name == 'train' else cfg['validation_sampling_seed']
        pairs,y[name] = sample_pairs(split.labels,count,c['positive_fraction'],pair_seed)
        stats = compute_relation_split_stats(split.trajectories)
        prepared = prepare_points(split.trajectories,stats,cfg['max_points'])
        rows[name] = make_rows(prepared,pairs,cfg)
        keys[name] = hash_strings(t.key for t in split.trajectories)
        np.savez_compressed(dest/f'{name}_pairs_generated.npz',pairs=pairs,y=y[name],
                            keys_sha256=np.asarray(keys[name]),sampling_seed=np.asarray(pair_seed))
        log('mlp_pairs_generated', split=name, pairs=len(pairs), positives=int(y[name].sum()),
            relation_rows=sum(len(r) for r in rows[name]))
    model = LightPairMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'],weight_decay=c['weight_decay'])
    best,trace = math.inf,[]
    for epoch in range(1,c['epochs']+1):
        model.train()
        start = perf_counter()
        order = np.random.default_rng(mlp_seed+epoch).permutation(len(rows['train']))
        loss_sum = 0.
        for batch_start in range(0,len(order),c['batch_size']):
            ids = order[batch_start:batch_start+c['batch_size']]
            x,mask = pack_rows(rows['train'],ids,device)
            target = torch.as_tensor(y['train'][ids],device=device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x,mask),target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite MLP loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            loss_sum += len(ids)*float(loss.detach().cpu())
        row = dict(epoch=epoch,training_bce=loss_sum/len(order))
        if epoch%c['validate_every'] == 0 or epoch == c['epochs']:
            value = mlp_val_loss(model,rows['val'],y['val'],c['batch_size'],device)
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite validation loss')
            if value < best:
                best, selected = value,epoch
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    epoch=epoch,seed=mlp_seed,selection='minimum fixed validation pair BCE; earliest tie',
                    validation_bce=value,source_keys_sha256=keys,provenance=provenance),dest/'pointwise_mlp.pt')
            row.update(validation_bce=value,selected_epoch=selected)
        sync(device)
        row['seconds'] = perf_counter()-start
        trace.append(row)
        log('mlp_training',**row)
        write_json(dest/'training.json',dict(initialization='fresh random weights',
            prior_run_dependency=False,uses_M=False,uses_K=False,parameters=c,
            trace=trace,seed=mlp_seed,provenance=provenance,
            validation_sampling_seed=cfg['validation_sampling_seed'],
            pair_counts={k:len(v) for k,v in y.items()}))
    model.load_state_dict(torch.load(dest/'pointwise_mlp.pt',map_location='cpu',weights_only=True)['model'])
    model.eval()
    return model
