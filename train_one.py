#!/usr/bin/env python3
"""Train one city/seed with its fixed kp/kc and write an isolated result."""
import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
from threadpoolctl import threadpool_limits

from cic.device import select_device, environment_info
from cic.io import read_json, write_json, sha256, heartbeat, log, code_fingerprint
from cic.protocol import load_protocol, save_protocol, training_config, validate_neural_checkpoint
from cic.runtime import binding, committed, commit, preserve_incomplete, run_lock, dependency_versions
from cic.training import train_encoder
from main import run_city


def run(root, city, seed, requested_device):
    root = Path(root).resolve()
    meta = read_json(root / 'experiment.json')
    config = read_json(root / 'config_snapshot.json')
    if code_fingerprint() != meta['code_sha256']:
        raise ValueError('Code changed after the run started')
    if city not in meta['cities'] or seed not in meta['training_seeds']:
        raise ValueError('City/seed is not part of this run')
    cfg = training_config(config, seed)
    spec = config['cities'][city]
    device = select_device(requested_device)
    prep = root / f'seed_{seed}' / 'preparation' / city
    dest = root / f'seed_{seed}' / 'main' / city
    prep_binding = binding(meta, city, seed, 'encoder')
    main_binding = binding(meta, city, seed, 'main')
    with run_lock(root / 'job_locks' / f'{city}_{seed}.lock'), threadpool_limits(limits=cfg['threads']):
        if committed(dest, main_binding):
            log('completed_job_reused', city=city, training_seed=seed)
            return
        if not committed(prep, prep_binding):
            preserve_incomplete(prep, root)
            prep.mkdir(parents=True)
            start = perf_counter()
            roles, manifest = load_protocol(spec['data'], config['protocol'])
            original = manifest['original']
            if original['csv_sha256'] != spec['data_sha256']:
                raise ValueError(f'{city}: CSV checksum mismatch')
            if {k: v['users'] for k, v in original['splits'].items()} != spec['expected_user_splits']:
                raise ValueError(f'{city}: user split counts differ from the configuration')
            if config['experiment_kind'] == 'formal' and original['trajectory_count_histogram'] != {'10': sum(spec['expected_user_splits'].values())}:
                raise ValueError(f'{city}: formal Top10 data must have ten trajectories per user')
            save_protocol(prep, roles, manifest)
            write_json(prep / 'environment.json', dict(**environment_info(device), dependencies=dependency_versions()))
            with heartbeat(f'{city}_seed{seed}_encoder'):
                encoder = train_encoder(roles['neural_train'], roles['val'], cfg, prep / 'encoder', device)
            checkpoint = torch.load(prep / 'encoder/encoder.pt', map_location='cpu', weights_only=True)
            validate_neural_checkpoint(checkpoint, manifest, cfg)
            fixed = dict(data=spec['data'], data_sha256=spec['data_sha256'],
                         encoder=str(prep / 'encoder/encoder.pt'), encoder_sha256=sha256(prep / 'encoder/encoder.pt'),
                         source_manifest=str(prep / 'data_manifest.json'), encoder_seed=seed,
                         encoder_max_points=cfg['max_points'],
                         preset_kp=spec['fixed_k'][str(seed)]['kp'], preset_kc=spec['fixed_k'][str(seed)]['kc'])
            write_json(prep / 'fixed_inputs.json', fixed)
            write_json(prep / 'complete.json', dict(city=city, training_seed=seed, seconds=perf_counter()-start))
            commit(prep, prep_binding)
            del encoder, checkpoint, roles, manifest
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            log('completed_encoder_reused', city=city, training_seed=seed)
        preserve_incomplete(dest, root)
        fixed = read_json(prep / 'fixed_inputs.json')
        run_city(city, fixed, cfg, config['protocol'], prep, dest, device)
        commit(dest, main_binding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.run, args.city, args.seed, args.device)
