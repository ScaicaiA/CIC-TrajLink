"""Engineering test using real data subsets and two epochs, never formal results."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cic.io import read_json, write_json, sha256
from cic.protocol import load_protocol
from cic.preprocessing import compute_relation_split_stats
from cic.retrieval import candidates


def run(output, cities):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read_json(ROOT / 'config/four_cities.json'))
    cfg['experiment_kind'] = 'engineering_smoke'
    cfg['training']['encoder'].update(epochs=2, batch_users=8)
    cfg['training']['mlp'].update(epochs=2, training_pairs=128, validation_pairs=64, validate_every=1, batch_size=32)
    cfg['training'].update(threads=2, score_batch=32, score_chunk=128, similarity_block_rows=16)
    for city in cities:
        spec = cfg['cities'][city]
        csv.field_size_limit(2**31-1)
        with gzip.open(ROOT / 'data' / f'{city}.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
        users = sorted({row[0] for row in rows})
        chosen = set(np.random.default_rng(42).choice(users, 60, replace=False))
        path = output / f'{city}_SMOKE_ONLY.csv'
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(row for row in rows if row[0] in chosen)
        spec.update(data=str(path), data_sha256=sha256(path), expected_user_splits=dict(train=36, val=12, test=12))
        spec.pop('archive')
        spec.pop('archive_sha256')
    write_json(output / 'config.json', cfg)
    command = [sys.executable, '-u', str(ROOT / 'run_main.py'), '--config', str(output / 'config.json'),
               '--output', str(output / 'experiment'), '--device', 'cpu', '--cities', *cities]
    with (output / 'driver.log').open('w') as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    experiment = output / 'experiment'
    summary = read_json(experiment / 'summary.json')
    assert summary['experiment_kind'] == 'engineering_smoke'
    assert not summary['full_main_experiment']
    assert summary['requested_jobs_complete'] and summary['sample_std_ddof'] == 1
    assert summary['completed'] == len(cities) * 3
    checks = []
    for city in cities:
        roles, manifest = load_protocol(output / f'{city}_SMOKE_ONLY.csv', cfg['protocol'])
        allocations, val_pairs = [], []
        encoder_hashes, mlp_hashes = [], []
        for seed in (42, 43, 44):
            prep = experiment / f'seed_{seed}/preparation' / city
            main = experiment / f'seed_{seed}/main' / city
            allocations.append(read_json(main / 'identity_allocation.json'))
            assert read_json(main / 'data_manifest.json') == manifest
            result = read_json(main / 'result.json')
            pair = cfg['cities'][city]['fixed_k'][str(seed)]
            assert (result['preset_kp'], result['preset_kc']) == (pair['kp'], pair['kc'])
            assert not result['uses_M'] and not result['uses_K'] and result['additional_scored_pairs'] == 0
            for file in (prep / 'encoder/encoder.pt', main / 'mlp/pointwise_mlp.pt'):
                ckpt = torch.load(file, map_location='cpu', weights_only=True)
                assert ckpt['seed'] == seed
                assert ckpt['provenance']['neural_train'] == manifest['roles']['neural_train']
            encoder_hashes.append(sha256(prep / 'encoder/encoder.pt'))
            mlp_hashes.append(sha256(main / 'mlp/pointwise_mlp.pt'))
            with np.load(main / 'mlp/val_pairs_generated.npz') as cache:
                val_pairs.append(cache['pairs'].copy())
                assert cache['sampling_seed'].item() == 42
            with np.load(main / 'calibration/observed_graph.npz') as graph:
                assert len(graph['trajectory_keys']) == 40
                np.testing.assert_array_equal(graph['trajectory_keys'], [t.key for t in roles['calibration'].trajectories])
                stats = compute_relation_split_stats(roles['calibration'].trajectories)
                expected, _ = candidates(roles['calibration'].trajectories, stats['poi_idf'], graph['embeddings'], pair['kp'], pair['kc'], 16)
                np.testing.assert_array_equal(graph['pairs'], expected)
            cal = read_json(main / 'calibration/training.json')
            assert cal['source_cohort'] == manifest['roles']['calibration']
            assert not cal['validation_labels_used_for_fitting'] and not cal['test_labels_used_for_fitting']
            assert len(cal['features']['feature_names']) == 14
            with np.load(main / 'test/partition.npz') as part:
                assert len(part['pred']) == 120
            checks.append(dict(city=city, seed=seed, passed=True))
        assert allocations[0] == allocations[1] == allocations[2]
        np.testing.assert_array_equal(val_pairs[0], val_pairs[1])
        np.testing.assert_array_equal(val_pairs[1], val_pairs[2])
        assert len(set(encoder_hashes)) == 3 and len(set(mlp_hashes)) == 3
    # --resume must not change a single completed model/result, or train another epoch.
    protected = [p for p in experiment.rglob('*') if p.is_file() and ('/main/' in str(p) or '/preparation/' in str(p))]
    before = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in protected}
    with (output / 'resume.log').open('w') as handle:
        subprocess.run(command + ['--resume'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    assert before == {str(p): (sha256(p), p.stat().st_mtime_ns) for p in protected}
    assert read_json(experiment / 'progress.json')['status'] == 'complete'
    assert 'main_job_start' not in (output / 'resume.log').read_text()
    # A missing main-stage commit simulates interruption before a stage finished.
    # The complete encoder must be reused; every remaining old file is archived.
    city = cities[0]
    prep = experiment / 'seed_42/preparation' / city
    main = experiment / 'seed_42/main' / city
    marker = main / '_complete.json'
    marker.unlink()
    archived_expected = {p.relative_to(main).as_posix(): sha256(p) for p in main.rglob('*') if p.is_file()}
    stable = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in protected if not p.is_relative_to(main)}
    with np.load(main / 'test/partition.npz') as part:
        first_pred = part['pred'].copy()
    with (output / 'partial_resume.log').open('w') as handle:
        subprocess.run(command + ['--resume'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    assert stable == {name: (sha256(Path(name)), Path(name).stat().st_mtime_ns) for name in stable}
    history = list((experiment / 'incomplete_attempts').glob(f'*/seed_42/main/{city}'))
    assert len(history) == 1
    assert archived_expected == {p.relative_to(history[0]).as_posix(): sha256(p) for p in history[0].rglob('*') if p.is_file()}
    retry_log = (output / 'partial_resume.log').read_text()
    assert 'completed_encoder_reused' in retry_log and 'encoder_training' not in retry_log
    with np.load(main / 'test/partition.npz') as part:
        np.testing.assert_array_equal(first_pred, part['pred'])
    assert read_json(experiment / 'progress.json')['status'] == 'complete'
    write_json(output / 'audit.json', dict(passed=True, scope='ENGINEERING ONLY: 60 users per city, 2 epochs, CPU',
               formal_experiment_run=False, checks=checks, resume_preserves_all_completed_files=True,
               interrupted_backend_reuses_complete_encoder=True, incomplete_stage_files_preserved=True,
               resumed_partition_identical=True))
    print(f'PASS {len(checks)} end-to-end jobs, protocol audit and non-destructive resume: {output / "audit.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cities', nargs='+', default=['NYC'], choices=['Istanbul', 'Tokyo', 'Moscow', 'NYC'])
    args = parser.parse_args()
    run(args.output, args.cities)
