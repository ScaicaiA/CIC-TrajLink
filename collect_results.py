#!/usr/bin/env python3
"""Export actual completed results and three-seed mean +/- sample deviation."""
import argparse
import csv
import math
from pathlib import Path
import statistics

from cic.io import read_json, write_json
from cic.protocol import EXPECTED_SEEDS, PROTOCOL_NAME
from cic.runtime import binding, committed


def summarize(rows, cities, seeds=EXPECTED_SEEDS):
    expected = {(city, seed) for city in cities for seed in seeds}
    actual = [(r['city'], r['seed']) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected or sorted(seeds) != EXPECTED_SEEDS:
        raise ValueError('Each requested city must have seeds 42, 43 and 44 exactly once')
    numeric = [k for k in rows[0] if k not in ('city', 'seed')]
    output = {}
    for city in cities:
        members = sorted((r for r in rows if r['city'] == city), key=lambda r: r['seed'])
        summary = {}
        for key in numeric:
            values = [r[key] for r in members]
            if all(v is None for v in values):
                summary[key] = dict(mean=None, sample_std=None, n=0)
            elif any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f'Inconsistent/missing metric {city}/{key}')
            else:
                summary[key] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values), n=3)
        output[city] = dict(seeds=list(seeds), metrics=summary)
    return output


def result_row(result, preparation):
    m = result['metrics']
    encoder = read_json(preparation / 'encoder/training.json')
    rss = result['inference_resources']['stage_rss_peak_bytes']
    gpu = result['device_memory'].get('gpu_peak_allocated_bytes')
    return dict(city=result['city'], seed=result['training_seed'], kp=result['preset_kp'], kc=result['preset_kc'],
        G_Acc_percent=100*m['Grouping_ACC'], Pairwise_Precision_percent=100*m['Pairwise_Precision'],
        Pairwise_Recall_percent=100*m['Pairwise_Recall'], Pairwise_F1_percent=100*m['Pairwise_F1'],
        NMI=m['NMI'], ARI=m['ARI'], clusters=m['n_pred_clusters'], singletons=m['singleton_clusters'],
        largest_cluster=m['max_cluster_size'], candidate_pairs=result['candidate_pairs'],
        saved_pair_percent=100*result['pair_reduction_fraction'],
        final_test_candidate_recall_percent=100*result['candidate_audit']['candidate_pair_recall'],
        inference_seconds=result['inference_seconds'],
        mlp_training_seconds=result['training_resources']['stage_seconds'],
        calibration_seconds=result['calibration_resources']['stage_seconds'],
        encoder_epochs_seconds=sum(r['seconds'] for r in encoder['trace']),
        preparation_seconds=read_json(preparation / 'complete.json')['seconds'],
        peak_RSS_MiB=rss/2**20 if rss is not None else None,
        peak_CUDA_MiB=gpu/2**20 if gpu is not None else None)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root, print_output=True, verify_files=True):
    root = Path(root).resolve()
    meta = read_json(root / 'experiment.json')
    config = read_json(root / 'config_snapshot.json')
    if meta['protocol']['name'] != PROTOCOL_NAME:
        raise ValueError('Unexpected experiment protocol')
    rows, shared_roles = [], {}
    for seed in meta['training_seeds']:
        folder = root / f'seed_{seed}'
        for city in meta['cities']:
            dest = folder / 'main' / city
            marker = dest / '_complete.json'
            if not marker.exists():
                continue
            expected = binding(meta, city, seed, 'main')
            if verify_files:
                committed(dest, expected)
                if not committed(folder / 'preparation' / city, binding(meta, city, seed, 'encoder')):
                    raise ValueError(f'Missing completed encoder stage for {city}/seed_{seed}')
            elif read_json(marker)['binding'] != expected:
                raise ValueError('Completed result has a different run signature')
            result = read_json(dest / 'result.json')
            if result['city'] != city or result['training_seed'] != seed or result['protocol'] != PROTOCOL_NAME:
                raise ValueError('Result belongs to a different city, seed or protocol')
            pair = config['cities'][city]['fixed_k'][str(seed)]
            if (result['preset_kp'], result['preset_kc']) != (pair['kp'], pair['kc']):
                raise ValueError('Result uses different fixed budgets')
            if result['role_manifest'] != shared_roles.setdefault(city, result['role_manifest']):
                raise ValueError('Identity/trajectory splits changed between seeds')
            if read_json(dest / 'configuration.json')['code_sha256'] != meta['code_sha256']:
                raise ValueError('Mixed code versions')
            rows.append(result_row(result, folder / 'preparation' / city))
    complete_cities = [city for city in meta['cities']
                       if sorted(r['seed'] for r in rows if r['city'] == city) == EXPECTED_SEEDS]
    summaries = summarize([r for r in rows if r['city'] in complete_cities], complete_cities) if complete_cities else {}
    total = len(meta['cities']) * len(meta['training_seeds'])
    full = (len(rows) == 12 and set(meta['cities']) == {'Istanbul', 'Tokyo', 'Moscow', 'NYC'}
            and sorted(meta['training_seeds']) == EXPECTED_SEEDS and meta['experiment_kind'] == 'formal')
    summary = dict(experiment_kind=meta['experiment_kind'], full_main_experiment=full,
                   requested_jobs_complete=len(rows) == total, completed=len(rows), total=total,
                   seed_count=len(meta['training_seeds']), sample_std_ddof=1,
                   ensemble=False, best_seed_selection=False, split_manifests_equal_across_seeds=True,
                   per_seed=rows, per_city=summaries)
    write_json(root / 'summary.json', summary)
    write_csv(root / 'per_seed_results.csv', rows)
    aggregate = []
    for city, data in summaries.items():
        row = dict(city=city, seeds='42 43 44')
        for key, stat in data['metrics'].items():
            row[key + '_mean'] = stat['mean']
            row[key + '_sample_std'] = stat['sample_std']
        aggregate.append(row)
    write_csv(root / 'mean_std_results.csv', aggregate)
    lines = [f"Experiment kind: {meta['experiment_kind']}", f'Completed jobs: {len(rows)}/{total}',
             'G-Acc/P/R/F1: percent. Mean +/- sample standard deviation (ddof=1), only for complete three-seed cities.',
             '', 'city       seed    kp    kc     G-Acc      P-F1     Recall   clusters   inference_s']
    for r in rows:
        lines.append(f"{r['city']:9s} {r['seed']:4d} {r['kp']:5d} {r['kc']:5d} "
                     f"{r['G_Acc_percent']:9.3f} {r['Pairwise_F1_percent']:9.3f} "
                     f"{r['Pairwise_Recall_percent']:9.3f} {r['clusters']:9d} {r['inference_seconds']:13.3f}")
    lines.extend(['', 'city         G-Acc mean +/- std       F1 mean +/- std'])
    for city, data in summaries.items():
        acc, f1 = data['metrics']['G_Acc_percent'], data['metrics']['Pairwise_F1_percent']
        lines.append(f"{city:9s} {acc['mean']:9.3f} +/- {acc['sample_std']:.3f}  {f1['mean']:9.3f} +/- {f1['sample_std']:.3f}")
    if not summaries:
        lines.append('No complete three-seed city yet; individual results are listed above.')
    (root / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if print_output:
        print('\n'.join(lines), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'runs/main')
    collect(parser.parse_args().root)
