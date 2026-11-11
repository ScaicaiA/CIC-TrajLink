#!/usr/bin/env python3
"""Run the four-city, three-seed main experiment with fixed retrieval budgets."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

from cic.device import select_device, environment_info
from cic.io import read_json, write_json, log, code_fingerprint
from cic.runtime import load_config, object_hash, dependency_versions, run_lock, binding, committed
from prepare_data import prepare

ROOT = Path(__file__).resolve().parent


def execute(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True)
        try:
            for line in child.stdout:
                handle.write(line)
                handle.flush()
                print(line, end='', flush=True)
            status = child.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def run(args):
    config_path, output = args.config.resolve(), args.output.resolve()
    config = load_config(config_path)
    cities = args.cities or list(config['cities'])
    seeds = args.seeds or config['training_seeds']
    if len(cities) != len(set(cities)) or any(c not in config['cities'] for c in cities):
        raise ValueError('Unknown or duplicate city')
    if len(seeds) != len(set(seeds)) or any(s not in config['training_seeds'] for s in seeds):
        raise ValueError('Unknown or duplicate training seed')
    device = select_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / '.run.lock'):
        prepare(config_path, cities)
        snapshot = copy.deepcopy(config)
        for spec in snapshot['cities'].values():
            for key in ('data', 'archive'):
                if key in spec:
                    spec[key] = str((config_path.parent / spec[key]).resolve())
        fingerprint = code_fingerprint()
        environment = dict(**environment_info(device), dependencies=dependency_versions())
        signature = object_hash(dict(config=snapshot, cities=cities, seeds=seeds,
                                     code=fingerprint, environment=environment))
        meta_path = output / 'experiment.json'
        if meta_path.exists():
            if not args.resume:
                raise FileExistsError('Output already exists. Use --resume with the same arguments, or choose another --output')
            meta = read_json(meta_path)
            if meta['run_signature'] != signature or read_json(output / 'config_snapshot.json') != snapshot:
                raise ValueError('Code, data paths, configuration, selected jobs or environment changed; choose a new output directory')
        else:
            existing = [p.name for p in output.iterdir() if p.name != '.run.lock']
            if existing:
                raise FileExistsError(f'Output is not an empty experiment directory: {existing}')
            meta = dict(protocol=config['protocol'], training_seeds=seeds, cities=cities,
                        experiment_kind=config['experiment_kind'], code_sha256=fingerprint,
                        run_signature=signature, all_runs_complete=False, budget_mode=config['budget_mode'],
                        paired_seeds=True, ensemble=False, best_seed_selection=False)
            write_json(output / 'config_snapshot.json', snapshot)
            write_json(output / 'environment.json', environment)
            write_json(meta_path, meta)
        from collect_results import collect
        total, completed, start = len(cities) * len(seeds), 0, perf_counter()
        try:
            for seed in seeds:
                for city in cities:
                    if code_fingerprint() != fingerprint:
                        raise ValueError('Code changed during this run')
                    progress = dict(status='running', city=city, training_seed=seed,
                                    completed=completed, total=total, pid=os.getpid())
                    write_json(output / 'progress.json', progress)
                    dest = output / f'seed_{seed}' / 'main' / city
                    if committed(dest, binding(meta, city, seed, 'main')):
                        log('completed_job_reused', city=city, training_seed=seed)
                    else:
                        pair = config['cities'][city]['fixed_k'][str(seed)]
                        log('main_job_start', city=city, training_seed=seed, **pair)
                        execute([sys.executable, '-u', str(ROOT / 'train_one.py'), '--run', str(output),
                                 '--city', city, '--seed', str(seed), '--device', str(device)],
                                output / 'logs' / f'{city}_seed{seed}.log')
                        if not committed(dest, binding(meta, city, seed, 'main')):
                            raise RuntimeError('Worker exited without a committed result')
                    completed += 1
                    write_json(output / 'progress.json', dict(progress, completed=completed))
                    collect(output, print_output=False, verify_files=False)
            collect(output, print_output=True, verify_files=True)
            meta.update(all_runs_complete=True, last_invocation_seconds=perf_counter()-start)
            write_json(meta_path, meta)
            write_json(output / 'progress.json', dict(status='complete', completed=completed, total=total,
                                                      summary=str(output / 'summary.txt')))
            log('main_experiment_complete', completed=completed, total=total, summary=str(output / 'summary.txt'))
        except BaseException as error:
            write_json(output / 'progress.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       completed=completed, total=total, error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/four_cities.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/main')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:N, cpu or auto')
    parser.add_argument('--cities', nargs='+', choices=['Istanbul', 'Tokyo', 'Moscow', 'NYC'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 43, 44])
    parser.add_argument('--resume', action='store_true')
    signal.signal(signal.SIGTERM, interrupted)
    run(parser.parse_args())
