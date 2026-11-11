#!/usr/bin/env python3
"""Compare newly computed metrics with the archived reference; never replace them."""
import argparse
from pathlib import Path
from cic.io import read_json, write_json

ROOT = Path(__file__).resolve().parent


def compare(run):
    run = Path(run).resolve()
    actual = read_json(run / 'summary.json')
    if actual['experiment_kind'] != 'formal':
        raise ValueError('Engineering smoke results are not comparable to the formal reference')
    reference = read_json(ROOT / 'reference/main_results.json')
    lookup = {(r['city'], r['seed']): r for r in reference['per_seed']}
    comparisons = []
    for row in actual['per_seed']:
        ref = lookup[(row['city'], row['seed'])]
        if (row['kp'], row['kc']) != (ref['kp'], ref['kc']):
            raise ValueError('Fixed budgets differ from the reference')
        item = dict(city=row['city'], seed=row['seed'])
        for key in ('G_Acc_percent', 'Pairwise_F1_percent'):
            item[key] = dict(actual=row[key], reference=ref[key], difference_percentage_points=row[key]-ref[key])
        comparisons.append(item)
    output = dict(full_main_experiment=actual['full_main_experiment'], rows=comparisons,
                  note='Signed differences, actual minus archived reference. Hardware/library differences may affect numerical results.')
    write_json(run / 'reference_comparison.json', output)
    for row in comparisons:
        acc, f1 = row['G_Acc_percent'], row['Pairwise_F1_percent']
        print(f"{row['city']:9s} seed {row['seed']}  G-Acc {acc['actual']:.3f} (delta {acc['difference_percentage_points']:+.3f} pp)"
              f"  F1 {f1['actual']:.3f} (delta {f1['difference_percentage_points']:+.3f} pp)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/main')
    compare(parser.parse_args().run)
