import copy
from pathlib import Path
import unittest
import numpy as np

from cic.data import Split, split_source_users, describe_split
from cic.preprocessing import TrajectoryRecord, TrajectoryPoint, compute_relation_split_stats
from cic.protocol import training_config, neural_provenance, validate_neural_checkpoint
from cic.io import read_json
from cic.training import sample_pairs, encoder_validation_batches
from collect_results import summarize

ROOT = Path(__file__).resolve().parents[1]


def source_fixture(n):
    users = [f'u{i:04d}' for i in range(n)]
    records = [TrajectoryRecord(f'{uid}_week{week}',
        [TrajectoryPoint(f'private_{uid}',0.,0.,0.,7,'c')])
        for uid in users for week in range(3)]
    return Split(records, np.repeat(np.arange(n),3), users)


class HoldoutTests(unittest.TestCase):
    def test_whole_user_counts_and_original_order_preserved(self):
        for n, expected in [(769,(692,77)),(1254,(1129,125)),(1183,(1065,118)),(287,(258,29))]:
            source = source_fixture(n)
            train,cal = split_source_users(source,42)
            self.assertEqual((len(train.users),len(cal.users)),expected)
            self.assertFalse(set(train.users)&set(cal.users))
            self.assertEqual(set(train.users)|set(cal.users),set(source.users))
            for split in (train,cal):
                self.assertEqual(len(split.trajectories),3*len(split.users))
                self.assertTrue(np.all(np.bincount(split.labels)==3))
                for i,uid in enumerate(split.users):
                    keys=[split.trajectories[j].key for j in np.flatnonzero(split.labels==i)]
                    self.assertEqual(keys,[f'{uid}_week{w}' for w in range(3)])

    def test_one_fixed_holdout_three_neural_seeds(self):
        cfg=read_json(ROOT/'config/four_cities.json')
        base=source_fixture(100)
        allocations=[]
        pair_sets=[]
        for seed in (42,43,44):
            c=training_config(cfg,seed)
            self.assertEqual(c['seed'],seed)
            self.assertNotIn('seed',c['mlp'])
            self.assertEqual(c['calibration_seed'],42)
            self.assertEqual(c['validation_sampling_seed'],42)
            train,cal=split_source_users(base,cfg['protocol']['calibration_split_seed'])
            allocations.append((train.users,cal.users))
            pairs,_=sample_pairs(train.labels,80,.25,seed)
            pair_sets.append(pairs)
            # Calibration-only POIs cannot enter training relation statistics.
            stats=compute_relation_split_stats(train.trajectories)
            self.assertTrue(all(f'private_{uid}' not in stats['poi_df'] for uid in cal.users))
            self.assertTrue(all(pairs.ravel()<len(train.trajectories)))
        self.assertEqual(allocations[0],allocations[1])
        self.assertEqual(allocations[1],allocations[2])
        self.assertFalse(np.array_equal(pair_sets[0],pair_sets[1]))

    def test_validation_sampling_unchanged_across_training_seeds(self):
        cfg=read_json(ROOT/'config/four_cities.json')
        source=source_fixture(24)
        pairs=[];batch_labels=[]
        sequences=[np.full((2,22),i,np.float32) for i in range(len(source.labels))]
        for seed in (42,43,44):
            c=training_config(cfg,seed)
            pairs.append(sample_pairs(source.labels,60,.25,c['validation_sampling_seed'])[0])
            batches=encoder_validation_batches(sequences,source.labels,8,c['validation_sampling_seed'])
            batch_labels.append([tuple(x[0,0] for x in seq) for seq,_ in batches])
        np.testing.assert_array_equal(pairs[0],pairs[1])
        np.testing.assert_array_equal(pairs[1],pairs[2])
        self.assertEqual(batch_labels[0],batch_labels[1])
        self.assertEqual(batch_labels[1],batch_labels[2])

    def test_checkpoint_rejects_old_users_and_wrong_seed(self):
        config=read_json(ROOT/'config/four_cities.json')
        cfg=training_config(config,42)
        train,cal=split_source_users(source_fixture(100))
        val=source_fixture(20)
        val.users=[f'val_{u}' for u in val.users]
        provenance=neural_provenance(train,val,cfg)
        manifest=dict(roles=dict(neural_train=describe_split(train),val=describe_split(val)))
        checkpoint=dict(provenance=provenance,seed=42)
        validate_neural_checkpoint(checkpoint,manifest,cfg)
        for wrong in ({'seed':42},dict(checkpoint,seed=43)):
            with self.assertRaises(ValueError):validate_neural_checkpoint(wrong,manifest,cfg)
        wrong=copy.deepcopy(checkpoint)
        wrong['provenance']['neural_train']=describe_split(source_fixture(100))
        with self.assertRaises(ValueError):validate_neural_checkpoint(wrong,manifest,cfg)

    def test_small_calibration_does_not_fall_back_to_validation(self):
        with self.assertRaises(ValueError):split_source_users(source_fixture(10))

    def test_main_calibration_has_no_validation_graph(self):
        source=(ROOT/'main.py').read_text()
        section=source[source.index("calibration = roles['calibration']"):source.index('# All source model fitting')]
        self.assertNotIn("roles['val']",section)
        self.assertIn('compute_relation_split_stats(calibration.trajectories)',section)
        self.assertIn('fit_calibration(graph,calibration,cfg',section)
        self.assertNotIn('induced_edges',source)
        self.assertNotIn('calibration_nodes',source)


class AggregateTests(unittest.TestCase):
    def test_mean_and_sample_std_not_best(self):
        rows=[dict(city='Tokyo',seed=s,acc=v,memory=None) for s,v in zip((42,43,44),(.6,.7,.8))]
        out=summarize(rows,['Tokyo'])['Tokyo']['metrics']
        self.assertAlmostEqual(out['acc']['mean'],.7)
        self.assertAlmostEqual(out['acc']['sample_std'],.1)
        self.assertEqual(out['acc']['n'],3)
        self.assertIsNone(out['memory']['mean'])

    def test_incomplete_duplicate_and_mixed_seed_rejected(self):
        rows=[dict(city='Tokyo',seed=s,acc=.7) for s in (42,43,44)]
        for bad in (rows[:2],rows+[rows[0]],rows[:2]+[dict(city='Tokyo',seed=45,acc=.9)]):
            with self.assertRaises(ValueError):summarize(bad,['Tokyo'])


if __name__=='__main__':
    unittest.main()
