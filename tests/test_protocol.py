import ast

from pathlib import Path

import unittest

import numpy as np

from fractions import Fraction

from cic.retrieval import candidates

from cic.preprocessing import TrajectoryPoint,TrajectoryRecord,compute_relation_split_stats

from cic.training import sample_pairs,encoder_validation_batches

from cic.data import split_users

from cic.partition import partition_probabilities

from cic.evaluation import evaluate_partition

from cic.calibration import fit_support, assemble

class ProtocolTests(unittest.TestCase):
    def test_fresh_unique_pairs_labels_and_reproducibility(self):
        y = np.repeat(np.arange(10),5)
        a,t = sample_pairs(y,120,.25,42)
        b,u = sample_pairs(y,120,.25,42)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(t,u)
        np.testing.assert_array_equal(t,y[a[:,0]]==y[a[:,1]])
        self.assertEqual(len(np.unique(a,axis=0)),120)
        self.assertEqual(int(t.sum()),30)
        self.assertTrue(np.all(a[:,0]<a[:,1]))

    def test_original_rounded_identity_splits(self):
        for n,counts in [(2090,(1254,418,418)),(1282,(769,256,257)),(1971,(1183,394,394)),(479,(287,96,96))]:
            out = split_users([str(i) for i in range(n)],42)
            self.assertEqual(tuple(map(len,out.values())),counts)
            self.assertFalse(set(out['train'])&set(out['test']))

    def test_single_user_remainder_encoder_batch(self):
        labels = np.repeat(np.arange(17),2)
        seq = [np.zeros((2,22),np.float32) for _ in labels]
        batches = encoder_validation_batches(seq,labels,16,42)
        self.assertEqual(sum(len(y) for _,y in batches),34)
        self.assertTrue(all(len(np.unique(y))>=2 for _,y in batches))

    def test_missing_edge_not_negative_and_isolate_preserved(self):
        edges = np.array([[0,1],[1,2]])
        p = np.array([.9,.9])
        pred,details = partition_probabilities(4,edges,p)
        self.assertEqual(pred[0],pred[2])  # via observed positive path; no invented 0--2 score
        self.assertNotEqual(pred[0],pred[3])
        self.assertEqual(details['additional_pairs'],0)
        self.assertAlmostEqual(details['expected_observed_pair_errors'],.2)
        self.assertFalse(details['uses_M'] or details['uses_K'])

    def test_hungarian_does_not_hide_fragmentation(self):
        metrics = evaluate_partition(np.array([0,0,1,1]),np.array([0,1,2,3]))
        self.assertEqual(metrics['Grouping_ACC'],.5)
        self.assertEqual(metrics['Pairwise_Recall'],0.)

    def test_calibration_rejects_single_class_instead_of_fabricating_edges(self):
        with self.assertRaises(ValueError):
            fit_support(np.zeros((4,6)),np.ones(4,dtype=int),42)

    def test_fourteen_feature_order_and_shared_support_only_on_observed_edges(self):
        records = [TrajectoryRecord(str(i),[TrajectoryPoint('p',0.,0.,0.,7,'c')]) for i in range(4)]
        e = np.zeros((4,32))
        e[:,0] = 1.
        pairs = np.array([[0,1],[0,2],[1,2]],np.int64)
        graph = dict(n=4,pairs=pairs,logits=np.array([1.,2.,3.]),embeddings=e)
        support = dict(feature_indices=list(range(6)),raw_coefficients=[0.]*6,raw_intercept=0.)
        x,info = assemble(graph,records,support,-2.)
        self.assertEqual(x.shape,(3,14))
        np.testing.assert_array_equal(x[:,0],graph['logits'])
        np.testing.assert_allclose(x[:,3],1.)
        np.testing.assert_allclose(x[:,6],.5)  # triangle: one shared neighbor, each degree two
        np.testing.assert_allclose(x[:,7:10],1.)
        np.testing.assert_allclose(x[:,10:],np.log(2.))
        self.assertEqual(info['additional_pairs'],0)
        self.assertEqual(info['common_neighbors']['zero_support_nodes'],1)

    def test_main_does_not_import_selector_or_call_candidate_recall(self):
        root = Path(__file__).resolve().parents[1]
        for p in [root/'main.py',root/'cic/inference.py',root/'cic/retrieval.py',root/'cic/training.py']:
            tree = ast.parse(p.read_text())
            modules = [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any('offline_k' in m or 'offline_select' in m for m in modules))
            calls = [n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertNotIn('minimum_union',calls)
            self.assertNotIn('positive_pairs',calls)
        text = (root/'main.py').read_text()
        self.assertLess(text.index('pred,report = infer('),text.index('metrics = evaluate_partition('))
