"""Regression checks for turn nomination, training splits, swaps and evidence metrics."""
import unittest
from collections import Counter

import numpy as np

from evaluate import percentages
from pamr.ranking import mixed_order, nominate, rerank, turn_ranks
from pamr.training import training_fold
from retrieve import EXAMPLE, run


class PipelineTests(unittest.TestCase):
    def test_limits_are_per_seed_and_per_anchor(self):
        owners = [0,0,0,21,22,23,24,25,26,27,28,1,29]
        edges = [(0,3),(0,4),(0,5),(0,6),(1,7),(2,8),(9,10),(11,12)]
        ranks = np.arange(1, len(owners)+1)
        support = nominate(owners, ranks, ranks*0, edges, range(20))
        self.assertEqual(support, Counter([21,22,23,25,29]))

    def test_rrf_uses_only_positive_bm25_and_stable_ties(self):
        dr, wr = turn_ranks([.9,.4,.4,.2], [0.,2.,2.,0.])
        self.assertEqual(dr.tolist(), [1,2,3,4])
        self.assertEqual(wr.tolist(), [0,1,2,0])
        self.assertEqual(mixed_order(dr,wr).tolist(), [1,2,0,3])

    def test_one_hop_and_empty_graph(self):
        dr = np.arange(1,4)
        self.assertEqual(nominate([0,21,22],dr,dr*0,[(0,1),(1,2)],{0}), {21:1})
        self.assertFalse(nominate([],[],[],[],range(20)))

    def test_swaps_use_two_graph_slots_and_retire_both_sessions(self):
        scores = np.zeros(24)
        scores[[11,12,13]] = [3,4,10]
        order = rerank(scores, {11,12,13})
        self.assertEqual(order[0], 12)
        self.assertEqual(order[12], 0)
        self.assertEqual(sorted(order), list(range(24)))
        self.assertTrue(all(order[j] == i for i,j in enumerate(order) if i != j))
        self.assertLessEqual(sum(i != j for i,j in enumerate(order)), 8)

    def test_unfilled_slots_and_threshold(self):
        scores = np.zeros(10)
        scores[8] = .40
        self.assertEqual(rerank(scores,set()), list(range(10)))
        scores[8] = .41
        self.assertEqual(rerank(scores,set())[0], 8)

    def test_fold_excludes_variants_and_evidence_conflicts(self):
        rows = [dict(qid=i, group=g, fold=f, abstention=a) for i,(g,f,a) in enumerate(
            [(0,1,False),(0,1,True),(1,2,False),(2,3,False),(3,4,True)])]
        train, held = training_fold(rows,{0:[1],1:[0],2:[],3:[]},1)
        self.assertEqual([r["qid"] for r in held],[0,1])
        self.assertEqual([r["qid"] for r in train],[3])

    def test_all_evidence_recall_and_missing_candidates(self):
        rows = [dict(new_first=1, new_last=3),dict(new_first=2, new_last=None),
                dict(new_first=None, new_last=None)]
        p = percentages(rows, "new")
        self.assertAlmostEqual(p["Success@1"],100/3)
        self.assertAlmostEqual(p["Success@5"],200/3)
        self.assertEqual(p["Recall@1"],0)
        self.assertAlmostEqual(p["Recall@5"],100/3)
        self.assertEqual(p["MRR"],50)

    def test_cached_example_rankings(self):
        result = run(EXAMPLE)
        self.assertEqual([r["ranking"][0] for r in result], ["A","E","F"])
        for r in result:
            self.assertEqual(sorted(r["ranking"]), sorted(r["initial"]))
            self.assertEqual(set(r["nominated"]), set("ABCDEF"))


if __name__ == "__main__":
    unittest.main()
