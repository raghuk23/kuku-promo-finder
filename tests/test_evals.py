"""Stage 6 eval metric tests."""
import unittest

from src.config import load_config
from src.evals import spearman, compare_runs, pick_review_set, human_agreement

CFG = load_config()


def rec(cid, ep, hook, cliff, stand, valid=True):
    return {"candidate_id": cid, "episode": ep, "status": "ok", "valid": valid,
            "hook_score": hook, "cliffhanger_score": cliff, "standalone_score": stand,
            "segments": [{"episode": ep, "start_s": 0.0, "end_s": 30.0}]}


class TestSpearman(unittest.TestCase):
    def test_perfect_and_inverse(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_ties_use_average_ranks(self):
        self.assertAlmostEqual(spearman([1, 1, 2], [1, 1, 2]), 1.0)

    def test_constant_input_is_undefined(self):
        self.assertIsNone(spearman([5, 5, 5], [1, 2, 3]))


class TestCompareRuns(unittest.TestCase):
    def test_identical_runs_agree_fully(self):
        a = [rec("c1", 1, 9, 8, 7), rec("c2", 1, 5, 5, 5), rec("c3", 2, 7, 7, 7)]
        m = compare_runs(a, [dict(r) for r in a], CFG)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["spearman_final"], 1.0)
        self.assertEqual(m["floor_agreement"], 1.0)
        self.assertEqual(m["mean_abs_diff"]["hook_score"], 0.0)
        self.assertEqual(m["top_per_episode_agreement"], 1.0)
        self.assertEqual(m["floor_confusion"], {"both_pass": 2, "a_only": 0, "b_only": 0, "neither": 1})

    def test_ranked_overlap_undefined_when_runs_cover_different_episodes(self):
        a = [rec("c1", 1, 9, 8, 7), rec("c3", 2, 7, 7, 7)]
        b = [rec("c1", 1, 9, 8, 7)]
        a[0]["rank"] = b[0]["rank"] = 1
        self.assertIsNone(compare_runs(a, b, CFG)["ranked_overlap"])

    def test_only_shared_valid_candidates_compared(self):
        a = [rec("c1", 1, 9, 8, 7), rec("c2", 1, 5, 5, 5), rec("c3", 2, 7, 7, 7, valid=False)]
        b = [rec("c1", 1, 8, 8, 7), rec("c2", 1, 6, 5, 5)]
        m = compare_runs(a, b, CFG)
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["mean_abs_diff"]["hook_score"], 1.0)


class TestReviewSet(unittest.TestCase):
    def test_top_bottom_random_disjoint_and_blind_shuffled(self):
        rs = [rec(f"c{i}", i, 1 + i % 10, 5, 5) for i in range(40)]
        for i, r in enumerate(rs):
            r["rank"] = i + 1 if i < 10 else None
            r["final_score"] = 10 - i * 0.2
        picked = pick_review_set(rs, n_top=10, n_bottom=5, n_random=10, seed=1)
        ids = [p["candidate_id"] for p in picked]
        self.assertEqual(len(ids), 25)
        self.assertEqual(len(set(ids)), 25)
        self.assertEqual({p["why_picked"] for p in picked}, {"top", "bottom", "random"})
        self.assertNotEqual(ids[:10], [f"c{i}" for i in range(10)])   # order is shuffled (blind)
        self.assertEqual(ids, [p["candidate_id"] for p in pick_review_set(rs, 10, 5, 10, seed=1)])


class TestHumanAgreement(unittest.TestCase):
    def test_floor_precision_recall_and_correlation(self):
        llm = [rec("c1", 1, 9, 9, 9), rec("c2", 2, 8, 8, 8), rec("c3", 3, 3, 3, 3), rec("c4", 4, 2, 2, 2)]
        human = [{"candidate_id": "c1", "human_score": "5"}, {"candidate_id": "c2", "human_score": "2"},
                 {"candidate_id": "c3", "human_score": "4"}, {"candidate_id": "c4", "human_score": "1"},
                 {"candidate_id": "c9", "human_score": ""}]
        m = human_agreement(llm, human, CFG)
        self.assertEqual(m["n_scored"], 4)
        self.assertAlmostEqual(m["floor_precision"], 0.5)   # c1 good, c2 not
        self.assertAlmostEqual(m["floor_recall"], 0.5)      # good = c1, c3; floor caught c1
        self.assertIsNotNone(m["spearman"])


if __name__ == "__main__":
    unittest.main()


def _load_evaluate_script():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_report_name_is_a_safe_filename():
    ev = _load_evaluate_script()
    assert ev.report_name(None, "Ablation: heuristics only vs LLM judge") == "ablation"
    assert ev.report_name(None, "Stability of the LLM judge") == "stability"
    assert ev.report_name("single_vs_two_orders", "anything") == "single_vs_two_orders"
