"""Stage 5 ranking tests."""
import unittest

from src.config import load_config
from src.rank import final_score, passes_floor, rank

CFG = load_config()


def rec(cid, ep, start, end, hook, cliff, stand, valid=True, status="ok"):
    return {"candidate_id": cid, "episode": ep, "status": status, "valid": valid,
            "segments": [{"episode": ep, "start_s": start, "end_s": end}],
            "hook_score": hook, "cliffhanger_score": cliff, "standalone_score": stand}


class TestScore(unittest.TestCase):
    def test_weighted_mean(self):
        self.assertAlmostEqual(final_score(rec("a", 1, 0, 30, 10, 5, 8), CFG), 0.45*10 + 0.30*5 + 0.25*8)

    def test_missing_standalone_renormalises(self):
        self.assertAlmostEqual(final_score(rec("a", 1, 0, 30, 10, 5, None), CFG), (0.45*10 + 0.30*5) / 0.75)


class TestFloor(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(passes_floor(rec("a", 1, 0, 30, 7, 7, 6), CFG)[0])

    def test_weak_hook_fails_with_reason(self):
        ok, why = passes_floor(rec("a", 1, 0, 30, 6, 9, 9), CFG)
        self.assertFalse(ok)
        self.assertTrue(any("hook" in w for w in why))

    def test_heuristic_without_standalone_is_judged_on_the_rest(self):
        self.assertTrue(passes_floor(rec("a", 1, 0, 30, 8, 8, None), CFG)[0])


class TestRank(unittest.TestCase):
    def test_overlapping_refined_cuts_are_deduplicated(self):
        a, b = rec("ep01_c1", 1, 0, 40, 9, 9, 9), rec("ep01_c2", 1, 5, 40, 8, 8, 8)
        out = rank([a, b], CFG, episodes=[1])
        self.assertEqual([r["candidate_id"] for r in out["ranked"]], ["ep01_c1"])
        self.assertEqual(b["duplicate_of"], "ep01_c1")

    def test_invalid_failed_and_below_floor_never_ranked(self):
        rs = [rec("ok", 1, 0, 30, 9, 9, 9), rec("inv", 2, 0, 30, 9, 9, 9, valid=False),
              rec("fail", 3, 0, 30, None, None, None, valid=False, status="judge_failed"),
              rec("weak", 4, 0, 30, 5, 9, 9)]
        out = rank(rs, CFG, episodes=[1, 2, 3, 4])
        self.assertEqual([r["candidate_id"] for r in out["ranked"]], ["ok"])
        self.assertIn("below quality floor", rs[3]["excluded_reason"])

    def test_max_two_per_episode_in_posting_order(self):
        rs = [rec(f"ep01_c{i}", 1, i * 30.0, i * 30.0 + 25, 9, 9, 9) for i in range(3)]
        rs.append(rec("ep02_c1", 2, 0, 25, 7, 7, 7))
        out = rank(rs, CFG, episodes=[1, 2])
        ids = [r["candidate_id"] for r in out["ranked"]]
        self.assertEqual(sum(i.startswith("ep01") for i in ids), 2)
        self.assertIn("ep02_c1", ids)
        self.assertIn("episode cap", rs[2]["excluded_reason"])

    def test_early_episode_bonus_breaks_ties_toward_episode_one(self):
        late, early = rec("ep30_c1", 30, 0, 30, 8, 8, 8), rec("ep01_c1", 1, 0, 30, 8, 8, 8)
        out = rank([late, early], CFG, episodes=[1, 30])
        self.assertEqual([r["candidate_id"] for r in out["ranked"]], ["ep01_c1", "ep30_c1"])
        self.assertGreater(early["posting_score"], early["final_score"])
        self.assertEqual(late["posting_score"], late["final_score"])

    def test_best_per_episode_covers_every_episode(self):
        rs = [rec("ep01_c1", 1, 0, 30, 9, 9, 9), rec("ep02_c1", 2, 0, 30, 5, 5, 5)]
        rows = {r["episode"]: r for r in rank(rs, CFG, episodes=[1, 2, 3])["best_per_episode"]}
        self.assertEqual(set(rows), {1, 2, 3})
        self.assertFalse(rows[1]["below_floor"])
        self.assertTrue(rows[2]["below_floor"])
        self.assertIsNone(rows[3]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
