"""Stage 4 validation tests: hard errors reject a promo, warnings flag it."""
import copy
import unittest

from src.config import load_config
from src.validate import validate_record, validate_all
from src.judge import judge_episode
from tests.test_judge import make_ep, cand, FakeClient, good_judgement

CFG = load_config()


def records(ep=None, **overrides):
    ep = ep or make_ep()
    cs = [cand(ep, "ep01_c1", 10, 40), cand(ep, "ep01_c2", 20, 50)]
    resp = {"judgements": [good_judgement(c.candidate_id, **overrides) for c in cs]}
    return ep, judge_episode(ep, cs, FakeClient(lambda *a: resp), CFG)


class TestValidateRecord(unittest.TestCase):
    def test_legal_record_has_no_errors(self):
        ep, rs = records()
        self.assertEqual(validate_record(rs[0], ep, CFG)["errors"], [])

    def test_start_not_on_a_cut_point(self):
        ep, rs = records()
        r = copy.deepcopy(rs[0]); r["segments"][0]["start_s"] = 12.3
        self.assertTrue(any("legal cut" in e for e in validate_record(r, ep, CFG)["errors"]))

    def test_too_long(self):
        ep, rs = records(make_ep(duration=90))
        r = copy.deepcopy(rs[0]); r["segments"][0]["start_s"], r["segments"][0]["end_s"] = 10.0, 75.0
        self.assertTrue(any("duration" in e for e in validate_record(r, ep, CFG)["errors"]))

    def test_past_episode_end(self):
        ep, rs = records()
        r = copy.deepcopy(rs[0]); r["segments"][0]["end_s"] = 65.0
        self.assertTrue(any("episode end" in e for e in validate_record(r, ep, CFG)["errors"]))

    def test_averaged_fractional_scores_are_valid(self):
        ep, rs = records()
        r = copy.deepcopy(rs[0]); r["hook_score"] = 7.5
        self.assertEqual(validate_record(r, ep, CFG)["errors"], [])

    def test_empty_hook(self):
        ep, rs = records(hook="")
        self.assertTrue(any("hook" in e for e in validate_record(rs[0], ep, CFG)["errors"]))

    def test_quote_present_in_clip_is_fine(self):
        ep, rs = records(cliffhanger='She asks "Who are you?"')
        self.assertEqual(validate_record(rs[0], ep, CFG)["warnings"], [])

    def test_single_quoted_lines_are_checked_too(self):
        # The real judge quotes with single quotes: 'Then let her suffer'.
        ep, rs = records(cliffhanger="She whispers 'I am the Duke of Ashcroft' and leaves.")
        self.assertTrue(any("quote not found" in x for x in validate_record(rs[0], ep, CFG)["warnings"]))
        ep, rs = records(cliffhanger="She asks 'Who are you?' before the cut.")
        self.assertEqual(validate_record(rs[0], ep, CFG)["warnings"], [])

    def test_apostrophes_are_not_mistaken_for_quotes(self):
        ep, rs = records(hook="Juliana's sister's betrayal is revealed at the altar's edge.")
        self.assertEqual(validate_record(rs[0], ep, CFG)["warnings"], [])

    def test_invented_quote_is_flagged(self):
        ep, rs = records(cliffhanger='He says "I am the Duke of Ashcroft."')
        w = validate_record(rs[0], ep, CFG)["warnings"]
        self.assertTrue(any("quote not found" in x for x in w))


class TestValidateAll(unittest.TestCase):
    def test_report_counts_and_marks_records(self):
        ep, rs = records()
        bad = copy.deepcopy(rs[1]); bad["segments"][0]["start_s"] = 12.3
        failed = copy.deepcopy(rs[0]); failed["status"] = "judge_failed"
        allr = [rs[0], bad, failed]
        rep = validate_all(allr, {ep.number: ep}, CFG)
        self.assertEqual((rep["judged_ok"], rep["valid"], rep["invalid"], rep["judge_failed"]), (2, 1, 1, 1))
        self.assertAlmostEqual(rep["timestamp_validity_rate"], 0.5)
        self.assertTrue(rs[0]["valid"])
        self.assertFalse(bad["valid"])
        self.assertFalse(failed["valid"])


if __name__ == "__main__":
    unittest.main()
