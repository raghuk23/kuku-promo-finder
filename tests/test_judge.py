"""Stage 3 judge tests. The LLM is replaced by a fake client: these tests check
our code's handling of whatever comes back, not the model's taste."""
import tempfile
import unittest

from src.config import load_config
from src.candidates import Candidate, cut_points
from src.llm import LLMClient, JsonCache
from src.judge import (build_user_prompt, parse_judgements, refine, judge_episode,
                       beat_id, PROMPT_VERSION)
from tests.test_candidates import ep_from

CFG = load_config()


def make_ep(duration=60):
    ep = ep_from([
        ("DIALOGUE", 11, 13, "You should turn around.", "MATRIARCH", ["Cold"]),
        ("DIALOGUE", 22, 24, "Who are you?", "JULIANA", ["Desperate"]),
        ("ACTION", 31, 33, "She slaps him."),
    ], duration=duration)
    ep.logline = "SECRET LOGLINE THE JUDGE MUST NOT SEE"
    return ep


def cand(ep, cid, start, end, score=0.5):
    idx = [b.beat_index for b in ep.beats if b.start_s < end and b.end_s > start]
    return Candidate(cid, ep.number, float(start), float(end), idx, {"hook": 1.0}, score)


def bid(ep, text, start=None):
    b = next(b for b in ep.beats if b.text == text and (start is None or b.start_s == start))
    return beat_id(b)


class FakeClient(LLMClient):
    name = "fake"

    def __init__(self, responder):
        self.responder, self.calls, self.last_model = responder, 0, "fake-model"

    def complete_json(self, system, user, schema):
        self.calls += 1
        return self.responder(system, user, schema)


def good_judgement(cid, **kw):
    j = {"candidate_id": cid, "hook": "A bride is told to turn around.", "hook_score": 8,
         "cliffhanger": "She asks who he is.", "cliffhanger_score": 7,
         "standalone_reason": "Clear situation.", "standalone_score": 9,
         "weakness": "Slow middle.", "why_it_works": "Public humiliation, instantly.",
         "start_beat": "", "end_beat": ""}
    j.update(kw)
    return j


class TestPrompt(unittest.TestCase):
    def setUp(self):
        self.ep = make_ep()
        self.cs = [cand(self.ep, "ep01_c1", 10, 40), cand(self.ep, "ep01_c2", 20, 50)]

    def test_lists_every_candidate_with_beat_ids_and_in_out_marks(self):
        p = build_user_prompt(self.ep, self.cs, CFG)
        for s in ("ep01_c1", "ep01_c2", "[b", "IN ", "OUT", "You should turn around."):
            self.assertIn(s, p)

    def test_withholds_logline_so_standalone_is_judged_cold(self):
        self.assertNotIn("SECRET LOGLINE", build_user_prompt(self.ep, self.cs, CFG))

    def test_order_seed_changes_order_not_content(self):
        cs = self.cs + [cand(self.ep, "ep01_c3", 30, 55)]
        plain = build_user_prompt(self.ep, cs, CFG)
        order = lambda p: [c for c in ("ep01_c1", "ep01_c2", "ep01_c3") if c in p]
        self.assertEqual(sorted(order(plain)), ["ep01_c1", "ep01_c2", "ep01_c3"])
        shuffled = {tuple(sorted((c.candidate_id for c in cs),
                                 key=lambda cid: build_user_prompt(self.ep, cs, CFG, order_seed=s).index(cid)))
                    for s in range(6)}
        self.assertGreater(len(shuffled), 1)


class TestParse(unittest.TestCase):
    def setUp(self):
        self.ep = make_ep()
        self.cs = [cand(self.ep, "ep01_c1", 10, 40), cand(self.ep, "ep01_c2", 20, 50)]

    def test_valid_response(self):
        js, issues = parse_judgements({"judgements": [good_judgement("ep01_c1"),
                                                      good_judgement("ep01_c2")]}, self.cs)
        self.assertEqual(set(js), {"ep01_c1", "ep01_c2"})
        self.assertEqual(issues, [])

    def test_missing_candidate_reported_not_invented(self):
        js, issues = parse_judgements({"judgements": [good_judgement("ep01_c1")]}, self.cs)
        self.assertNotIn("ep01_c2", js)
        self.assertTrue(any("ep01_c2" in i and "missing" in i for i in issues))

    def test_unknown_candidate_ignored_and_reported(self):
        js, issues = parse_judgements({"judgements": [good_judgement("ep01_c1"), good_judgement("ep01_c2"),
                                                      good_judgement("ep99_c9")]}, self.cs)
        self.assertNotIn("ep99_c9", js)
        self.assertTrue(any("ep99_c9" in i for i in issues))

    def test_out_of_range_score_clamped_and_reported(self):
        js, issues = parse_judgements({"judgements": [good_judgement("ep01_c1", hook_score=14),
                                                      good_judgement("ep01_c2")]}, self.cs)
        self.assertEqual(js["ep01_c1"]["hook_score"], 10)
        self.assertTrue(any("hook_score" in i for i in issues))


class TestRefine(unittest.TestCase):
    def setUp(self):
        self.ep = make_ep()
        self.starts, self.ends = cut_points(self.ep, CFG)
        self.c = cand(self.ep, "ep01_c1", 10, 40)

    def test_snaps_to_nearest_legal_cut_points(self):
        s, e, note = refine(self.ep, self.c, bid(self.ep, "Who are you?"),
                            bid(self.ep, "Medium shot.", start=40.0), CFG)
        self.assertEqual((s, e), (20.0, 45.0))      # line at 22s snaps to the 20s cut
        self.assertIn(s, self.starts)
        self.assertIn(e, self.ends)

    def test_rejects_illegal_length_and_keeps_original(self):
        same = bid(self.ep, "Medium shot.", start=30.0)
        s, e, note = refine(self.ep, self.c, same, same, CFG)
        self.assertEqual((s, e), (10.0, 40.0))
        self.assertIn("rejected", note)

    def test_unknown_or_empty_beat_keeps_original(self):
        self.assertEqual(refine(self.ep, self.c, "b999", "", CFG)[:2], (10.0, 40.0))


class TestJudgeEpisode(unittest.TestCase):
    def setUp(self):
        self.ep = make_ep()
        self.cs = [cand(self.ep, "ep01_c1", 10, 40), cand(self.ep, "ep01_c2", 20, 50)]
        self.starts, self.ends = cut_points(self.ep, CFG)

    def respond_all(self, *_):
        return {"judgements": [good_judgement(c.candidate_id) for c in self.cs]}

    def test_end_to_end_with_fake_client(self):
        res = judge_episode(self.ep, self.cs, FakeClient(self.respond_all), CFG)
        self.assertEqual([r["status"] for r in res], ["ok", "ok"])
        for r in res:
            self.assertEqual(r["judge"], "fake-model")
            self.assertEqual(r["prompt_version"], PROMPT_VERSION)
            seg = r["segments"][0]
            self.assertIn(seg["start_s"], self.starts)
            self.assertIn(seg["end_s"], self.ends)
            self.assertEqual(seg["episode"], self.ep.number)

    def test_cache_avoids_second_call(self):
        client = FakeClient(self.respond_all)
        with tempfile.TemporaryDirectory() as d:
            cache = JsonCache(d)
            judge_episode(self.ep, self.cs, client, CFG, cache=cache)
            judge_episode(self.ep, self.cs, client, CFG, cache=cache)
        self.assertEqual(client.calls, 1)

    def test_malformed_response_marks_failed_never_fills_scores(self):
        res = judge_episode(self.ep, self.cs, FakeClient(lambda *a: {"nope": 1}), CFG)
        for r in res:
            self.assertEqual(r["status"], "judge_failed")
            self.assertIsNone(r["hook_score"])

    def test_client_exception_marks_failed(self):
        def boom(*a):
            raise Exception("404 NOT_FOUND")
        res = judge_episode(self.ep, self.cs, FakeClient(boom), CFG)
        self.assertTrue(all(r["status"] == "judge_failed" for r in res))

    def test_heuristic_judge_needs_no_client(self):
        res = judge_episode(self.ep, self.cs, None, CFG)
        for r in res:
            self.assertEqual(r["judge"], "heuristic")
            self.assertTrue(1 <= r["hook_score"] <= 10)
            self.assertIsNone(r["standalone_score"])   # heuristics cannot judge this
            self.assertTrue(r["hook"])


if __name__ == "__main__":
    unittest.main()
