"""Stage 3 judge tests. The LLM is replaced by a fake client: these tests check
our code's handling of whatever comes back, not the model's taste."""
import tempfile
import unittest

from src.config import load_config
from src.candidates import Candidate, cut_points
from src.llm import LLMClient, JsonCache, QuotaExhaustedError
from src.judge import (build_user_prompt, parse_judgements, refine, judge_episode,
                       judge_batch, beat_id, PROMPT_VERSION)
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

    def test_refinement_that_cuts_out_the_quoted_cliffhanger_is_reverted(self):
        # Real case (ep17_c4): the model shortened its cut, then described a line
        # that was no longer inside it.
        c = self.cs[0]                                   # 10 -> 40, contains "Who are you?" at 22
        start_late = bid(self.ep, "Medium shot.", start=25.0)  # start at 25s drops the 22s line
        resp = {"judgements": [good_judgement(c.candidate_id, cliffhanger="She asks 'Who are you?'",
                                              start_beat=start_late,
                                              end_beat=bid(self.ep, "Medium shot.", start=45.0)),
                               good_judgement(self.cs[1].candidate_id)]}
        r = judge_episode(self.ep, self.cs, FakeClient(lambda *a: resp), CFG)[0]
        self.assertEqual((r["segments"][0]["start_s"], r["segments"][0]["end_s"]), (10.0, 40.0))
        self.assertIn("reverted", r["refinement_note"])

    def test_cache_avoids_second_call(self):
        client = FakeClient(self.respond_all)
        with tempfile.TemporaryDirectory() as d:
            cache = JsonCache(d)
            judge_episode(self.ep, self.cs, client, CFG, cache=cache)
            judge_episode(self.ep, self.cs, client, CFG, cache=cache)
        self.assertEqual(client.calls, 1)

    def test_cached_answer_from_a_different_model_is_rejudged_when_model_is_required(self):
        # Strict mode: one global ranking must come from one judge.
        with tempfile.TemporaryDirectory() as d:
            cache = JsonCache(d)
            judge_episode(self.ep, self.cs, FakeClient(self.respond_all), CFG, cache=cache)
            strict = FakeClient(self.respond_all)
            strict.required_model = strict.last_model = "primary-model"
            res = judge_episode(self.ep, self.cs, strict, CFG, cache=cache)
            self.assertEqual(strict.calls, 1)
            self.assertEqual(res[0]["judge"], "primary-model")
            again = FakeClient(self.respond_all)
            again.required_model = again.last_model = "primary-model"
            judge_episode(self.ep, self.cs, again, CFG, cache=cache)
            self.assertEqual(again.calls, 0)          # now cached under the right model

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


class TestJudgeBatch(unittest.TestCase):
    """Free tier: 20 requests/day/model. Several episodes per call keeps a full
    show inside one day's quota."""

    def setUp(self):
        self.e1, self.e2 = make_ep(), make_ep()
        self.e2.number = 2
        for b in self.e2.beats:
            b.episode = 2
        self.c1 = [cand(self.e1, "ep01_c1", 10, 40)]
        self.c2 = [cand(self.e2, "ep02_c1", 20, 50), cand(self.e2, "ep02_c2", 10, 35)]
        self.all = self.c1 + self.c2

    def respond_all(self, *_):
        return {"judgements": [good_judgement(c.candidate_id) for c in self.all]}

    def test_one_call_judges_several_episodes(self):
        client = FakeClient(self.respond_all)
        res = judge_batch([(self.e1, self.c1), (self.e2, self.c2)], client, CFG)
        self.assertEqual(client.calls, 1)
        self.assertEqual([(r["episode"], r["status"]) for r in res], [(1, "ok"), (2, "ok"), (2, "ok")])

    def test_batch_prompt_keeps_each_episode_separate(self):
        seen = {}
        def capture(system, user, schema):
            seen["user"] = user
            return self.respond_all()
        judge_batch([(self.e1, self.c1), (self.e2, self.c2)], FakeClient(capture), CFG)
        self.assertIn("Episode 1:", seen["user"])
        self.assertIn("Episode 2:", seen["user"])

    def test_single_episode_batch_prompt_is_unchanged(self):
        seen = {}
        def capture(system, user, schema):
            seen["user"] = user
            return {"judgements": [good_judgement("ep01_c1")]}
        judge_batch([(self.e1, self.c1)], FakeClient(capture), CFG)
        self.assertEqual(seen["user"], build_user_prompt(self.e1, self.c1, CFG))

    def test_daily_quota_raises_so_the_run_can_stop(self):
        def quota(*a):
            raise Exception("429 RESOURCE_EXHAUSTED requests per day limit exceeded on Free Tier")
        with self.assertRaises(QuotaExhaustedError):
            judge_batch([(self.e1, self.c1)], FakeClient(quota), CFG)


if __name__ == "__main__":
    unittest.main()
