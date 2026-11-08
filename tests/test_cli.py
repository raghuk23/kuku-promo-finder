"""End-to-end: one command produces every raw output. Uses the no-LLM heuristic
provider, so it is free and offline."""
import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from src.cli import main
from tests.test_parse import TITLE_CASE, DATA_DIR, FILES

OUTPUTS = ("promos_ranked.json", "promos_ranked.md", "best_per_episode.csv", "all_candidates.json",
           "validation_report.json", "parse_report.json", "run_summary.json")


@unittest.skipUnless(FILES, "input pack not present (it is not committed)")
class TestFullRunHeuristic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        main(["--input", DATA_DIR, "--out", cls.tmp.name, "--provider", "heuristic", "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def test_every_output_written(self):
        for f in OUTPUTS:
            self.assertTrue(os.path.exists(self.p(f)), f)

    def test_ranked_promos_have_what_the_brief_asks_for(self):
        ranked = json.load(open(self.p("promos_ranked.json")))["promos"]
        self.assertTrue(1 <= len(ranked) <= 10)
        for i, r in enumerate(ranked, 1):
            self.assertEqual(r["rank"], i)
            self.assertTrue(r["segments"] and all({"episode", "start", "end"} <= set(s) for s in r["segments"]))
            for k in ("hook", "cliffhanger", "why_it_works", "rank_reason"):
                self.assertTrue(r[k], k)

    def test_all_timestamps_valid(self):
        self.assertEqual(json.load(open(self.p("validation_report.json")))["timestamp_validity_rate"], 1.0)

    def test_best_per_episode_has_all_fifty(self):
        rows = list(csv.DictReader(open(self.p("best_per_episode.csv"))))
        self.assertEqual(sorted(int(r["episode"]) for r in rows), list(range(1, 51)))

    def test_summary_records_judge_and_funnel(self):
        s = json.load(open(self.p("run_summary.json")))
        self.assertEqual(s["provider"], "heuristic")
        self.assertEqual(s["funnel"]["candidates"], 250)
        self.assertEqual(s["funnel"]["windows"], 12127)


class TestEpisodeArg(unittest.TestCase):
    def test_trailing_comma_and_spaces_are_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "episode-001.txt")
            open(f, "w").write(TITLE_CASE)
            s = main(["--input", f, "--out", os.path.join(d, "o"), "--provider", "heuristic",
                      "--episodes", " 1, ", "--quiet"])
            self.assertEqual(s["funnel"]["episodes"], 1)


class TestDegenerateInput(unittest.TestCase):
    def test_single_file_with_no_camera_cuts_finishes_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "episode-001.txt")
            open(f, "w").write(TITLE_CASE)   # one camera beat -> no legal windows
            out = os.path.join(d, "out")
            main(["--input", f, "--out", out, "--provider", "heuristic", "--quiet"])
            self.assertEqual(json.load(open(os.path.join(out, "promos_ranked.json")))["promos"], [])
            rows = list(csv.DictReader(open(os.path.join(out, "best_per_episode.csv"))))
            self.assertEqual(rows[0]["note"], "no valid candidate")


@unittest.skipUnless(FILES, "input pack not present (it is not committed)")
class TestBatchingAndQuota(unittest.TestCase):
    def run_with(self, client, *extra):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        with mock.patch("src.cli.make_client", return_value=client):
            s = main(["--input", DATA_DIR, "--out", d.name, "--episodes", "1,2,3,4,5,6,7",
                      "--no-cache", "--quiet", *extra])
        return s, json.load(open(os.path.join(d.name, "all_candidates.json")))

    def test_episodes_are_judged_in_batches(self):
        from tests.test_judge import FakeClient, good_judgement
        import re as _re
        client = FakeClient(lambda s, u, sc: {"judgements": [
            good_judgement(cid) for cid in _re.findall(r"### (ep\d+_c\d+)", u)]})
        s, recs = self.run_with(client, "--episodes-per-call", "3", "--order-seeds", "none")
        self.assertEqual(client.calls, 3)                      # 7 episodes / 3 per call
        self.assertTrue(all(r["status"] == "ok" for r in recs))

    def test_two_orderings_double_the_calls_and_are_recorded(self):
        from tests.test_judge import FakeClient, good_judgement
        import re as _re
        client = FakeClient(lambda s, u, sc: {"judgements": [
            good_judgement(cid) for cid in _re.findall(r"### (ep\d+_c\d+)", u)]})
        s, recs = self.run_with(client, "--episodes-per-call", "3", "--order-seeds", "none,1")
        self.assertEqual(client.calls, 6)
        self.assertEqual(s["order_seeds"], [None, 1])
        self.assertTrue(all(r["orders_judged"] == 2 for r in recs))

    def test_daily_quota_stops_the_run_and_still_writes_outputs(self):
        from tests.test_judge import FakeClient
        def quota(*a):
            raise Exception("429 RESOURCE_EXHAUSTED requests per day limit exceeded on Free Tier")
        client = FakeClient(quota)
        s, recs = self.run_with(client, "--episodes-per-call", "3")
        self.assertEqual(client.calls, 1)                      # stopped, no pointless retries
        self.assertTrue(all(r["status"] == "judge_failed" for r in recs))
        self.assertTrue(all("daily quota" in r["failure_reason"] for r in recs))
        self.assertIn("daily quota", s["stopped_early"])


if __name__ == "__main__":
    unittest.main()
