"""Stage 2 candidate generation tests."""
import itertools
import os
import unittest

from src.parse import Episode, Scene, Beat, load_show
from src.config import load_config
from src.candidates import (adaptive_k, iou, nms, cut_points, enumerate_windows,
                            load_lexicon, score_window, generate_candidates, Candidate)

CFG = load_config()
LEX = load_lexicon()


def ep_from(specs, duration=None, number=1, camera_every=5):
    """specs: (type, start, end, text[, speaker[, tags]]). A camera shot grid
    every `camera_every` seconds is added so legal cut points exist."""
    dur = duration or max(s[2] for s in specs)
    grid = [("CAMERA", float(t), float(min(t + camera_every, dur)), "Medium shot.")
            for t in range(0, int(dur), camera_every)]
    beats = []
    for i, s in enumerate(sorted(grid + list(specs), key=lambda s: (s[1], s[2]))):
        typ, st, en, tx = s[:4]
        sp = s[4] if len(s) > 4 else None
        tags = list(s[5]) if len(s) > 5 else []
        beats.append(Beat(i, number, 1, typ, sp, sp, tags, tx, float(st), float(en)))
    return Episode(number=number, scenes=[Scene(1, "X", 0.0, float(dur))], beats=beats)


def cand(start, end, score, ep=1):
    return Candidate(candidate_id=f"t{start}", episode=ep, start_s=start, end_s=end,
                     beat_indices=[], components={}, heuristic_score=score)


class TestAdaptiveK(unittest.TestCase):
    def test_matches_plan_formula(self):
        self.assertEqual(adaptive_k(70, CFG), 3)
        self.assertEqual(adaptive_k(118, CFG), 5)
        self.assertEqual(adaptive_k(201, CFG), 8)

    def test_clamped(self):
        self.assertEqual(adaptive_k(10, CFG), 3)
        self.assertEqual(adaptive_k(900, CFG), 8)


class TestIoU(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou((0, 30), (0, 30)), 1.0)

    def test_disjoint(self):
        self.assertAlmostEqual(iou((0, 30), (40, 60)), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(iou((0, 30), (15, 45)), 15 / 45)


class TestNMS(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = cand(0, 30, 0.9), cand(2, 32, 0.8), cand(40, 70, 0.7)

    def test_suppresses_heavy_overlap_keeps_distinct(self):
        self.assertEqual(nms([self.c, self.b, self.a], k=3, iou_thresh=0.5), [self.a, self.c])

    def test_stops_at_k(self):
        self.assertEqual(nms([self.a, self.b, self.c], k=1, iou_thresh=0.5), [self.a])


class TestCutPoints(unittest.TestCase):
    def test_camera_boundaries_minus_those_inside_a_spoken_line(self):
        ep = ep_from([("DIALOGUE", 8, 12, "Hello there.", "A")], duration=20)
        starts, ends = cut_points(ep, CFG)
        self.assertEqual(starts, [0.0, 5.0, 15.0])   # 10 is mid-line
        self.assertEqual(ends, [5.0, 15.0, 20.0])


class TestEnumerateWindows(unittest.TestCase):
    def test_only_legal_lengths_at_legal_points(self):
        ep = ep_from([], duration=60)
        w = enumerate_windows(ep, CFG)
        self.assertIn((0.0, 20.0), w)
        self.assertIn((0.0, 60.0), w)
        self.assertIn((40.0, 60.0), w)
        self.assertNotIn((45.0, 60.0), w)            # 15s: too short
        self.assertTrue(all(20 <= e - s <= 60 for s, e in w))


class TestScoring(unittest.TestCase):
    def comp(self, ep, t0, t1):
        return score_window(ep, t0, t1, LEX, CFG)[0]

    def test_slap_and_screamed_line_beats_establishing_shot_as_hook(self):
        ep = ep_from([
            ("ACTION", 0, 2, "She slaps him across the face."),
            ("DIALOGUE", 1, 3, "How dare you!", "A", ["Screaming"]),
            ("CAMERA", 20, 23, "Establishing wide shot of the manor."),
        ], duration=45)
        self.assertGreater(self.comp(ep, 0, 25)["hook"], 0.5)
        self.assertEqual(self.comp(ep, 20, 45)["hook"], 0.0)

    def test_intense_tag_alone_is_a_weaker_hook_than_the_briefs_archetypes(self):
        # Brief: a hook is "a slap, a reveal or a shocking line". An emotion tag
        # alone is common in this show (28% of lines) so it earns less.
        tag_only = ep_from([("DIALOGUE", 1, 3, "We should go.", "A", ["Cold"])], duration=25)
        slap = ep_from([("ACTION", 0, 2, "She slaps him.")], duration=25)
        reveal = ep_from([("CAMERA", 0, 2, "Her eyes widen as the veil falls, revealing his scar.")], duration=25)
        shocking = ep_from([("DIALOGUE", 1, 3, "You married a beggar.", "A")], duration=25)
        self.assertAlmostEqual(self.comp(tag_only, 0, 25)["hook"], 0.6)
        for ep in (slap, reveal, shocking):
            self.assertEqual(self.comp(ep, 0, 25)["hook"], 1.0)

    def test_push_in_and_tears_are_not_physical_conflict(self):
        ep = ep_from([
            ("CAMERA", 0, 2, "Push-in close-up on her face."),
            ("ACTION", 0, 3, "Tears roll down her cheeks."),
        ], duration=25)
        self.assertEqual(self.comp(ep, 0, 25)["hook"], 0.0)

    def test_ominous_sting_is_a_cliffhanger_peaceful_smile_is_not(self):
        ominous = ep_from([("SFX", 22, 25, "Ominous drone swells to a cliffhanger peak.")], duration=25)
        peaceful = ep_from([("CAMERA", 22, 25, "Close-up on her peaceful, contented smile.")], duration=25)
        self.assertGreater(self.comp(ominous, 0, 25)["cliffhanger"], 0.4)
        self.assertEqual(self.comp(peaceful, 0, 25)["cliffhanger"], 0.0)

    def test_unanswered_question_at_the_cut_raises_cliffhanger(self):
        q = ep_from([("DIALOGUE", 22, 24, "Who are you really?", "A")], duration=25)
        s = ep_from([("DIALOGUE", 22, 24, "Who you are does not matter.", "A")], duration=25)
        self.assertGreater(self.comp(q, 0, 25)["cliffhanger"], self.comp(s, 0, 25)["cliffhanger"])

    def test_speakers(self):
        two = ep_from([("DIALOGUE", 2, 4, "Go.", "A"), ("DIALOGUE", 5, 7, "No.", "B")], duration=25)
        one = ep_from([("DIALOGUE", 2, 4, "Go.", "A"), ("DIALOGUE", 5, 7, "Now.", "A")], duration=25)
        self.assertEqual(self.comp(two, 0, 25)["speakers"], 1.0)
        self.assertEqual(self.comp(one, 0, 25)["speakers"], 0.0)

    def test_length_fit(self):
        ep = ep_from([], duration=60)
        self.assertEqual(self.comp(ep, 0, 35)["length_fit"], 1.0)
        self.assertAlmostEqual(self.comp(ep, 0, 20)["length_fit"], 0.5)
        self.assertAlmostEqual(self.comp(ep, 0, 60)["length_fit"], 0.5)

    def test_total_is_weighted_sum_of_components(self):
        ep = ep_from([("DIALOGUE", 1, 3, "How dare you!", "A", ["Furious"])], duration=30)
        comps, total = score_window(ep, 0, 30, LEX, CFG)
        w = CFG["heuristic_weights"]
        self.assertAlmostEqual(total, sum(w[k] * comps[k] for k in w))


DATA_DIR = os.environ.get("KUKU_DATA_DIR", os.path.join(
    os.path.dirname(__file__), "..", "..", "I Married a Broken Soldier. He was a Duke"))


@unittest.skipUnless(os.path.isdir(DATA_DIR), "input pack not present (it is not committed)")
class TestRealCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps, _ = load_show(DATA_DIR)
        cls.cands = {e.number: generate_candidates(e, LEX, CFG) for e in cls.eps}

    def test_every_episode_gets_adaptive_k_candidates(self):
        short = {e.number: (len(self.cands[e.number]), adaptive_k(e.duration_s, CFG))
                 for e in self.eps if len(self.cands[e.number]) != adaptive_k(e.duration_s, CFG)}
        self.assertEqual(short, {})

    def test_all_candidates_legal(self):
        for e in self.eps:
            starts, ends = cut_points(e, CFG)
            for c in self.cands[e.number]:
                self.assertTrue(20 <= c.duration_s <= 60, c)
                self.assertIn(c.start_s, starts)
                self.assertIn(c.end_s, ends)
                self.assertLessEqual(c.end_s, e.duration_s)

    def test_no_two_candidates_overlap_beyond_nms_threshold(self):
        for cs in self.cands.values():
            for a, b in itertools.combinations(cs, 2):
                self.assertLessEqual(iou((a.start_s, a.end_s), (b.start_s, b.end_s)), 0.5)

    def test_ids_and_total(self):
        self.assertEqual(self.cands[6][0].candidate_id, "ep06_c1")
        total = sum(len(c) for c in self.cands.values())
        self.assertTrue(200 <= total <= 300, total)

    def test_deterministic(self):
        again = generate_candidates(self.eps[0], LEX, CFG)
        self.assertEqual([(c.start_s, c.end_s) for c in again],
                         [(c.start_s, c.end_s) for c in self.cands[self.eps[0].number]])


if __name__ == "__main__":
    unittest.main()
