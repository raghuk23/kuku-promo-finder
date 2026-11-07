"""Stage 1 parser tests.

Fixtures mirror the real input pack's format, including the two quirks found
during recon: episode 6 uses UPPERCASE field labels, and speaker labels can
carry stacked parenthetical tags like `(Off-screen) (Vulnerable, Begging)`.
"""
import os
import glob
import unittest

from src.parse import (to_seconds, parse_episode_text, parse_episode_file,
                       normalise_speakers, load_show, parse_report)

TITLE_CASE = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A bride is abandoned at the altar.
CONTINUITY NOTES: First episode.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
Doors part.

**MATRIARCH (Cold, Commanding):**
Start: (00:11.30) | End: (00:13.10)
"You should turn around, Juliana."

**LADY JULIANA (Off-screen) (Vulnerable, Begging):**
Start: (00:13.50) | End: (00:14.30)
"Why?"

SCENE 02
Scene heading: EXT. STEPS – DAY
Scene start: 00:20.00
Scene end: 01:05.50

Actions and dialogue:

**ACTION:**
Start: (00:21.00) | End: (00:24.00)
She runs down the steps,
tears streaming.

**SOUND FX:**
Start: (01:00.00) | End: (01:05.50)
Ominous drone swells.

--- SCRIPT END ---
"""

# Episode 6 in the real pack uses uppercase field labels.
UPPER_CASE = (TITLE_CASE
              .replace("Scene heading:", "SCENE HEADING:")
              .replace("Scene start:", "SCENE START:")
              .replace("Scene end:", "SCENE END:")
              .replace("Actions and dialogue:", "ACTIONS AND DIALOGUE:"))


class TestTimestamps(unittest.TestCase):
    def test_minutes_seconds_hundredths(self):
        self.assertAlmostEqual(to_seconds("02:18.00"), 138.0)
        self.assertAlmostEqual(to_seconds("00:13.50"), 13.5)
        self.assertAlmostEqual(to_seconds("01:27.21"), 87.21)


class TestEpisodeHeader(unittest.TestCase):
    def test_number_and_logline(self):
        ep = parse_episode_text(TITLE_CASE)
        self.assertEqual(ep.number, 1)
        self.assertEqual(ep.logline, "A bride is abandoned at the altar.")


class TestScenes(unittest.TestCase):
    def test_scene_headings_and_bounds(self):
        ep = parse_episode_text(TITLE_CASE)
        self.assertEqual(len(ep.scenes), 2)
        self.assertEqual(ep.scenes[0].heading, "INT. CATHEDRAL – DAY")
        self.assertAlmostEqual(ep.scenes[1].start_s, 20.0)
        self.assertAlmostEqual(ep.scenes[1].end_s, 65.5)

    def test_duration_is_last_scene_end(self):
        self.assertAlmostEqual(parse_episode_text(TITLE_CASE).duration_s, 65.5)

    def test_uppercase_labels_parse_identically(self):
        # Regression for episode 6: a case-sensitive parser drops it silently.
        a, b = parse_episode_text(TITLE_CASE), parse_episode_text(UPPER_CASE)
        self.assertEqual([s.heading for s in a.scenes], [s.heading for s in b.scenes])
        self.assertEqual(len(a.beats), len(b.beats))
        self.assertEqual(len(b.beats), 5)


class TestBeats(unittest.TestCase):
    def setUp(self):
        self.beats = parse_episode_text(TITLE_CASE).beats

    def test_block_types(self):
        self.assertEqual([b.type for b in self.beats],
                         ["CAMERA", "DIALOGUE", "DIALOGUE", "ACTION", "SFX"])

    def test_beat_timing(self):
        self.assertAlmostEqual(self.beats[1].start_s, 11.3)
        self.assertAlmostEqual(self.beats[1].end_s, 13.1)

    def test_dialogue_text_has_quotes_stripped(self):
        self.assertEqual(self.beats[1].text, "You should turn around, Juliana.")

    def test_stacked_parenthetical_tags(self):
        b = self.beats[2]
        self.assertEqual(b.tags, ["Off-screen", "Vulnerable, Begging"])

    def test_speaker_is_raw_label_before_show_level_normalisation(self):
        b = self.beats[2]
        self.assertEqual(b.speaker, "LADY JULIANA")
        self.assertEqual(b.speaker_raw, "LADY JULIANA")

    def test_non_dialogue_has_no_speaker(self):
        self.assertIsNone(self.beats[0].speaker)

    def test_multiline_text_is_joined(self):
        self.assertEqual(self.beats[3].text, "She runs down the steps, tears streaming.")

    def test_beats_know_their_scene(self):
        self.assertEqual([b.scene for b in self.beats], [1, 1, 1, 2, 2])

    def test_beat_index_is_sequential(self):
        self.assertEqual([b.beat_index for b in self.beats], [0, 1, 2, 3, 4])


def _dialogue(speaker):
    return ('**%s (Calm):**\nStart: (00:01.00) | End: (00:02.00)\n"Hello."\n\n' % speaker)


def _ep_with(*speakers):
    body = "".join(_dialogue(s) for s in speakers)
    return parse_episode_text("EPISODE NUMBER: Episode 9\n\n--- SCRIPT START ---\n\n"
                              "SCENE 01\nScene heading: X\nScene start: 00:00.00\n"
                              "Scene end: 00:30.00\n\n" + body + "--- SCRIPT END ---\n")


class TestSpeakerNormalisation(unittest.TestCase):
    """`LADY X` merges into `X` only when `X` also speaks bare somewhere in the
    show -- evidence they are one person. Surnames (LADY ASHCROFT vs LORD
    ASHCROFT in the real pack) must stay distinct."""

    def test_merges_when_bare_name_speaks_elsewhere_in_show(self):
        a, b = _ep_with("JULIANA"), _ep_with("LADY JULIANA")
        normalise_speakers([a, b])
        self.assertEqual(b.beats[0].speaker, "JULIANA")
        self.assertEqual(b.beats[0].speaker_raw, "LADY JULIANA")

    def test_no_merge_without_bare_name_evidence(self):
        ep = _ep_with("LADY ASHCROFT", "LORD ASHCROFT")
        normalise_speakers([ep])
        self.assertEqual([x.speaker for x in ep.beats], ["LADY ASHCROFT", "LORD ASHCROFT"])

    def test_descriptive_labels_untouched(self):
        ep = _ep_with("LADY IN GOLD GOWN", "GOWN")
        normalise_speakers([ep])
        self.assertEqual(ep.beats[0].speaker, "LADY IN GOLD GOWN")


class TestAnomalies(unittest.TestCase):
    def test_label_without_timestamp_is_reported_not_fatal(self):
        broken = TITLE_CASE.replace("Start: (00:00.00) | End: (00:04.80)\n", "", 1)
        ep = parse_episode_text(broken)
        self.assertEqual(len(ep.beats), 4)
        self.assertTrue(any("no timestamp" in a for a in ep.anomalies))

    def test_end_before_start_is_reported(self):
        broken = TITLE_CASE.replace("End: (00:13.10)", "End: (00:05.00)", 1)  # starts 11.30
        ep = parse_episode_text(broken)
        self.assertTrue(any("end before start" in a for a in ep.anomalies))

    def test_clean_episode_has_no_anomalies(self):
        self.assertEqual(parse_episode_text(TITLE_CASE).anomalies, [])


# --- Real input pack (confidential, lives outside the repo) -----------------
DATA_DIR = os.environ.get(
    "KUKU_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "I Married a Broken Soldier. He was a Duke"))
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))


@unittest.skipUnless(FILES, "input pack not present (it is not committed)")
class TestRealInputPack(unittest.TestCase):
    """Numbers verified independently during recon with grep."""

    @classmethod
    def setUpClass(cls):
        cls.eps = [parse_episode_file(f) for f in FILES]

    def test_all_fifty_episodes_numbered_1_to_50(self):
        self.assertEqual(sorted(e.number for e in self.eps), list(range(1, 51)))

    def test_total_beats_matches_timestamp_line_count(self):
        self.assertEqual(sum(len(e.beats) for e in self.eps), 3727)

    def test_episode_6_is_not_dropped(self):
        ep6 = next(e for e in self.eps if e.number == 6)
        self.assertEqual(len(ep6.scenes), 2)
        self.assertGreater(len(ep6.beats), 50)
        self.assertAlmostEqual(ep6.duration_s, 115.0)

    def test_duration_range(self):
        d = sorted(e.duration_s for e in self.eps)
        self.assertAlmostEqual(d[0], 70.0, delta=1)
        self.assertAlmostEqual(d[-1], 201.0, delta=1)

    def test_real_aliases(self):
        normalise_speakers(self.eps)
        sp = {b.speaker for e in self.eps for b in e.beats if b.type == "DIALOGUE"}
        self.assertNotIn("LADY JULIANA", sp)
        self.assertIn("JULIANA", sp)
        self.assertTrue({"LADY ASHCROFT", "LORD ASHCROFT"} <= sp)

    def test_load_show_sorts_and_normalises(self):
        eps, merges = load_show(DATA_DIR)
        self.assertEqual([e.number for e in eps], list(range(1, 51)))
        self.assertEqual(merges["LADY JULIANA"], "JULIANA")

    def test_parse_report(self):
        eps, merges = load_show(DATA_DIR)
        r = parse_report(eps, merges)
        self.assertEqual(r["episodes"], 50)
        self.assertEqual(r["beats"], 3727)
        self.assertEqual(r["anomaly_count"], 0)
        ep6 = next(x for x in r["per_episode"] if x["episode"] == 6)
        self.assertEqual(ep6["scenes"], 2)
        self.assertAlmostEqual(ep6["duration_s"], 115.0)
        self.assertEqual(set(ep6["beat_types"]), {"CAMERA", "ACTION", "DIALOGUE", "SFX"})

    def test_no_anomalies_in_real_pack(self):
        bad = {e.number: e.anomalies for e in self.eps if e.anomalies}
        self.assertEqual(bad, {})


if __name__ == "__main__":
    unittest.main()
