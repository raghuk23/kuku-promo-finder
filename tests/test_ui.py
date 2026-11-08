"""Streamlit app helpers and a headless render of the app itself."""
import io
import os
import unittest
import zipfile

from src.config import ROOT
from src.ui import run_pipeline, load_run, zip_dir


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def screenplay(n=1, duration=60):
    """A small but realistic episode: camera grid, conflict lines, two speakers."""
    L = [f"EPISODE NUMBER: Episode {n}", "EPISODE LOGLINE: x", "CONTINUITY NOTES: y", "",
         "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. HALL – DAY",
         "Scene start: 00:00.00", f"Scene end: {mmss(duration)}", "", "Actions and dialogue:", ""]
    for t in range(0, duration, 5):
        L += ["**CAMERA:**", f"Start: ({mmss(t)}) | End: ({mmss(t + 5)})", "Medium shot.", ""]
        if t % 10 == 0:
            L += ["**LADY A (Furious):**", f"Start: ({mmss(t + 1)}) | End: ({mmss(t + 3)})",
                  '"How dare you betray me?"', ""]
        else:
            L += ["**B (Cold):**", f"Start: ({mmss(t + 1)}) | End: ({mmss(t + 3)})", '"Who are you?"', ""]
    return ("\n".join(L + ["--- SCRIPT END ---"]) + "\n").encode("utf-8")


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.res = run_pipeline([("episode-001.txt", screenplay(1)), ("episode-002.txt", screenplay(2, 75))],
                                provider="heuristic")

    def test_uploaded_files_produce_ranked_promos(self):
        self.assertTrue(self.res["promos"])
        self.assertEqual(self.res["summary"]["provider"], "heuristic")
        self.assertEqual(self.res["summary"]["funnel"]["episodes"], 2)

    def test_script_for_each_ranked_promo_is_available(self):
        p = self.res["promos"][0]
        lines = self.res["scripts"][p["candidate_id"]]
        self.assertTrue(any("How dare you" in l or "Who are you" in l for l in lines))

    def test_log_lines_are_streamed(self):
        self.assertTrue(any("parsed 2 episodes" in l for l in self.res["log"]))

    def test_outputs_zip(self):
        names = zipfile.ZipFile(io.BytesIO(zip_dir(self.res["out_dir"]))).namelist()
        self.assertIn("promos_ranked.json", names)
        self.assertIn("best_per_episode.csv", names)

    def test_load_run_reads_a_finished_output_folder(self):
        again = load_run(self.res["out_dir"])
        self.assertEqual([p["candidate_id"] for p in again["promos"]],
                         [p["candidate_id"] for p in self.res["promos"]])
        self.assertEqual(again["scripts"], {})


class TestAppRenders(unittest.TestCase):
    def test_app_starts_without_errors(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertTrue(any("promo" in t.value.lower() for t in at.title))


    def test_upload_and_run_flow(self):
        """The interview path: upload new screenplays, pick the free judge, run."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
        at.sidebar.radio[0].set_value("Run on new screenplays").run()
        fu = at.sidebar.file_uploader[0]
        fu.upload("episode-001.txt", screenplay(1), "text/plain")
        fu.upload("episode-002.txt", screenplay(2, 75), "text/plain")
        at.run()
        at.sidebar.button[0].click().run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        headings = [m.value for m in at.markdown if m.value.startswith("#### ")]
        self.assertTrue(headings and "Episode" in headings[0], headings)


if __name__ == "__main__":
    unittest.main()
