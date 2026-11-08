"""Helpers behind the Streamlit app. The app calls the same runner as the CLI,
so the two can never disagree."""
import csv
import io
import json
import os
import tempfile
import zipfile

from .cli import main
from .judge import fmt_ts
from .parse import load_show


def load_run(out_dir: str) -> dict:
    """Read a finished output folder (e.g. the bundled outputs/gemini)."""
    def j(name):
        with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
            return json.load(fh)
    with open(os.path.join(out_dir, "best_per_episode.csv"), encoding="utf-8") as fh:
        best = list(csv.DictReader(fh))
    return {"promos": j("promos_ranked.json")["promos"], "summary": j("run_summary.json"),
            "validation": j("validation_report.json"), "records": j("all_candidates.json"),
            "best": best, "out_dir": out_dir, "scripts": {}, "log": []}


def _clip_lines(ep, segments):
    out = []
    for s in segments:
        for b in ep.beats:
            if s["start_s"] - 0.05 <= b.start_s < s["end_s"] - 0.05:
                who = (b.speaker_raw or "") if b.type == "DIALOGUE" else b.type
                text = f'"{b.text}"' if b.type == "DIALOGUE" else b.text
                out.append(f"{fmt_ts(b.start_s)}  {who}: {text}")
    return out


def run_pipeline(files, provider, model=None, api_key=None, episodes_per_call=None, log=None) -> dict:
    """files: [(filename, bytes)]. Inputs live in a temp folder that is deleted
    before this returns; only the outputs folder remains (for download)."""
    lines = []

    def _log(*x):
        s = " ".join(str(i) for i in x)
        lines.append(s)
        if log:
            log(s)

    out_dir = tempfile.mkdtemp(prefix="promo_out_")
    with tempfile.TemporaryDirectory(prefix="promo_in_") as d:
        for i, (name, data) in enumerate(files):
            safe = os.path.basename(name) or f"episode-{i + 1}.txt"
            if not safe.lower().endswith(".txt"):
                safe += ".txt"
            with open(os.path.join(d, safe), "wb") as fh:
                fh.write(data)
        argv = ["--input", d, "--out", out_dir, "--provider", provider]
        if model:
            argv += ["--model", model]
        if episodes_per_call:
            argv += ["--episodes-per-call", str(episodes_per_call)]
        main(argv, log=_log, api_key=api_key or None)
        res = load_run(out_dir)
        eps, _ = load_show(d)
        by_ep = {e.number: e for e in eps}
        recs = {r["candidate_id"]: r for r in res["records"]}
        for p in res["promos"]:
            r = recs[p["candidate_id"]]
            res["scripts"][p["candidate_id"]] = _clip_lines(by_ep[r["episode"]], r["segments"])
    res["log"] = lines
    return res


def zip_dir(out_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, name), arcname=name)
    return buf.getvalue()
