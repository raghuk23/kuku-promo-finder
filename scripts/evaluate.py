"""Quality checks over finished runs. No model calls -- reads output folders.

  stability     compare two Gemini runs of the same candidates (e.g. shuffled order)
  ablation      heuristic-only run vs LLM run
  review-sheet  write the 25 clips for a blind human review
  human         score agreement once eval/review_sheet.csv is filled in

  .venv/bin/python scripts/evaluate.py ablation --a outputs/heuristic --b outputs/gemini
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ROOT, load_config  # noqa: E402
from src.evals import compare_runs, human_agreement, pick_review_set  # noqa: E402
from src.judge import fmt_ts  # noqa: E402
from src.parse import load_show  # noqa: E402

EVAL = os.path.join(ROOT, "eval")


def load(run):
    with open(os.path.join(run, "all_candidates.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.0f}%" if pct else f"{x:.2f}"


def compare_md(title, m, a_name, b_name, extra=""):
    d, c = m["mean_abs_diff"], m["floor_confusion"]
    return "\n".join([
        f"# {title}", "", f"Compared **{m['n']}** candidates judged validly in both `{a_name}` and `{b_name}`.", "",
        "| Metric | Value |", "|---|---|",
        f"| Spearman rank correlation of final scores | {fmt(m['spearman_final'])} |",
        f"| Mean absolute difference, hook / cliffhanger / standalone (1-10 scale) | "
        f"{fmt(d['hook_score'])} / {fmt(d['cliffhanger_score'])} / {fmt(d['standalone_score'])} |",
        f"| Candidates with identical scores | {fmt(m['exact_match_rate'], True)} |",
        f"| Same quality-floor decision | {fmt(m['floor_agreement'], True)} |",
        f"| Same best candidate per episode ({m['episodes_compared_for_top']} episodes) | "
        f"{fmt(m['top_per_episode_agreement'], True)} |",
        f"| Overlap of posted top lists | "
        f"{'n/a (runs cover different episodes)' if m['ranked_overlap'] is None else str(m['ranked_overlap']) + ' of ' + str(m['ranked_sizes'][0])} |", "",
        "Quality floor decisions:", "",
        f"| | {b_name} passes | {b_name} rejects |", "|---|---|---|",
        f"| **{a_name} passes** | {c['both_pass']} | {c['a_only']} |",
        f"| **{a_name} rejects** | {c['b_only']} | {c['neither']} |", "", extra]).rstrip() + "\n"


def report_name(name, title):
    """File stem for a report: --name if given, else the title's first word (letters/digits only)."""
    return name or re.sub(r"[^a-z0-9_]+", "", title.lower().split()[0])


def cmd_compare(a, title, extra_fn=None):
    cfg = load_config()
    A, B = load(a.a), load(a.b)
    m = compare_runs(A, B, cfg)
    extra = extra_fn(A, B) if extra_fn else ""
    name = report_name(a.name, title)
    os.makedirs(EVAL, exist_ok=True)
    with open(os.path.join(EVAL, f"{name}.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    md = compare_md(title, m, os.path.basename(os.path.normpath(a.a)), os.path.basename(os.path.normpath(a.b)), extra)
    with open(os.path.join(EVAL, f"{name}.md"), "w") as fh:
        fh.write(md)
    print(md)


def ties_note(A, B):
    tops = lambda R: sorted((r for r in R if r.get("rank")), key=lambda r: r["rank"])
    note = []
    for name, R in (("A", A), ("B", B)):
        t = tops(R)
        if t:
            note.append(f"- Run {name}: posted top {len(t)} has {len({round(r['final_score'], 2) for r in t})} "
                        f"distinct final scores (range {min(r['final_score'] for r in t):.2f}-"
                        f"{max(r['final_score'] for r in t):.2f}).")
    return "Ability to separate the top picks:\n\n" + "\n".join(note) if note else ""


def cmd_review_sheet(a):
    recs = load(a.run)
    picked = pick_review_set(recs, a.n_top, a.n_bottom, a.n_random, a.seed)
    eps, _ = load_show(a.input)
    by = {e.number: e for e in eps}
    os.makedirs(EVAL, exist_ok=True)
    rows, md, key = [], ["# Blind review clips (confidential: contains script text, not committed)", "",
                         "Score each clip 1-5 in eval/review_sheet.csv: *would I post this as a promo?* "
                         "5 = post today, 3 = maybe, 1 = never. Judge as a stranger to the show.", ""], {}
    for n, r in enumerate(picked, 1):
        s = r["segments"][0]
        rows.append({"clip": n, "candidate_id": r["candidate_id"], "episode": r["episode"],
                     "start": fmt_ts(s["start_s"]), "end": fmt_ts(s["end_s"]), "human_score": "", "notes": ""})
        key[r["candidate_id"]] = r["why_picked"]
        md += [f"## Clip {n}: episode {r['episode']}, {fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} "
               f"({s['end_s'] - s['start_s']:.0f}s)", ""]
        for b in by[r["episode"]].beats:
            if s["start_s"] - 0.05 <= b.start_s < s["end_s"] - 0.05:
                who = (b.speaker_raw or "") if b.type == "DIALOGUE" else b.type
                md.append(f"- `{fmt_ts(b.start_s)}` **{who}**: {b.text}")
        md.append("")
    with open(os.path.join(EVAL, "review_sheet.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(EVAL, "review_key.json"), "w") as fh:
        json.dump(key, fh, indent=2)
    with open(os.path.join(EVAL, "review_clips.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {len(rows)} clips: eval/review_sheet.csv (fill in), eval/review_clips.md (read)")


def cmd_human(a):
    cfg = load_config()
    with open(os.path.join(EVAL, "review_sheet.csv")) as fh:
        rows = list(csv.DictReader(fh))
    runs = [(n, r) for n, r in (x.split("=", 1) for x in a.runs) if os.path.isdir(r)]
    out = {name: human_agreement(load(run), rows, cfg) for name, run in runs}
    with open(os.path.join(EVAL, "human_agreement.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    n = max((m.get("n_scored", 0) for m in out.values()), default=0)
    lines = ["# Human agreement", "",
             f"Clips scored blind by a human: **{n}** (1-5, 'would I post this'; 4+ counts as good). "
             "The 25 mix the single-order top 10, its 5 weakest, and 10 random candidates.", "",
             "| Judge | Spearman vs human | Floor precision | Floor recall |", "|---|---|---|---|"]
    for name, m in out.items():
        lines.append(f"| {name} | {fmt(m.get('spearman'))} | {fmt(m.get('floor_precision'), True)} | "
                     f"{fmt(m.get('floor_recall'), True)} |")
    lines += ["", "Floor precision: of clips the judge would post, the share the human rated 4+. "
              "Floor recall: of clips the human rated 4+, the share the judge would post. "
              f"Small sample (n={n}): treat differences of a few points as noise."]
    with open(os.path.join(EVAL, "human_agreement.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("stability", "ablation"):
        p = sub.add_parser(c)
        p.add_argument("--a", required=True)
        p.add_argument("--b", required=True)
        p.add_argument("--name")
    p = sub.add_parser("review-sheet")
    p.add_argument("--run", default="outputs/gemini")
    p.add_argument("--input", default=os.path.join(ROOT, "..", "I Married a Broken Soldier. He was a Duke"))
    p.add_argument("--n-top", type=int, default=10)
    p.add_argument("--n-bottom", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("human")
    p.add_argument("--runs", nargs="+", default=["heuristic=outputs/heuristic",
                                                 "llm_single_order=outputs/gemini_single_order",
                                                 "llm_two_orders=outputs/gemini"])
    a = ap.parse_args()
    if a.cmd == "stability":
        cmd_compare(a, "Stability of the LLM judge")
    elif a.cmd == "ablation":
        cmd_compare(a, "Ablation: heuristics only vs LLM judge", ties_note)
    elif a.cmd == "review-sheet":
        cmd_review_sheet(a)
    else:
        cmd_human(a)


if __name__ == "__main__":
    main()
