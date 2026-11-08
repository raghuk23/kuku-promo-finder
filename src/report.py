"""Write the raw outputs. Nothing here edits a judgement; it only formats."""
import csv
import json
import os
from typing import List

from .judge import fmt_ts

PROMO_KEYS = ("rank", "candidate_id", "episode", "segments", "total_duration_s", "hook", "cliffhanger",
              "why_it_works", "rank_reason", "final_score", "posting_score", "hook_score",
              "cliffhanger_score", "standalone_score", "standalone_reason", "weakness", "judge",
              "refinement_note", "original_cut", "warnings")


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def rank_reason(r: dict) -> str:
    s = r.get("standalone_score")
    parts = f"hook {r['hook_score']}, cliffhanger {r['cliffhanger_score']}" + (f", standalone {s}" if s else "")
    boost = " + early-episode boost" if (r.get("posting_score") or 0) > (r.get("final_score") or 0) + 1e-9 else ""
    return f"Score {r['final_score']:.1f}/10 ({parts}){boost}. {r.get('why_it_works') or ''}".strip()


def promo_view(r: dict) -> dict:
    segs = [{k: s[k] for k in ("episode", "start", "end", "duration_s")} for s in r["segments"]]
    v = dict(r, segments=segs, rank_reason=rank_reason(r),
             total_duration_s=round(sum(s["duration_s"] for s in segs), 2),
             final_score=round(r["final_score"], 2), posting_score=round(r["posting_score"], 2),
             warnings=(r.get("validation") or {}).get("warnings", []))
    return {k: v.get(k) for k in PROMO_KEYS}


def write_all(out_dir: str, *, show: str, summary: dict, parse_rep: dict, records: List[dict],
              ranked: dict, validation: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    promos = [promo_view(r) for r in ranked["ranked"]]
    _dump(os.path.join(out_dir, "promos_ranked.json"),
          {"show": show, "judge": summary["judges_used"], "prompt_version": summary["prompt_version"],
           "promos": promos})
    _dump(os.path.join(out_dir, "all_candidates.json"), records)
    _dump(os.path.join(out_dir, "validation_report.json"), validation)
    _dump(os.path.join(out_dir, "parse_report.json"), parse_rep)
    _dump(os.path.join(out_dir, "run_summary.json"), summary)

    cols = ["episode", "candidate_id", "start", "end", "duration_s", "final_score", "hook_score",
            "cliffhanger_score", "standalone_score", "below_floor", "floor_reasons", "posted_rank",
            "hook", "cliffhanger", "why_it_works", "note"]
    with open(os.path.join(out_dir, "best_per_episode.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked["best_per_episode"])

    md = [f"# Promo shortlist: {show}", "",
          f"Judge: {', '.join(summary['judges_used']) or 'n/a'} · prompt `{summary['prompt_version']}` · "
          f"{summary['funnel']['passed_floor']} promos passed the quality floor across "
          f"{summary['funnel']['episodes_with_a_promo']} of {summary['funnel']['episodes']} episodes · "
          f"generated {summary['finished_at']}", ""]
    if not promos:
        md.append("_No candidate passed the quality floor._")
    else:
        md += ["| # | Ep | Cut | Length | Score | Hook | Cliffhanger |", "|---|---|---|---|---|---|---|"]
        for p in promos:
            cut = " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"])
            md.append(f"| {p['rank']} | {p['episode']} | {cut} | {p['total_duration_s']:.0f}s | "
                      f"{p['final_score']:.1f} | {p['hook']} | {p['cliffhanger']} |")
        for p in promos:
            md += ["", f"## {p['rank']}. Episode {p['episode']} · "
                   + " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"]), "",
                   f"- **Hook:** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}",
                   f"- **Why it works:** {p['why_it_works']}", f"- **Why this rank:** {p['rank_reason']}",
                   f"- **Weakness:** {p['weakness']}", f"- **Cut:** {p['refinement_note']}"]
            if p["warnings"]:
                md.append(f"- **Check:** {'; '.join(p['warnings'])}")
    with open(os.path.join(out_dir, "promos_ranked.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
