"""Stage 5: score, apply the quality floor, dedupe, and choose a posting order.

Pure code with every number in config.yaml. Each record ends up with a plain
`excluded_reason` if it did not make the list, so an editor can see why.
"""
from collections import Counter, defaultdict
from typing import List, Optional

from .candidates import iou
from .judge import fmt_ts

FIELDS = (("hook", "hook_score"), ("cliffhanger", "cliffhanger_score"), ("standalone", "standalone_score"))


def final_score(rec: dict, cfg: dict) -> Optional[float]:
    """Weighted mean of the judge's 1-10 scores; renormalised if one is missing
    (the heuristic judge cannot score standalone-ness)."""
    w = cfg["ranking"]["weights"]
    avail = {k: rec.get(f) for k, f in FIELDS if rec.get(f) is not None}
    if "hook" not in avail:
        return None
    return sum(w[k] * v for k, v in avail.items()) / sum(w[k] for k in avail)


def passes_floor(rec: dict, cfg: dict):
    floor, reasons = cfg["ranking"]["quality_floor"], []
    for k, f in FIELDS:
        v = rec.get(f)
        if v is None:
            if k != "standalone":
                reasons.append(f"{k} missing")
            continue
        if v < floor[k]:
            reasons.append(f"{k} {v} < {floor[k]}")
    return not reasons, reasons


def _span(r):
    s = r["segments"][0]
    return s["start_s"], s["end_s"]


def rank(records: List[dict], cfg: dict, episodes: List[int]) -> dict:
    R = cfg["ranking"]
    for r in records:
        r.update(final_score=None, floor_pass=False, floor_reasons=[], duplicate_of=None,
                 posting_score=None, rank=None, excluded_reason=None)

    usable = []
    for r in records:
        if r.get("status") != "ok":
            r["excluded_reason"] = "judge failed"
        elif not r.get("valid"):
            r["excluded_reason"] = "failed validation"
        else:
            r["final_score"] = final_score(r, cfg)
            r["floor_pass"], r["floor_reasons"] = passes_floor(r, cfg)
            usable.append(r)

    by_score = lambda r: (-r["final_score"], -(r.get("hook_score") or 0), r["episode"], _span(r)[0])
    kept = defaultdict(list)
    for r in sorted(usable, key=by_score):
        dup = next((k for k in kept[r["episode"]] if iou(_span(r), _span(k)) > R["dedup_iou"]), None)
        if dup:
            r["duplicate_of"] = dup["candidate_id"]
            r["excluded_reason"] = f"duplicate of {dup['candidate_id']} (overlap after refinement)"
        else:
            kept[r["episode"]].append(r)

    eligible = [r for r in usable if r["duplicate_of"] is None]
    for r in eligible:
        if not r["floor_pass"]:
            r["excluded_reason"] = "below quality floor: " + "; ".join(r["floor_reasons"])
    pool = [r for r in eligible if r["floor_pass"]]
    for r in pool:
        bonus = R["early_episode_bonus"] * max(0.0, 1 - (r["episode"] - 1) / R["early_episode_window"])
        r["posting_score"] = r["final_score"] + bonus
    pool.sort(key=lambda r: (-r["posting_score"],) + by_score(r))

    ranked, per_ep = [], Counter()
    for r in pool:
        if per_ep[r["episode"]] >= R["max_per_episode"]:
            r["excluded_reason"] = f"episode cap (max {R['max_per_episode']} per episode in posting order)"
        elif len(ranked) >= R["top_n"]:
            r["excluded_reason"] = f"passed the floor; ranked below top {R['top_n']}"
        else:
            ranked.append(r)
            per_ep[r["episode"]] += 1
            r["rank"] = len(ranked)

    best = []
    for e in episodes:
        mine = [r for r in usable if r["episode"] == e]
        if not mine:
            best.append({"episode": e, "candidate_id": None, "below_floor": None,
                         "note": "no valid candidate"})
            continue
        r = min(mine, key=by_score)
        s, t = _span(r)
        best.append({"episode": e, "candidate_id": r["candidate_id"], "start": fmt_ts(s), "end": fmt_ts(t),
                     "duration_s": round(t - s, 2), "final_score": round(r["final_score"], 2),
                     "hook_score": r.get("hook_score"), "cliffhanger_score": r.get("cliffhanger_score"),
                     "standalone_score": r.get("standalone_score"), "below_floor": not r["floor_pass"],
                     "floor_reasons": "; ".join(r["floor_reasons"]), "posted_rank": r["rank"],
                     "hook": r.get("hook"), "cliffhanger": r.get("cliffhanger"),
                     "why_it_works": r.get("why_it_works"), "note": ""})

    funnel = {"judged_records": len(records),
              "judge_failed": sum(r.get("status") != "ok" for r in records),
              "failed_validation": sum(r["excluded_reason"] == "failed validation" for r in records),
              "duplicates_removed": sum(r["duplicate_of"] is not None for r in records),
              "below_floor": sum(1 for r in eligible if not r["floor_pass"]),
              "passed_floor": len(pool), "ranked": len(ranked),
              "episodes_with_a_promo": len({r["episode"] for r in pool})}
    return {"ranked": ranked, "pool": pool, "best_per_episode": best, "funnel": funnel}
