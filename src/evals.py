"""Stage 6: evaluation metrics. Pure functions over pipeline records.

Three questions, each answerable with real runs:
1. Stability -- does the judge give the same answer when candidate order changes?
2. Ablation  -- does the LLM rank differently from the free heuristics?
3. Human     -- does either agree with a person deciding "would I post this"?
"""
import random
from collections import defaultdict
from typing import List, Optional

from .rank import final_score, passes_floor

SCORE_KEYS = ("hook_score", "cliffhanger_score", "standalone_score")


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    """Rank correlation with average ranks for ties; None if undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def _usable(records):
    return {r["candidate_id"]: r for r in records if r.get("status") == "ok" and r.get("valid")}


def compare_runs(a: List[dict], b: List[dict], cfg: dict) -> dict:
    """Compare two judgements of the same candidates (run A vs run B)."""
    A, B = _usable(a), _usable(b)
    ids = sorted(set(A) & set(B))
    fa = {i: final_score(A[i], cfg) for i in ids}
    fb = {i: final_score(B[i], cfg) for i in ids}
    diffs = {}
    for k in SCORE_KEYS:
        d = [abs(A[i][k] - B[i][k]) for i in ids if A[i].get(k) is not None and B[i].get(k) is not None]
        diffs[k] = (sum(d) / len(d)) if d else None
    pa = {i: passes_floor(A[i], cfg)[0] for i in ids}
    pb = {i: passes_floor(B[i], cfg)[0] for i in ids}
    conf = {"both_pass": sum(pa[i] and pb[i] for i in ids), "a_only": sum(pa[i] and not pb[i] for i in ids),
            "b_only": sum(pb[i] and not pa[i] for i in ids), "neither": sum(not pa[i] and not pb[i] for i in ids)}
    by_ep = defaultdict(list)
    for i in ids:
        by_ep[A[i]["episode"]].append(i)
    best = lambda f, run, cs: max(cs, key=lambda i: (f[i], run[i].get("hook_score") or 0, i))
    eps = [cs for cs in by_ep.values() if len(cs) >= 2]
    top_agree = (sum(best(fa, A, cs) == best(fb, B, cs) for cs in eps) / len(eps)) if eps else None
    # Top lists are only comparable when both runs ranked the same episodes.
    same_eps = {r["episode"] for r in a} == {r["episode"] for r in b}
    ra = {r["candidate_id"] for r in a if r.get("rank")}
    rb = {r["candidate_id"] for r in b if r.get("rank")}
    return {
        "n": len(ids),
        "spearman_final": spearman([fa[i] for i in ids], [fb[i] for i in ids]),
        "mean_abs_diff": diffs,
        "exact_match_rate": (sum(all(A[i].get(k) == B[i].get(k) for k in SCORE_KEYS) for i in ids) / len(ids))
                            if ids else None,
        "floor_agreement": (sum(pa[i] == pb[i] for i in ids) / len(ids)) if ids else None,
        "floor_confusion": conf,
        "top_per_episode_agreement": top_agree,
        "episodes_compared_for_top": len(eps),
        "ranked_overlap": len(ra & rb) if same_eps else None, "ranked_sizes": [len(ra), len(rb)],
    }


def pick_review_set(records: List[dict], n_top: int, n_bottom: int, n_random: int, seed: int) -> List[dict]:
    """The clips a human scores: the posted top N, the weakest, and a random
    sample of the rest -- shuffled so the reviewer cannot tell which is which."""
    pool = [r for r in records if r.get("status") == "ok" and r.get("valid") and r.get("final_score") is not None]
    top = sorted((r for r in pool if r.get("rank")), key=lambda r: r["rank"])[:n_top]
    rest = [r for r in pool if r not in top]
    bottom = sorted(rest, key=lambda r: (r["final_score"], r["candidate_id"]))[:n_bottom]
    rest = [r for r in rest if r not in bottom]
    rng = random.Random(seed)
    rand = rng.sample(sorted(rest, key=lambda r: r["candidate_id"]), min(n_random, len(rest)))
    picked = ([dict(r, why_picked="top") for r in top] + [dict(r, why_picked="bottom") for r in bottom]
              + [dict(r, why_picked="random") for r in rand])
    rng.shuffle(picked)
    return picked


def human_agreement(records: List[dict], rows: List[dict], cfg: dict, good_at: int = 4) -> dict:
    """rows: [{'candidate_id', 'human_score' (1-5, blank = not scored)}]."""
    R = _usable(records)
    scored = [(row["candidate_id"], int(row["human_score"])) for row in rows
              if str(row.get("human_score", "")).strip() and row["candidate_id"] in R]
    if not scored:
        return {"n_scored": 0}
    human = [h for _, h in scored]
    model = [final_score(R[c], cfg) for c, _ in scored]
    passed = [passes_floor(R[c], cfg)[0] for c, _ in scored]
    good = [h >= good_at for h in human]
    tp = sum(p and g for p, g in zip(passed, good))
    return {
        "n_scored": len(scored),
        "spearman": spearman(model, human),
        "floor_precision": (tp / sum(passed)) if any(passed) else None,
        "floor_recall": (tp / sum(good)) if any(good) else None,
        "human_good": sum(good), "floor_passed": sum(passed),
    }
