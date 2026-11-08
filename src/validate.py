"""Stage 4: hard checks on every judged promo. Pure code.

Errors reject a promo (it can never reach the ranked list). Warnings keep it
but flag it for a human. The headline number for the doc is
`timestamp_validity_rate`: the share of judged promos whose timestamps pass
every check, machine-verified on 100% of outputs.
"""
import re
from collections import Counter
from typing import Dict, List

from .candidates import cut_points
from .parse import Episode

# Double quotes, or single quotes that are not apostrophes (not touching a letter
# on the outside): the live judge writes both 'Then let her suffer' and "...".
QUOTE = re.compile(r'["“]([^"”]{3,}?)["”]|(?<!\w)[\'‘]([^\'’\n]{3,}?)[\'’](?!\w)')
TIMESTAMP_ERRORS = ("legal cut", "duration", "episode end", "not before", "episode mismatch", "no segments")


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?;:—–-…'\"")


def quotes_not_in_clip(texts: dict, ep: Episode, segs: list, eps: float) -> list:
    """[(field, quote)] for every quoted line in `texts` that is not spoken or
    written anywhere inside the clip's segments."""
    clip = _norm(" ".join(b.text for b in ep.beats for s in segs
                          if b.start_s < s["end_s"] - eps and b.end_s > s["start_s"] + eps))
    missing = []
    for k, text in texts.items():
        for dq, sq in QUOTE.findall(str(text or "")):
            q = dq or sq
            if _norm(q) not in clip:
                missing.append((k, q))
    return missing


def _on(t, points):
    return any(abs(t - p) <= 1e-6 for p in points)


def validate_record(rec: dict, ep: Episode, cfg: dict) -> dict:
    W, eps = cfg["window"], cfg["window"]["boundary_eps_s"]
    errors, warnings = [], []
    if rec.get("status") != "ok":
        return {"errors": [f"not judged ({rec.get('status')})"], "warnings": []}

    starts, ends = cut_points(ep, cfg)
    segs = rec.get("segments") or []
    if not segs:
        errors.append("no segments")
    total = 0.0
    for i, s in enumerate(segs):
        a, b = s["start_s"], s["end_s"]
        if s.get("episode") != ep.number:
            errors.append(f"segment {i}: episode mismatch")
        if not _on(a, starts):
            errors.append(f"segment {i}: start {a} is not a legal cut point")
        if not _on(b, ends):
            errors.append(f"segment {i}: end {b} is not a legal cut point")
        if a >= b:
            errors.append(f"segment {i}: start not before end")
        if b > ep.duration_s + eps:
            errors.append(f"segment {i}: ends after episode end ({ep.duration_s}s)")
        total += b - a
    if segs and not (W["min_s"] - 1e-6 <= total <= W["max_s"] + 1e-6):
        errors.append(f"duration {total:.1f}s outside {W['min_s']}-{W['max_s']}s")

    for k in ("hook", "cliffhanger", "why_it_works"):
        if not str(rec.get(k) or "").strip():
            errors.append(f"empty {k}")
    for k in ("hook_score", "cliffhanger_score", "standalone_score"):
        v = rec.get(k)
        if v is None and k == "standalone_score":
            continue
        if not isinstance(v, int) or not 1 <= v <= 10:
            errors.append(f"{k} invalid ({v!r})")

    # Grounding: anything the judge puts in quotes must actually be in the clip.
    texts = {k: rec.get(k) for k in ("hook", "cliffhanger", "why_it_works", "weakness")}
    for k, q in quotes_not_in_clip(texts, ep, segs, eps):
        warnings.append(f"{k}: quote not found in clip: {q!r}")
    return {"errors": errors, "warnings": warnings}


def validate_all(records: List[dict], episodes: Dict[int, Episode], cfg: dict) -> dict:
    judged = valid = ts_ok = failed = 0
    err_types, warn_types, refine, judges = Counter(), Counter(), Counter(), Counter()
    examples = []
    for r in records:
        judges[r.get("judge")] += 1
        if r.get("status") != "ok":
            failed += 1
            r["valid"] = False
            r["validation"] = {"errors": [f"not judged ({r.get('status')})"], "warnings": []}
            continue
        judged += 1
        v = validate_record(r, episodes[r["episode"]], cfg)
        r["validation"], r["valid"] = v, not v["errors"]
        valid += r["valid"]
        ts_ok += not any(t in e for e in v["errors"] for t in TIMESTAMP_ERRORS)
        for e in v["errors"]:
            err_types[re.sub(r"[\d.]+", "N", e)] += 1
        for w in v["warnings"]:
            warn_types[w.split(":")[0] + ": quote not found"] += 1
        if (v["errors"] or v["warnings"]) and len(examples) < 20:
            examples.append({"candidate_id": r["candidate_id"], **v})
        note = (r.get("refinement_note") or "unchanged").split(":")[0].split(" from")[0]
        refine[note] += 1
    return {
        "records": len(records), "judged_ok": judged, "judge_failed": failed,
        "valid": valid, "invalid": judged - valid,
        "timestamp_validity_rate": (ts_ok / judged) if judged else None,
        "valid_rate": (valid / judged) if judged else None,
        "error_types": dict(err_types), "warning_types": dict(warn_types),
        "warnings_total": sum(warn_types.values()),
        "refinements": dict(refine), "judges": dict(judges), "examples": examples,
    }
