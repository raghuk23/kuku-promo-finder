"""Stage 2: cut legal candidate windows and score them cheaply. No AI.

The LLM never sees a raw episode. It sees only the few windows this stage
proposes, and every window here is legal by construction:
- it starts on a camera-shot start and ends on a camera-shot end (where an
  editor would actually cut), never in the middle of a spoken line;
- it is 20-60 seconds long (the brief's hard bounds).

Scoring is a transparent weighted sum of named components so that we can see
*why* a window ranked where it did, and so the LLM's added value can be
measured against this baseline (the ablation in the quality report).
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml

from .config import ROOT
from .parse import Episode


@dataclass
class Candidate:
    candidate_id: str
    episode: int
    start_s: float
    end_s: float
    beat_indices: List[int]
    components: dict
    heuristic_score: float

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


# ---------------------------------------------------------------- lexicon
def load_lexicon(path=None) -> dict:
    with open(path or os.path.join(ROOT, "lexicon.yaml"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def rx(words):
        return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

    return {
        "tags_high": {w.lower() for w in raw["tags_high"]},
        "tags_medium": {w.lower() for w in raw["tags_medium"]},
        **{k: rx(raw[k]) for k in ("conflict", "physical", "cliffhanger", "resolution", "establishing")},
    }


# ---------------------------------------------------------------- geometry
def adaptive_k(duration_s: float, cfg: dict) -> int:
    c = cfg["candidates"]
    k = int(duration_s / c["seconds_per_candidate"] + 0.5)  # round half up
    return max(c["k_min"], min(c["k_max"], k))


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def nms(cands: List[Candidate], k: int, iou_thresh: float) -> List[Candidate]:
    """Greedy non-maximum suppression over time intervals, as in object detection."""
    kept: List[Candidate] = []
    for c in sorted(cands, key=lambda c: (-c.heuristic_score, c.start_s, c.end_s)):
        if all(iou((c.start_s, c.end_s), (x.start_s, x.end_s)) <= iou_thresh for x in kept):
            kept.append(c)
            if len(kept) == k:
                break
    return kept


def cut_points(ep: Episode, cfg: dict) -> Tuple[List[float], List[float]]:
    """Legal start/end times: camera-shot boundaries not inside a spoken line."""
    eps = cfg["window"]["boundary_eps_s"]
    cams = [b for b in ep.beats if b.type == "CAMERA"]
    lines = [b for b in ep.beats if b.type == "DIALOGUE"]

    def clean(ts):
        return sorted(t for t in {round(x, 2) for x in ts}
                      if t <= ep.duration_s + eps
                      and not any(d.start_s < t - eps and d.end_s > t + eps for d in lines))

    return clean(b.start_s for b in cams), clean(b.end_s for b in cams)


def enumerate_windows(ep: Episode, cfg: dict) -> List[Tuple[float, float]]:
    lo, hi = cfg["window"]["min_s"], cfg["window"]["max_s"]
    starts, ends = cut_points(ep, cfg)
    out = []
    for s in starts:
        for e in ends:
            d = e - s
            if d > hi + 1e-6:
                break
            if d >= lo - 1e-6:
                out.append((s, e))
    return out


# ---------------------------------------------------------------- scoring
def _tag_level(tags, lex) -> float:
    words = set(re.findall(r"[a-z]+", " ".join(tags).lower()))
    if words & lex["tags_high"]:
        return 1.0
    if words & lex["tags_medium"]:
        return 0.5
    return 0.0


def _beat_features(ep: Episode, lex: dict) -> Dict[int, dict]:
    """Per-beat signals, computed once per episode."""
    feats = {}
    for b in ep.beats:
        conflict = b.type in ("DIALOGUE", "ACTION") and bool(lex["conflict"].search(b.text))
        physical = b.type in ("ACTION", "CAMERA") and bool(lex["physical"].search(b.text))
        tag = _tag_level(b.tags, lex) if b.type == "DIALOGUE" else 0.0
        cliff_marker = bool(lex["cliffhanger"].search(b.text))
        # The brief's hook archetypes -- "a slap, a reveal or a shocking line" --
        # earn full marks. An intense emotion tag alone is common in this show
        # (28% of lines), so it earns at most 0.6.
        archetype = conflict or physical or (cliff_marker and b.type in ("ACTION", "CAMERA"))
        feats[b.beat_index] = {
            "signal": 1.0 if archetype else 0.6 * tag,
            "tag": tag,
            "hit": conflict or physical,
            "cliff_marker": cliff_marker,
            "resolution": bool(lex["resolution"].search(b.text)),
            "establishing": b.type == "CAMERA" and bool(lex["establishing"].search(b.text)),
        }
    return feats


def score_window(ep: Episode, t0: float, t1: float, lex: dict, cfg: dict, _feats=None):
    """Return (components, total). Every component is in [0, 1]."""
    W = cfg["window"]
    eps = W["boundary_eps_s"]
    f = _feats if _feats is not None else _beat_features(ep, lex)
    inside = [b for b in ep.beats if b.start_s < t1 - eps and b.end_s > t0 + eps]
    head = [b for b in inside if b.start_s < t0 + W["hook_s"]]
    tail = [b for b in inside if b.end_s > t1 - W["cliff_s"]]
    lines = [b for b in inside if b.type == "DIALOGUE"]

    # Hook: strongest signal in the first 3 seconds; a slow establishing shot with
    # no line spoken costs points.
    hook = max((f[b.beat_index]["signal"] for b in head), default=0.0)
    if any(f[b.beat_index]["establishing"] for b in head) and not any(b.type == "DIALOGUE" for b in head):
        hook -= 0.3

    # Cliffhanger: explicit markers and an unanswered question push up; signs of
    # resolution (smiles, fading out) push down.
    tail_lines = [b for b in tail if b.type == "DIALOGUE"]
    cliff = 0.5 * max((f[b.beat_index]["signal"] for b in tail), default=0.0)
    if any(f[b.beat_index]["cliff_marker"] for b in tail):
        cliff += 0.5
    if tail_lines and tail_lines[-1].text.rstrip().endswith(("?", "—", "...", "…")):
        cliff += 0.3
    if any(f[b.beat_index]["resolution"] for b in tail):
        cliff -= 0.4

    dur = t1 - t0
    intensity = (sum(f[b.beat_index]["tag"] for b in lines) / len(lines)) if lines else 0.0
    hits = sum(1 for b in inside if f[b.beat_index]["hit"])
    conflict = min(1.0, (hits / (dur / 10.0)) / 1.5)
    speakers = 1.0 if len({b.speaker for b in lines}) >= 2 else 0.0
    lo, hi = W["sweet_spot_s"]
    if dur < lo:
        length_fit = 0.5 + 0.5 * (dur - W["min_s"]) / (lo - W["min_s"])
    elif dur > hi:
        length_fit = 0.5 + 0.5 * (W["max_s"] - dur) / (W["max_s"] - hi)
    else:
        length_fit = 1.0

    clip = lambda x: round(max(0.0, min(1.0, x)), 4)
    comps = {"hook": clip(hook), "cliffhanger": clip(cliff), "intensity": clip(intensity),
             "conflict": clip(conflict), "speakers": speakers, "length_fit": clip(length_fit)}
    w = cfg["heuristic_weights"]
    total = sum(w[k] * comps[k] for k in w)
    return comps, total


# ---------------------------------------------------------------- pipeline
def generate_candidates(ep: Episode, lex: dict, cfg: dict) -> List[Candidate]:
    feats = _beat_features(ep, lex)
    eps = cfg["window"]["boundary_eps_s"]
    scored = []
    for s, e in enumerate_windows(ep, cfg):
        comps, total = score_window(ep, s, e, lex, cfg, _feats=feats)
        idx = [b.beat_index for b in ep.beats if b.start_s < e - eps and b.end_s > s + eps]
        scored.append(Candidate("", ep.number, s, e, idx, comps, round(total, 4)))
    kept = nms(scored, adaptive_k(ep.duration_s, cfg), cfg["candidates"]["nms_iou"])
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"ep{ep.number:02d}_c{i}"
    return kept
