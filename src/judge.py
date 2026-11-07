"""Stage 3: the LLM judges Stage 2's candidates. This is the only AI step.

What the model is given: each candidate's beats (with ids), plus a few beats of
context either side. What it is NOT given: the episode logline or story so far
-- so "makes sense on its own" is judged the way a stranger scrolling past
would judge it.

What the model may return: scores, short texts, and optionally a better start /
end *beat id*. It never returns a timestamp. Code resolves a beat id to the
nearest legal camera cut and re-checks the 20-60s bound; anything else is
rejected and logged.
"""
import random
from typing import Dict, List, Optional, Tuple

from .candidates import Candidate, cut_points
from .parse import Beat, Episode

PROMPT_VERSION = "judge-v1"
SCORE_KEYS = ("hook_score", "cliffhanger_score", "standalone_score")
TEXT_KEYS = ("hook", "cliffhanger", "standalone_reason", "weakness", "why_it_works")

SYSTEM = """You are a senior promo editor for vertical microdramas (1-2 minute episodes, \
the style of ReelShort and DramaBox). You pick clips from the show itself to post on \
Instagram, TikTok and YouTube to pull in NEW viewers.

A promo is a 20 to 60-second clip. A good promo:
- opens on a hook: something in the first 3 seconds that makes a scrolling viewer stop, \
like a slap, a reveal or a shocking line;
- ends on a cliffhanger: it cuts at an unresolved moment, so the viewer needs to know \
what happens next;
- makes sense on its own to someone who has never seen the show.

You will see candidate clips as screenplay beats. Each beat has an id like [b17], a time \
range, and a type: CAMERA (what the shot shows), ACTION, SFX, or a CHARACTER speaking. \
Lines marked IN are inside the current cut; lines marked OUT are nearby context you may \
extend into. You are deliberately NOT told the story so far: judge exactly as a stranger \
would. You are reading a script, not watching video, so treat CAMERA and ACTION beats as \
real visual content -- a slap or a reveal shot can be the hook even with no dialogue.

Score each candidate 1-10 on three criteria, using these anchors:
hook_score -- 9-10: the first 3 seconds contain a slap or physical shock, a reveal, or a \
line that is shocking even out of context. 5-6: emotional but not arresting. 1-3: \
establishing shots, calm talk, nothing happening.
cliffhanger_score -- 9-10: the cut lands mid-confrontation, right after a reveal, on a \
threat, or on an unanswered question. 5-6: tension, but partly resolved. 1-3: ends on \
resolution -- a smile, an embrace, a fade-out.
standalone_score -- 9-10: a stranger instantly understands who wants what and why it \
matters. 5-6: the gist is clear but a key relationship is unexplained. 1-3: depends on \
earlier episodes.

Use the full range. Candidates differ; most will NOT deserve 8+. Compare them against \
each other.

For each candidate also write:
- hook: <=20 words, what the viewer sees/hears in the first 3 seconds;
- cliffhanger: <=20 words, the unresolved moment at the cut;
- standalone_reason: one short clause;
- weakness: the single biggest reason this could fail as a promo;
- why_it_works: one line.
Use only events that appear in the beats. Never invent events. If you quote a line, \
quote it exactly.

Optionally improve the cut: if a better start or end exists among the listed beats \
(for example starting on the slap instead of before it, or ending on the unanswered \
line), return that beat's id in start_beat / end_beat. Otherwise return empty strings. \
Never return timestamps. The clip must stay 20-60 seconds; the cut will be snapped to \
the nearest camera cut.

Example (synthetic, not from this show): a clip opening on "CAMERA: Wide establishing \
shot of a city skyline" and ending on "ACTION: They embrace, smiling" -> hook 2, \
cliffhanger 1. A clip opening on "ACTION: She throws the wine in his face" and cutting \
on "I know what you did to my sister." -> hook 9, cliffhanger 9.

Return exactly one judgement per candidate_id."""

_STR = {"type": "string"}
_SCORE = {"type": "integer", "minimum": 1, "maximum": 10}
SCHEMA = {
    "type": "object",
    "properties": {"judgements": {"type": "array", "items": {
        "type": "object",
        # Text before each score, so the model states its reason before scoring.
        "properties": {"candidate_id": _STR, "hook": _STR, "hook_score": _SCORE,
                       "cliffhanger": _STR, "cliffhanger_score": _SCORE,
                       "standalone_reason": _STR, "standalone_score": _SCORE,
                       "weakness": _STR, "why_it_works": _STR,
                       "start_beat": _STR, "end_beat": _STR},
        "required": ["candidate_id", "hook", "hook_score", "cliffhanger", "cliffhanger_score",
                     "standalone_reason", "standalone_score", "weakness", "why_it_works",
                     "start_beat", "end_beat"]}}},
    "required": ["judgements"],
}


# ---------------------------------------------------------------- rendering
def fmt_ts(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def beat_id(b: Beat) -> str:
    return f"b{b.beat_index}"


def _eps(cfg):
    return cfg["window"]["boundary_eps_s"]


def shown_beats(ep: Episode, c: Candidate, cfg: dict) -> List[Tuple[Beat, bool]]:
    """Beats in the cut plus `context_pad_s` either side, each flagged IN/OUT."""
    pad, eps = cfg["judge"]["context_pad_s"], _eps(cfg)
    out = []
    for b in ep.beats:
        if b.end_s > c.start_s - pad and b.start_s < c.end_s + pad:
            inside = c.start_s - eps <= b.start_s < c.end_s - eps
            out.append((b, inside))
    return out


def _cut_beats(ep, c, cfg) -> Tuple[str, str]:
    eps = _eps(cfg)
    shown = shown_beats(ep, c, cfg)
    ins = [b for b, i in shown if i] or [b for b, _ in shown]
    start = next((b for b in ins if b.type == "CAMERA" and abs(b.start_s - c.start_s) < eps), ins[0])
    end = next((b for b in reversed(ins) if b.type == "CAMERA" and abs(b.end_s - c.end_s) < eps), ins[-1])
    return beat_id(start), beat_id(end)


def beat_line(b: Beat, inside: bool) -> str:
    who = (b.speaker_raw or b.speaker) if b.type == "DIALOGUE" else b.type
    tags = f" ({', '.join(b.tags)})" if b.tags else ""
    text = f'"{b.text}"' if b.type == "DIALOGUE" else b.text
    mark = "IN " if inside else "OUT"
    return f"{mark} [{beat_id(b)}] {fmt_ts(b.start_s)}-{fmt_ts(b.end_s)} {who}{tags}: {text}"


def build_user_prompt(ep: Episode, cands: List[Candidate], cfg: dict,
                      order_seed: Optional[int] = None) -> str:
    order = list(cands)
    if order_seed is not None:  # used by the stability eval to detect position bias
        random.Random(order_seed).shuffle(order)
    lines = [f"Episode {ep.number}: {len(order)} candidate promos. Judge every one."]
    for c in order:
        sb, eb = _cut_beats(ep, c, cfg)
        lines.append(f"\n### {c.candidate_id} | current cut [{sb}] {fmt_ts(c.start_s)} -> "
                     f"[{eb}] {fmt_ts(c.end_s)} ({c.duration_s:.1f}s)")
        lines.extend(beat_line(b, i) for b, i in shown_beats(ep, c, cfg))
    return "\n".join(lines)


# ---------------------------------------------------------------- parsing
def parse_judgements(data, cands: List[Candidate]) -> Tuple[Dict[str, dict], List[str]]:
    """Validate the model's JSON. Never invents a missing judgement."""
    ids = {c.candidate_id for c in cands}
    items = data.get("judgements") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}, ["response has no 'judgements' list"]
    out, issues = {}, []
    for j in items:
        if not isinstance(j, dict):
            issues.append("non-object judgement ignored")
            continue
        cid = j.get("candidate_id")
        if cid not in ids:
            issues.append(f"unknown candidate_id {cid!r} ignored")
            continue
        if cid in out:
            issues.append(f"{cid}: duplicate judgement, first kept")
            continue
        rec, ok = {}, True
        for k in SCORE_KEYS:
            try:
                v = int(round(float(j.get(k))))
            except (TypeError, ValueError):
                issues.append(f"{cid}: {k} is not a number ({j.get(k)!r})")
                ok = False
                break
            if not 1 <= v <= 10:
                issues.append(f"{cid}: {k}={v} out of range, clamped")
                v = max(1, min(10, v))
            rec[k] = v
        if not ok:
            continue
        for k in TEXT_KEYS:
            rec[k] = str(j.get(k) or "").strip()
        if not rec["hook"] or not rec["cliffhanger"]:
            issues.append(f"{cid}: empty hook or cliffhanger text")
        rec["start_beat"] = str(j.get("start_beat") or "").strip()
        rec["end_beat"] = str(j.get("end_beat") or "").strip()
        out[cid] = rec
    for cid in sorted(ids - set(out)):
        issues.append(f"{cid}: missing from response")
    return out, issues


# ---------------------------------------------------------------- refinement
def refine(ep: Episode, c: Candidate, start_beat: str, end_beat: str, cfg: dict):
    """Turn an optional beat-id suggestion into a legal cut. Returns (start_s, end_s, note)."""
    orig = (c.start_s, c.end_s)
    if not start_beat and not end_beat:
        return orig + ("unchanged",)
    starts, ends = cut_points(ep, cfg)
    shown = {beat_id(b): b for b, _ in shown_beats(ep, c, cfg)}
    s, e = orig
    for which, bid in (("start", start_beat), ("end", end_beat)):
        if not bid:
            continue
        b = shown.get(bid)
        if b is None:
            return orig + (f"refinement ignored: unknown {which} beat {bid!r}",)
        if which == "start":
            s = min(starts, key=lambda t: (abs(t - b.start_s), t))
        else:
            e = min(ends, key=lambda t: (abs(t - b.end_s), t))
    lo, hi = cfg["window"]["min_s"], cfg["window"]["max_s"]
    d = e - s
    if s >= e or not (lo - 1e-6 <= d <= hi + 1e-6):
        return orig + (f"refinement rejected: {fmt_ts(s)}->{fmt_ts(e)} is {d:.1f}s",)
    if (s, e) == orig:
        return orig + ("unchanged",)
    return s, e, f"refined from {fmt_ts(orig[0])}->{fmt_ts(orig[1])}"


# ---------------------------------------------------------------- records
def _segment(ep, s, e):
    return {"episode": ep.number, "start_s": s, "end_s": e, "start": fmt_ts(s),
            "end": fmt_ts(e), "duration_s": round(e - s, 2)}


def _base_record(ep, c, judge, prompt_version):
    return {
        "candidate_id": c.candidate_id, "episode": ep.number, "judge": judge,
        "prompt_version": prompt_version, "status": None, "failure_reason": None,
        "original_cut": _segment(ep, c.start_s, c.end_s),
        "segments": [_segment(ep, c.start_s, c.end_s)], "refinement_note": None,
        "hook_score": None, "hook": None, "cliffhanger_score": None, "cliffhanger": None,
        "standalone_score": None, "standalone_reason": None,
        "why_it_works": None, "weakness": None,
        "heuristic_score": c.heuristic_score, "components": c.components, "judge_issues": [],
    }


def _heuristic_record(ep, c, cfg):
    """No-LLM judge: Stage 2 components mapped to 1-10, texts lifted from the beats.
    Cannot judge "makes sense on its own", so standalone_score stays None."""
    W, eps = cfg["window"], _eps(cfg)
    rec = _base_record(ep, c, "heuristic", None)
    ins = [b for b in ep.beats if c.start_s - eps <= b.start_s < c.end_s - eps]
    spoken = [b for b in ins if b.type in ("DIALOGUE", "ACTION")] or ins
    head = [b for b in spoken if b.start_s < c.start_s + W["hook_s"]] or spoken[:1]
    tail = [b for b in spoken if b.end_s > c.end_s - W["cliff_s"]] or spoken[-1:]
    comps = c.components
    top = sorted(comps, key=lambda k: -comps.get(k, 0))[:2]
    rec.update(status="ok", refinement_note="unchanged",
               hook_score=1 + round(9 * comps.get("hook", 0)),
               cliffhanger_score=1 + round(9 * comps.get("cliffhanger", 0)),
               hook=head[0].text if head else "", cliffhanger=tail[-1].text if tail else "",
               standalone_reason="not assessed by heuristics",
               why_it_works="heuristic: strongest signals " + ", ".join(top),
               weakness="heuristic judge: cannot assess context, implied conflict or visual payoff")
    return rec


def judge_episode(ep: Episode, cands: List[Candidate], client, cfg: dict,
                  cache=None, order_seed: Optional[int] = None) -> List[dict]:
    if not cands:
        return []
    if client is None:
        return [_heuristic_record(ep, c, cfg) for c in cands]

    user = build_user_prompt(ep, cands, cfg, order_seed=order_seed)
    key = cache.key(client.name, PROMPT_VERSION, SYSTEM, user) if cache else None
    hit = cache.get(key) if cache else None
    model, js, issues = None, {}, []
    if hit:
        model = hit["model"]
        js, issues = parse_judgements(hit["data"], cands)
    else:
        for attempt in range(2):  # one retry if the reply parses to nothing usable
            try:
                data = client.complete_json(SYSTEM, user, SCHEMA)
            except Exception as e:  # noqa: BLE001 -- recorded on every candidate
                issues = [f"llm call failed: {type(e).__name__}: {str(e)[:300]}"]
                break
            model = getattr(client, "last_model", None) or client.name
            js, issues = parse_judgements(data, cands)
            if js:
                if cache:
                    cache.set(key, {"model": model, "data": data})
                break

    results = []
    for c in cands:
        rec = _base_record(ep, c, model or client.name, PROMPT_VERSION)
        rec["judge_issues"] = [i for i in issues if c.candidate_id in i
                               or not any(x.candidate_id in i for x in cands)]
        j = js.get(c.candidate_id)
        if j is None:
            rec.update(status="judge_failed",
                       failure_reason="; ".join(rec["judge_issues"]) or "no judgement returned")
        else:
            s, e, note = refine(ep, c, j["start_beat"], j["end_beat"], cfg)
            rec.update(status="ok", segments=[_segment(ep, s, e)], refinement_note=note,
                       **{k: j[k] for k in SCORE_KEYS + TEXT_KEYS},
                       suggested_beats={"start": j["start_beat"], "end": j["end_beat"]})
        results.append(rec)
    return results
