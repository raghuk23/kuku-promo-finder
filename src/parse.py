"""Stage 1: parse a microdrama screenplay into timed beats.

Pure code, no AI. Deterministic, free, and the single source of truth for every
timestamp the pipeline ever emits. Downstream stages (including the LLM) may
only *refer* to beats by index; they never produce times themselves.

Tolerances, each driven by something observed in the real input pack:
- Field labels are matched case-insensitively (episode 6 uses UPPERCASE).
- Speaker labels may stack parenthetical tags: `NAME (Off-screen) (Snide)`.
- `LADY X` and `X` are the same character when the show gives evidence of it
  (JULIANA 352 blocks vs LADY JULIANA 110) -- see normalise_speakers().
Anything the parser cannot interpret is recorded in `Episode.anomalies`
rather than raised or silently dropped.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

_TIME = r"(\d+(?::\d+)+(?:\.\d+)?)"
TS_PAIR = re.compile(r"^start:\s*\(\s*" + _TIME + r"\s*\)\s*\|\s*end:\s*\(\s*" + _TIME + r"\s*\)\s*$", re.I)
LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
FIELD = re.compile(r"^(episode number|episode logline|continuity notes|scene heading|scene start|scene end)\s*:\s*(.*)$", re.I)
SCENE = re.compile(r"^scene\s+(\d+)\s*$", re.I)
SECTION = re.compile(r"^actions and dialogue\s*:\s*$", re.I)
MARKER = re.compile(r"^-+\s*script\s+(start|end)\s*-+$", re.I)
TAG = re.compile(r"\(([^()]*)\)")

NON_DIALOGUE = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX"}
QUOTES = "\"“”"


def to_seconds(ts: str) -> float:
    """'MM:SS.ff' (or 'HH:MM:SS.ff') -> seconds."""
    total = 0.0
    for part in ts.strip().split(":"):
        total = total * 60 + float(part)
    return round(total, 3)


@dataclass
class Scene:
    index: int
    heading: str = ""
    start_s: Optional[float] = None
    end_s: Optional[float] = None


@dataclass
class Beat:
    beat_index: int
    episode: int
    scene: Optional[int]
    type: str                  # CAMERA | ACTION | DIALOGUE | SFX
    speaker: Optional[str]     # DIALOGUE only; merged across the show by normalise_speakers()
    speaker_raw: Optional[str]
    tags: List[str]
    text: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class Episode:
    number: Optional[int]
    logline: str = ""
    continuity: str = ""
    source: Optional[str] = None
    scenes: List[Scene] = field(default_factory=list)
    beats: List[Beat] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        ends = [s.end_s for s in self.scenes if s.end_s is not None]
        if ends:
            return max(ends)
        return max((b.end_s for b in self.beats), default=0.0)


def _label_parts(label: str):
    """'LADY JULIANA (Off-screen) (Begging)' -> ('DIALOGUE', 'JULIANA', 'LADY JULIANA', [...])"""
    tags = [t.strip() for t in TAG.findall(label)]
    name = " ".join(label.split("(", 1)[0].split()).upper()
    if name in NON_DIALOGUE:
        return NON_DIALOGUE[name], None, None, tags
    return "DIALOGUE", name, name, tags


def parse_episode_text(text: str, source: Optional[str] = None) -> Episode:
    ep = Episode(number=None, source=source)
    scene: Optional[Scene] = None
    pending = None  # dict(label, ts, lines, line_no) for the block being read

    def flush():
        nonlocal pending
        if pending is None:
            return
        p, pending = pending, None
        if p["ts"] is None:
            ep.anomalies.append(f"line {p['line_no']}: no timestamp for block '{p['label']}' (dropped)")
            return
        start, end = p["ts"]
        if end < start:
            ep.anomalies.append(f"line {p['line_no']}: end before start in block '{p['label']}'")
        btype, speaker, speaker_raw, tags = _label_parts(p["label"])
        body = " ".join(p["lines"]).strip()
        if btype == "DIALOGUE":
            body = body.strip(QUOTES).strip()
        ep.beats.append(Beat(
            beat_index=len(ep.beats), episode=ep.number, scene=scene.index if scene else None,
            type=btype, speaker=speaker, speaker_raw=speaker_raw, tags=tags,
            text=body, start_s=start, end_s=end))

    for line_no, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()

        if MARKER.match(s) or SECTION.match(s):
            flush()
            continue

        m = FIELD.match(s)
        if m:
            flush()
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "episode number":
                num = re.search(r"\d+", val)
                ep.number = int(num.group()) if num else None
            elif key == "episode logline":
                ep.logline = val
            elif key == "continuity notes":
                ep.continuity = val
            elif scene is None:
                ep.anomalies.append(f"line {line_no}: '{m.group(1)}' outside any scene")
            elif key == "scene heading":
                scene.heading = val
            elif key == "scene start":
                scene.start_s = to_seconds(val)
            elif key == "scene end":
                scene.end_s = to_seconds(val)
            continue

        m = SCENE.match(s)
        if m:
            flush()
            scene = Scene(index=int(m.group(1)))
            ep.scenes.append(scene)
            continue

        m = LABEL.match(s)
        if m:
            flush()
            pending = {"label": m.group(1).strip(), "ts": None, "lines": [], "line_no": line_no}
            continue

        if pending is None:
            continue  # blank lines / header prose between blocks

        if pending["ts"] is None:
            m = TS_PAIR.match(s)
            if m:
                pending["ts"] = (to_seconds(m.group(1)), to_seconds(m.group(2)))
                continue
        if not s:
            if pending["lines"]:
                flush()
            continue
        pending["lines"].append(s)

    flush()
    for b in ep.beats:  # header may follow nothing, but keep episode id consistent
        b.episode = ep.number
    return ep


def parse_episode_file(path: str) -> Episode:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    ep = parse_episode_text(text, source=os.path.basename(path))
    if ep.number is None:
        m = re.search(r"episode-(\d+)", os.path.basename(path), re.I)
        if m:
            ep.number = int(m.group(1))
            for b in ep.beats:
                b.episode = ep.number
            ep.anomalies.append("episode number missing from header; taken from filename")
    return ep


TITLES = ("LADY", "LORD", "SIR")


def normalise_speakers(episodes: List[Episode]) -> dict:
    """Merge `TITLE X` into `X` when bare `X` also speaks somewhere in the show.

    Evidence-based rather than a fixed rule, so it transfers to a new show:
    `LADY JULIANA` merges because `JULIANA` speaks on her own, while
    `LADY ASHCROFT` and `LORD ASHCROFT` stay distinct because nobody speaks as
    plain `ASHCROFT`. Mutates beats in place; returns the merge map applied.
    """
    names = {b.speaker_raw for e in episodes for b in e.beats if b.speaker_raw}
    merges = {}
    for n in names:
        title, _, rest = n.partition(" ")
        if title in TITLES and rest in names:
            merges[n] = rest
    for e in episodes:
        for b in e.beats:
            if b.speaker_raw in merges:
                b.speaker = merges[b.speaker_raw]
    return merges


def load_show(data_dir: str):
    """Parse every .txt in `data_dir` as one show. Returns (episodes, merges)."""
    import glob
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    parsed = [parse_episode_file(p) for p in paths]
    # Uploads may lack both a header and an "episode-N" filename: number them in
    # filename order after the highest known number, and say so.
    used = {e.number for e in parsed if e.number is not None}
    nxt = max(used, default=0) + 1
    for e in parsed:
        if e.number is None:
            e.number, nxt = nxt, nxt + 1
            for b in e.beats:
                b.episode = e.number
            e.anomalies.append(f"no episode number in header or filename; assigned {e.number} by file order")
    episodes = sorted(parsed, key=lambda e: e.number)
    merges = normalise_speakers(episodes)
    return episodes, merges


def parse_report(episodes: List[Episode], merges: dict) -> dict:
    """Machine-readable health check. A malformed new show shows up here as
    anomalies or a suspicious beat count instead of disappearing silently."""
    from collections import Counter
    per = []
    for e in episodes:
        per.append({
            "episode": e.number,
            "source": e.source,
            "scenes": len(e.scenes),
            "beats": len(e.beats),
            "duration_s": e.duration_s,
            "beat_types": dict(Counter(b.type for b in e.beats)),
            "anomalies": e.anomalies,
        })
    return {
        "episodes": len(episodes),
        "beats": sum(len(e.beats) for e in episodes),
        "anomaly_count": sum(len(e.anomalies) for e in episodes),
        "speaker_merges": merges,
        "per_episode": per,
    }
