"""Run the whole pipeline on a screenplay folder (a show) or a single episode file.

    .venv/bin/python -m src.cli --input "<folder or .txt>" --out outputs/gemini
    .venv/bin/python -m src.cli --input ... --out outputs/heuristic --provider heuristic
    LLM_PROVIDER=openai_compatible LLM_MODEL=<id> .venv/bin/python -m src.cli --input ...
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

from .candidates import enumerate_windows, generate_candidates, load_lexicon
from .config import ROOT, load_config
from .judge import PROMPT_VERSION, failed_records, judge_batch
from .llm import JsonCache, QuotaExhaustedError, make_client
from .parse import load_show, normalise_speakers, parse_episode_file, parse_report
from .rank import rank
from .report import write_all
from .validate import validate_all


def _load(path):
    if os.path.isdir(path):
        return load_show(path)
    ep = parse_episode_file(path)
    return [ep], normalise_speakers([ep])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find and rank promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt files, or one .txt")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--provider", help="gemini | openai_compatible | heuristic (overrides config/env)")
    ap.add_argument("--model", help="model id (overrides config/env)")
    ap.add_argument("--episodes", help="comma-separated episode numbers to run, e.g. 1,2,3")
    ap.add_argument("--order-seed", type=int, help="shuffle candidate order in prompts (stability eval)")
    ap.add_argument("--no-cache", action="store_true", help="always call the model")
    ap.add_argument("--episodes-per-call", type=int, help="episodes judged per model call (default: config)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="let other models answer if the primary is overloaded (mixes judges)")
    ap.add_argument("--config", help="alternative config.yaml")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))

    started, t0 = datetime.now(timezone.utc), time.time()
    cfg, lex = load_config(a.config), load_lexicon()
    if a.allow_fallback:
        cfg["llm"]["strict_model"] = False
    episodes, merges = _load(a.input)
    if a.episodes:
        keep = {int(x) for x in a.episodes.split(",") if x.strip()}
        episodes = [e for e in episodes if e.number in keep]
    prep = parse_report(episodes, merges)
    log(f"parsed {prep['episodes']} episodes, {prep['beats']} beats, {prep['anomaly_count']} anomalies")

    client = make_client(cfg, provider=a.provider, model=a.model)
    provider = "heuristic" if client is None else client.name.split(":")[0]
    cache = None if (a.no_cache or client is None) else JsonCache(os.path.join(ROOT, ".cache", "judge"))
    log(f"judge: {'heuristic (no LLM)' if client is None else client.name}")

    items, windows = [], 0
    for ep in episodes:
        windows += len(enumerate_windows(ep, cfg))
        items.append((ep, generate_candidates(ep, lex, cfg)))
    n_cands = sum(len(cs) for _, cs in items)
    per_call = a.episodes_per_call or cfg["judge"].get("episodes_per_call", 1)
    batches = [items[i:i + per_call] for i in range(0, len(items), per_call)]
    log(f"{n_cands} candidates in {len(episodes)} episodes -> {len(batches)} judge call(s), "
        f"{per_call} episodes per call")

    records, stopped = [], None
    for bi, batch in enumerate(batches, 1):
        eps_txt = ",".join(str(ep.number) for ep, _ in batch)
        if stopped:
            records.extend(failed_records(batch, f"not judged: {stopped}", "none"))
            continue
        te = time.time()
        try:
            res = judge_batch(batch, client, cfg, cache=cache, order_seed=a.order_seed)
        except QuotaExhaustedError as e:
            stopped = f"daily quota exhausted ({str(e)[:200]})"
            log(f"  call {bi}/{len(batches)} (eps {eps_txt}): STOPPED, {stopped}")
            log("  Re-run after the quota resets (finished calls are cached), or pass --model <other model>.")
            records.extend(failed_records(batch, f"not judged: {stopped}", "none"))
            continue
        records.extend(res)
        ok = sum(r["status"] == "ok" for r in res)
        fail = next((r["failure_reason"] for r in res if r["status"] != "ok"), None)
        log(f"  call {bi}/{len(batches)} (eps {eps_txt}): {ok}/{len(res)} judged"
            f"{' by ' + res[0]['judge'] if res else ''} [{time.time() - te:.1f}s]"
            + (f"  FAILED: {fail[:160]}" if fail else ""))

    by_num = {e.number: e for e in episodes}
    validation = validate_all(records, by_num, cfg)
    ranked = rank(records, cfg, episodes=[e.number for e in episodes])
    judges = sorted({r["judge"] for r in records if r.get("status") == "ok"})
    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1), "input": os.path.abspath(a.input),
        "provider": provider, "judges_used": judges, "prompt_version": PROMPT_VERSION,
        "order_seed": a.order_seed, "cache": cache is not None, "episodes_per_call": per_call,
        "judge_calls": len(batches), "stopped_early": stopped,
        "llm_usage": getattr(client, "usage", None),
        "funnel": {"episodes": len(episodes), "windows": windows, "candidates": n_cands, **ranked["funnel"]},
        "timestamp_validity_rate": validation["timestamp_validity_rate"],
        "config": cfg,
    }
    show = os.path.basename(os.path.normpath(a.input))
    write_all(a.out, show=show, summary=summary, parse_rep=prep, records=records,
              ranked=ranked, validation=validation)
    f = summary["funnel"]
    log(f"\nfunnel: {f['windows']} windows -> {f['candidates']} candidates -> {f['passed_floor']} passed floor "
        f"-> {f['ranked']} ranked | timestamp validity {validation['timestamp_validity_rate']}")
    log(f"outputs written to {a.out}")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
