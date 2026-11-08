"""Run the whole pipeline on a screenplay folder (a show) or a single episode file.

    .venv/bin/python -m src.cli --input "<folder or .txt>" --out outputs/gemini
    .venv/bin/python -m src.cli --input ... --out outputs/heuristic --provider heuristic
    LLM_PROVIDER=openai_compatible LLM_MODEL=<id> .venv/bin/python -m src.cli --input ...
"""
import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from .candidates import enumerate_windows, generate_candidates, load_lexicon
from .config import ROOT, load_config
from .judge import PROMPT_VERSION, failed_records, judge_batch_ensemble
from .llm import CacheOnlyClient, JsonCache, QuotaExhaustedError, make_client
from .parse import load_show, normalise_speakers, parse_episode_file, parse_report
from .rank import rank
from .report import write_all
from .validate import validate_all


def _load(path):
    if os.path.isdir(path):
        return load_show(path)
    ep = parse_episode_file(path)
    return [ep], normalise_speakers([ep])


def main(argv=None, *, log=None, api_key=None):
    """`log` receives progress lines (used by the Streamlit app); `api_key` is
    passed to the client directly, never via the command line or environment."""
    ap = argparse.ArgumentParser(description="Find and rank promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt files, or one .txt")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--provider", help="gemini | openai_compatible | heuristic (overrides config/env)")
    ap.add_argument("--model", help="model id (overrides config/env)")
    ap.add_argument("--episodes", help="comma-separated episode numbers to run, e.g. 1,2,3")
    ap.add_argument("--order-seed", type=int, help="judge in ONE shuffled order (stability eval)")
    ap.add_argument("--order-seeds", help="orderings to average, e.g. 'none,1' (default: config)")
    ap.add_argument("--no-cache", action="store_true", help="always call the model")
    ap.add_argument("--episodes-per-call", type=int, help="episodes judged per model call (default: config)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="let other models answer if the primary is overloaded (mixes judges)")
    ap.add_argument("--config", help="alternative config.yaml")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if log is None:
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

    client = make_client(cfg, provider=a.provider, model=a.model, api_key=api_key)
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
    if a.order_seed is not None:
        seeds = [a.order_seed]
    elif a.order_seeds:
        seeds = [None if x.strip().lower() in ("none", "null", "") else int(x) for x in a.order_seeds.split(",")]
    else:
        seeds = cfg["judge"].get("order_seeds") or [None]
    if client is None:
        seeds = [None]
    log(f"{n_cands} candidates in {len(episodes)} episodes -> {len(batches)} batch(es) x {len(seeds)} "
        f"ordering(s), {per_call} episodes per call")

    records, stopped = [], None
    for bi, batch in enumerate(batches, 1):
        eps_txt = ",".join(str(ep.number) for ep, _ in batch)
        te = time.time()
        try:
            res = judge_batch_ensemble(batch, client, cfg, cache=cache, seeds=seeds)
        except QuotaExhaustedError as e:
            if not stopped:
                stopped = f"daily quota exhausted ({str(e)[:200]})"
                log(f"  batch {bi}/{len(batches)} (eps {eps_txt}): {stopped}")
                log("  Continuing on cached answers only. Re-run after the quota resets to fill the gaps.")
                client = CacheOnlyClient(client)
            res = getattr(e, "partial", None)
            if not res:
                records.extend(failed_records(batch, f"not judged: {stopped}", "none"))
                log(f"  batch {bi}/{len(batches)} (eps {eps_txt}): not judged (no cached answer)")
                continue
        records.extend(res)
        ok = sum(r["status"] == "ok" for r in res)
        fail = next((r["failure_reason"] for r in res if r["status"] != "ok"), None)
        orders = max((r.get("orders_judged") or 0 for r in res), default=0)
        log(f"  batch {bi}/{len(batches)} (eps {eps_txt}): {ok}/{len(res)} judged, {orders} ordering(s)"
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
        "judge_calls": len(batches) * len(seeds), "stopped_early": stopped, "order_seeds": seeds,
        "orders_judged": dict(Counter(r.get("orders_judged", 0) for r in records)),
        "order_unstable_candidates": sum(bool(r.get("order_unstable")) for r in records if r["status"] == "ok"),
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
