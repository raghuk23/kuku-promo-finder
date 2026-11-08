"""Streamlit front end. Run:  .venv/bin/streamlit run app.py

A thin layer over the same runner as the CLI (src/cli.py), so the app and
the command line always agree.
"""
import os

import streamlit as st

from src.config import ROOT, load_config
from src.ui import load_run, run_pipeline, zip_dir

st.set_page_config(page_title="Promo finder", page_icon="🎬", layout="wide")
CFG = load_config()
PROVIDERS = {"Heuristic (free, no key)": "heuristic", "Gemini": "gemini",
             "OpenAI-compatible API": "openai_compatible"}


def bundled_runs():
    base = os.path.join(ROOT, "outputs")
    if not os.path.isdir(base):
        return []
    order = {"gemini": 0, "heuristic": 1}
    runs = [d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "promos_ranked.json"))]
    return sorted(runs, key=lambda d: (order.get(d, 9), d))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render(res):
    s, v, f = res["summary"], res["validation"], res["summary"]["funnel"]
    judge = ", ".join(s.get("judges_used") or []) or s.get("provider")
    st.caption(f"Judge: **{judge}** · prompt `{s.get('prompt_version')}` · {s.get('finished_at', '')}")
    c = st.columns(6)
    c[0].metric("Episodes", f["episodes"])
    c[1].metric("Legal windows", f"{f['windows']:,}")
    c[2].metric("Candidates judged", f"{f['candidates'] - f['judge_failed']}/{f['candidates']}")
    c[3].metric("Passed quality floor", f["passed_floor"])
    c[4].metric("Ranked", f["ranked"])
    c[5].metric("Timestamp validity", pct(v.get("timestamp_validity_rate")))
    if s.get("stopped_early"):
        st.warning(f"Run stopped early: {s['stopped_early'][:200]}. Unjudged episodes are listed as "
                   "'judge failed'; re-run after the quota resets (finished calls are cached).")

    tabs = st.tabs(["Posting order", "Best per episode", "All candidates", "Run details"])
    with tabs[0]:
        if not res["promos"]:
            st.info("No candidate passed the quality floor. See *Best per episode* for the closest misses.")
        for p in res["promos"]:
            cut = " + ".join(f"{x['start']}–{x['end']}" for x in p["segments"])
            with st.container(border=True):
                st.markdown(f"#### {p['rank']}. Episode {p['episode']} · {cut} · {p['total_duration_s']:.0f}s")
                a, b = st.columns(2)
                a.markdown(f"**Hook (first 3s)**  \n{p['hook']}")
                b.markdown(f"**Cliffhanger (the cut)**  \n{p['cliffhanger']}")
                st.markdown(f"**Why it works:** {p['why_it_works']}  \n**Why this rank:** {p['rank_reason']}")
                st.markdown(f"**Weakness:** {p['weakness']}")
                stand = p.get("standalone_score")
                st.caption(f"hook {p['hook_score']}/10 · cliffhanger {p['cliffhanger_score']}/10 · "
                           f"standalone {stand if stand is not None else 'n/a'}/10 · "
                           f"score {p['final_score']} · cut: {p['refinement_note']}")
                for w in p.get("warnings") or []:
                    st.warning(f"Check: {w}")
                if p.get("order_unstable"):
                    st.info("Borderline: the judge's post/don't-post call changed when candidates were "
                            "shown in a different order. Scores are the average of both orderings.")
                script = res["scripts"].get(p["candidate_id"])
                if script:
                    with st.expander("Script for this clip"):
                        st.code("\n".join(script), language=None)
    with tabs[1]:
        st.caption("The best candidate in every episode, including ones below the quality floor.")
        cols = ["episode", "candidate_id", "start", "end", "duration_s", "final_score", "hook_score",
                "cliffhanger_score", "standalone_score", "below_floor", "posted_rank", "hook", "cliffhanger", "note"]
        st.dataframe([{k: r.get(k) for k in cols} for r in res["best"]], hide_index=True, width="stretch")
    with tabs[2]:
        st.caption("Every candidate with its scores and, if it was not posted, the reason.")
        rows = [{"candidate_id": r["candidate_id"], "episode": r["episode"], "status": r["status"],
                 "start": r["segments"][0].get("start"), "end": r["segments"][0].get("end"),
                 "score": None if r.get("final_score") is None else round(r["final_score"], 2),
                 "hook": r.get("hook_score"), "cliff": r.get("cliffhanger_score"),
                 "standalone": r.get("standalone_score"), "rank": r.get("rank"),
                 "excluded_reason": r.get("excluded_reason"), "cut": r.get("refinement_note")}
                for r in res["records"]]
        st.dataframe(rows, hide_index=True, width="stretch")
    with tabs[3]:
        u = s.get("llm_usage") or {}
        st.markdown(f"- Judge calls: **{s.get('judge_calls', 'n/a')}** ({s.get('episodes_per_call', 'n/a')} episodes per call)\n"
                    f"- Tokens: {u.get('input_tokens', 0):,} in · {u.get('output_tokens', 0):,} out · "
                    f"{u.get('thinking_tokens', 0):,} thinking\n"
                    f"- Cut suggestions from the judge: {v.get('refinements')}\n"
                    f"- Quote warnings (judge quoted a line not in the clip): {v.get('warnings_total', 0)}\n"
                    f"- Elapsed: {s.get('elapsed_s')}s")
        if res["log"]:
            st.code("\n".join(res["log"]), language=None)
        st.download_button("Download all outputs (.zip)", zip_dir(res["out_dir"]),
                           file_name="promo_outputs.zip", mime="application/zip")


st.title("Microdrama promo finder")
st.caption("Finds 20–60 second promo clips in screenplays and ranks which to post first. "
           "Every timestamp comes from the screenplay parser; the LLM only judges clips that code has "
           "already cut on camera shots, so it cannot invent a timestamp.")

with st.sidebar:
    mode = st.radio("Show", ["Example results", "Run on new screenplays"])
    if mode == "Run on new screenplays":
        label = st.selectbox("Judge", list(PROVIDERS))
        provider = PROVIDERS[label]
        model = api_key = None
        if provider != "heuristic":
            model = st.text_input("Model", value=CFG["llm"]["model"] if provider == "gemini" else "")
            api_key = st.text_input("API key", type="password",
                                    help="Used for this run only and never stored. Leave blank to use the "
                                         "server's .env key, if any.")
        per_call = st.number_input("Episodes per model call", 1, 10, CFG["judge"]["episodes_per_call"],
                                   help="The Gemini free tier allows 20 calls per day per model.")
        files = st.file_uploader("Screenplay .txt files (one per episode)", type=["txt"],
                                 accept_multiple_files=True)
        st.caption("Uploaded files are processed in a temporary folder and deleted after the run.")
        go = st.button("Find promos", type="primary", disabled=not files)

if mode == "Example results":
    runs = bundled_runs()
    if not runs:
        st.info("No bundled results found in outputs/. Switch to *Run on new screenplays*.")
    else:
        run = st.selectbox("Bundled run (raw outputs on the assignment's 50-episode show)", runs)
        render(load_run(os.path.join(ROOT, "outputs", run)))
else:
    if go:
        with st.status("Running the pipeline...", expanded=True) as status:
            try:
                st.session_state["result"] = run_pipeline(
                    [(f.name, f.getvalue()) for f in files], provider, model=model or None,
                    api_key=api_key or None, episodes_per_call=per_call, log=status.write)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001 -- shown to the user, not swallowed
                status.update(label="Failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
    if st.session_state.get("result"):
        render(st.session_state["result"])
    elif not go:
        st.info("Upload one or more episode screenplays in the sidebar, pick a judge, and press *Find promos*. "
                "The heuristic judge needs no key and runs in seconds.")
