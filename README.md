# Kuku promo finder

Reads microdrama screenplays and returns a ranked shortlist of promo clips
(20–60s) for Instagram, TikTok and YouTube. For each promo you get the episode,
the start and end timestamps of every segment, the hook, the cliffhanger, one
line on why it works, and why it ranks where it does.

**Design in one line:** code owns the clock, the LLM owns the taste. Every
timestamp comes from the screenplay parser. The model only judges clips that
code has already cut to legal points, and it can refer to beats only by id, so
it cannot invent a timestamp.

## Run it (about 60 seconds to first output)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Free, offline, no key: heuristic judge only
.venv/bin/python -m src.cli --input "<folder of episode .txt files>" --out outputs/heuristic --provider heuristic

# With the LLM judge: put GEMINI_API_KEY=... in .env (free key from aistudio.google.com/apikey)
.venv/bin/python scripts/smoke_test.py          # check the key and model with one tiny call
.venv/bin/python -m src.cli --input "<folder or one .txt>" --out outputs/gemini
```

`--input` takes a whole show (a folder) or one episode file. Use
`--episodes 1,2,3` to run a subset.

### Web app

```bash
.venv/bin/streamlit run app.py
```

- **Example results** shows the bundled raw outputs for the 50-episode show.
- **Run on new screenplays** takes uploaded `.txt` episodes and a judge choice
  (heuristic with no key, Gemini, or any OpenAI-compatible API), and returns
  the ranked promos with the script for each clip and a .zip of all outputs.

A pasted API key goes straight to the client for that run. It is never
written to the environment or to disk. Uploaded files are processed in a
temporary folder that is deleted after the run.

**Deploy (Streamlit Community Cloud, free):** push the repo to GitHub, then
on share.streamlit.io create an app from it with main file `app.py` and
Python 3.12. Don't put a key in Secrets on a public app: the free tier allows
only 20 calls per day per model, and anyone could use them up. Visitors paste
their own key or use the heuristic judge.

### Swapping the model

No code changes are needed. Pick a provider and model per run:

```bash
.venv/bin/python -m src.cli --input ... --out ... --provider gemini --model gemini-3.6-flash
LLM_PROVIDER=openai_compatible LLM_MODEL=<model id> OPENAI_COMPAT_API_KEY=... .venv/bin/python -m src.cli ...
```

`openai_compatible` works with any `/chat/completions` endpoint, such as
OpenRouter, Groq, or Anthropic's OpenAI-compatible endpoint. Set the base URL
in `config.yaml`. Every result records which model judged it.

## How it works

| Stage | What | AI? |
|---|---|---|
| 1. Parse | Screenplay → timed beats. Case-insensitive labels, logs anomalies | no |
| 2. Candidates | Every 20–60s window between camera cuts (never mid-line), scored on hook / cliffhanger / intensity / conflict / speakers / length, deduped with NMS → 3–8 per episode | no |
| 3. Judge | LLM scores each candidate 1–10 on hook, cliffhanger and standalone, writes the texts, and may move a cut by beat id | **yes** |
| 4. Validate | Every cut is on a legal camera cut, 20–60s, inside the episode; quoted lines must exist in the clip | no |
| 5. Rank | Weighted score, pre-registered quality floor, dedupe after refinement, max 2 per episode, small early-episode boost | no |

All tunable numbers live in `config.yaml`, and all trigger words live in
`lexicon.yaml`.

## Outputs (`--out` folder)

| File | Contents |
|---|---|
| `promos_ranked.md` / `.json` | The posting order: top 10, with hook, cliffhanger, why it works, why this rank, weakness |
| `best_per_episode.csv` | The best candidate in every episode, flagged `below_floor` when it isn't good enough to post |
| `all_candidates.json` | Every candidate with its scores, validation result, and the reason it was excluded |
| `validation_report.json` | Timestamp validity rate, error and warning counts, refinements applied or refused |
| `parse_report.json` | Per-episode parse health |
| `run_summary.json` | Judge used, prompt version, real token usage, the funnel, and a config snapshot |

## Quality checks

```bash
.venv/bin/python scripts/evaluate.py ablation  --a outputs/heuristic --b outputs/gemini
.venv/bin/python scripts/evaluate.py stability --a outputs/gemini --b outputs/gemini_shuffled
.venv/bin/python scripts/evaluate.py review-sheet     # 25 clips for a blind human review
.venv/bin/python scripts/evaluate.py human            # after filling in eval/review_sheet.csv
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests that use the real input pack skip themselves when the pack isn't present.

## Confidentiality

The input pack is Kuku's content, shared for this assignment only. It is not
in this repo: `.gitignore` excludes `*.txt`. The raw outputs quote individual
lines, as the brief's required hook and cliffhanger fields need.
