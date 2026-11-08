# Stability of the LLM judge

Compared **250** candidates judged validly in both `gemini_single_order` and `gemini`.

| Metric | Value |
|---|---|
| Spearman rank correlation of final scores | 0.95 |
| Mean absolute difference, hook / cliffhanger / standalone (1-10 scale) | 0.36 / 0.36 / 0.36 |
| Candidates with identical scores | 10% |
| Same quality-floor decision | 92% |
| Same best candidate per episode (50 episodes) | 72% |
| Overlap of posted top lists | 8 of 10 |

Quality floor decisions:

| | gemini passes | gemini rejects |
|---|---|---|
| **gemini_single_order passes** | 58 | 17 |
| **gemini_single_order rejects** | 2 | 173 |
