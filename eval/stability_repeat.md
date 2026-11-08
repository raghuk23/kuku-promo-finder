# Stability of the LLM judge

Compared **54** candidates judged validly in both `gemini` and `gemini_repeat`.

| Metric | Value |
|---|---|
| Spearman rank correlation of final scores | 1.00 |
| Mean absolute difference, hook / cliffhanger / standalone (1-10 scale) | 0.00 / 0.00 / 0.00 |
| Candidates with identical scores | 100% |
| Same quality-floor decision | 100% |
| Same best candidate per episode (10 episodes) | 100% |
| Overlap of posted top lists | n/a (runs cover different episodes) |

Quality floor decisions:

| | gemini_repeat passes | gemini_repeat rejects |
|---|---|---|
| **gemini passes** | 24 | 0 |
| **gemini rejects** | 0 | 30 |
