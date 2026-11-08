# Stability of the LLM judge

Compared **54** candidates judged validly in both `gemini` and `gemini_shuffled`.

| Metric | Value |
|---|---|
| Spearman rank correlation of final scores | 0.64 |
| Mean absolute difference, hook / cliffhanger / standalone (1-10 scale) | 0.87 / 0.83 / 0.83 |
| Candidates with identical scores | 6% |
| Same quality-floor decision | 74% |
| Same best candidate per episode (10 episodes) | 40% |
| Overlap of posted top lists | n/a (runs cover different episodes) |

Quality floor decisions:

| | gemini_shuffled passes | gemini_shuffled rejects |
|---|---|---|
| **gemini passes** | 14 | 10 |
| **gemini rejects** | 4 | 26 |
