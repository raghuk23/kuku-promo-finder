# Ablation: heuristics only vs LLM judge

Compared **250** candidates judged validly in both `heuristic` and `gemini`.

| Metric | Value |
|---|---|
| Spearman rank correlation of final scores | 0.29 |
| Mean absolute difference, hook / cliffhanger / standalone (1-10 scale) | 2.99 / 2.45 / n/a |
| Candidates with identical scores | 0% |
| Same quality-floor decision | 65% |
| Same best candidate per episode (50 episodes) | 30% |
| Overlap of posted top lists | 2 of 10 |

Quality floor decisions:

| | gemini passes | gemini rejects |
|---|---|---|
| **heuristic passes** | 29 | 57 |
| **heuristic rejects** | 31 | 133 |

Ability to separate the top picks:

- Run A: posted top 10 has 1 distinct final scores (range 10.00-10.00).
- Run B: posted top 10 has 9 distinct final scores (range 8.32-9.30).
