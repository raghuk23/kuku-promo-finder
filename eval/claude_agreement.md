# Agreement with a second LLM (Claude, blind)

Clips scored blind by a second LLM (Claude, blind): **25** (1-5, 'would I post this'; 4+ counts as good). The 25 mix the single-order top 10, its 5 weakest, and 10 random candidates.

| Judge | Spearman vs reviewer | Floor precision | Floor recall |
|---|---|---|---|
| heuristic | 0.40 | 67% | 50% |
| llm_single_order | 0.81 | 82% | 75% |
| llm_two_orders | 0.82 | 82% | 75% |

Floor precision: of clips the judge would post, the share the reviewer rated 4+. Floor recall: of clips the reviewer rated 4+, the share the judge would post. Small sample (n=25): treat differences of a few points as noise.
