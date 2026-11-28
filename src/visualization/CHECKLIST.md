# 8. Visualizations

These are concrete plots worth planning.

| Plot | What to show | What it helps answer |
|---|---|---|
| **Learning curves** | Training action tokens on x-axis; success and failure-penalized moves in separate panels; individual seed lines | Does entropy improve learning speed or final performance? |
| **Success by move budget** | Move budget on x-axis; fraction identified by that move on y-axis | Does the policy reach correct answers sooner? |
| **Entropy reduction curves** | Move number versus normalized candidate entropy; individual examples plus an aggregate | Does it gather information earlier? State how ended games are handled. |
| **Paired game comparison** | Baseline moves versus entropy-condition moves, with a diagonal reference | Which games improve or deteriorate? |
| **Effect estimates with intervals** | Move reduction and success difference, with confidence intervals and target thresholds | Are improvements practically meaningful and sufficiently precise? |
| **Error breakdown** | Counts/rates of semantic errors, malformed output, repeats, unsupported questions, and wrong identification | What actually causes failures? |
| **Runtime breakdown** | Time spent generating, parsing, executing the environment, and updating | Where is compute being spent? |
| **Training-signal diagnostics** | Reward variation and zero-advantage-group rate over updates | Is the baseline receiving enough signal to learn? |
| **Optional target-level heatmap** | Animal/profile versus condition, colored by success or moves | Are overall gains driven by a few easy or difficult targets? |
