# 6. Experiment design and evaluation

Decide these before inspecting final test results.

- [ ] **State the primary comparison:** entropy condition versus the action-cost baseline.
- [ ] **Define practical success thresholds:** for example, the abstract’s 10% reduction in failure-penalized moves with no more than a two-percentage-point success loss.
- [ ] **Give learning-efficiency and discounting hypotheses their own thresholds.**
- [ ] **Freeze the experiment matrix:** conditions, seeds, datasets, candidate count, and shared settings.
- [ ] **Keep model, parser, SFT initialization, decoding, and training budget fixed across the main reward comparison.**
- [ ] **Separate development decisions from final evaluation:** use development games for debugging and tuning.
- [ ] **Define checkpoint selection:** use a common-budget endpoint for the primary comparison.
- [ ] **Evaluate the same game IDs across conditions.**
- [ ] **Define exclusions and retries:** distinguish task failures from incomplete runs or infrastructure errors.
- [ ] **Keep artificial-data experiments separate:** use toy data for correctness first; treat training on artificial data as an additional experiment if retained.
- [ ] **Optional:** test reserved animal profiles, masked names, and canonical question wording.

Record these metrics:

| Area | Metrics |
|---|---|
| Identification | Success@20; success@5/8/12; final identification accuracy |
| Efficiency | Failure-penalized moves; questions; candidate guesses; total actions |
| Parser | Schema validity; semantic accuracy; unsupported/ambiguous rate; retry rate |
| Question behavior | Repeats; invalid questions; unique predicates; information gained per question |
| Learning | Task metrics versus training action tokens; learning-curve area; zero-advantage-group rate |
| Runtime | Policy/parser latency; end-to-end game latency; token throughput |
| Resources | Peak GPU memory; GPU-hours; parser API calls/tokens; total cost |

Keep **question count, candidate-guess count, and total action count separate**. Also report moves across failures through the agreed penalty; reporting only successful games can make a weak policy appear efficient.
