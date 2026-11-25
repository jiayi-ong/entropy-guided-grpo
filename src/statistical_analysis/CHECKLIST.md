# 7. Statistical analysis

Start with effect sizes and uncertainty, then choose tests that fit the experiment.

- [ ] **Calculate absolute and relative improvement:** moves saved per game, percentage reduction in moves, and success differences in percentage points.
- [ ] **Calculate paired confidence intervals:** preserve matching by game ID.
- [ ] **Account for repeated target animals:** use target-grouped bootstrap comparisons rather than assuming every generated game is independent.
- [ ] **Show training-seed variation separately:** hundreds of games do not replace independent training runs.
- [ ] **Plan sample size using development results:** estimate variation, choose the minimum useful effect, and assess whether the available seeds/games can detect it.
- [ ] **Use simulation-based power checks where practical:** preserve the target and seed structure.
- [ ] **Use analytical paired-test power as a rough cross-check:** its inputs must reflect paired differences and the number of independent pairs. [statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.solve_power.html)
- [ ] **Use a paired t-test only with a defensible independent analysis unit and suitable assumptions:** matching the same games does not remove all target/seed dependence. SciPy’s paired test operates on related measurements. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
- [ ] **Treat ANOVA as optional:** if used, account for repeated measurements; do not apply ordinary independent-group ANOVA to all game rows.
- [ ] **Check the success guardrail explicitly:** a non-significant success difference does not establish that any loss is acceptably small.
- [ ] **Limit multiple comparisons:** identify the primary hypothesis and label other findings secondary or exploratory.
- [ ] **Report inconclusive results honestly:** distinguish “the estimated effect is promising” from “the evidence establishes the target effect.”
