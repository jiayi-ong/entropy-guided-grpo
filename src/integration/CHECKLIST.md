# 9. Interfaces, integration, and minimal project management

Keep this small enough for notebooks and supporting Python files.

- [ ] **Agree on shared records:** game configuration, parsed action, environment transition, episode result, and run configuration.
- [ ] **Define one authoritative owner for each calculation:** beliefs, rewards, counting, and metrics.
- [ ] **Document counting rules:** actual moves, reward-charged moves, failure-penalized moves, and training action tokens.
- [ ] **Test one successful game through the whole system.**
- [ ] **Test one problematic game:** unsupported question → rephrase → wrong guess → failure.
- [ ] **Test one complete GRPO group:** include an equal-return case and verify safe zero-advantage handling.
- [ ] **Test saved-result regeneration:** another teammate reproduces metrics without running training.
- [ ] **Use one repository and short branches:** review changes to interfaces, reward math, and masking.
- [ ] **Add minimal CPU CI:** imports, environment/reward tests, schema validation, and synthetic mask checks.
- [ ] **Keep one GPU smoke notebook:** rerun it after changes to the training stack or rollout integration.
- [ ] **Record code/configuration/model versions and seeds for every run.**
- [ ] **Store checkpoints and large outputs outside Git.**
- [ ] **Prepare the final reproducible package:** executed analysis notebook, predictions, manifests, configuration, checkpoint references, and concise instructions.

Before full training, the essential evidence is: **correct scripted games, a sufficiently accurate parser, an SFT-versus-greedy comparison, a working GRPO update, and a measured runtime estimate**.
