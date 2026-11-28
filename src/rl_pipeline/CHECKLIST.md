# 5. RL pipeline and compute

The pipeline loads game manifests and connects existing components; it should not generate a separate definition of the test data or game rules.

- [ ] **Pin the training environment:** Python, PyTorch/CUDA, Transformers, TRL, PEFT, Accelerate, and the selected rollout backend.
- [ ] **Load the policy and tokenizer:** record exact model revisions and chat templates.
- [ ] **Configure ordinary LoRA and bf16:** follow the professor’s recommendation and verify actual memory use.
- [ ] **Configure the frozen reference policy:** confirm that it represents the intended SFT checkpoint.
- [ ] **Implement the rollout adapter:** policy action → parser → environment → public response → next action.
- [ ] **Verify multi-turn support in the pinned TRL version.**
- [ ] **Verify loss masking:** train on policy-generated action tokens; environment responses and supplied context receive zero direct policy loss.
- [ ] **Verify token alignment:** generated token IDs, behavior log probabilities, masks, and rewards refer to the same rollout.
- [ ] **Verify GRPO grouping:** group members share the initial roster, target, display order, and prompt, while maintaining independent environment state.
- [ ] **Check reward aggregation:** each game’s return is applied to that game’s policy tokens once.
- [ ] **Run a tiny complete training update:** generate games, calculate advantages, backpropagate, and update the intended LoRA parameters.
- [ ] **Track zero-advantage groups:** inspect how frequently games within a group receive identical returns. TRL exposes reward-standard-deviation and zero-standard-deviation diagnostics for this purpose. [TRL documentation](https://huggingface.co/docs/trl/v0.26.0/grpo_trainer)
- [ ] **Track training health:** loss, KL divergence, gradient norm, reward components, success, invalid actions, and non-finite values.
- [ ] **Check context/output limits:** ensure full games fit; record token-limit endings separately.
- [ ] **Test checkpoint save, reload, and resume.**
- [ ] **Benchmark a representative pilot:** include policy generation, parsing, environment execution, updates, and evaluation.
- [ ] **Estimate full compute needs:** GPU-hours, elapsed time, peak memory, storage, API calls, retries, and API cost.
- [ ] **Identify bottlenecks:** distinguish slow generation, slow parsing, low GPU utilization, and oversized batches.
- [ ] **Set shared run limits:** action-token cap, resource ceiling, evaluation frequency, and checkpoint rule.
- [ ] **Prepare fallbacks:** smaller batches first; then a shared smaller model, simpler actions, or lower token cap if required.
- [ ] **Run conditions across multiple seeds:** use matched evaluation games and comparable training budgets.

**Completion check:** a short run completes, changes the intended parameters, saves usable outputs, and supports a credible full-run cost estimate.
