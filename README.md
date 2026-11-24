# entropy-guided-grpo

An academic project investigating whether entropy-reduction rewards help a small language model ask informative questions and identify hidden animals more efficiently in Twenty Questions.

## Planned approach

- Build a deterministic game environment using UCI Zoo animal attributes.
- Use a frozen parser to map natural-language questions to attribute tests.
- Warm-start the policy with supervised fine-tuning, then compare GRPO with and without an entropy bonus, giving both conditions an incentive to finish sooner.
- Evaluate identification success, failure-penalized moves, learning efficiency, and compute cost using controlled baselines, ablations, and multiple training seeds.

## Status

Initial repository setup. Implementation details, experiment configurations, and work allocation are being finalized.
