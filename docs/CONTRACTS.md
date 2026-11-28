# Shared boundaries

CHECKLIST.md files describe future project work. This document describes the
smaller implemented slice; checklist items are not claims of completed work.

## Invariants

1. The environment alone answers questions and updates surviving candidates.
2. Player and parser receive the original public table and relevant history.
   Neither receives the target, filtered set, entropy, or rewards.
3. Each policy action consumes one move; internal parser retries consume none.
   Correct guesses end games. Wrong guesses continue. Singleton candidate sets
   still require an explicit guess.
4. Each GRPO group has G complete games with the same roster, target, order,
   and prompt. State and samples are independent. Infrastructure failures do not
   become losing games or partially accepted groups.
5. One complete-game return supplies one group-relative advantage to that
   trajectory's policy tokens. This is trajectory-level shaping, not per-turn
   advantage assignment.
6. New policy tokens are counted and trained once. Initial prompts, supplied
   feedback, padding, and copied history have zero direct policy loss. Context
   can still influence future predictions and their gradients.
7. Group size divides each local microbatch. Native TRL owns reward normalization,
   GRPO loss, gradient accumulation, optimization, and DDP synchronization.

## Records

| Record | Producer | Meaning |
| --- | --- | --- |
| GameSpec | Game/data developer | Fixed roster and private target |
| Observation | Environment | Public roster, normalized action/answer history, remaining moves |
| Predicate / Action | Parser | Exactly one question predicate or guess |
| ParseResult / ParseAttempt | Parser | Final interpretation and individual API attempts |
| Transition / StepResult | Environment | Public observation separated from private evidence |
| GeneratedTurn | Policy backend | Exact sampled IDs and aligned behavior log probabilities |
| Trajectory | Driver | Complete game, token mask, and transition evidence |
| RewardBreakdown | Scorer | Named components whose sum is the scalar return |

The fixture executes equality predicates. Operator and negation fields make
future numeric predicates possible without changing the record shape. The API
validator initially accepts Boolean equality only; widen it and the environment
together when extending the question language.

Raw fixture actions are Q0..Q4 and G0..G31, referring to displayed column and row
order. Free-form mode accepts one yes/no question; guesses still use G IDs and
bypass semantic parsing. Changing this language requires a contract-version
change and renewed integration tests.

## Parser failures

valid carries an action. unsupported, ambiguous, and invalid are interpreted game
events without an action. failed means the parser could not produce a usable
interpretation; the driver stops the incomplete batch.

ParseAttempt preserves stage, HTTP status, provider error code, request ID,
duration, and token usage. Stages distinguish transport, service, request setup,
output validation, and success. A 404 is a request failure, not a wrong question.
Temporary 429/server errors have bounded retries. Spend/credit failures and a
Retry-After exceeding the configured bound stop retries.

HTTP 200 with malformed JSON or an invented attribute is an output validation
failure. Correct interpretation of a question about unavailable data is
unsupported. Schema-valid but semantically wrong interpretation needs an audit
set; type validation alone cannot detect it.

Attempts remain separate from the final action outcome. Cache hits have no new
attempts. The future evaluation developer chooses aggregation and denominators
from this raw evidence, including incomplete coverage rather than silent drops.

The included frontier adapter uses Anthropic Messages. Set an exact model ID and
ANTHROPIC_API_KEY. No live request runs in CI. Other providers implement the same
QuestionParser interface without changes to the environment or trainer.

## Token and trainer protocol

plain-transcript-v1 is a prefix-preserving text transcript, not native tool calling.
The driver encodes the public prompt once, appends exact sampled IDs, then appends
encoded environment feedback with mask 0. It never reconstructs older sampled
tokens from decoded text. Training and inference use this same protocol. Native
chat-template support is future work rather than an implicit loader behavior.

GameGRPOTrainer overrides only _generate_single_turn in pinned TRL 0.29.0.
That release's ordinary Transformers path does not call the custom rollout hook,
so the narrow override returns the native generation tuple plus env_mask and
scalar rewards. The library handles the remaining training algorithm. Upgrade
only after rerunning group, mask, update, and resume tests.

One fresh generation batch supplies one optimizer update, optionally through
gradient accumulation. No rollout-reuse buffer spans completed optimizer steps.
Native sampler/RNG recovery therefore identifies the next manifest rows.

## Extending a component

- Environment: implement GameEnvironment or extend TableGameEnvironment. Preserve
  authoritative beliefs and public/private separation.
- Parser: implement parse_batch with one result per input in the same order.
  Preserve statuses and diagnostics.
- Reward: implement score using transitions. Do not mutate traces or duplicate
  belief updates. New reward randomness must have explicit checkpointed state.
- Policy: implement generate_batch over exact prefixes. A different sampling
  distribution may require explicit probability-ratio handling.
- Evaluation: reuse load_inference_policy and play_games, without an optimizer
  or trainer. Group labels there can be ordinary bookkeeping IDs.

Records are plain dataclasses; ABCs define only replaceable operations;
orchestration stays in functions. There is no registry.
