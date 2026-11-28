# 1. Game environment and data

The environment owns the game state, answers, and candidate filtering.

- [ ] **Define a common dataset format:** candidate ID, display name, attributes, attribute types, allowed values, and short attribute definitions.
- [ ] **Create small artificial datasets for testing:** include balanced attributes, numeric attributes, duplicate profiles, and deliberately uninformative questions.
- [ ] **Load and audit UCI Zoo:** check duplicate names, duplicate attribute profiles, unusual labels, and numeric attributes such as legs.
- [ ] **Make dataset loading interchangeable:** artificial data and UCI Zoo should enter the same environment through the common format.
- [ ] **Generate game configurations:** candidate roster, hidden target, display order, candidate count, and random seed.
- [ ] **Create train, development, and final test manifests:** keep repeated rosters out of different splits; reserve unseen profiles separately if retained.
- [ ] **Define the game rules:** allowed questions, candidate guesses, wrong guesses, repeated questions, unsupported questions, rephrasing, and the move limit.
- [ ] **Implement `reset()` and `step()`:** return the public observation, reward, episode status, and diagnostic information. Gymnasium separates normal termination from truncation; preserve that distinction and record the specific ending reason. [Gymnasium documentation](https://gymnasium.farama.org/api/env/)
- [ ] **Build the public game context:** original candidate table, attribute definitions, question/answer history, and remaining moves.
- [ ] **Keep private information separate:** hidden target, surviving candidate set, entropy, and reward calculations must not enter policy or parser prompts.
- [ ] **Implement candidate filtering:** update the surviving set after a valid answer; handle wrong guesses according to the agreed rules.
- [ ] **Calculate belief entropy:** under the uniform surviving-candidate assumption, calculate entropy from the surviving candidate count.
- [ ] **Create deterministic scripted games:** for example, eight candidates reduced to four, two, and one should produce entropy values of three, two, one, and zero bits.
- [ ] **Test edge cases:** unknown answers, repeated predicates, negation, numeric comparisons, invalid candidate IDs, singleton sets, and move-limit failures.
- [ ] **Save complete game traces:** question, parsed action, answer, surviving count, entropy before/after, move number, and ending reason.

**Completion check:** the same game and scripted actions reproduce the same trace, and every answer/filtering step agrees with hand calculations.
