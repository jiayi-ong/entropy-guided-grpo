# 3. Reward design

Keep reward calculations in one place. The environment supplies authoritative state changes; the reward function converts them into scores.

- [ ] **Implement terminal correctness:** correct identification versus failure.
- [ ] **Implement the shared action cost:** charge questions, rephrases, invalid actions, and candidate guesses consistently.
- [ ] **Define failure charging:** early abandonment must not receive an advantage from avoiding later action costs.
- [ ] **Implement information gain:** calculate entropy before minus entropy after an eligible answered question.
- [ ] **Normalize information gain:** make reward scale comparable across candidate-set sizes.
- [ ] **Implement discounting:** use the actual policy move index, including earlier unsuccessful questions and rephrases.
- [ ] **Specify zero-bonus events:** guesses, repeats, unknown answers, and invalid questions.
- [ ] **Separate reward components in logs:** terminal reward, action cost, information bonus, and total return.
- [ ] **Define step versus episode rewards:** avoid adding a cumulative reward repeatedly at every step.
- [ ] **Check numerical bounds and scaling:** inspect the relative size of correctness, move cost, and information reward.
- [ ] **Test reward exploits:** early abandonment, repeated questions, excessive candidate guessing, unsupported questions, and collecting information without making a final identification.
- [ ] **Check the undiscounted sum:** understand when it becomes constant across successful games.

Use the following comparisons:

| Condition | Purpose |
|---|---|
| Correctness only | Optional diagnostic showing the original sparse-reward behavior |
| Correctness + action cost | Main baseline; already rewards finishing sooner |
| Correctness + action cost + discounted information | Main entropy condition |
| Same reward with undiscounted information | Optional discounting ablation |
| Action-cost baseline + information-free bonus | Optional control for extra reward variation |

An information-free bonus control becomes particularly useful if you want to attribute gains specifically to information content. A per-move cost can still yield identical returns when every game fails at the same move limit.

**Completion check:** all reward components match hand calculations and cannot be increased through an obvious counting or termination loophole.
