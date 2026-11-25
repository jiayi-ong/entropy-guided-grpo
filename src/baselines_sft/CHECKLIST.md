# 4. Controller baselines and SFT

This deserves its own work area because it determines whether RL has room to improve.

- [ ] **Implement a random legal-question controller.**
- [ ] **Implement a greedy expected-information-gain controller:** choose the question before seeing the hidden target’s answer.
- [ ] **Run the original policy without training.**
- [ ] **Generate training-only demonstrations:** use the greedy controller and reviewed question phrasings.
- [ ] **Include full dialogues and final guesses:** use the same message/action format as later rollouts.
- [ ] **Train the shared SFT checkpoint.**
- [ ] **Evaluate original, SFT-only, random, and greedy on identical development games.**
- [ ] **Check SFT headroom:** compare success, failure-penalized moves, and question quality.
- [ ] **Define the response to a ceiling:** make one justified development-stage adjustment or report that SFT already solves the chosen task well.
- [ ] **Freeze the shared initialization:** every RL condition starts from the same SFT checkpoint.

**Completion check:** you have a baseline table before committing to the main RL runs.
