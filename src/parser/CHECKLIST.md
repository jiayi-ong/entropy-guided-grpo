# 2. Parser

Keep the parser independent of a particular dataset by passing it the dataset’s attribute definitions and allowed predicates.

- [ ] **Define the parser input:** question text, attribute names/descriptions, types, allowed operators, and any necessary public context.
- [ ] **Define the output schema:** action type, attribute, operator, value, negation, and parse status.
- [ ] **Separate questions from candidate guesses:** guesses should resolve to listed candidate IDs through a simple validation path.
- [ ] **Write the parser prompt:** explain supported questions, unavailable attributes, ambiguity, and rejection behavior.
- [ ] **Implement Pydantic validation:** reject missing fields, invalid operators, wrong value types, and unknown attributes.
- [ ] **Implement semantic validation:** check whether the structured predicate preserves the original meaning, including negation and thresholds.
- [ ] **Create reviewed aliases:** do not automatically treat “legs” and “limbs,” or “feline” and “cat,” as equivalent. Accept an alias only when the dataset definitions support that interpretation.
- [ ] **Handle numeric questions:** distinguish `legs == 4`, `legs >= 4`, and `legs > 0`.
- [ ] **Handle unsupported or ambiguous meanings:** return explicit statuses such as `unsupported`, `ambiguous`, or `invalid`; avoid silently choosing an interpretation.
- [ ] **Add bounded retries:** specify the number of retries and what information the retry receives. A parser retry on unchanged text should not become an extra policy move.
- [ ] **Implement a simple model-call wrapper:** support the chosen local parser or frontier-model API without building a general client framework.
- [ ] **Freeze parser configuration:** model/version, prompt, schema, temperature, output limit, seed where supported, and reasoning effort where applicable.
- [ ] **Measure repeatability:** run fixed inputs repeatedly and check whether interpretations change.
- [ ] **Add caching:** include question, relevant context, schema version, prompt version, and model configuration in the cache key.
- [ ] **Build a manually reviewed parser evaluation set:** cover paraphrases, negation, numbers, unsupported questions, and ambiguity.
- [ ] **Measure schema validity and semantic accuracy separately:** valid structure does not establish correct interpretation.
- [ ] **Track failures by cause:** configuration/endpoint errors such as 404; timeouts/rate limits; OOM/resource errors; malformed output; and incorrect interpretation.
- [ ] **Define recovery rules:** technical failures should not silently become “unknown” game answers or policy failures.
- [ ] **Prepare a fallback:** predefined question IDs if reliable language interpretation cannot be achieved.

**Completion check:** at least 95% semantic accuracy on the development audit, with no known systematic error in numeric predicates or negation.
