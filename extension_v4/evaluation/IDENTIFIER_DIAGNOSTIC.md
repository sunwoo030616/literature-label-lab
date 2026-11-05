# Post-hoc identifier preservation diagnostic

Written after reading the six original agent traces, before replaying modified plans.

Observed: the English statin question named 9 studies/3449 patients, but planner-v1 emitted only `statins COVID-19 study patients`. The retrieved source changed. The Korean counterpart retained 3449 and found the target.

Replay saved raw plans through an optional deterministic guard. A single explicitly supplied PMID binds lookup to that ID; multiple IDs require disambiguation. Unrequested PMID lookups fail closed. Search queries keep 3+-digit numeric anchors found in the question, including commas/Korean suffixes. This does not preserve all small counts, named entities or semantics.

Unchanged plans reuse original answers. Only changed plans are searched again through MCP and answered with the unchanged evidence-first-v1 prompt. Keep original plans/results. Report same-task post-hoc improvement, not new blind generalization. This isolates the effect of preserving explicit identifiers without new stochastic planner outputs.
