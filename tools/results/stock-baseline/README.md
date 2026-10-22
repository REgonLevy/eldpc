# Stock baseline: ldpc 2.4.1 at its common settings

`stock-baseline.json`: the stock decoder that eldpc is compared with, ldpc 2.4.1 `BpOsdDecoder` at its common
settings (`ms30`: min-sum alpha 0.625, 30 iterations, parallel schedule, OSD-0), and the LER records the stock
comparisons use. eldpc's `ms30` preset is bit-identical to it (`verify/`), so the two have one LER.

How it was produced:

- latency: the Python `decode()` call, one syndrome per call, timed per shot in one process alone on one
  isolated CPU of machine A's host (Ryzen 7 9850X3D, bench mode; tuning/KNOWLEDGE.md section 0.2), on a fresh
  latency shot set per problem; the `ms30` rows were capped at 540 s (hence 7,103 shots on bb144 xyz). The stock
  observables equal eldpc's on every timed shot.
- LER: fresh sets of 1M shots each (four per row), decoded by eldpc at the same config (bit-identical to stock);
  99 % two-sided Clopper-Pearson intervals. The pooled `a03125c3` row adds the five fresh-shot runs of that
  preset on the identical circuit.

Cited by: tuning/KNOWLEDGE.md sections 4.2, 5.2, 5.3 and 5.5; README.md ("Results"); docs/CLAIMS.md. The "1,900-2,140x"
of KNOWLEDGE 5.2 is computed from this record's 72.8 ms, the C-ABI stock row of
`tuning/data/knob_facts.json: bench` (65.0 ms) and the A-stock `a03125c3` p50 of
`tools/results/final-bench/final-bench.json` (34.02 µs).
