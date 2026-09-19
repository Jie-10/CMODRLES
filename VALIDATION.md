# Validation notes

The package was structurally checked against the current PlatEMO
`DRLOS-EMCMO` implementation.

Checked items:

- `CMODRLES` inherits directly from `ALGORITHM`.
- Main loop uses `Algorithm.NotTerminated(...)`; no `NotTerminated2` call remains.
- The previous `CMODRLES_DQN < handle` class is removed.
- DRL state is stored as two values, action as one value, reward as one value,
  and next state as two values: `Data = [s1 s2 a r s1_next s2_next]`.
- Action 1 maps to AP and action 2 maps to DP through `auxIndex = action + 1`.
- Candidate ordering for MP is `[MP, O_MP, O_aux]`, so auxiliary reward credit
  starts at candidate index `2N+1`.
- MP and the selected auxiliary task each generate N evaluated offspring per
  complete iteration.
- `Dropout` contains the same DRLOS-EMCMO network utility implementation style
  (`trainmodel`, `testNet`, `updatemodel`, etc.).

MATLAB is not installed in the packaging environment, so an actual PlatEMO
end-to-end execution was not performed here. Run a short DAS-CMOP smoke test
before launching the full experimental batch.
