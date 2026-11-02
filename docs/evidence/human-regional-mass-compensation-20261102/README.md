# Bounded regional source compensation and collector fix

The prior strict routing graph was short by 0.00029156037632 kg in 6,359 Skin cells. This opt-in extension admits selected original Skin parents through same-material source-face adjacency within two region hops. It does not edit geometry, mass, quadrature weights, labels, or Skin/Fat density bounds. Only selected parent-to-child routes within 0.075 m are added; their entire original source capacity is 0.0003117525663607096 kg, below the 0.000325 kg policy budget. Existing shared source capacities prevent double spending. Carrier expansion and solver-free native import recheck the policy.

The full trial exposed a separate numerical error: a rounded-down global supply-minus-minimum collector cap could strand positive source residuals. The collector is redundant because source arcs already bound supply. It now uses a ceiling of total source supply, while every minimum, maximum, source ledger and closure check remains mandatory. The three-cell regression failed before this fix and passes afterward; genuinely insufficient receiver capacity still fails.

Carrier validation also inverts an already validated explicit allowlist once instead of scanning all children for every source parent. Child order and nearest distances are preserved; legacy region-only support retains its original path. The regression compares indexed and predicate-scan flow/mass/distances exactly.

Nine focused C++ controls pass with strict warnings and assertions enabled. The native executable builds with the compensation and collector changes. Full allocation and deterministic replay are pending in full-allocation-003 on SSH Numi. These results establish source software checks only: no physical admission or native motion is claimed.

Files preserve the explicit policy, actual compile/run commands, the failed full trial, before-fix counterexample, and focused control outputs. The unchanged accepted geometry is candidate 1020. Native integration and motion remain separate gates.
