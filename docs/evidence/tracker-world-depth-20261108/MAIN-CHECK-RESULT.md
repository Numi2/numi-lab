# Main compatibility check

Main source 55216b9b427b18f25885098496f092a8ed7b0f97 was built in its clean checkout; the exact retained depth fixture was compiled separately against that source and runtime. Two captured runs and the capture-free control completed. The normalized verifier passed all checks: 55 accepted, three rejected without reset, two rejected with reset, three mixed steps, and nonvacuous visible depth in accepted and rejected cases. Captures, physical outputs and transitions repeated and the control matched. The archived main captures were independently extracted, hash-checked and reanalyzed with the same result.

This check covers main and the integration branch separately; it does not merge them or establish equivalence of their other rendering behavior. It is a zero-action, oracle-mask fixture with forced capacity failures, not full sensor or policy transaction qualification.
