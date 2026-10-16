# Source 172: warm CGLS finite-breakdown decline

This overlay is based on frozen source169 `a5ccffa52ffe368b49c877094a5081b7ee246ebf`. It is default-off. If an explicitly enabled warm-start CGLS solve reaches the exact finite `BadDenominator` condition, and the selected environment is healthy, the original FGMRES certificate is still present, and every relevant CGLS row/scalar is finite and structurally valid, the optional refinement is declined. The original FGMRES solution, certificate, and line-search state are preserved. Mixed error bits, nonfinite data, invalid dimensions, unhealthy status, or missing certificate keep the ordinary fail-closed behavior.

The implementation uses a private marker between finalization and installation. The row validator checks the exact error signature and all private CGLS rows; the installer rechecks the marker and original certificate after the operator action. It labels the trace as a finite breakdown decline, not as an installed refinement. No tolerance or physical operator was changed.

The candidate differs from source169 in exactly four files; the checker Metal source is byte-identical. `source-172.patch` is the standalone source delta and `source-overlay.json` is the canonical hash manifest. The temporary compile tree binds the two candidate CGLS Metal includes; depfiles record those resolved paths.

## Validation

Both production and checker Metal translation units compiled successfully to AIR. Objective-C++ syntax checks passed for `runtime.mm` and the checker host. The production Metal translation unit emitted two pre-existing maybe-uninitialized warnings in the frozen `fgmres.metalinc` (`sharedValid` and `refinementValid`). The checker itself was not executed, no metallib was linked, and no GPU or physical transaction was run.

## Integration order

Apply the separately sealed source171 checkpoint-arena/header change and this source172 patch onto source169, resolve only any mechanical overlap, then configure/build the combined source173 and run its CPU tests plus production/checker Metal compile. Before any trajectory test, run the compiled warm-start kernel checker and confirm the exact finite-breakdown decline and legacy-fail-closed cases. Only after that should root create a fresh warm-enabled control0 checkpoint and resume it with matching source/program/environment inputs. Do not reuse a checkpoint created with the warm-start option absent.

The source173 integration and any native/checkpoint qualification are outside this overlay’s evidence.
