# Parallel Human contact validation

This increment replaces the one-thread-per-human scan of all 413,997 FEM/body contact samples with one GPU thread per sample, followed by bounded 256-sample block summaries and deterministic provisional-status consumption. Force, tangent, reaction, acceptance rules, and physiological state ownership are unchanged. Persistent scratch buffers stay GPU-resident; CPU code only allocates and encodes. ABI17 prevents mismatched host/shader scratch bindings.

The 20-stage existing owner suite passes, including 1,605 small-face/feature/rigid-pose probes, the synthetic 86-body/413,488-pair capacity fixture, stale-ABI rejection, and zero/partial/multiple-block invalid-sample tests. CPU tests cover the UINT32_MAX last-block arithmetic without a giant allocation. These are contact-owner tests, not a full anatomical qualification.

The exact complete-scene source preflight passes its intentional source-only stop. The matched native candidate returns the same byte-identical common-field failure receipt as ABI16: zero accepted physical steps, material object 15, Newton iteration 5, diagnostic negative infinity and 12 backtracks. Controller/physiology accepted state has not advanced in this failure path.

The single matched process run took 89.836592 s versus 133.927499 s (32.92% less wall time). This is failure latency including initialization, not sustained or accepted simulation throughput. No 0.1x claim is made. The unresolved skin/fat geometry, supported equilibrium, complete rollback qualification and long coupled runs remain blockers.

Source-map, exact build/launch inputs, test receipts and failed native traces are retained below. The shared dirty checkout and its normal index were not changed by publication.
