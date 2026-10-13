# Optional Human host phase diagnostics

Set `NUMI_HUMAN_HOST_INNER_PROFILE=1` to record bounded first-eight host intervals in articulated submit/run, native Brain motor callbacks, and the separate HumanIO transaction owner. Unset, zero, and other values disable diagnostics. Disabled tracing makes no clock reads. The fixed-size tracer buffers output until its own lock scope unwinds; outer callers may still hold their own locks. Parent intervals include nested diagnostic overhead. These are host intervals, not GPU durations or performance qualification.

All owners preserve their existing phase, generation, source identity, lease and dispatch rules. Transaction callback image/symbol attribution is emitted only with diagnostics enabled. Standard and current split-encoder paths remain intact.

Native M4 default eight-step and split nine-step matched pairs preserve every accepted sensor, motor and standing-progress row, actual first/final commit summaries, final geometry sections and all four images. The ninth split root produces no host diagnostic marks, demonstrating the sampling boundary. The standing path bypasses the separate HumanIO callback; that owner is exercised separately by the existing exact-v2 CPU contract and native exact receipt and motor/sensor transaction tests. All three pass with diagnostics disabled and enabled; the enabled tests produce32 HumanIO phase marks and disabled tests produce none.

Reports pin actual tested source file hashes and binary/library inputs; their source base is d954e0b7 with the exact source-file changes published alongside this receipt. No throughput gain, GPU execution duration, new ten-second stance qualification, or anatomical qualification is claimed.
