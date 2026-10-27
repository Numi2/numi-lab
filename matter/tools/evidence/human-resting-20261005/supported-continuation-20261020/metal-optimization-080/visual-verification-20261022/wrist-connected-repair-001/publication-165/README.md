# P165 - bounded connected-refinement evidence

This increment preserves completed offline trials 608-614 and their failure records. The new retained-field construction reaches 1419 surface / 0 contained / 0 self pairs across 185 banks. Its candidate SHA-256 is b9d88e147e365a040825b8b476e8d90adf7ed17c00da2870dca76363c47199c3. This remains static construction evidence.

## Recorded outcomes

| Trial | Result | Interpretation |
|---|---:|---|
| 608 | 661 self pairs | New-cut trial failed the self gate. |
| 609 v1 / 610-001 | Wrong argv; preparation failure | Retained as failed setup evidence. |
| 609 v2 / 610-002 | 10 unit tests passed; 94 self pairs | Corrected merge path ran, but the candidate still fails the self gate. |
| 611 runs 001-002 | No gain | Bounded harmonic searches did not improve the parent candidate. |
| 612 raw LP | 48 self pairs; fit report write failed with ENOSPC | Candidate SHA-256 is 1d146b8d7bfc41aeb5d1f29843a19736328d0632693f4b3b5ba22fce05afaff0; the zero-byte report and salvage receipt are both retained. |
| 612 retained candidate | 2207 surface / 337 contained / 0 self; 185 banks | Exact full construction checks and P175 finalization passed. It remains unselected because contained pairs remain. |
| 614 fresh unretained proposal | Failed the cheap F32 area gate | Retained as a negative. |
| 614 retained-field candidate | 1419 surface / 0 contained / 0 self; 185 banks | Full545 and strict final state passed. Five focused tests passed. This is the new best construction checkpoint. |

The 612 owner wrapper emitted a diagnostic NameError after the successful full check. The actual check execution receipt records rc=0 and unchanged source pins; this is not labeled as an audit failure.

## Evidence and limits

The run directories preserve candidates, launch and execution receipts, raw self-gate evidence, and failed setup attempts. The 608 and 610-002 large fit reports are supplied as lossless zstd archives with restoration receipts and a tested restore helper. The 612 ENOSPC receipt preserves the complete candidate despite the missing fit report.

All results are static construction evidence. native_admission=false; accepted_physical_steps=0. No physical or biological admission is claimed. The previous global checkpoint was 1576 / 0 / 0.

P165 is parented to P164 commit 392e6cc19544e08096ea28a7ff9411e706623ea1, whose manifest SHA-256 is 757e95d2a92d9bd69d01fbaa76c625de5393fa78bf0ad0b09aa2b63b7b74b236.
