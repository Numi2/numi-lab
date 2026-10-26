# Connected skin clearance: local step ownership and witness-constrained repair

All execution was on SSH Numi. This is offline geometry construction. Native admission is false and accepted physical steps are zero.

## Checked results

| Candidate | Skin-bone surface pairs | Contained node-bank pairs | Exact self-pairs |
|---|---:|---:|---:|
| 415 baseline | 3,255 | 391 | 0 |
| 417 uniform backtracking | 3,253 | 384 | 0 |
| 427 witness-constrained scale solve | 3,239 | 347 | 0 |

427 retains a maximum 0.437498 mm correction from 415. The final four-trial repair took 55.66 seconds, including the complete 185-bank point and surface checks. This timing excludes the preceding fit and unsuccessful trials. The source displacement maximum is 11.813831 mm, below the declared 12 mm construction cap. All 41,733 current/reference area checks pass, 169 non-coordinate arrays are byte-identical, 237 typed anchors are unchanged, and coordinates outside the declared support are unchanged.

284 surface pair identities are newly introduced despite the net reduction of 16 surface pairs; 300 are resolved. No contained pair is introduced. Maximum contained depth falls 2.693716 to2.382812 mm; summed squared depth falls 0.000172241861 to0.000132489809 m². These counts do not constitute zero whole-body clearance.

## Implementation

418 did not exercise local behavior: its false uniform flag still selected the global 275 helper, reproducing 415 bit-for-bit.419 now validates mode, helper identity/hash and callable contract before launching pinned 373. The graded 423 path additionally binds context arrays to exact baseline and trial hashes.

423 applies graded per-node proposal attenuation using baseline edge lengths and proposal amplitudes. It propagates witness reductions through minimum-plus graph distances, while exempting fixed anchors as reduction sources. Its kappa = 0.1 is an explicit engineering construction parameter.

426 adds current exact failure witnesses as frozen 026 separating rows. It solves a bounded linear program for proposal multipliers, preserving the graded edge budgets and maximizing retained displacement without increasing any existing multiplier. It reuses the existing exact gates; the linear certificate alone cannot accept a candidate.427 reaches self 0 after 10→6→one area failure→0 and passes the complete typed-bank checks. The final area failure is retained, not concealed.

## Failed attempts and verification

420 uses the actual 183 local reducer, retaining a larger correction but ending with 13 self-pairs after eight exact audits.422 removes five of those through two ancestry-preserving flips and stops at 8.424 uses graded attenuation but ends with 10 self-pairs;425 removes six and stops at 4. These branches are rejected.

424's final all-source hash assertion failed because a test file changed during execution. All runtime source pins were unchanged. The precise differing test hashes and the failed assertion are retained; this run is not represented as an immutable all-source run.427 has unchanged runtime/input pins.

24 focused tests plus3 parameter subtests pass: 19 for mode binding, context validation, wrapper dispatch and graded attenuation; 5 for the scale LP, witness constraints, zero feasibility and propagation. Independent full geometry verification is separate evidence. An initial toy test exposed anchor-source propagation; that behavior was corrected before 424. No image or native visual verification was performed.

## Limits and recovery

Zero whole-body clearance remains unmet. Inner skin/fat cells, thickness and regional mass, conservative mechanics transfer, retained moving poses and native 16/60-second runs remain unqualified. 45 known head boundary edges remain open. Mixed-source anatomy remains a reference assembly.

Exact commands, candidates, failed trials, owner/source hashes and audits are retained in existing NPZ/JSON evidence and four manifest-bound archives, restored locally with content hashes, sizes, modes, owners and nanosecond timestamps checked. The initial 50 MiB packaging check failed for the 55,309,664-byte420-422 archive; it was independently verified under an explicit 64 MiB bound without rewriting its bytes. Previous publication 128 was independently recovered from GitHub. The shared dirty checkout and normal Git index are preserved.
