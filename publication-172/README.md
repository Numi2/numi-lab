# P172 connected-surface construction and diagnostics

P172 extends the verified P171 prefix at commit `61c81e589e7062d8cfb371afb2f6d0dfdc4fcfc1` (manifest SHA-256 `f25ea591e67e1429538b96fa3e1f1a14b28e3251c3cfd81b4b3e1d61c0ce156a`). It keeps a strict 580 run-008 checkpoint, the failed/non-improving v8 run-006 attempt, a tested component-selection proposal, and source-embedding diagnostics.

## Construction evidence

- 580 run-008 passed the exact construction checks at 782 surface pairs, zero contained points and zero self pairs (185 banks, 237 anchors). Candidate SHA-256: `58488b5d329c3506134c290c6a3435518e1ad75878ac451fe2a0d3179996bc48`. The state is not a native or physical admission.
- v8 run-006 started from that 782 checkpoint. Its first fit candidate failed with 221 self pairs. The retained candidate later passed full construction checks at 784 surface pairs, zero contained points and zero self pairs, but its `construction_progress` flag was false and it did not improve 782. The coordinator stopped during finalization; the prepared disk-guard cut was not run. This is a failed/non-improving branch, not a replacement checkpoint.
- The v1 631 selector rejected certified empty-AABB rows because it required a `result` object. The v2 selector adds that bounded case; its root test receipt records 8 passing tests. The v2 component proposal passed the strict check at 772/0/0 on all 185 banks with 237 anchors (candidate SHA-256 `668ae678559450363eaaa99a1bb0d6d3323cb1fde0056104b664d85836facbdd`). It remains an unselected component proposal. Its independent v2 peer review is included.
- The registered-source-embedding output is an inert diagnostic attached to P171's 796 candidate, not the 782 candidate. It does not select or modify a candidate.

The known 45 pre-existing head boundary edges and two loops remain; P172 does not claim whole-skin closure. All included outcomes are construction-only: `native_admission` is false and `accepted_physical_steps` is zero. No physical, anatomical, or native qualification is claimed.

The temporary-object-cache recovery record covers 18 obsolete cache files. The archived payload was restored on the host before those temporary files were retired; the record says the archive had not yet been published remotely at that time. P171's retirement proof records six immutable remote recoveries, four retired plaintext reports, and two retained reports awaiting quantization preparation.

One 49 MB v8 fit report is represented by a lossless zstd archive and a fresh restore proof. The plaintext report was retained while preparing P172. Use the helper and receipt in `evidence/compression-preservation/619-v8-run-006/` to restore it.

`publication-manifest.json` lists each package member except itself with its exact path, size, mode and SHA-256.
