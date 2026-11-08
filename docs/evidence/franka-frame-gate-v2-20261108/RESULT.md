# Corrected Franka frame-admission result

The preregistered fixture completed with exit 0 on 2026-11-08 from 00:05:31.918576 to 00:05:31.988788 UTC. Preflight and postflight verified the exact executable, owner dylib, shader, asset and author hashes declared in the plan.

The isolated owner selected direct-winner rendering. The PRESENT/PRESENT step completed with both world statuses zero, and the inspected winner plane had zero nonempty winners. A subsequent fixture-supplied MISSING/PRESENT status was rejected before the observation hook admitted work; the captured private visual history and sequence stayed unchanged at 1. A following PRESENT/PRESENT step completed with both world statuses zero and a second admitted/finished hook. Exact output and compressed snapshots accompany this report.

Frame statuses were supplied by the fixture, not by a capture-completion owner. This is a whole-batch admission test on an isolated research owner, not a production implementation. It does not establish independent per-environment retry, mixed acceptance, physical rollback, capture command failure handling, all actor state restoration, policy fingerprint migration, or production missing-frame behavior. The earlier wrong-binding asset failure and first authoring failure remain retained. The camera deliberately produced an empty scene; empty pixels are not used to infer capture status.
