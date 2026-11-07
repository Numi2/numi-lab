# Result: proposed mixed missing-frame gate failed

The pinned generic process ran 2026-11-07 23:51:17.647214–23:51:17.867611 UTC and exited 8, with matching preflight/postflight input hashes. Its original synthetic baseline and the three sensor-invalid/no-noise-flicker cases passed.

With the proposed conservative rule and edge flicker forced to one, the measured pixel was 0.0816327; the adjacent no-FRAME pixel was also filled to 0.0816327 instead of the required far value 1.0. The frame-present edge receiver was filled to 0.0816327 as proposed, and the geometry-only pixel without a depth-valid neighbor stayed at 1.0. A fully absent frame stayed far everywhere. Thus this first shader candidate does not satisfy the proposed mixed-frame contract.

The source documentation does not define this edge condition, so the test evaluates an explicit engineering contract rather than an existing promised behavior. It is a materialized-plane synthetic test, not proof of a live renderer condition. The direct-winner shortcut has no independent per-pixel frame signal in this fixture; no missing-frame conclusion applies to that path. Product source and observation fingerprints remain unchanged.

