# Legacy CCD acceptance repair, 7 November 2026

The first containment change makes an unresolved swept query reject the step
regardless of advancement-method flags or speculative-envelope coverage.
Both the rigid and procedural-rod owning acceptance kernels use this rule.
The envelope marker remains diagnostic; it is no longer a clearance certificate.
Both directional experimental options remain disabled by default.

The [historical results](CCD_INTERVAL_LOCAL_EXPERIMENT_20261107.md) and their
raw evidence remain unchanged. In that hybrid configuration the legacy path
accepted the 250 and177 rad/s rotating plate sweeps with no normal impulse,
despite an analytic minimum plane gap of approximately −210 micrometres.
This containment rejects those casts with complete q/v restoration at cap16.

The complete MetalWorld regression now covers six fixtures, all three search
selectors, and caps16/128 (36 runs). It requires zero accepted unresponded
swept hits, rejection of every unresolved cast, exact repeated q/v, exact
restoration on rejection, and acceptance of the legacy translating impact,
two clear rotating controls and ordinary clear translation. The rotating clear
controls have centre height0.23m; they are not stringent grazing tests.
The new `metalrobo.ccd.acceptance` CTest passed on ssh numi, Apple M6.
[Evidence](evidence/ccd-acceptance-containment-20261107/analysis.json).

This is containment, not completion of useful impact resolution. The rotating
impact fixtures still reject. A subsequent geometric repair must resolve real
impacts and admit certified grazing misses at the production cap16; simply
rejecting uncertain motion cannot satisfy that requirement. No performance
optimization, default CCD-mode change or contact-law change is included here.
