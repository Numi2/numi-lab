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

At containment commit 977eb61, the complete MetalWorld regression covered six fixtures, all three search
selectors, and caps16/128 (36 runs). It requires zero accepted unresponded
swept hits, rejection of every unresolved cast, exact repeated q/v, exact
restoration on rejection, and acceptance of the legacy translating impact,
two clear rotating controls and ordinary clear translation. The rotating clear
controls have centre height0.23m; they are not stringent grazing tests.
The new `metalrobo.ccd.acceptance` CTest passed on ssh numi, Apple M6.
[Evidence](evidence/ccd-acceptance-containment-20261107/analysis.json).

That commit is containment, not completion of useful impact resolution. Its rotating
impact fixtures still reject. A subsequent geometric repair must resolve real
impacts and admit certified grazing misses at the production cap16; simply
rejecting uncertain motion cannot satisfy that requirement. No performance
optimization, default CCD-mode change or contact-law change is included here.

## Subsequent geometric and contact-frame repair

The follow-up keeps acceptance independent of both experimental selectors.
It adds a supported, whole-interval certificate for a centred single floating
box against a static plane. It also fixes hybrid manifold compilation: geometry,
local anchors and contact Jacobians now use the same selected impact pose.
Previously the anchors were transformed using the pre-impact body pose.
A retained event trace showed a positive pre-solve normal velocity where the
rotating lower edge was approaching the plane.

The certificate multiplies each vertex's plane projection by the positive
quaternion normalization denominator. The resulting numerator and denominator
are quadratic polynomials in normalized-linear quaternion interpolation.
Outward Bernstein subdivision encloses the entire interval.
The monotone atan2 map converts that curve parameter to the floating integrator's
constant-angular-speed endpoint curve; translation is enclosed over the whole
time interval. The normalization denominator is bounded below throughout.

For the single floating body's discrete partial-step path
x(t)=x0+t(v0+t*a), R(t)=Exp(t*(omega0+t*b))*R0,
the endpoint curve also carries an envelope
H^2/4 * (|a| + radius*|b|), since t*(H-t)<=H^2/4 and
the rotation exponential's differential has norm at most one. Projection-owned
metadata restricts this path to a centred shape, one floating body without
joints, a finite acceleration envelope, and a segment below a half-turn.
A static-plane ownership marker prevents treating an unprojected moving plane
as a fixed support. Other shapes retain the existing search and strict
unresolved rejection; this certificate does not claim arbitrary articulated paths.

A resolved event needs a negative guarded-clearance witness, a fully certified clear prefix,
and a safe left endpoint inside the existing contact activation band.
Separated surfaces must make positive certified progress. Exhausted, uncertain
or unrepresentable intervals reject; a speculative envelope cannot override
that result. Each vertex uses the configured cap16 (cap128 is diagnostic);
no cap, event budget or velocity-iteration budget was raised.

### Native result and limits

Apple M6, ssh numi: 288 complete-step observations, including repeated runs,
cover 16 fixtures, scales1/0.5, caps1/16/128 and all three search selectors.
At cap16, the legacy selector now completes the original rotating impacts:

| Angular speed | Accepted normal impulse |
| --- | --- |
| 250 rad/s | 1.169145465 N s |
| 177 rad/s | 0.705331564 N s |
| Unused 211 rad/s | 0.877037704 N s |
| Unused 293 rad/s | 1.316815019 N s |

Rotating near-misses, thinner near-misses, far controls and ordinary clear
translation accept without impulse at both scales under cap16. The forced
cap1 exhaustion controls reject and restore. Every rejected trial restores
q/v exactly and retains an empty initial manifold cache; repeated q/v/status
are identical. No accepted unresponded swept crossing remains in this matrix.

The independent FP64 observer checks every literal accepted segment, with
100001 endpoint-geodesic samples, a whole-interval Lipschitz enclosure, the
discrete acceleration envelope and a 0.2 micrometre rounding allowance.
All 174 accepted observations pass; the smallest enclosed gap is 11.095 micrometres.
This is geometric qualification of these supported fixtures, not force/torque
accuracy or arbitrary engine trajectories.

**The broader all-impact completion gate failed.** Both 0.4 mm thick rotating
impact stresses still reject at caps16 and128 (typed unresolved/event-budget
failure), with restoration. They are retained as failures, not counted as
successfully resolved contact. The acceptance CTest separately requires
the original and unused-rotation impacts, all clear controls, and safe handling
of those remaining stresses; its report explicitly exposes the failed
all-impact gate. CTest1/1 passed. Neither experimental default was enabled.
There is no performance claim.

[Final observations](evidence/ccd-acceptance-resolution-20261107/world-steps.json),
[acceptance analysis](evidence/ccd-acceptance-resolution-20261107/analysis.json),
[independent geometry](evidence/ccd-acceptance-resolution-20261107/geometry.json).
The retained archive includes failed certificates, the invalid old-binary
attempt, zero-time-event failures, source patches and build logs. An earlier
anchor-frame result that appeared to complete both impacts preceded correction
of a stack-exhaustion bookkeeping defect; it is superseded by the whole-interval
checks above and is not qualification evidence.
