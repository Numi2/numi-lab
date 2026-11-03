# Compensated reference surface construction

The production surface builder dropped finite positive reference-mode triangles below a
fixed1e-12 area floor and could collapse them after rounded world-position projection.
It also bounded rounded world endpoints instead of the represented compensated sweep.

The fix measures reference triangles using local compensated differences and retains
every finite positive area. It encloses represented current and previous positions plus
the optional private base with outward bounds and outward thickness padding. Authored
exclusions, topology identities, force laws, mass, physical collision thresholds and
accepted-state gates remain unchanged. Nonreference area admission retains the legacy path.
More valid faces can produce more contact work; this is a geometry correctness repair,
not a claimed performance optimization.

The actual GPU builder regression failed with the original shader, then passed16cases
and16canaries: translated world-coordinate collapse, positive subfloor triangles,
nonfinite/degenerate inputs, private-base bounds, exclusions and legacy behavior.
The complete native FGMRES suite passed811433checks on SSH Numi, AppleM6/16GiB.

Integrated021 accepts6steps (93.75microseconds), passing018's root5 rejection.
It then failsroot6 at vascularrow0:1.177861395e-5 above unchanged1e-5.
The last workspace has371882surface primitives (about275more),659430contacts and
3637613candidates; capacity is not exceeded. Sources/binaries stayed fixed during execution.
This is still an unbalanced release stress test, not supported rest or sustained motion,
and is far below0.1x. No anatomical/mass qualification is inferred from this short run.

A diagnostic-only selector allows a bounded contact trace threshold in(0,1] and honors
an explicit root even when automatic failure capture is enabled. It never enters a
physical kernel or changes alpha. Trial022 uses it to capture the new limiting contact.
No images or visual verification were used.
