# Rejected tiled tangent-cache layout (59)

The complete 32-root Human comparison preserved all 1,798,453 endpoint records
and numerical traces exactly, with the same 144 active solves / 2,304 columns /
62 exhausted budgets. It was slower in this matched pair:

| Configuration | Body wall time | Simulated time | Mechanics/body factor |
| --- | ---: | ---: | ---: |
| Production control 304b | 231.374 s | 0.5 ms | 0.000002161x |
| Tile32 candidate 305 | 255.355 s | 0.5 ms | 0.000001958x |

This is 10.36% more body time in one pair, not an estimate of long-run variance.
The candidate is retired from active code. Its exact scoped patch is retained;
there is no new production cache-layout flag or persistent allocation.

The candidate transposed 81-float element slots into 32-element tiles, with
checked partial-tile padding and unchanged arithmetic order. Native test
coupons used two environments, 74 elements each, selectors 2 and 3, and three
steps per selector. All 30 complete state-array comparisons were byte-exact.
The full kernel contract passed 2,743,467 checks after standalone test factories
were made to resolve the new optional function constant. Earlier factory
SIGABRT failures are retained; they were pipeline-construction failures, not
accepted physics results. The initial regional fixture also failed its existing
one-object admission guard and was corrected without weakening the guard.
A generic-VM cold-compilation attempt was stopped; the successful coupon used
the existing proven stateless specialization.

Both native runs used the same geometry, mass, contact, timestep, physiology
and solver settings, with predictor disabled. All execution was on SSH Numi's
physical Apple M6. This release prefix is not supported rest, native presentation,
sustained physiology or 0.1x qualification.

A tile layout alone does not improve the current mixture of element and
node-incidence cache consumers. The measured failed attempt is retained to avoid
repeating it or carrying its implementation bloat into production.
