# Native stage precision candidate: unqualified

This isolated candidate is disabled by default. The extra stage low word is
private Newton scratch, reset at each working solve. In the narrowly admitted
midpoint fixture it is used in the applied correction, determinant line-search
trial geometry, stage momentum residual, endpoint velocity reflection and
full-step certificate. Unsupported diagnostics, deformable contacts, parallel
backtracking and residual Armijo paths are rejected. The existing endpoint low
word remains transactional accepted state. This is an implementation snapshot,
not a passing test or a performance result; force-bearing rollout, accurate
independent comparison, replay and rejection qualification are pending.
