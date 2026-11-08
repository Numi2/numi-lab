# Authoritative articulated sweeps and float motion margins

The hybrid sweep fast path used stationary articulated geometry when that
collider lacked its own CCD flag, even if it paired with a CCD sphere. It also
encoded a Boolean in a motion-margin field read as float bits. The revision
uses future FK for all hybrid articulated colliders and encodes the actual
float margin in the remaining speculative fast path.

All128 paired native projection conditions pass in both modes: hybrid geometry
and margin are independent of own-CCD flag, source flags remain intact, flagged
FK is byte exact, speculative geometry changes only its margin bits and that
margin matches the known speed envelope. Repetition and the4096 contact-time
records are exact.

All six runs on four unused physical initial poses complete3840 accepted
transitions with zero rejection. Fully sampled32/64 positive arms pass the
predeclared grip/lift/release and penetration gates. Open/no-contact/zero-friction
controls fail hold as predicted. Chunk1/8 final q/v/scene is byte exact. Chunk8
provides interval endpoint samples; its full32 hold impulse samples are not
observed and are not claimed.

The full-trajectory numerical gate FAILS:32vs64 differs by118.168mm, primarily
during release/landing; holding differs by at most2.662mm. Therefore the whole
mechanical gesture is not numerically qualified. The128 attempt was rejected
before physics by the helper selector, and native admission was also capped64.
A separately sealed source integration adds128 support and requires exact
legacy32/64 replay before its actual convergence test. No policy/profile was
promoted and no tolerance/event budget was increased.
