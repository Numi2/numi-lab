# Initial release initialization audit

This is a read-only source/evidence diagnosis for the supported-pose release in native trial 025. It does not change the physical model or qualify the short run as sustained support.

## Run and source binding

The run is `build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-contact-precision-025`. `provenance.json` binds host `numi.local`, base revision `7fcc98136014d001164dfbc839f5015ff6d5e29d`, executable SHA-256 `83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e`, metallib SHA-256 `582a030a67312ee00f3358e4a48e8c7741e7d2b67584fe5d496f58f578532535`, suite-library SHA-256 `7255db47f2c88261f855abcd6de4abcb2546baeea9467e5cbf645f412f1eee49`, and probe source SHA-256 `a9bc047d16d53843b6473fc28f0e7580dc04ee3e8dbd3df54dfdb89f03d6ed84`.

The retained stdout reports `resting_initialization=dynamic_release`, activation cap 0.01, normalized residual RMS 123.280248507, maximum generalized acceleration residual 531.331390008, `balanced=false`, and `qualification=initialization_transient_not_static_equilibrium`. The same run preserved the authored pose to within 1.86264514923e-9 in q; the two explicit shoulder q overrides are 36/74 = 0.10. These are a bounded candidate initialization and a transient release, not a static-equilibrium or clinical claim.

## Owning path

`apps/numilab_human_myosim_visual_probe.mm:5147–5317` owns `compileStaticStandActivation` and the dynamic-release output/gate. `:5281–5288` accepts the dynamic candidate when its residual is no worse than the initial candidate and skips the ordinary acceleration threshold; `:5310–5317` prints the explicit unbalanced qualification.

For `preserveRestingInitialPosture`, `:6280–6285` keeps the authored q instead of ground-aligning it. The compile call at `:6379–6387` passes `nullptr` for static support contacts in that mode, and `:6395–6402` requires the resulting support-force vector to be zero. Thus this offline recruitment solve includes no bed reaction. The later runtime bed contact remains the physical contact owner. Do not synthesize a support wrench at a positive gap: the related retained 022 accepted-trace summary gives a minimum contact gap of +15.2555903696 micrometers, and the 025 setup intentionally starts separated to avoid a roundoff impulse.

`src/core/NumiHumanMuscleEquilibrium.cpp:1187–1338` resolves muscle paths, gravity target, supplied passive coordinate couplings, muscle forces, and the acceleration residual for recruitment. That compile occurs before the runtime whole-body FEM skin/fat attachments, fascia, and registered ligament owners are bound; those later force owners therefore do not explain the reported compile-time residual directly.

## Muscle/tendon initialization

The NHMYO2 compliant fiber/tendon zero-state path in `src/core/MujocoMuscleReference.cpp:742–805` solves a stationary fiber/tendon balance before the ordinary update. Its code comment explicitly says the zero-state sentinel does not impose a hidden 2% tendon prestretch. This rules out that specific artificial initialization strain. It does not establish that the payload tendon slack lengths match this registered pose; no per-muscle path, slack-length, passive-force, or active-force breakdown is retained, so any such contribution remains unquantified.

## Next diagnostic

Before choosing an initialization correction, expose the maximum-residual DOF and its exact row decomposition at this same authored q: gravity target, active muscle force, zero-activation passive muscle/tendon force, passive coordinate force, equality/limit reactions, and residual/acceleration. The compiled result already carries aggregate generalized muscle, passive-coordinate, gravity, and residual vectors plus per-muscle total/passive tendon forces. This report can identify an owner mismatch without changing activation limits, adding assistance, moving the pose, or assigning a bed reaction while the physical contact is positively separated.
