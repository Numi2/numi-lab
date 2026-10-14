# Synthetic skin-wound traction v5: retained solver failure

This bounded native Metal study tested equal-and-opposite traction on two bite
patches of one deterministic, homogeneous, uncalibrated synthetic skin coupon.
It is a simulation stress test, not skin calibration, a suturing procedure, or
clinical evidence.

## Registered protocol

- Coupon: authored 30 x 24 x 1.5 mm block, 16 mm incision, 0.6 mm initial gap,
  24 x 8 x 2 cells, 2,304 tetrahedra, and 12 top-surface lip pairs.
- Loaded arm: 10 mN maximum per bite site, with a half-cosine ramp across all
  60 steps at 1 ms per step. Solver tolerances and validity limits were fixed.
- Control: the v4 zero-force record was checksum-verified and replayed into the
  v5 notebook. It is schedule-invariant, but it is not an independent control
  replicate.
- Evidence level: simulation only. No needle, thread, robot, puncture, tear,
  tissue healing, measured material fit, biological replicate, or clinical
  qualification was included.

## Outcome

The loaded run accepted six of 60 samples (0 through 5), then stopped on the
next step with `NM_STATUS_NONLINEAR_SOLVER_FAILURE` (code 10). The mixed FEM
volume-residual diagnostic reached its unchanged `1e-4` limit. Maximum
displacement in the last accepted state was 9.53 micrometres, minimum `J` was
0.999636, and the maximum reported solver residual was `2.97e-6`. The final
accepted force was 0.176 mN per site; the planned 10 mN was never reached.

The last accepted loaded gap was 0.594221 mm, compared with the reused v4
zero-force value of 0.600000 mm. This 5.78 micrometre partial-trajectory
difference is descriptive only: the loaded treatment did not reach its
registered endpoint, so the prediction remains inconclusive. The smoother
ramp moved the observed stop from v4 step 3 to v5 step 6; it did not produce a
complete treatment trajectory. No solver threshold was relaxed.

The six-frame MP4 displays only accepted states at one frame per second. Its
0.006 seconds of simulated time therefore plays for six seconds. It is labeled
inconclusive and explicitly does not show wound closure:

- [Watch the v5 solver-stop clip](https://numi-lab-research.vercel.app/media/synthetic-skin-traction-v5-inconclusive.mp4)
- [View the final accepted state](https://numi-lab-research.vercel.app/media/synthetic-skin-traction-v5-inconclusive-poster.png)

## Evidence identity

- Mac mini run: `/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v5-20261004/study-v5-replay`
- Notebook registration SHA-256: `2096e553b8a88392800d4d0914f6ec563ebb782345d55401689a8fef94d078c4`
- Analysis SHA-256: `aeb297224795b364aabd3f95ca7f83c31db54a5bc81b986b066b176585cce71a`
- Revision SHA-256: `00dac18be68d408bfd0a9ecc259e4002e95b8a648fb6621daaf921108c31ed8e`
- Treatment receipt SHA-256: `8692760d6cb4d071fd3d5f83ed8be7dba178528ad4e1583f98df5b44c05a7324`
- Probe source SHA-256: `a6429d2d176607a362f82fec36639f8dd1965b5c1a43dbb98deef2dff80ce840`
- Probe binary SHA-256: `4003c5371309e435827fb3f4d476cdba3989dce82715931342e01d816efafdf2`
- Accepted trajectory CSV SHA-256: `f107f5728367bfdda0db4179725ef65bc037934831e88dd85e7eccba7fa55b5a`

The next useful step is a solver-focused mixed-volume residual diagnosis under
bounded tests. Further load-schedule trials should wait until a stable native
trajectory passes the existing checks.
