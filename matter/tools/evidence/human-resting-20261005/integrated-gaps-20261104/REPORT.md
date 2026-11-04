# Integrated Human solver and memory work, 2026-11-04

## Result and scope

The complete current Human scene still falls far short of 0.1x.
The verified default change removes 426,089,952 bytes (406.35 MiB) of
persistent duplicate immutable tetrahedron storage. It does not accelerate
the measured body run materially. The 32 accepted roots cover only
0.000500000023749 simulated seconds of a release transient, not supported
rest, a complete breath, a heartbeat, or sustained qualification.

Trial 238: body 228.965443333 s, 0.000002183735748x. The comparable 234
body time is 229.075407625 s; this is noise-level change, not a claimed
speedup. Initialization-inclusive 238 elapsed time is approximately 416 s.
Presentation qualification is pending. The scene is balanced=false; this
is not evidence of an equilibrated supported checkpoint.

## Implemented changes

* Proven-empty FEM transport assembly: replace four whole-mesh inactive
  field passes with one explicit publication of unchanged transport
  solution and zero residual scratch. Admission uses the existing proof
  that no FEM transport objects or field boundaries are active. Mechanical
  pressure and vascular/physiological equations remain active.
* Owner-incidence certification, opt-in
  NM_MIXED_CERTIFICATE_OWNER_INCIDENCE=1: reuse freshly generated
  deterministic node/contact incidence rather than scan every contact
  independently for every continuum owner. Validate encoded slots/corners,
  select the first supporting corner per owner, and retain failure
  witnesses. This path assumes the existing generator's support/owner
  invariants and complete fresh incidence; it is not a new collision
  detector or a reusable stale contact cache. Default remains off pending
  broader layout qualification.
* Bounded column telemetry, opt-in NM_FGMRES_COLUMN_TRACE=1 with the
  existing selected Newton trace root: capture active column residual
  history on the same GPU timeline, then read once at completion. No
  per-column host synchronization or encoder-splitting timing. Distinguish
  scaled Arnoldi estimates from fresh physical residuals.
* Direct sparse block assembly, experimental
  NM_FEM_ASSEMBLED_XYZ_DIRECT_EDGE=1: accumulate the existing deterministic
  contribution order directly into block CSR, eliminating the 144-float
  per-element intermediate. Preserve tangent, product/sum validity and
  physical certificates. Incompatible with graph factor reuse requiring
  that intermediate. Default remains off.
* Immutable tetrahedron sharing, default on: accepted, candidate and
  checkpoint views alias immutable reference/connectivity/material
  records when no mutation, puncture, cohesive or mutable-topology path
  is admitted. Dynamic positions, pressure and material history remain
  independently owned. Restore still checks exact immutable layout;
  borrowed mutation commands reject before encoding.
  NM_FEM_SHARE_IMMUTABLE_TETRAHEDRA=0 retains the comparison path.

## Native results

| Trial | Result | Actual work | Interpretation |
|---|---|---|---|
| 232 | 32 accepted | 144 solves / 2,304 columns / 62 budget exhaustions | Empty-field + owner-incidence paths; no whole-run gain |
| 233 | 0 accepted | capacity rejection | 2M candidate capacity was insufficient; 4M retained |
| 234 | 32 accepted | 144 / 2,304 / 62 | Late-root column/alpha diagnostic |
| 235 | 25 accepted, root 25 rejects | 85 / 1,360 / 32 | Existing materialized sparse assembly |
| 236 | Same root 25 rejection | 85 / 1,360 / 32 | Direct CSR exactly matches 235 compact outputs/failure; removes 852,179,904 bytes |
| 237 | 31 accepted, step 32 contact rejection | 172 / 2,752 / 79 | Contact proof-composition experiment failed; production change removed |
| 238 | 32 accepted | 144 / 2,304 / 62 | Immutable element sharing; all ten compact CSV traces byte-identical to 234 |
| 239 | 0 accepted, vascular line-search rejection | 1 solve / 7 columns / 0 budget exhaustions | Experimental thin-cell/regional composite is not enabled in this source increment |

All ten compact traces of 232 and 234 also match exactly. The parallel
owner certificate changes 12 floating reduction outputs by at most
1.119e-7 relative; it retains the same accepted trajectory. Direct CSR
has only been qualified against its equally rejecting reference through
root 25, not promoted as the Human production setting.

After Krylov allocation, Metal reports 5,935,087,616 bytes in 234 and
5,082,890,240 in 238. This phase difference includes upload/transient
lifetimes; only 426,089,952 bytes are identified as the persistent sharing
saving. Do not add these two figures as separate gains.

## Diagnosis and next work

Root 28 uses sixteen active Newton solves, each with sixteen columns.
Its later scaled Arnoldi residual falls only to approximately 28.8% of
the initial norm; several accepted line-search fractions are about
0.000346. Inner solve cost, conditioning and globalization remain coupled.
No early-stopping omission is claimed; completed-work suppression exists.

The new current-operator thin-cell/regional trial covers 30,768 nodes in
8,290 local patches (11,335 selected cells). It reaches the linear target
in seven columns, then rejects the shared vascular line search. Its
failure is retained before any promotion. Stronger preconditioning must
be tested against physical nonlinear progress, not column count alone.

Remaining: supported equilibrium initialization, efficient valid local
and interface solves, conditioning of retained thin cells, compact
material operator storage, fresh contact certification attribution,
inactive Krylov backing layout and sustained coupled qualification.
Do not retry the retired larger-timestep path or the failed aggressive
66x coordinate reduction as accepted solutions. No 0.1x result, full
anatomy motion qualification, or clinical validation is claimed.

## Validation and reproducibility

Native device: SSH Numi, Apple M6, Mac18,5, 16 GB, macOS 27.0 (26A425).
Builds use the working Metal.xctoolchain bin directory on PATH, without
requiring a full Xcode install. Source execution worktree:
    /Users/numi/Work/skin-fat-native-integration-209
baseline f75955f08d6edf0f85027eac7614017ac9fbc996, with exact dirty
execution patches retained per trial. Publication intentionally preserves
unrelated changes already present on main.

Passing relevant native checks:
* deterministic contact incidence and owner enumeration: 7,911 checks;
* FGMRES true residual, scaling, empty fields, telemetry, assembled
  products and ICB paths: 2,665,138 checks after removal of trial 237;
* direct sparse assembly component: 5,123 checks;
* immutable one/two-environment storage, restore/tamper and borrowed
  mutation admission: 362 checks;
* complete material/frame/accepted-state suite, including late rejection,
  rollback, replay, reset and immutable restore: PASS.

Each run directory retains the launch argv/environment, source and
binary/library/input hashes, exact patch, initialization/body timing,
compact traces, resource guard and execution result. Replay command:
    python3 run_native_plan.py install-motion
Review the frozen paths and hashes and use a fresh output directory.
The driver pipeline archive is a mutable cache; its initial hash belongs
to that exact run, not later reuse. Large immutable mass preparation
dumps are hard-linked, not regenerated for each trial. Current
source-carrier mass inputs are protected after verified restoration from
the archived cold evidence; see the recovery receipts. Failed trials
remain evidence, not defaults.
