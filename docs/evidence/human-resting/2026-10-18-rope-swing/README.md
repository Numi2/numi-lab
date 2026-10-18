# Native rope swing: eight-angle video

Executed on SSH Numi / Apple M6, 2026-10-18. Successful ten-second exploratory demonstration, not full Human or longer-motion qualification.

The existing articulated Metal contact solver supplies all rope impulses. A prescribed GPU trajectory moves the upper anchor by up to 0.30 m laterally and 0.30 m in depth, with a four-second period and three-second smooth ramp. The ideal rope is a massless unilateral 0.8 m length limit attached to the existing torso site. It has no distributed bending, rope collision or resolved harness contact. The positional moving-plane drive does not explicitly supply support velocity. There are no added body forces or root assistance.

All eight camera azimuths view the same continuing simulation. Each view lasts 1.25 simulated seconds. The original native movie is unretimed, 50.1067 seconds; the convenience preview is uniformly accelerated to 24 seconds. It is not a real-time performance claim. The actual physical run completed 10.000000475 simulated seconds in 48.289181875 wall seconds: 0.20708573x.

Four accepted geometry captures verify visual anchor and torso-site correspondence below 0.14 micrometres. The sampled chord length is 0.7998316–0.8004404 m. All presented frames have zero collapsed/nonfinite triangles. Other support contact normal impulses are zero. Current skin025, wrist source registration, back repair, mass, joint model, and physiological owners are retained. Wrists move through their existing physical DOFs; digits remain rigid reference anatomy.

Earlier attempts are retained. The longer run still rejects a muscle158 collapsed triangle at 11.710 simulated seconds. The shorter demo passes; it does not resolve that longer-motion defect or qualify swing-pose anatomical intersections.

The scene-specific extension is compiled from the frozen ABI41 source. The shared dirty application changes were preserved; this evidence patch is against the frozen source, not a patch for current main. The two source conditioners use existing Human closed-surface precision and receipt owners: seven muscle158 sub-quarter-micrometre edges and four bone49 sub-half-micrometre edges, with paired intercostal conditioning and regenerated tendon identity. Local closure/volume checks pass; physical vertex/binding state and tendon endpoint bytes remain unchanged. These are passive rendering precision repairs, not new dynamics.

Exact replay on SSH Numi, choosing a fresh output directory:

    /Users/n/numi-human-prep-venv-20261005/bin/python3 /Users/n/numi-lab-native-replay-publish-20261017/tools/launch_human_rope.py /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/invocation-009.json --output /Users/n/rope-swing-next-run --exclusive-wrapper /Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py

See native-invocation.json for hashes/configuration, swing-verification.json for numerical checks, attempts.json for failures, source/ for preparation and media-export instruments, and retained-artifacts.json for movie/binary identities. Large native assets and movies remain on SSH Numi; current evidence has not been retired or claimed archived remotely.
