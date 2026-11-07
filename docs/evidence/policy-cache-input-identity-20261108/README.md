# Policy evaluation cache input identity

The pre-fix CPU reproduction in `before.txt`, against parent revision
`ef33187235bdbb6958fe4ab12ee5f5194c2251d2`, replaced WorldPack bytes at the
same path and observed the same evaluation contract. The post-fix check in
`actual-world-copy-after.txt` used a copy of the retained authored synthetic
Franka WorldPack, changed its bytes at the same path, and then deleted it.
The contract changed and the missing input was rejected. This is an identity
check, not a physics evaluation.

`cpu-tests.txt` records the complete selector unit suite. Its real `_evaluate`
cache regression confirms a repeated unchanged direct input reuses evidence,
replacement invokes the evaluator again, and a missing input is rejected.
Other tests check each supported direct file option, invalidate the old
contract schema, and verify that a missing external URDF mesh reaches and
fails the evaluator instead of reusing evidence. Unresolved multi-file paths
and implicit Numifly inputs always bypass persistent cache reuse; this does
not establish or hash their full dependency closure. Manifest arguments are
also preserved when projecting trainer arguments to evaluation.

The selector checks top-level bytes before and after evaluation. It does not
establish atomicity for dependencies that can change during evaluation, nor
does this repair add dependent shared-library or environment identity. These
CPU tests do not qualify a native rollout, learned policy, or Window UI.
