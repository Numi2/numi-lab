# Frozen capitate attachment audit and inferred retargeting proposals

This evidence has two deliberately separate outputs:

- `source-slot-provenance-audit.json` rejects identity-preserving transfer of all six frozen sites. The old `source_vertex` integers were reused as emitted slots for new vertices. Five now trace to opposite-side replacement ancestry; the sixth traces to a generated right-hand vertex rather than its old source identity. The original target, stable ID, donor, stiffness, area, and rejection reasons remain recorded.
- `inferred-capitate-retargeting-proposals.json` proposes one separately named new inferred retarget for site 9. It binds a barycentric point on an exact, non-inferred C014 D face with same-side 3rd/4th metacarpal owners and positive capitate contribution. It preserves donor body 49, original kA and tributary area, and declares the changed zero-strain target. It does not claim to recover the frozen source identity or to qualify the attachment.

The five other sites have no proposal under the declared local screen. The nearest exact same-side hand surface is either too far from the old target or has no capitate contribution nearby. In particular, site 2 has a very close same-side fifth-metacarpal point, but the triangle has zero capitate weight; the nearest palm triangle with any capitate contribution is about 33 mm away. These sites remain unresolved.

Run both generators on Numi with:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python \
  /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/capitate-attachment-correspondence-001/source-slot-provenance-002/audit_capitate_source_slot_provenance.py
/Users/n/numi-human-prep-venv-20261005/bin/python \
  /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/capitate-attachment-correspondence-001/source-slot-provenance-002/generate_inferred_capitate_retargeting_proposals.py
```

The donor frame audit found no matrix or translation mismatch. The previous `wrist_carpals`-only nearest search selected distant faces because its region labels/filter were stale; the new proposals classify actual triangle corners from named source bone weights and retain exact face ancestry. The locality and normal checks here are proposal-selection screens, not physical admission gates.

No application header or runtime file is changed. The output requires compiler-owner review and downstream surface/force tests before any integration.
