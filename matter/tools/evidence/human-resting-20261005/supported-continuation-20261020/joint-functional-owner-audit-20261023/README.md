# Joint ownership audit

Run `python3 audit_joint_coverage.py --output <fresh-receipt.json>` on SSH Numi. The checker reads the retained native rigid payload, source joint/body manifest, source oracle and registered bone manifest. It asserts actual q/v ownership and parent/child body chains for both shoulders, elbows, knees, ankles, subtalar joints and aggregate MTP joints; it also prevents hand digit coordinates from being mislabeled as toes.

Separate toe meshes exist, but the current source has one aggregate toe body per foot and no independent toe/IP articulation. Muscle generalized forces in this receipt are static source-oracle measurements, not native motion qualification. Ligament and elbow receipts retain their original revision/scope; regional skin/fat deformation is not qualified by this checker. The native rigid file hash differs from the manifest's payload hash while their embedded upstream archive identity matches; both exact identities are recorded.
