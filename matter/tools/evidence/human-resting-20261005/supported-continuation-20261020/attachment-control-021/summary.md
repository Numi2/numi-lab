# Attachment identity control 021

Outcome: failed after 18 accepted coupled rows out of 32 requested at 125 microseconds (0.00225000010687 s accepted of the requested 4 ms mechanics-only horizon). This is a completed failure control, not a pass or sustained-contact result. Process wall time was 328.822 s; it is not a real-time throughput claim.

Contact over the accepted trace: summed and maximum normal impulse were 0 N s and 0 N s; maximum peak penetration was 0 m; minimum recorded contact gap was 1.52555903696e-05 m. sustained_contact=false because recorded normal impulse is zero. Root assistance maxima were 0 N and 0 Nm.

Mass at the final accepted row: reference positive-body 82.7159423828 kg, candidate 82.7159423828 kg, external 0 kg, body-mass residual 0 kg. Maximum absolute mass residual was 0 kg. Transfer momentum and angular-momentum residual maxima were 0 kg m/s and 0 kg m2/s.

The first rejected segment began after accepted step 18. Matter status 5 (invalid deformation) was reported for passive-tissue object 15, global tet 21728; stderr identifies the BodyParts3D whole-body skin-fat bonded shell, local tet 17981. Rejected-trial J=0.34004101157188416 and accepted effective J=0.36239880769482324. The accepted coordinates are the rolled-back effective state, not the rejected candidate.

The frozen attachment selection reports 20 rows and includes source vertex 8467 / donor body 139. The transfer diagnostic reports active_detail_vertices=54551 and unreferenced_detail_vertices=7399; these are raw transfer-accounting fields, not a claim that 7399 anatomical skin vertices are missing. maximum projection was 0.0425848176 m at source vertex 14622, donor body 128, coarse representative 859, face 1690. All 20 site identities and this max-projection witness are in selected_stdout.log.

## Run binding

- Repository HEAD recorded at packaging: c927871bafc1194f8ef314fc0687e4d123d11dd8.
- Invocation source digest: c7b626d0d152ddff0bb7b574aae393b6bc01f25c466416602e88c1dcf3148c37; executable digest: af112c6cf363382b21beafb7d28ccdeb592b4c69811dac1aed02416bc22f09a8; provenance executable digest: af112c6cf363382b21beafb7d28ccdeb592b4c69811dac1aed02416bc22f09a8.
- Per-file source digests and exact input-asset hashes are preserved in provenance.json and input-assets-sha256.txt. This is the source/binary control captured before the subsequent elastic edits; these recorded hashes bind it to that earlier build. The current working tree has since changed and is not represented by this control.
This is the historical 021 hard-20 attachment control. The captured invocation predates the later elastic edits and does not contain NUMI_HUMAN_SKIN_HARD_ATTACHMENT_REFERENCE=1; reproducing this hard control with the later executable requires setting that environment variable.\n\n- The executable path has since been reused by a later build: archived run/provenance SHA-256=af112c6cf363382b21beafb7d28ccdeb592b4c69811dac1aed02416bc22f09a8, current path SHA-256=83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e. The archived invocation and contemporaneous provenance agree; the current executable is not evidence for the 021 run. No executable, metallib, input payload, or 283 MB MRV pack was copied into this package.
- Invocation SHA-256: f0f87ea4f8c9451d6e868986dc9aff07f56625a634b24882bbf06101892906a6; provenance SHA-256: f6b718dcb988a515657facd5b7260193236379e8cb400e094d428b9593232ff7; accepted CSV SHA-256: a922292d444b4b66ab71fb1e0bbef0a4bc16a74fee08e7d42f458690f22bfc5a; failure receipt SHA-256: c0be71edf6e3dd9c418afaa1fd7773ef19c8825679c005e5d34668a0ac173319.
