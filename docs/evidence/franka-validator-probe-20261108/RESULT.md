# Read-only staged validation result

The diagnostic process ran from 00:40:49.740834 to 00:40:50.184630 UTC on 2026-11-08 and exited 2. Pinned inputs matched before and after. It reproduced the pre-submit rejection without publication and the subsequent unpublished status-17 result. The const callback retained the raw staged records before validation.

Both world and contact statuses had code 12, MR_STEP_UNSUPPORTED, no successful physics substeps and zero required raw contacts. Both transitions had policy revision 0, while the installed policy revision required by the validator was 1; this mismatch is sufficient to trigger the reported malformed-transition validation failure. The captured actor, critic, latent, log-probability and value arrays contained no nonfinite values. Raw status and transition records are preserved alongside the exact outputs.

The two unsupported physics statuses are independently unresolved. These data do not establish a contact-capacity overflow or its cause. The callback did not change validation or publication, no mixed arm was run, and no sensor transaction or GPU-failure restoration is established.
