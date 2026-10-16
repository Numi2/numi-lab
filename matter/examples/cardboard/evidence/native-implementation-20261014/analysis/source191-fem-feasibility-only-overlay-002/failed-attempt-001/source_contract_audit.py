#!/usr/bin/env python3
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
runtime=(root/"matter/src/runtime.mm").read_text()
contact=(root/"matter/src/metal/contact.metalinc").read_text()
policy=(root/"matter/include/numi/matter/deformable_contact_trial_policy.h").read_text()
checks={
 "host_flag_is_existing_full_contact_admission": "femFullContactResidualArmijoEnabled ? 1u : 0u" in runtime,
 "host_binds_new_buffer20": "[encoder setBytes:&fullResidualContactFeasibilityOnly" in runtime and "atIndex:20u" in runtime,
 "kernel_reads_buffer20": "constant uint& fullResidualArmijoEnabled [[buffer(20)]]" in contact,
 "FEM_and_MPM_support_bits_are_recorded_after_validation": "addSupport(\n                    supportMask, 2u)" in contact and "addSupport(\n                supportMask, 1u)" in contact,
 "exact_FEM_only_mask_fails_closed": "supportMask == femBit" in policy,
 "full_sweep_requires_exact_requested_fraction": "trialTestedFraction != 1.0f" in contact and "!trialCertified" in contact,
 "finite_barrier_still_required": "if (finiteTrial &&\n                        (femFeasibilityOnly || trialEnergy <= armijoRight))" in contact,
 "global_private_trial_path_remains": "if (femFullContactResidualArmijoEnabled)" in runtime and "encodeArmijoDeformableContact(" in runtime,
 "global_residual_measure_remains": "nm_fem_armijo_measure_trial" in runtime,
 "final_mixed_certificate_remains": "nm_mixed_certify" in runtime,
}
out={"schema":"numi.cardboard.source191-002.source-contract-audit.v1","checks":checks,"passed":all(checks.values())}
print(json.dumps(out,indent=2,sort_keys=True))
if not out["passed"]: sys.exit(1)
