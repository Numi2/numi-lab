#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "source/matter/src/human_respiration.metal"
baseline = Path("/Users/n/numi-human-separation-20261018/C/blood-transfer-208/shader-preflight/HumanRespiration208.metal")
output = root / "native-sustained-capacity-001/HumanRespirationSustained.metal"
receipt_path = output.with_name("shader-flatten-receipt.json")
source_text = source.read_text()
base_text = baseline.read_text()
source_start = source_text.index("inline float3x3 nmHumanBloodMatrixFromRows(")
source_end = source_text.index("kernel void nm_human_respiration_predict(", source_start)
flat_start = base_text.index("kernel void nm_human_blood_mpm_handoff(")
flat_end = base_text.index("kernel void nm_human_respiration_predict(", flat_start)
updated = base_text[:flat_start] + source_text[source_start:source_end] + base_text[flat_end:]
dispatch_old = "    nm_u32 particleCount;\n    nm_u32 materialStateStride;\n} NMHumanRespirationDispatch;"
dispatch_new = "    nm_u32 particleCount;\n    nm_u32 materialStateStride;\n    // Source-verified SI bulk coefficient for the high-shear blood proxy.\n    // This occupies the prior 16-byte ABI tail padding.\n    float bloodMpmBulkModulus;\n} NMHumanRespirationDispatch;"
if updated.count(dispatch_old) != 1:
    raise RuntimeError("flattened respiration dispatch ABI anchor is not unique")
updated = updated.replace(dispatch_old, dispatch_new, 1)
output.write_text(updated)
sha = lambda data: hashlib.sha256(data).hexdigest()
source_bytes = source.read_bytes()
flat_bytes = output.read_bytes()
receipt = {
    "baseline_flat_source": str(baseline),
    "baseline_flat_source_sha256": sha(baseline.read_bytes()),
    "flattened_sha256": sha(flat_bytes),
    "flattened_source": str(output),
    "hand_off_msl_source": str(source),
    "hand_off_msl_source_sha256": sha(source_bytes),
    "inserted_helpers": [
        "nmHumanBloodMatrixFromRows", "nmHumanBloodMatrixToRows",
        "nmHumanBloodDeterminantSafe", "nmHumanBloodRowDifference",
        "nmHumanBloodAngularVelocity", "nmHumanBloodAffineNormSquared",
        "nmHumanBloodMergeKineticAdmissible", "nmHumanBloodMergeDeformation",
        "nmHumanBloodMergeStatesCompatible",
    ],
    "method": "The 208 runtime preflight is already expanded to one source unit. This deterministic update substitutes the isolated respiration source's namespaced coalescence helper and handoff-kernel block, mirrors the source-bound bulk coefficient into the existing dispatch tail padding, avoids duplicate ABI expansion, and retains an include-free runtime input.",
    "replaced_function": "nm_human_blood_mpm_handoff",
    "schema": "numi.sustained_blood_flattened_shader.v6",
    "source_header_sha256": sha((root / "source/matter/include/numi/matter/human_respiration.h").read_bytes()),
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2))
