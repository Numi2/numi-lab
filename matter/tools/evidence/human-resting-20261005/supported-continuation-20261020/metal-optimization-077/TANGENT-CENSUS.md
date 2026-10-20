# Human FEM material tangent census

Input log: /Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-077/global-icb-census-1/stderr.log
Binary SHA256: c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828
Execution source HEAD (provenance correction): 245a7610ab21ed7c510bb4b8594e8f4b4b99de31
Invocation source_head (inherited template field, not authoritative): aadcd68e0747540978edd88a23af0cd114932942
Working-source runtime.mm SHA256 (source-sha256.json): 8326124b6aa399bfd2ec18ce92a408167760d8a129320f904023a67742801c08
Working-source FEM shader SHA256 (source-sha256.json): 15dbcdf7423e1daef484a908e1d62d19d958be8817357960e61fdfb1785aff95

The runtime emitted this census once while uploading the compiled Human FEM world with NM_FEM_TANGENT_CENSUS=1. Instruction and opcode counts are from compiled scalar bytecode descriptors; tetrahedra are active object/tet assignments. This is source-work accounting, not shader instruction execution, call counts, cache hits, projection iterations, factorization counts, or timing.

Estimated tangent visits use the requested formula: active_fem_tetrahedra * 9 * tangent_instruction_count, where tangent_instruction_count sums the nine compiled tangent scalar programs. This is a potential bytecode-work weight across nine deformation directions, not a runtime visit count. The stress weight assumes one stress-program-set visit per active tetrahedron only as a reference.

## Totals

- Compiled material descriptors: **20**.
- FEM tetrahedron slots: **54,133 stored; 54,133 active**.
- Constitutive families: bytecode=19, neo_hookean=1.
- Stateful descriptors: **0**; active implicit-state FEM tetrahedra: **0**.
- Active generic implicit-projection tetrahedra (potential generic bytecode state solve): **0**; this is static eligibility, not observed fallback calls.
- Canonical tangent candidates/enabled: **2/0**.
- Shared 15-state factor candidates/enabled: **0/0**.
- Estimated tangent bytecode visits over nine directions: **3,554,868,033**.
- Reference stress weight (active tets * stress instructions): **86,706,189**; tangent weight for one direction: **394,985,337**.

## Per-material breakdown ranked by estimated tangent visits

| Rank | Material | Family/projection | Active tets | States / implicit | Canonical candidate/enabled | Stress instructions (exp/log) | Tangent instructions (exp/log) | Estimated tangent visits |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | human_optic_nerve_passive_reference_proxy | bytecode/generic | 15,098 | 0/0 | 0/0 | 1,926 (18/18) | 8,808 (45/45) | 1,196,848,656 |
| 2 | human_skin_dermis_frequency_specific_effective | bytecode/generic | 13,857 | 0/0 | 1/0 | 1,134 (0/9) | 4,956 (0/9) | 618,077,628 |
| 3 | human_subcutaneous_adipose_low_strain_effective | bytecode/generic | 13,857 | 0/0 | 1/0 | 1,134 (0/9) | 4,956 (0/9) | 618,077,628 |
| 4 | human_passive_organ_reference_proxy | bytecode/generic | 7,564 | 0/0 | 0/0 | 1,926 (18/18) | 8,808 (45/45) | 599,613,408 |
| 5 | numi_human_regional_myofascia_6_transverse_isotropic | bytecode/generic | 576 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 106,453,440 |
| 6 | numi_human_regional_myofascia_7_transverse_isotropic | bytecode/generic | 576 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 106,453,440 |
| 7 | numi_reduced_abdominal_aortic_wall_v1 | neo_hookean/generic | 1,152 | 0/0 | 0/0 | 1,755 (0/18) | 8,154 (0/18) | 84,540,672 |
| 8 | human_prostate_passive_reference_proxy | bytecode/generic | 442 | 0/0 | 0/0 | 1,926 (18/18) | 8,808 (45/45) | 35,038,224 |
| 9 | numi_human_regional_myofascia_8_transverse_isotropic | bytecode/generic | 144 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 26,613,360 |
| 10 | numi_human_regional_myofascia_9_transverse_isotropic | bytecode/generic | 144 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 26,613,360 |
| 11 | numi_human_regional_myofascia_5_transverse_isotropic | bytecode/generic | 129 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 23,841,135 |
| 12 | numi_human_regional_myofascia_2_transverse_isotropic | bytecode/generic | 126 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 23,286,690 |
| 13 | numi_human_regional_myofascia_10_transverse_isotropic | bytecode/generic | 108 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 19,960,020 |
| 14 | numi_human_regional_myofascia_11_transverse_isotropic | bytecode/generic | 108 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 19,960,020 |
| 15 | numi_human_regional_myofascia_0_transverse_isotropic | bytecode/generic | 54 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 9,980,010 |
| 16 | numi_human_regional_myofascia_1_transverse_isotropic | bytecode/generic | 54 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 9,980,010 |
| 17 | numi_human_regional_myofascia_3_transverse_isotropic | bytecode/generic | 54 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 9,980,010 |
| 18 | numi_human_regional_myofascia_4_transverse_isotropic | bytecode/generic | 54 | 0/0 | 0/0 | 4,023 (9/9) | 20,535 (18/9) | 9,980,010 |
| 19 | inferred_registered_left_ACL_reference_v1_febio_reference | bytecode/generic | 18 | 0/0 | 0/0 | 5,598 (45/54) | 29,538 (144/153) | 4,785,156 |
| 20 | inferred_registered_right_ACL_reference_v1_febio_reference | bytecode/generic | 18 | 0/0 | 0/0 | 5,598 (45/54) | 29,538 (144/153) | 4,785,156 |

## Family-weighted tangent visits

| Family | Materials | Active tets | Estimated visits |
| --- | ---: | ---: | ---: |
| bytecode | 19 | 52,981 | 3,470,327,361 |
| neo_hookean | 1 | 1,152 | 84,540,672 |

The earlier 8.06-second tangent-build profile remains unattributed by material: this census does not measure dynamic call counts or disjoint GPU stage time.
