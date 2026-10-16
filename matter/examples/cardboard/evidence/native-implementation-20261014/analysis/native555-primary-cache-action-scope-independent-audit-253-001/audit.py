#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
OUT = Q / "analysis/native555-primary-cache-action-scope-independent-audit-253-001"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "snapshot221_frozen": Q / "snapshots/source-qualification-checkpoint-proof-221-003/frozen-source.json",
    "runtime221": Q / "snapshots/source-qualification-checkpoint-proof-221-003/matter/src/runtime.mm",
    "fem221": Q / "snapshots/source-qualification-checkpoint-proof-221-003/matter/src/metal/fem.metalinc",
    "build221": Q / "build-source221-checkpoint-proof-003/source-build.json",
    "probe221": Q / "build-source221-checkpoint-proof-003/numi-matter-cardboard-probe",
    "metallib221": Q / "build-source221-checkpoint-proof-003/shaders/NumiMatter.metallib",
    "prediction555": Q / "solver/control19-newton3-resume-diagnostic-555/prediction.json",
    "receipt555": Q / "solver/control19-newton3-resume-diagnostic-555/receipt.json",
    "capture555": Q / "solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture",
    "snapshot241_frozen": Q / "snapshots/source-strict-refined-contact-observers-241-004/frozen-source.json",
    "runtime241": Q / "snapshots/source-strict-refined-contact-observers-241-004/matter/src/runtime.mm",
    "fem241": Q / "snapshots/source-strict-refined-contact-observers-241-004/matter/src/metal/fem.metalinc",
    "mp_report555": Q / "analysis/native555-mp-root-independent-adjudication-002/report.json",
    "mp_manifest555": Q / "analysis/native555-mp-root-independent-adjudication-002/manifest.json",
    "cached_operator_report251": Q / "analysis/native555-cached-native-tangent-conditioning-251-003/report.json",
    "cached_operator_manifest251": Q / "analysis/native555-cached-native-tangent-conditioning-251-003/manifest.json",
    "cached_operator_vectors555": Q / "analysis/native555-cached-operator-001/operator.vectors.npz",
    "cached_operator_launch555": Q / "analysis/native555-cached-operator-001/launch.json",
    "publication040_readme": Q / "analysis/publication040-reproduction-002/README.md",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lines(path):
    return path.read_text(errors="replace").splitlines()

def excerpt(path, line_no):
    ls = lines(path)
    return ls[line_no - 1].strip()

p221 = json.loads(FILES["snapshot221_frozen"].read_text())
p241 = json.loads(FILES["snapshot241_frozen"].read_text())
build = json.loads(FILES["build221"].read_text())
pred = json.loads(FILES["prediction555"].read_text())
receipt = json.loads(FILES["receipt555"].read_text())
mp = json.loads(FILES["mp_report555"].read_text())
cached = json.loads(FILES["cached_operator_report251"].read_text())
mp_manifest = json.loads(FILES["mp_manifest555"].read_text())
cached_manifest = json.loads(FILES["cached_operator_manifest251"].read_text())

assert pred["source_snapshot"] == str(Q / "snapshots/source-qualification-checkpoint-proof-221-003")
assert build["source_snapshot"] == pred["source_snapshot"]
assert p221["source_sha256"]["matter/src/runtime.mm"] == sha(FILES["runtime221"])
assert p241["source_sha256"]["matter/src/runtime.mm"] == sha(FILES["runtime241"])
assert p241["source_sha256"]["matter/src/metal/fem.metalinc"] == sha(FILES["fem241"])
assert sha(FILES["capture555"]) == "c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53"
assert pred["executed_build_verified"] is True
assert receipt["source_inputs_unchanged"] is True
assert p221["source_sha256"]["matter/src/metal/fem.metalinc"] == sha(FILES["fem221"])
assert pred["argv"][0] == str(FILES["probe221"])
assert pred["argv"][pred["argv"].index("--metallib") + 1] == str(FILES["metallib221"])
assert "--cache-fem-algorithmic-tangent" in pred["argv"]
assert build["artifacts_sha256"][str(FILES["probe221"])] == sha(FILES["probe221"])
assert build["artifacts_sha256"][str(FILES["metallib221"])] == sha(FILES["metallib221"])
assert build["artifacts_sha256"][str(FILES["probe221"])] == "8b8666d804a50a55fd016cb1a6faf7f2b5d014a843b4a4d3cfda75469034c30d"
assert build["artifacts_sha256"][str(FILES["metallib221"])] == "02fbde247e241ce40958795ceaf0259aee6a6fb57c09374b0a7d31e088cb3e2b"

checks = [
    ("runtime221", 13093, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 13118, "[encoder setBuffer:state.dummy offset:0u atIndex:23u];"),
    ("runtime221", 13119, "[encoder setBuffer:state.dummy offset:0u atIndex:24u];"),
    ("runtime221", 13120, "const std::uint32_t useCache = 0u;"),
    ("runtime241", 13469, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime241", 13494, "[encoder setBuffer:state.dummy offset:0u atIndex:23u];"),
    ("runtime241", 13495, "[encoder setBuffer:state.dummy offset:0u atIndex:24u];"),
    ("runtime241", 13496, "const std::uint32_t useCache = 0u;"),
    ("runtime241", 20613, "if (state.mixedSolverValue.executionBudgets.w == 1u) {"),
    ("runtime241", 20620, 'dispatchThreads(tangentCacheBuilder,'),
    ("runtime241", 20813, "const std::uint32_t useCache ="),
    ("runtime241", 20860, "const std::uint32_t useCache ="),
    ("runtime241", 21189, "const std::uint32_t useCache ="),
    ("runtime241", 21671, "metadata.tangentCacheUsed ="),
    ("runtime241", 22730, "metadata.tangentCacheUsed ="),
    ("runtime241", 20894, 'dispatchThreads("nm_fem_apply_operator_elements",'),
    ("runtime241", 20947, "const std::uint32_t useCache ="),
    ("runtime241", 22808, 'dispatchThreads("nm_fem_apply_operator_elements",'),
    ("runtime241", 22837, "const std::uint32_t useCache ="),
    ("runtime241", 23366, "const std::uint32_t useCache ="),
    ("runtime241", 23616, 'dispatchThreads("nm_fem_apply_operator_elements",'),
    ("runtime241", 23655, "const std::uint32_t useCache ="),
    ("runtime241", 23825, 'dispatchThreads("nm_fem_apply_operator_elements",'),
    ("runtime241", 23823, "const std::uint32_t useCacheForCapturedAction ="),
    ("runtime241", 24689, 'dispatchThreads("nm_fem_apply_operator_elements",'),
    ("runtime241", 24728, "const std::uint32_t useCache = cacheMode;"),
    ("runtime241", 24830, "const std::uint32_t useCache ="),
]
for key, no, want in checks:
    assert want in excerpt(FILES[key], no), (key, no, excerpt(FILES[key], no))

checks221_production = [
    ("runtime221", 20094, "if (state.mixedSolverValue.executionBudgets.w == 1u) {"),
    ("runtime221", 20101, "dispatchThreads(tangentCacheBuilder,"),
    ("runtime221", 20259, '"nm_fgmres_build_tangent_fem_preconditioner"'),
    ("runtime221", 20294, "const std::uint32_t useCache ="),
    ("runtime221", 20304, '"nm_fgmres_build_tangent_fem_block_preconditioner"'),
    ("runtime221", 20341, "const std::uint32_t useCache ="),
    ("runtime221", 20375, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 20428, "const std::uint32_t useCache ="),
    ("runtime221", 20670, "const std::uint32_t useCache ="),
    ("runtime221", 21152, "metadata.tangentCacheUsed ="),
    ("runtime221", 22211, "metadata.tangentCacheUsed ="),
    ("runtime221", 22289, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 22318, "const std::uint32_t useCache ="),
    ("runtime221", 22817, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 22847, "const std::uint32_t useCache ="),
    ("runtime221", 23097, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 23136, "const std::uint32_t useCache ="),
    ("runtime221", 23304, "const std::uint32_t useCacheForCapturedAction ="),
    ("runtime221", 23306, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 23337, "[encoder setBytes:&useCacheForCapturedAction"),
    ("runtime221", 24170, 'dispatchThreads("nm_fem_apply_operator_elements"'),
    ("runtime221", 24209, "const std::uint32_t useCache = cacheMode;"),
    ("runtime221", 24311, "const std::uint32_t useCache ="),
    ("runtime221", 24349, "action(4u, state.femSolution, state.femBaseCGLSQ);"),
    ("runtime221", 24379, "action(3u, state.femBaseCGLSB, state.femBaseCGLSS);"),
]
for key, no, want in checks221_production:
    assert want in excerpt(FILES[key], no), (key, no, excerpt(FILES[key], no))

kernel = lines(FILES["fem241"])
kernel_checks = {
    3166: "device const float* tangentCache [[buffer(23)]],",
    3167: "device const uint* tangentCacheValid [[buffer(24)]],",
    3168: "constant uint& useTangentCache [[buffer(25)]],",
    3240: "tangentCacheValid[global] == 1u;",
    3245: "const bool requestTransposeCache = useTangentCache == 3u;",
    3246: "const bool requestStrictCache = useTangentCache == 1u ||",
    3248: "const bool requestFallbackCache = useTangentCache == 2u;",
    3250: "(requestFallbackCache && cacheEntryValid)) {",
    3251: "if (!cacheEntryValid) {",
    3280: "evaluateAlgorithmicStressTangent(",
}
for no, want in kernel_checks.items():
    assert want in kernel[no - 1], (no, kernel[no - 1].strip())

assert cached["schema"] == "numi.cardboard.captured-native-tangent-linear-conditioning.v1"
assert cached["checks"]["assembled_vs_native_action"]["relative_l2"] == 0.000597271628787105
assert mp["native_A_delta_vs_negative_MP_center_residual_derivative"]["relative_l2"] == \
       1.1972403495054814e-05
assert mp_manifest["inputs_sha256"]["capture"] == sha(FILES["capture555"])
assert cached_manifest["inputs_sha256"][str(FILES["capture555"])] == sha(FILES["capture555"])
assert cached_manifest["inputs_sha256"][str(FILES["cached_operator_vectors555"])] == \
       sha(FILES["cached_operator_vectors555"])

line_evidence = []
for key, no, _ in checks:
    line_evidence.append({"file": str(FILES[key]), "line": no, "text": excerpt(FILES[key], no)})
for key, no, _ in checks221_production:
    line_evidence.append({"file": str(FILES[key]), "line": no, "text": excerpt(FILES[key], no)})
for no in sorted(kernel_checks):
    line_evidence.append({"file": str(FILES["fem241"]), "line": no,
                          "text": kernel[no - 1].strip()})

report = {
    "schema": "numi.cardboard.native555-primary-cache-action-scope-audit.v1",
    "verdict": "PASS_SOURCE_SCOPE_IDENTIFIED; native cached-kernel action parity remains unmeasured",
    "classification": "read-only source and report audit; no native dispatch, no production edit",
    "source_bindings": {
        "source221_snapshot": str(Q / "snapshots/source-qualification-checkpoint-proof-221-003"),
        "source221_frozen_sha256": sha(FILES["snapshot221_frozen"]),
        "source221_runtime_sha256": sha(FILES["runtime221"]),
        "source221_fem_kernel_sha256": sha(FILES["fem221"]),
        "source221_build_record_sha256": sha(FILES["build221"]),
        "source221_probe_sha256": sha(FILES["probe221"]),
        "source221_metallib_sha256": sha(FILES["metallib221"]),
        "source221_build_artifact_map_sha256": {
            str(FILES["probe221"]): build["artifacts_sha256"][str(FILES["probe221"])],
            str(FILES["metallib221"]): build["artifacts_sha256"][str(FILES["metallib221"])],
        },
        "source241_snapshot": str(Q / "snapshots/source-strict-refined-contact-observers-241-004"),
        "source241_frozen_sha256": sha(FILES["snapshot241_frozen"]),
        "source241_runtime_sha256": sha(FILES["runtime241"]),
        "source241_fem_kernel_sha256": sha(FILES["fem241"]),
        "native555_prediction_sha256": sha(FILES["prediction555"]),
        "native555_receipt_sha256": sha(FILES["receipt555"]),
        "native555_capture_sha256": sha(FILES["capture555"]),
    },
    "diagnostic_fd_action": {
        "source221_explicit_use_cache": 0,
        "source221_dispatch_line": 13093,
        "source221_dummy_cache_buffer_lines": [13118, 13119],
        "source221_use_cache_line": 13120,
        "source241_same_behavior_lines": [13469, 13494, 13495, 13496],
        "meaning": "Captured production_A_delta_v_impulse_Ns is the direct constitutive evaluator output. The raw tangent-cache bytes and valid flags are separately snapshotted; that snapshot does not make this action a cache application.",
    },
    "native555_prediction_cache_policy": {
        "argv_contains_cache_fem_algorithmic_tangent": "--cache-fem-algorithmic-tangent" in pred["argv"],
        "source221_strict_cache_primary_calls": [
            {"lines": [22289, 22318], "role": "Ordinary FGMRES element action; strict mode 1 iff executionBudgets.w == 1."},
            {"lines": [22817, 22847], "role": "Restart/augmentation operator action; strict mode 1 iff cache policy enabled."},
            {"lines": [23097, 23136], "role": "Fresh true-residual action; strict mode 1 iff cache policy enabled."},
            {"lines": [23304, 23306, 23337], "role": "Optional captured action receives mode 1 when cache policy enabled."},
            {"lines": [24170, 24209, 24349, 24379], "role": "Base-Cauchy action uses caller-specified forward mode 4 and transpose mode 3."},
        ],
        "source221_cache_construction_and_preconditioners": [
            {"lines": [20094, 20101], "role": "Build the per-tet tangent cache only when executionBudgets.w == 1."},
            {"lines": [20259, 20294], "role": "Tangent FEM preconditioner requests strict cache when enabled."},
            {"lines": [20304, 20341], "role": "Block FEM preconditioner requests strict cache when enabled."},
            {"lines": [20375, 20428], "role": "Regional RAP uses mode 2 when enabled; valid entries use cache while invalid entries can use direct fallback."},
            {"lines": [20670], "role": "Graph preconditioner build receives strict cache mode when enabled."},
            {"lines": [24311], "role": "CGLS node-block preconditioner receives strict cache mode when enabled."},
            {"lines": [21152, 22211], "role": "Cache-use metadata markers; these report policy state and are not operator dispatches."},
        ],
        "interpretation": "The actual native555 command enabled the cache policy and the normal solver call sites in its exact source221 build pass cache modes. The separate operator-FD diagnostic at lines 13093 and 13120 hardcodes mode 0 and dummy cache buffers, so its recorded action is not one of those production cached actions.",
    },
    "source221_production_call_sites": [
        {"path": "matter/src/runtime.mm", "lines": [20259, 20294], "role": "FEM tangent preconditioner uses strict mode 1 when cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [20304, 20341], "role": "Block tangent preconditioner uses strict mode 1 when cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [20375, 20428], "role": "Regional RAP applies mode 2, allowing direct fallback for an invalid cache entry."},
        {"path": "matter/src/runtime.mm", "lines": [22289, 22318], "role": "Primary FGMRES operator uses strict mode 1 when enabled."},
        {"path": "matter/src/runtime.mm", "lines": [22817, 22847], "role": "Restart/augmentation operator uses strict mode 1 when enabled."},
        {"path": "matter/src/runtime.mm", "lines": [23097, 23136], "role": "Fresh true residual uses strict mode 1 when enabled."},
        {"path": "matter/src/runtime.mm", "lines": [23304, 23306, 23337], "role": "Captured action sets mode from cache policy."},
        {"path": "matter/src/runtime.mm", "lines": [24170, 24209, 24349, 24379], "role": "Base-Cauchy forward/transpose calls use modes 4 and 3."},
        {"path": "matter/src/runtime.mm", "lines": [24311], "role": "CGLS node-block preconditioner receives strict mode 1 when cache policy is enabled."},
        {"path": "matter/src/runtime.mm", "lines": [21152, 22211], "role": "Metadata markers, not dispatches."},
    ],
    "source241_production_cache_call_sites": [
        {"path": "matter/src/runtime.mm", "lines": [20613, 20620, 20638, 20640], "role": "Build per-tet tangent cache when executionBudgets.w == 1."},
        {"path": "matter/src/runtime.mm", "lines": [20813, 20860], "role": "Scalar and block tangent preconditioner builders use strict mode 1 iff cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [21189], "role": "Graph preconditioner builder receives strict cache mode iff cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [20894, 20944, 20947], "role": "Regional RAP operator, mode 2 when cache policy is enabled; valid entry uses cache, invalid entry falls back to the direct evaluator."},
        {"path": "matter/src/runtime.mm", "lines": [22808, 22837], "role": "Ordinary FGMRES element action, strict mode 1 iff cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [23336, 23366], "role": "Restart/augmentation FGMRES element action, strict mode 1 iff cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [23616, 23655], "role": "Fresh true-residual element action, strict mode 1 iff cache policy enabled."},
        {"path": "matter/src/runtime.mm", "lines": [23825, 23823], "role": "Optional captured CGLS action, mode tracks cache policy."},
        {"path": "matter/src/runtime.mm", "lines": [24689, 24728, 24868, 24898], "role": "Base-Cauchy actions; forward mode 4 and transpose mode 3."},
        {"path": "matter/src/runtime.mm", "lines": [24830], "role": "CGLS right-scale block builder receives strict cache mode when enabled."},
        {"path": "matter/src/runtime.mm", "lines": [21671, 22730], "role": "Metadata markers reporting cache policy; not action dispatches."},
    ],
    "kernel_contract": {
        "cache_validity": "tangentCacheValid[global] must equal 1 for a cache entry",
        "strict_modes": [1, 3, 4],
        "strict_invalid_entry": "Marks action invalid; no direct fallback.",
        "regional_mode": 2,
        "regional_invalid_entry": "Falls through to evaluateAlgorithmicStressTangent for that tet.",
        "direct_mode": 0,
    },
    "native555_metrics_scope": {
        "mp_report": str(FILES["mp_report555"]),
        "native_action_field": "production_A_delta_v_impulse_Ns",
        "MP_center_vs_saved_analytic_residual_derivative": {
            "relative_l2": 4.180940342245761e-11,
            "absolute_l2_Ns": 1.1410907703757011e-15,
            "maximum_abs_Ns": 7.645985361572094e-16,
            "target_l2_Ns": 2.72926824342768e-05,
        },
        "native_A_delta_vs_negative_MP_center_residual_derivative": {
            "relative_l2": 1.1972403495054814e-05,
            "absolute_l2_Ns": 3.267590065656604e-10,
            "maximum_abs_Ns": 1.1577149093769124e-10,
            "target_l2_Ns": 2.7292682434285462e-05,
        },
        "native_A_delta_vs_negative_MP_center_residual_derivative_relative": 1.1972403495054814e-05,
        "correct_interpretation": "Comparison for the uncached direct evaluator output only; not a comparison of native cache application against the MP action.",
        "cached_tangent_cpu_reconstruction": str(FILES["cached_operator_report251"]),
        "cached_reconstructed_operator_vs_direct_action": {
            "absolute_l2_Ns": 1.630114457310513e-08,
            "relative_l2": 0.000597271628787105,
            "maximum_absolute_Ns": 4.7987774014032e-09,
        },
        "cached_reconstructed_operator_vs_direct_action_relative_l2": 0.000597271628787105,
        "cached_tangent_cpu_action_vs_FP64_reference": {
            "absolute_l2_Ns": 3.6972129389110115e-09,
            "relative_l2": 0.00013546649841987232,
            "maximum_absolute_Ns": 1.3473194985735565e-09,
            "scope": "offline CPU evaluation/reconstruction from captured cached tangent; not a native Metal action call",
        },
        "condition_estimate_of_cached_cpu_assembly": {
            "matrix_1norm_kg": 0.48968685976191895,
            "inverse_1norm_estimate_per_kg": 19429730.17413329,
            "condition_1norm_estimate": 9514483.554992734,
            "condition_times_FP32_epsilon": 1.1342148780822754,
        },
        "correct_interpretation_cached": "CPU-assembled operator from captured cache compared with the uncached direct native action; the FD trace did not execute the cached Metal action on the captured direction.",
        "native_cached_kernel_action_parity": "not measured by native555",
    },
    "additive_erratum_targets": [
        {"path": str(FILES["mp_report555"]), "change": "Retain values but qualify 1.197e-5 as direct uncached evaluator versus MP action."},
        {"path": str(FILES["cached_operator_report251"]), "change": "Retain 5.9727e-4 but label it CPU cached-matrix reconstruction versus uncached direct action, not a native cached-kernel action comparison."},
        {"path": str(FILES["publication040_readme"]), "change": "Narrow source-derivative-accuracy text: 555 covers direct evaluator scope; cached application parity remains open."},
        {"path": "publication039/040 archived 555 capture and commands", "change": "Append the scope note; 555 suffix-history parity is unaffected."},
    ],
    "line_evidence": line_evidence,
    "limitations": [
        "Native555 was built against source221-003; the prediction argv enables the cache policy and source221 primary solver call sites pass cache modes, while the FD diagnostic alone hardcodes useCache=0.",
        "Source241 is an independently audited later continuation that retains the FD diagnostic useCache=0 behavior.",
        "Native596's planned FD capture using the same encodeFEMOperatorFDTrace route remains uncached unless source253 explicitly selects and reports the cache mode.",
        "This audit establishes no cached-kernel numerical parity, full nonlinear convergence, or physical qualification.",
    ],
    "all_production_files_unchanged": True,
}
(OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
manifest = {
    "schema": "numi.cardboard.native555-primary-cache-action-scope-audit-manifest.v1",
    "inputs_sha256": {k: sha(v) for k, v in FILES.items()},
    "outputs_sha256": {
        "audit.py": sha(OUT / "audit.py"),
        "report.json": sha(OUT / "report.json"),
        "README.md": sha(OUT / "README.md"),
    },
    "source221_sha256": p221["source_sha256"],
    "source241_sha256": p241["source_sha256"],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({"output": str(OUT), "report_sha256": sha(OUT/"report.json"),
                  "manifest_sha256": sha(OUT/"manifest.json"),
                  "audit_sha256": sha(OUT/"audit.py")}, sort_keys=True))
