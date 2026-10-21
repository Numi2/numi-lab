#!/usr/bin/env python3
"""Small source contract for the opt-in graph PSD contact augmentation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runtime = (ROOT / "matter/src/runtime.mm").read_text()
shader = (ROOT / "matter/src/metal/fgmres.metalinc").read_text()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


# 1. Strict opt-in, rejected malformed values, and selector-6 graph admission.
require("femHuman6GraphFullPSDContactBlocksEnabled = false;" in runtime,
        "the feature is not default-off")
require('std::getenv("NM_FGMRES_GRAPH_FULL_PSD_CONTACT_BLOCKS")' in runtime and
        'std::strcmp(graphPSDContactSetting, "0") != 0' in runtime and
        'std::strcmp(graphPSDContactSetting, "1") != 0' in runtime,
        "the environment option must accept only 0 or 1")
require("!human6GraphRequested" in runtime and
        "world.mixedSolver.executionBudgets.z != 6u" in runtime and
        "candidate->femHuman6GraphFullPSDContactBlocksEnabled =" in runtime and
        "graphPSDContactRequested &&\n                candidate->femHuman6GraphSchwarzEnabled" in runtime,
        "the opt-in must require and remain subordinate to selector-6 graph admission")
print("PASS strict default-off selector-6 admission")

# 2. Matrix lifecycle order: cached rows, generic fallback/full assembly,
#    optional addition, any parity copy, then graph factorization.
cached = runtime.index('dispatchThreads(\n                        "nm_fgmres_graph_build_patch_matrices_cached_rows"')
generic_fallback = runtime.index('"nm_fgmres_graph_build_patch_matrices",\n                        state.femRegionalGraphCachedFallbackDispatch', cached)
augment_guard = runtime.index(
    "if (state.femHuman6GraphFullPSDContactBlocksEnabled)", generic_fallback)
augment = runtime.index('"nm_fgmres_graph_add_psd_contact_blocks"', augment_guard)
parity = runtime.index("if (captureGraphMatrixParity)", augment)
copy = runtime.index('dispatchThreads("nm_fgmres_graph_matrix_copy"', parity)
factor = runtime.index('dispatchGroups256("nm_fgmres_graph_factor_patches"', copy)
require(cached < generic_fallback < augment_guard < augment < parity < copy < factor,
        "augmentation is not placed after matrix assembly and before copy/factor")
require("[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];\n                if (state.femHuman6GraphFullPSDContactBlocksEnabled)" in runtime,
        "the graph assembly writes are not barriered before contact augmentation")
print("PASS post-assembly/pre-factor placement and barrier")

# 3. Kernel uses the existing per-patch completed-factor predicate, skips
#    identity/fixed rows and due-aware dormant rows, and binds actual contact data.
start = shader.index("kernel void nm_fgmres_graph_add_psd_contact_blocks(")
end = shader.index("kernel void nm_fgmres_graph_build_patch_matrices_cached_rows(", start)
body = shader[start:end]
require("constant NMPrimalContactArguments& contact [[buffer(1)]]" in body and
        "device float* patchMatrices [[buffer(7)]]" in body,
        "kernel arguments do not target primal contact data and graph matrices")
require("fgmresGraphMayReuseCompletedFactor(" in body and
        "priorFactorValid[factorIndex])) return;" in body,
        "completed graph factors could receive the augmentation twice")
require("node.restAndFixed.w != 0.0f || nodeCoverage[row.w] == 0u) return;" in body and
        "dueAwareRows != 0u && !objectDue(" in body,
        "fixed/uncovered or due-aware inactive graph rows are not skipped")
print("PASS reuse, fixed-row, and due-row guards")

# 4. Fail-closed bounds/finite checks and local same-node 3x3 insertion only.
helper_start = shader.index("inline bool fgmresGraphAddContactBlockRow(")
helper_end = shader.index("// Diagonal of the exact IPC blocks", helper_start)
helper = shader[helper_start:helper_end]
require("matrixBase > matrixFloatCount" in helper and
        "entries > matrixFloatCount - matrixBase" in helper and
        "!isfinite(value) || !isfinite(old) || !isfinite(next[axis])" in helper,
        "local matrix write bounds or finite checks are missing")
require("matrixBase = environment * matrixFloatCount + patch.z" in body and
        "totalMatrixFloatCount" in body and
        "matrixFloatCount > 0xffffffffu / dispatch.environmentCount" in body,
        "environment-strided matrix capacity is not overflow-checked")
require("row * n +\n            localNode * 3u + axis" in helper,
        "contact entries are not confined to the row node's xyz columns")
require("NM_STATUS_NONFINITE_RESULT" in body,
        "invalid contact/matrix data does not fail closed")
print("PASS bounded finite same-node xyz insertion")

# 5. The Metal fixture covers x-z coupling and uses a per-case matrix range,
#    so multiple cases exercise bounds as well as the row helper.
fixture_start = shader.index("kernel void nm_fgmres_check_graph_psd_contact_block(")
fixture_end = shader.index("kernel void nm_fgmres_build_tangent_fem_block_preconditioner_cached(", fixture_start)
fixture = shader[fixture_start:fixture_end]
require("device float* patchMatrices [[buffer(6)]]" in fixture and
        "fgmresGraphAddContactBlockRow(" in fixture and
        "global * 36u, count * 36u" in fixture,
        "multi-case Metal row fixture kernel is incomplete")
print("PASS rotated-normal multi-case Metal row fixture source")
print("graph PSD contact source contract: 5 checks PASS")
