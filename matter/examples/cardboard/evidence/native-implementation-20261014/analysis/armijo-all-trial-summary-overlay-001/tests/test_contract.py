#!/usr/bin/env python3
"""Static contract checks for the source-115 all-trial summary overlay."""
from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
BASE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-private-ccd-fracture-115/matter/src/runtime.mm")
CANDIDATE = PACKAGE / "overlay/matter/src/runtime.mm"
STATUS_HEADER = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-private-ccd-fracture-115/matter/include/numi/matter/shared.h")
EXPECTED_BASE_SHA = "3c467b04e72512b53a5ba9e1f3ee4132f293b0d2e6ac22acc0c3936512319e74"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    base = BASE.read_text()
    candidate = CANDIDATE.read_text()
    header = STATUS_HEADER.read_text()
    require(sha(BASE) == EXPECTED_BASE_SHA, "base runtime hash mismatch")

    # The ABI has eight uint32 words plus one float4, with 16-byte alignment.
    status_body = re.search(
        r"typedef struct NM_ALIGN16 NMMatterStatusGPU \{(.*?)\} NMMatterStatusGPU;",
        header,
        re.S,
    )
    require(status_body is not None, "status ABI declaration missing")
    uint_words = len(re.findall(r"\bnm_u32\s+\w+\s*;", status_body.group(1)))
    has_float4 = len(re.findall(r"\bnm_float4\s+\w+\s*;", status_body.group(1)))
    require((uint_words, has_float4) == (8, 1), "unexpected status ABI fields")
    status_bytes = uint_words * 4 + 16
    require(status_bytes == 48, "expected status ABI byte size changed")

    expected_stages = [
        "geometry_validation",
        "bulk_material_mechanical",
        "cohesive_residual",
        "private_rigid_contact",
        "deformable_sweep_ccd",
        "deformable_self_narrowphase",
        "free_body_full_residual",
        "measured_predecision",
    ]
    stage_block = re.search(
        r"femCoupledTrialSummaryStageNames\{\{(.*?)\}\};", candidate, re.S
    )
    require(stage_block is not None, "stage table missing")
    stages = re.findall(r'"([a-z_]+)"', stage_block.group(1))
    require(stages == expected_stages, f"stage table mismatch: {stages}")

    # Status slots are trial-major, then stage, then environment. Decision
    # status is a separate trial-major array; metric/alpha records are distinct.
    for budget in (1, 16, 24, 32):
        for environments in (1, 2, 4):
            status_per_environment = status_bytes
            metric_bytes = environments * 16
            status_bytes_env = environments * status_per_environment
            alpha_bytes = environments * 4
            stage_bytes = budget * len(expected_stages) * status_bytes_env
            decision_bytes = budget * status_bytes_env
            # Loop-entry, predecision, and postdecision each export one metric;
            # the final metrics and FEM line-search are in the fixed tail.
            per_trial = 3 * metric_bytes + status_bytes_env + alpha_bytes
            final_bytes = 3 * status_bytes_env + 2 * metric_bytes + alpha_bytes
            total = stage_bytes + decision_bytes + budget * per_trial + final_bytes
            arrays = 8 + 5 * budget
            # Independent closed form for the current 48-byte status ABI,
            # 16-byte metric vectors, and 4-byte environment fractions.
            require(total == environments * (180 + 532 * budget),
                    "summary byte model disagrees with closed-form layout")
            require(total <= 32 * 1024 * 1024, "summary byte model exceeds 32 MiB")
            require(arrays <= 192, "summary array model exceeds cap")
            require(arrays == (8 + 5 * budget), "unexpected descriptor count")
    require("maximumBudget = 32u" in candidate, "compiled budget ceiling absent")
    require("captureArrayLimit = 192u" in candidate, "descriptor cap absent")
    require("femCoupledCaptureCapBytes" in candidate, "aggregate byte cap absent")
    require("state.device.maxBufferLength" in candidate, "device extent cap absent")
    require("totalBytes > static_cast<std::uint64_t>(\n                        state.device.maxBufferLength)" in candidate,
            "whole-payload device extent check absent")

    # Admission is opt-in, paired, source-scope bounded, and mutually exclusive
    # with the other captures. It never changes the physical operator or gate.
    for token in (
        'NM_FEM_COUPLED_TRIAL_SUMMARY_SELECTOR',
        'NM_FEM_COUPLED_TRIAL_SUMMARY_PATH',
        'summary_cannot_share_full_trial_or_center_readback',
        'summary_requires_full_contact_residual_armijo',
        'femCoupledCaptureScopeFailure()',
        'NM_FEM_OPERATOR_FD_SELECTOR',
        'physical_state_changed\\":false',
    ):
        require(token in candidate, f"missing admission contract token: {token}")

    # Existing full trial capture stays disabled in summary mode, preserving its
    # old schema and avoiding a second readback into the same command buffer.
    require(candidate.count('if (!femCoupledTrialSummaryMode &&\n                        femCoupledTrialCaptureSelected') == 3,
            "full-trial capture hooks are not all excluded in summary mode")
    require("if (femCoupledTrialCaptureSelected &&\n                femCoupledTrialSummaryMode) {\n                captureTrialSummaryStatus" in candidate,
            "summary callback allocation is not opt-in")
    require('stage_status_checkpoints\\":8' in candidate,
            "summary enabled record reports the wrong stage count")
    require("femCoupledTrialSummaryBudget) * 9u" in candidate,
            "summary does not require all 8 stage and 1 decision status copies")

    loop = candidate.rindex("encodeFEMResidualArmijoBaseStatus();")
    start_copy = candidate.index("captureTrialSummaryStart(trialIndex)", loop)
    candidate_stage = candidate.index('"nm_fem_armijo_validate_geometry"', loop)
    geom_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 0u', loop)
    bulk_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 1u', loop)
    cohesive_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 2u', loop)
    rigid_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 3u', loop)
    full_residual_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 6u', loop)
    measure = candidate.index('"nm_fem_armijo_measure_trial"', loop)
    predecision_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 7u', measure)
    decision = candidate.index('"nm_fem_armijo_decide"', measure)
    postdecision_copy = candidate.index('captureTrialSummaryPostDecision(trialIndex)', decision)
    global_status_copy = candidate.index('captureTrialSummaryDecisionStatus(trialIndex)', decision)
    contact_helper = candidate.index('const auto encodeArmijoDeformableContact =', 10000)
    sweep_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 4u', contact_helper)
    self_copy = candidate.index('captureTrialSummaryStatus(trialIndex, 5u', sweep_copy)
    contact_call = candidate.index('if (!encodeArmijoDeformableContact(', rigid_copy)
    require(start_copy < candidate_stage < geom_copy < bulk_copy < cohesive_copy <
            rigid_copy < contact_call < full_residual_copy < measure <
            predecision_copy < decision < postdecision_copy < global_status_copy,
            "summary readback points do not follow production stage order")
    require(sweep_copy < self_copy, "sweep CCD checkpoint is after endpoint narrowphase")
    measure_close = candidate.index('});', measure)
    require(candidate[measure_close + 3:predecision_copy].find(
                '[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]') == -1,
            "summary added an off-mode barrier after measurement")
    require(candidate.index("captureTrialSummaryFinal()", global_status_copy) >
            global_status_copy, "final state arrays are not captured after all trials")
    require(candidate.index("!finishFEMCoupledTrialCapture()", global_status_copy) >
            global_status_copy, "summary payload is not finalized after all trials")

    # The summary schema labels raw status and metric fields, includes both
    # accepted fractions, and binds the packed shapes and hashes.
    for token in (
        "numi.matter.coupled_armijo_trial_summary.v1",
        "armijo_trial_status_checkpoints",
        "armijo_trial_global_status_after_decision",
        "armijo_final_environment_line_search",
        "stage_and_decision_status_copy_count",
        "status_copy_count_scope",
        'shape',
        'metrics_float4',
        'summaryMode ? ".summary.bin" : ".coupled.bin"',
        'summaryMode ? ".summary.json" : ".coupled.json"',
    ):
        require(token in candidate, f"missing summary schema token: {token}")

    # Diagnostic-only implementation is limited to runtime readbacks and
    # metadata. The shader source and physics source are not edited by this pack.
    decision_dispatch = re.compile(r'dispatchThreads\s*\(\s*"nm_fem_armijo_decide"')
    require(len(decision_dispatch.findall(base)) ==
            len(decision_dispatch.findall(candidate)),
            "unexpected production decision kernel call-count delta")
    require(decision_dispatch.search(base) and decision_dispatch.search(candidate),
            "production decision call missing")
    kernel_dispatch = re.compile(
        r'dispatch(?:Threads|Groups32)\s*\(\s*"([A-Za-z0-9_]+)"')
    require(Counter(kernel_dispatch.findall(base)) ==
            Counter(kernel_dispatch.findall(candidate)),
            "diagnostic overlay changed production kernel dispatch inventory")
    print("PASS source-bound trial summary contract: 8 stages; budgets 1/16/24/32; envs 1/2/4; bytes <= 32 MiB; descriptors <= 192; capture order/config/schema/final fraction checked")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
