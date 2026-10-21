#!/usr/bin/env python3
"""Source and arithmetic contracts for opt-in true-residual FGMRES mode.

These tests bind control flow and admission behavior; they do not claim GPU or
physical qualification. Run with:
  python3 matter/tools/test_true_residual_certification_contract.py
"""
from pathlib import Path
import math
import re
import unittest


HERE = Path(__file__).resolve().parent
MATTER = HERE.parents[0]
RUNTIME = (MATTER / "src/runtime.mm").read_text()
FGMRES = (MATTER / "src/metal/fgmres.metalinc").read_text()
NATIVE_CHECK = (HERE / "fgmres_true_residual_native_check.mm").read_text()
ROOT = HERE.parents[1]
COUPLED_OWNER = (ROOT / "src/metal/MetalNumanXCoupledHuman.mm").read_text()
COUPLED_ADAPTER = (ROOT / "src/metal/MetalNumanXHumanMatter.mm").read_text()


def section(source: str, start: str, end: str) -> str:
    a = source.index(start)
    b = source.index(end, a + len(start))
    return source[a:b]


def adjudicate_true_residual(
    *, arnoldi_estimate: float, true_norm: float, initial_norm: float,
    forcing: float, relative_tolerance: float, absolute_tolerance: float,
    nonlinear_scale: float, defer_completion: bool, replace_residual: bool,
) -> tuple[bool, bool]:
    """Return (completion_latched, restart_residual_replaced)."""
    values = (arnoldi_estimate, true_norm, initial_norm, forcing,
              relative_tolerance, absolute_tolerance, nonlinear_scale)
    if not all(math.isfinite(x) and x >= 0.0 for x in values):
        raise ValueError("nonfinite/negative input is outside this contract")
    threshold = max(absolute_tolerance,
                    relative_tolerance * max(nonlinear_scale, 1.0))
    target = min(0.5, max(relative_tolerance, forcing)) * initial_norm
    accepted = true_norm == 0.0 or true_norm <= target or true_norm <= threshold
    # With the new option, the Arnoldi estimate cannot latch completion before
    # the accumulated correction receives its fresh residual evaluation.
    if defer_completion:
        completion = accepted if replace_residual else False
    else:
        completion = (arnoldi_estimate <= target or
                      arnoldi_estimate <= threshold)
    return completion, bool(replace_residual)


class TrueResidualCertificationContract(unittest.TestCase):
    def test_explicit_opt_in_is_separate_from_observer_selectors(self):
        self.assertIn('"NM_FEM_TRUE_RESIDUAL_CERTIFY"', RUNTIME)
        self.assertIn('std::strcmp(certification, "1") == 0', RUNTIME)
        self.assertIn('std::strcmp(certification, "0") != 0', RUNTIME)
        assign = section(RUNTIME, "femTrueResidualEnabled = state.femHumanHybridCoordinatesEnabled ||",
                         "dispatchThreads(\"nm_vascular_begin_working_set\"")
        self.assertIn("state.femHumanHybridCoordinatesEnabled", assign)
        self.assertIn("femTrueResidualCertificationRequested", assign)
        self.assertIn("femTrueResidualRootSelected", assign)
        # Selector-based readback remains an observer and is disabled when the
        # independent certification option is the only opt-in.
        trace = section(RUNTIME, "const auto encodeFEMTrueResidualTrace =",
                        "const auto encodeFEMNewtonTrace =")
        self.assertRegex(trace, r"!femTrueResidualEnabled\s*\|\|\s*!femTrueResidualRootSelected")

    def test_capability_failure_is_fail_closed(self):
        caps = section(RUNTIME, "const auto femTrueResidualCapabilityFailure =",
                       "const auto femResidualArmijoScope =")
        for guard in (
            "microtick != 0u", "state.enableVascularDense45",
            "cohesiveFaceCount > 4096u", "tetrahedronCount > (extendedPureArena ? 131072u : 32768u)",
            "femNewtonTraceScope(", "femTrueResidualElementOperator == nil",
            "femTrueResidualRightHandSide == nil",
            "femTrueResidualMetrics == nil", "femTrueResidualStatuses == nil",
            "femAlgorithmicTangentCache == nil",
        ):
            self.assertIn(guard, caps)
        cert = section(RUNTIME, "if (femTrueResidualCertificationRequested) {",
                       "const auto emitFEMNewtonTraceSkip =")
        self.assertIn("femTrueResidualCapabilityFailure()", cert)
        self.assertIn("ownership->preDynamicsOpen = false", cert)
        self.assertIn("return diagnostics", cert)

    def test_indirect_mass_action_dispatch_is_fail_closed(self):
        owner = section(COUPLED_OWNER, "bool validMassQuery(",
                        "bool validInverseQuery(")
        adapter = section(COUPLED_ADAPTER, "const bool hasIndirectDispatch =",
                          "switch (query.operation)")
        # Negative capability and environment cases.
        for guard in (
            "hasIndirectCapability != hasIndirectArguments",
            "!state.config.enableIndirectMassActionDispatch",
            "pass.environmentCount != 1u",
            "(pass.capabilities & indirectCapability) == 0u",
            "query.operation != MetalNumanXCoupledHumanOperation::massAction",
            "query.indirectDispatchArgumentsByteCapacity !=",
        ):
            self.assertIn(guard, owner + adapter)
        # A capability without the same transaction, epoch and slot lease fails.
        for field in ("TransactionFingerprint", "LinearizationEpoch",
                      "SlotGeneration"):
            self.assertIn(f"query.indirectDispatch{field} !=", owner)
            self.assertIn(f"query.indirectDispatch{field} !=", adapter)
        # Pointer/address mismatch, undersized storage, aliased vectors, or
        # overlap with protected resources all fail before owner dispatch.
        self.assertIn("query.indirectDispatchArgumentsGPUAddress !=", adapter)
        self.assertIn("!validBuffer(", owner)
        self.assertIn("MR_NUMANX_COUPLED_HUMAN_INDIRECT_DISPATCH_ARGUMENT_BYTES",
                      owner)
        self.assertIn("query.input == query.output", owner)
        self.assertIn("indirectArgumentsRegion", owner)
        self.assertIn("queryRegionsIsolated(", owner)
        self.assertIn("inputRegion, outputRegion, indirectArgumentsRegion", owner)

    def test_cycle_cap_rejects_certification_not_silent_skips(self):
        check = section(RUNTIME, "if (femTrueResidualEnabled && restartCycleCount > 64u)",
                        "const bool femCoupledCaptureNewtonSelected =")
        self.assertRegex(check, r"if \(femTrueResidualCertificationRequested\s*\|\|\s*state.femHumanHybridCoordinatesEnabled\)")
        self.assertIn("restart_cycle_cap_64", check)
        self.assertIn("return diagnostics", check)
        self.assertIn("femTrueResidualEnabled = false", check)
        self.assertIn("femTrueResidualRestartEnabled = false", check)
        self.assertIn("columnsThisCycle = std::min(", RUNTIME)
        self.assertIn("linearIterationBudget - iterationOffset", RUNTIME)

    def test_same_operator_action_precedes_true_residual_decision(self):
        cycle = section(RUNTIME, "for (std::uint32_t restartCycle = 0u;",
                        "// Capture an explicit same-generation J*delta")
        # Schur elimination may perform additional full-A actions before the
        # final residual branch. Scope this ordering contract to the actual
        # same-generation certification sequence, whose full action must still
        # follow its clear and precede the unchanged residual decision.
        certification = cycle[cycle.index("if (femTrueResidualEnabled) {"):]
        cert_order = [
            certification.index('"nm_fgmres_clear_true_operator"'),
            certification.index("encodeFullDirectOperator(trueResidualOperatorInput"),
            certification.index('"nm_fgmres_measure_true_residual"'),
        ]
        self.assertEqual(cert_order, sorted(cert_order))
        legacy_operator = section(
            certification, 'dispatchThreads("nm_fem_apply_operator_elements"',
            'dispatchGlobalMeasurement("nm_fgmres_measure_true_residual"',
        )
        self.assertLess(legacy_operator.index('"nm_fem_apply_operator_elements"'),
                        legacy_operator.index('"nm_fgmres_gather_nodes"'))
        self.assertRegex(cycle, r"const std::uint32_t useCache =\s*numi::matter::detail::femPrimaryOperatorCacheMode\(\s*state.mixedSolverValue.executionBudgets.w\)")
        self.assertIn("state.femAlgorithmicTangentCache", cycle)
        self.assertIn("state.femAlgorithmicTangentCacheValid", cycle)
        cohesive = section(
            RUNTIME,
            "const auto encodeAddCohesiveOperator = [&](",
            "const auto encodeFEMHumanAttachmentKinematics",
        )
        self.assertIn("if (state.cohesiveFacesPaged)", cohesive)
        self.assertIn(
            'dispatchThreads("nm_fem_apply_cohesive_operator_accumulator"',
            cohesive,
        )
        self.assertIn(
            'dispatchThreads("nm_fgmres_add_cohesive_operator"',
            cohesive,
        )
        true_residual_cohesive = cycle.index(
            "encodeAddCohesiveOperator(trueResidualOperatorMicro,"
        )
        self.assertLess(
            true_residual_cohesive,
            cycle.index('"nm_fgmres_measure_true_residual"'),
        )
        self.assertIn("bindPrimalContactArguments(12u)", cycle)
        self.assertLess(certification.index('"nm_fgmres_apply_free_rigid"'),
                        certification.index('"nm_fgmres_measure_true_residual"'))
        self.assertLess(certification.index('"nm_fgmres_apply_primal_rigid_contacts"'),
                        certification.index('"nm_fgmres_measure_true_residual"'))
        self.assertIn("allowConvergedTrueResidualOperator = 1u", cycle)
        self.assertIn("[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]", cycle)
        # The admitted free-BOX path uses the same candidate, contact argument
        # buffer, point Jacobians, and six-row operator arena as production.

    def test_operator_prefix_icb_preserves_fem_preconditioner_dependencies(self):
        producer = section(
            RUNTIME,
            "if (buildFEMPreconditionerThisCycle) {",
            "if (state.vascularValue.layout.cavities.x != 0u",
        )
        self.assertIn('"nm_fgmres_build_preconditioner"', producer)
        self.assertNotIn("femPrefixPreconditionerICBForCurrentSolve", producer)

        graph = section(
            RUNTIME,
            "NSUInteger prefixOrdinal = 0u;",
            'auto cmd = makeCommand(prefixGraph, prefixOrdinal,\n                        "nm_fem_apply_operator_elements")',
        )
        self.assertNotIn('"nm_fgmres_build_preconditioner"', graph)
        ordered = [
            graph.index('"nm_fgmres_precondition_empty_fields"'),
            graph.index('"nm_fgmres_apply_coupled_tangent_block"'),
            graph.index('"nm_fgmres_precondition_patches"'),
            graph.index('"nm_fgmres_precondition_coarse"'),
            graph.index('"nm_fem_human_attachment_map_direction"'),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertIn("femSelector6BasePreconditionerICBForCurrentSolve", graph)
        self.assertIn("!useAcceptedZ0ForCurrentRoot", RUNTIME)
        self.assertIn("!candidate->femHuman6SchurCondensedFGMRESEnabled", RUNTIME)
        self.assertIn("!candidate->femHuman6GraphSchwarzEnabled", RUNTIME)

        selector0 = section(
            RUNTIME,
            "const bool selector0PreconditionerICBEligible =",
            "const bool selector6BasePreconditionerICBEligible =",
        )
        self.assertIn("executionBudgets.z == 0u", selector0)
        self.assertIn("!candidate->femHumanGlobalCoarseEnabled", selector0)
        self.assertNotIn("vascularValue.layout.cavities", selector0)
        selector6 = section(
            RUNTIME,
            "const bool selector6BasePreconditionerICBEligible =",
            "const bool prefixPreconditionerICBEligible =",
        )
        self.assertIn("executionBudgets.z == 6u", selector6)
        self.assertNotIn("vascularValue.layout.cavities", selector6)
        self.assertIn("[cmd setBarrier]", RUNTIME)

        resources = section(
            RUNTIME,
            "for (id<MTLBuffer> resource : {",
            "}) if (resource != nil) [encoder useResource:resource usage:rw];",
        )
        self.assertIn("state.fieldBoundaries", resources)
        self.assertIn("state.femRegionalNodeBlockInverse", resources)
        self.assertTrue("maxCommandCount:prefixCapacity" in RUNTIME, "prefix allocation must use its checked capacity")
        self.assertTrue("? 12u : 7u" in RUNTIME and "constexpr NSUInteger tailCapacity = 11u" in RUNTIME, "recorded prefix/tail capacities must cover the admitted commands")

        ranges = section(
            FGMRES,
            "kernel void nm_fgmres_prepare_operator_icb_ranges(",
            "kernel void nm_fgmres_cgs2_prepare_range(",
        )
        self.assertIn(
            "ranges[0] = uint2(0u, active ? prefixCommandCount : 0u);",
            ranges,
        )
        self.assertIn("states[0].diagnostics.z <= 0.5f", ranges)
        self.assertNotIn("states[0].diagnostics.z =", ranges)

    def test_every_recorded_prefix_and_tail_pipeline_supports_indirect_commands(self):
        prefix = section(
            RUNTIME,
            "NSUInteger prefixOrdinal = 0u;",
            "NSUInteger tailOrdinal = 0u;",
        )
        tail = section(
            RUNTIME,
            "NSUInteger tailOrdinal = 0u;",
            "id<MTLBuffer> rangeBuffer = state.fgmresOperatorICBExecutionRange;",
        )
        recorded_prefix = set(re.findall(
            r'makeCommand\(prefixGraph,\s*prefixOrdinal,\s*"([^"]+)"',
            prefix,
        ))
        recorded_tail = set(re.findall(
            r'makeCommand\(tailGraph,\s*tailOrdinal,\s*"([^"]+)"',
            tail,
        ))
        self.assertTrue(recorded_prefix)
        self.assertTrue(recorded_tail)

        operator_classifier = section(
            RUNTIME,
            "const auto isOperatorICBKernel =",
            "const auto isFEMPrefixPreconditionerICBKernel =",
        )
        prefix_classifier = section(
            RUNTIME,
            "const auto isFEMPrefixPreconditionerICBKernel =",
            "const char* asyncPipelineSetting=",
        )
        names = lambda source: set(re.findall(
            r'std::strcmp\(name,\s*"([^"]+)"\)\s*==\s*0', source
        ))
        indirect_operator = names(operator_classifier)
        indirect_prefix = names(prefix_classifier)
        missing_prefix = recorded_prefix - indirect_operator - indirect_prefix
        self.assertFalse(missing_prefix,
            f"prefix ICB commands missing indirect PSOs: {sorted(missing_prefix)}")
        self.assertTrue(recorded_tail.issubset(indirect_operator),
            f"tail ICB commands missing indirect PSOs: {sorted(recorded_tail - indirect_operator)}")

        record_start = RUNTIME.index("NSUInteger prefixOrdinal = 0u;")
        command_start = RUNTIME.rfind("const auto makeCommand =", 0, record_start)
        command_end = RUNTIME.index("const auto finish =", record_start)
        make_command = RUNTIME[command_start:command_end]
        capability = make_command.index("pipeline.supportIndirectCommandBuffers")
        fetch = make_command.index("indirectComputeCommandAtIndex")
        set_pipeline = make_command.index("setComputePipelineState")
        self.assertLess(capability, fetch)
        self.assertLess(capability, set_pipeline)
        self.assertIn("matter_fgmres_icb_pipeline_rejected name=%s", make_command)
        self.assertIn("supportIndirectCommandBuffers_false", make_command)

    def test_graph_solve_binds_contact_promotion_in_every_runtime_path(self):
        # These buffers were added to the kernel after its direct and cached
        # consumers existed. Native kernel fixtures cannot detect omitted
        # production encoder bindings.
        marker = r'(?:dispatchGroups256\(\s*"nm_fgmres_graph_solve_patches",\s*patchCount,|dispatchGraphPatchSolve\(\s*[^,]+,)'
        starts = [m.start() for m in re.finditer(marker, RUNTIME)]
        self.assertEqual(len(starts), 3)
        for start in starts:
            end = RUNTIME.index("});", start)
            body = RUNTIME[start:end]
            self.assertRegex(body, r"atIndex:24u")
            self.assertRegex(body, r"atIndex:25u")
        # The additional cached ICB graph kernel may use
        # the admitted 32-thread capacity-96 variant.
        self.assertIn('"nm_fgmres_graph_solve_patches_capacity96_threads32"', RUNTIME)
        start = RUNTIME.index(
            'auto graphCmd = makeCommand(prefixGraph, prefixOrdinal,')
        end = RUNTIME.index("graphCmd = makeCommand(prefixGraph, prefixOrdinal,", start + 30)
        body = RUNTIME[start:end]
        self.assertIn("bind(graphCmd, 24u,", body)
        self.assertIn("scalar(graphCmd, 25u,", body)

    def test_true_residual_controls_latch_and_preserves_unmodified_thresholds(self):
        finalize = section(FGMRES, "kernel void nm_fgmres_finalize_cycle(",
                          "kernel void nm_fgmres_restart_residual_nodes(")
        self.assertIn("deferCompletionToTrueResidual", finalize)
        self.assertIn("used > 0u", finalize)
        self.assertRegex(finalize,
                         r"state\.diagnostics\.z = \(deferCompletionToTrueResidual != 0u && used > 0u\)\s*\? 0\.0f")
        measure = section(FGMRES, "kernel void nm_fgmres_measure_true_residual(",
                          "// Add the six-node native cohesive-face directional force")
        self.assertIn("const float target = fgmresResidualTarget(", measure)
        target = section(FGMRES, "inline float fgmresResidualTarget(",
                         "inline uint fgmresMatrixIndex(")
        self.assertIn("clamp(forcing, forcingFloor, 0.5f) * initialNorm", target)
        self.assertIn("min(relativeTarget, vascularResidualTolerance)", target)
        self.assertIn("vascularUnknownCount == 0u", target)
        self.assertIn("solver.residualTolerances.x", measure)
        self.assertIn("state.diagnostics.z = trueNorm == 0.0f || trueNorm <= target ||", measure)
        self.assertIn("trueNorm <= nonlinearThreshold", measure)
        self.assertIn("productionStatuses[environment] = diagnosticStatuses[environment]", measure)
        self.assertIn("dispatch.rigidGeneralizedCapacity != 6u", measure)
        self.assertIn("layout.unknownCount != expectedUnknownCount", measure)
        self.assertIn("fgmresRigidIndex(dispatch, environment, dof)", measure)
        self.assertIn("rigidResidual.x * rigidResidual.x", measure)
        self.assertIn("residual[row] = float4(value, 0.0f, 0.0f, 0.0f)", measure)

    def test_false_arnoldi_convergence_is_rejected_and_true_restart_is_used(self):
        estimate = 9.7661823e-6
        true_norm = 2.15781765e-4
        initial = 7.67543679e-4
        forcing = 0.015625
        target = forcing * initial
        self.assertLess(estimate, target)
        self.assertGreater(true_norm, target)
        complete, replaced = adjudicate_true_residual(
            arnoldi_estimate=estimate, true_norm=true_norm,
            initial_norm=initial, forcing=forcing,
            relative_tolerance=1e-7, absolute_tolerance=1e-7,
            nonlinear_scale=0.0,
            defer_completion=True, replace_residual=True)
        self.assertFalse(complete)
        self.assertTrue(replaced)
        # The same observation-only mode does not alter existing FGMRES latch.
        observed_complete, observed_replaced = adjudicate_true_residual(
            arnoldi_estimate=estimate, true_norm=true_norm,
            initial_norm=initial, forcing=forcing,
            relative_tolerance=1e-7, absolute_tolerance=1e-7,
            nonlinear_scale=0.0,
            defer_completion=False, replace_residual=False)
        self.assertTrue(observed_complete)
        self.assertFalse(observed_replaced)

    def test_true_absolute_gate_and_unresolved_finite_floor(self):
        complete, _ = adjudicate_true_residual(
            arnoldi_estimate=1e-9, true_norm=8e-8, initial_norm=1e-3,
            forcing=1e-2, relative_tolerance=1e-7,
            absolute_tolerance=1e-7, defer_completion=True,
            nonlinear_scale=0.0,
            replace_residual=True)
        self.assertTrue(complete)
        complete, _ = adjudicate_true_residual(
            arnoldi_estimate=1e-9, true_norm=1.19929e-5,
            initial_norm=7.67544e-4, forcing=0.015625,
            relative_tolerance=1e-7, absolute_tolerance=1e-7,
            nonlinear_scale=100.0,
            defer_completion=True, replace_residual=True)
        self.assertFalse(complete)

    def test_nonfinite_residual_is_not_certified(self):
        with self.assertRaises(ValueError):
            adjudicate_true_residual(
                arnoldi_estimate=0.0, true_norm=float("inf"),
                initial_norm=1.0, forcing=0.1, relative_tolerance=1e-7,
                absolute_tolerance=1e-7, nonlinear_scale=1.0,
                defer_completion=True, replace_residual=True)

    def test_unhealthy_diagnostic_status_has_single_lane_writer(self):
        measure = section(FGMRES, "kernel void nm_fgmres_measure_true_residual(",
                          "// Add the six-node native cohesive-face directional force")
        early = section(measure, "if (!environmentHealthy(diagnosticStatuses, environment)) {",
                        "const uint totalNodes")
        self.assertRegex(early,
                         r"if \(lane == 0u && propagateFailure != 0u &&")
        self.assertEqual(early.count("productionStatuses[environment] ="), 1)

    def test_native_kernel_fixture_covers_state_machine_and_failure_rollback(self):
        for kernel in (
            "nm_fgmres_finalize_cycle",
            "nm_fgmres_accumulate",
            "nm_fgmres_accumulate_fields",
            "nm_fgmres_measure_true_residual",
        ):
            self.assertIn(f"numi_matter_metal::{kernel}", NATIVE_CHECK)
        self.assertIn("production accumulation did not form the authored correction",
                      NATIVE_CHECK)
        self.assertIn("false Arnoldi convergence", NATIVE_CHECK)
        self.assertIn("same-operator true residual did not replace", NATIVE_CHECK)
        self.assertIn("nonfinite action did not propagate", NATIVE_CHECK)
        self.assertIn("partially replaced the restart residual", NATIVE_CHECK)
        self.assertIn("overwrote an existing production failure", NATIVE_CHECK)
        self.assertIn("full_FEM_operator_qualification=false", NATIVE_CHECK)


    def test_true_residual_refined_pure_arena_keeps_legacy_coupled_bounds(self):
        arena = (HERE.parent / "include" / "numi" / "matter" / "detail" /
                 "true_residual_arena_plan.hpp").read_text()
        for token in ("n>(extendedPureFEM?32768u:8192u)", "t>(extendedPureFEM?131072u:32768u)",
                      "c>4096u", "e>4u",
                      "sizeof(NMMatterStatusGPU)",
                      "sizeof(NMFEMElementVectorGPU)",
                      "sizeof(NMCohesiveFaceVectorGPU)",
                      "maximumAggregateBytes", "maxBufferLength",
                      "femTrueResidualCheckedMultiply",
                      "femTrueResidualCheckedAdd"):
            self.assertIn(token.replace(" ", ""), arena.replace(" ", ""))
        runtime = RUNTIME
        self.assertIn("femNewtonTraceScope(false, 4096u)", runtime)
        self.assertIn("femNewtonTraceScope(\n                    state.dispatch.rigidGeneralizedCapacity == 6u,\n                    extendedPureArena ? 32768u : 8192u)", runtime)
        allocations = section(runtime, "const detail::FEMTrueResidualArenaShape trueResidualArenaShape{",
                             "candidate->femConstraintReactions =")
        for name in ("femTrueResidualRightHandSide", "femTrueResidualMetrics",
                     "femTrueResidualStatuses", "femTrueResidualElementOperator",
                     "femTrueResidualCohesiveOperator", "femTrueResidualOperatorValue"):
            self.assertIn(name, allocations)
        self.assertIn("trueResidualResidentAdditionFits", allocations)
        self.assertIn("aggregateBytes", allocations)

    def test_free_box_true_residual_proxy_scope_keeps_bounds_and_composition_exact(self):
        runtime = RUNTIME
        scope = section(runtime, "const auto femNewtonTraceScope =",
                        "const auto femTrueResidualCapabilityFailure =")
        for token in ("state.dispatch.environmentCount != 1u",
                      "state.dispatch.rigidGeneralizedCapacity != 6u",
                      "state.dispatch.rigidQCapacity != 0u",
                      "state.requiresCoupledCandidate",
                      "NM_MATTER_CONTACT",
                      "state.dispatch.femNodeCount > 4096u",
                      "state.fgmresLayout.supportContactCount != 0u",
                      "state.fgmresLayout.vascularUnknownCount != 0u",
                      "state.dispatch.fieldBoundaryCount != 0u",
                      "state.fgmresLayout.unknownCount != expectedUnknownCount",
                      "state.dispatch.rigidProxyCount !=",
                      "state.rigidProxyLayout.size()",
                      "state.dispatch.contactPairCount !=",
                      "state.contactPairLayout.size()",
                      "validateFreeBoxCapsuleProxyScope(proxyScope)"):
            self.assertIn(token, scope if token in scope else runtime)
        capability = section(runtime, "const auto femTrueResidualCapabilityFailure =",
                             "const auto femResidualArmijoScope =")
        self.assertIn("femNewtonTraceScope(\n                    state.dispatch.rigidGeneralizedCapacity == 6u,\n                    extendedPureArena ? 32768u : 8192u)",
                      capability)
        helper = (MATTER / "include" / "numi" / "matter" / "detail" /
                  "free_box_true_residual_proxy_admission.hpp").read_text()
        for token in ("in.proxyCount==0u||in.proxyCount>16u",
                      "p.flags==NM_RIGID_PRESCRIBED_TRANSLATION",
                      "p.shapeKind==NM_RIGID_CAPSULE",
                      "p.shapeKind==NM_RIGID_BOX&&prescribed",
                      "p.flags==NM_RIGID_DYNAMIC",
                      "p.generalizedFreeBodyIndex!=0u",
                      "freeBoxes!=1u",
                      "kFreeBoxTrueResidualMaxPrescribedBodies=8u",
                      "prescribedBoxes!=0u&&!prescribedBoxPair",
                      "p.objectIndex!=0u",
                      "p.continuumNode>=in.femNodeCount",
                      "p.rigidProxy>=in.proxyCount",
                      "!freeBoxPair||(capsules!=0u&&!capsulePair)"):
            self.assertIn(token, "".join(helper.split()))
        self.assertIn("NM_RIGID_CAPSULE", helper)
        self.assertNotIn("NM_RIGID_ARTICULATED", helper)

    def test_proxy_admission_checker_has_supported_and_fail_closed_cases(self):
        checker = (HERE / "cardboard_free_box_true_residual_proxy_admission_check.cpp").read_text()
        for token in ("legacy single dynamic BOX must admit without capsule endpoint state",
                      "single BOX still requires current body state",
                      "single BOX pairs alone must admit",
                      "capsule without BOX must reject",
                      "valid capsule plus one free box must admit",
                      "authored 12-proxy eight-prescribed-body shipping role map must admit",
                      "missing prescribed BOX pair class must reject",
                      "prescribed shipping boxes require endpoint state",
                      "unmapped shipping proxy index must reject",
                      "ninth distinct prescribed tool body must reject",
                      "more than one dynamic free BOX must reject",
                      "box-only pairs must reject", "capsule-only pairs must reject",
                      "dynamic capsule must reject", "articulated BOX must reject",
                      "adaptive capsule must reject", "second BOX must reject",
                      "sphere proxy must reject",
                      "proxy count above full-contact bound must reject"):
            self.assertIn(token, checker)

    def test_native_box_fixture_uses_both_contact_blocks_and_tail_controls(self):
        coupled = (HERE / "cardboard_coupled_free_box_operator_check.mm").read_text()
        for required in (
            "nm_fgmres_measure_true_residual",
            "fixture.nativeAction(center, mixedDirection)",
            "active BOX contact did not apply the rigid-side direction to FEM rows",
            "active BOX contact did not apply the FEM-side direction to rigid rows",
            "rigidGeneralizedCapacity == 6u",
            "layout.unknownCount = 2u * dispatch.environmentCount * dispatch.femNodeCount + 6u",
            "omittedTailNorm2",
            "wrongSignNorm2",
            "six scalar BOX rows",
            "restart_tail_replaced=true",
            "zero full-layout residual did not meet the unchanged gate",
            "physical_transactions=0",
        ):
            self.assertIn(required, coupled)
        for required in ("checkPrescribedProxyRows",
                         "prescribed BOX",
                         "prescribed capsule",
                         "shipping_proxy_role_fixture",
                         "prescribed_contact_blocks_altered_generalized_rows=false"):
            self.assertIn(required, coupled)

if __name__ == "__main__":
    unittest.main(verbosity=2)
