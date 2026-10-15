#!/usr/bin/env python3
"""Source-bound contract checks for complete selector 5/6 runtime snapshots."""
from pathlib import Path
import hashlib

BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-hybrid-status-includes-129')
PKG = Path(__file__).resolve().parent
CAND = PKG / 'candidate/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm'
EXPECTED_BASE = '10fdbe2f68d4f31ee767ca3c949d5bee8c7834fe82b73f466d3460827c2fc461'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_path = BASE / 'matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm'
source = source_path.read_text()
candidate = CAND.read_text()
assert sha(source_path) == EXPECTED_BASE
assert sha(CAND) != EXPECTED_BASE
assert 'selector_runtime_outcome label=' in candidate
assert 'selector_runtime_snapshot_arrays label=' in candidate
assert 'selector_runtime_snapshot_metadata label=' in candidate
for token in (
    'statuses_raw_hex=', 'certificates_raw_hex=', 'fem_nodes_raw_hex=',
    'selector_runtime_status label=', 'selector_runtime_certificate label=',
    'resident_bytes_reported=', 'message_hex=', 'error_hex=',
):
    assert token in candidate, token

# Every vector in RuntimeStateSnapshot is emitted with an element count and
# exact raw bytes. Keep this list tied to the frozen 129 ABI.
fields = {
    'sutureProxyEdges': 'suture_proxy_edges', 'particles': 'particles',
    'femNodes': 'fem_nodes', 'femFields': 'fem_fields', 'vascularState': 'vascular_state',
    'vascularClock': 'vascular_clock', 'femTopologyNodes': 'fem_topology_nodes',
    'femTopologyTetrahedra': 'fem_topology_tetrahedra', 'cohesiveFaces': 'cohesive_faces',
    'punctureChannels': 'puncture_channels', 'topologyStates': 'topology_states',
    'statuses': 'statuses', 'solverCertificates': 'solver_certificates',
    'mpmActiveNodeIndices': 'mpm_active_node_indices', 'mpmNodeToActive': 'mpm_node_to_active',
    'mpmActiveNodeCounts': 'mpm_active_node_counts',
    'rigidGeneralizedCandidate': 'rigid_generalized_candidate', 'learnedWeights': 'learned_weights',
    'adaptive': 'adaptive', 'schedulers': 'schedulers', 'reactions': 'reactions',
    'rigidStates': 'rigid_states', 'contactSamples': 'contact_samples',
    'deformableContactFailures': 'deformable_contact_failures', 'contactHistories': 'contact_histories',
    'humanSupportHistories': 'human_support_histories',
    'humanSupportConsequences': 'human_support_consequences',
    'deformableContactHistories': 'deformable_contact_histories',
    'particleMaterialState': 'particle_material_state', 'femMaterialState': 'fem_material_state',
    'identification': 'identification', 'environmentParameters': 'environment_parameters',
    'femRegionalGalerkinLinearizationNodes': 'regional_linearization_nodes',
    'femRegionalGalerkinLinearizationMaterialState': 'regional_linearization_material_state',
    'femRegionalGalerkinLinearizationSchedulerExponent': 'regional_linearization_scheduler',
    'femRegionalGalerkinLinearizationLocalFHigh': 'regional_F_high',
    'femRegionalGalerkinLinearizationLocalFLow': 'regional_F_low',
    'femRegionalGalerkinBasisP': 'regional_basis_P',
    'femRegionalGalerkinEffectiveBasisP': 'regional_effective_basis_P',
    'femRegionalGalerkinOperatorAP': 'regional_operator_AP',
    'femRegionalGalerkinPivotedRAP': 'regional_pivoted_RAP',
    'femRegionalGalerkinPivots': 'regional_pivots',
    'femRegionalGalerkinFactorValid': 'regional_factor_valid',
    'femRegionalGalerkinNodeBlockInverse': 'regional_node_block_inverse',
    'femRegionalRestartGenerationsByEnvironment': 'restart_generations_by_environment',
    'femRegionalRestartAttemptCountsByEnvironment': 'restart_attempt_counts_by_environment',
    'femRegionalRestartTargetScalesByEnvironment': 'restart_target_scales_by_environment',
    'femRegionalRestartRawNormSquaredByEnvironment': 'restart_raw_norm_squared_by_environment',
    'femRegionalRestartModeScalesByEnvironment': 'restart_mode_scales_by_environment',
    'femRegionalRestartModeValidByEnvironment': 'restart_mode_valid_by_environment',
    'femRegionalRestartAugmentedFactorValidByEnvironment': 'restart_augmented_factor_valid_by_environment',
    'femRegionalRestartModesQByEnvironment': 'restart_modes_Q_by_environment',
    'femRegionalRestartOperatorAQByEnvironment': 'restart_operator_AQ_by_environment',
    'femRegionalRestartPivotedRAPByEnvironment': 'restart_pivoted_RAP_by_environment',
    'femRegionalRestartPivotsByEnvironment': 'restart_pivots_by_environment',
    'femRegionalRestartModeQ': 'restart_mode_Q',
    'femRegionalRestartOperatorAQ': 'restart_operator_AQ',
    'femRegionalRestartPivotedRAP': 'restart_pivoted_RAP',
    'femRegionalRestartPivots': 'restart_pivots',
}
for member, name in fields.items():
    assert f'EMIT_SNAPSHOT_VECTOR({name}, {member});' in candidate, (member, name)

for scalar in (
    'sutureProxyBindingRevision', 'coupledTimestepMultiplier', 'coupledTimestepDivisor',
    'fgmresIterationBudgetOverride', 'newtonIterationBudgetOverride',
    'identificationGeneration', 'identificationCheckpoint', 'identificationAdvanced',
    'allocationGeneration', 'learnedWeightRevision', 'materialStateStride',
    'femRegionalGalerkinDiagnosticsAvailable', 'femRegionalGalerkinModeCount',
    'femRegionalGalerkinDiagnosticReadbackBytes',
    'femRegionalRestartAugmentationDiagnosticsAvailable', 'femRegionalRestartBaseModeCount',
    'femRegionalRestartAugmentedModeCount', 'femRegionalRestartDiagnosticEnvironmentCount',
    'femRegionalRestartDiagnosticNodesPerEnvironment', 'femRegionalRestartDiagnosticModeStride',
):
    assert f'snapshot.{scalar}' in candidate, scalar

start = candidate.index('void checkSelector5And6AcceptanceParity(')
end = candidate.index('\n\n} // namespace', start)
parity = candidate[start:end]
positions = [
    parity.index('fiveState = runOneAcceptedRelease'),
    parity.index('sixState = runOneAcceptedRelease'),
    parity.index('printSelectorRuntimeOutcome("selector5"'),
    parity.index('printSelectorRuntimeOutcome("selector6"'),
    parity.index('need(acceptedOutcome(fiveState'),
]
assert positions == sorted(positions), positions
assert parity.count('catch (const std::exception& error)') >= 2
for token in (
    'state.statuses[0].code == NM_STATUS_SUCCESS',
    'state.solverCertificates[0].validity.w > 0.5f',
    'relativeCorrection > 0.0f', 'restAndFixed.w == 0.0f',
    'restAndFixed.w == 1.0f', 'loadedNodeMotion > 1.0e-8',
):
    assert token in candidate, token
assert 'runOneAcceptedRelease' in source and 'selector_runtime_outcome label=' not in source
print(f'PASS: frozen base; {len(fields)} snapshot vectors and required scalars emitted; both selector outcomes precede unchanged gates')
