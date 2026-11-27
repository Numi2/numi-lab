#pragma once

// Match the fused Metal argument buffer to the existing task and world arenas.
// Keep the arguments here so host binding and shader indexing share one list.
#define MR_RESIDENT_TASK_ARGUMENTS(X) \
    X(0, const MRTaskDispatchGPU, dispatch, kTaskDispatch) \
    X(1, const MRTaskProgramHeaderGPU, program, kTaskProgramHeader) \
    X(2, const uchar, arena, kTaskProgramArena) \
    X(3, const MRMetalWorldDispatchGPU, world, kWorldDispatch) \
    X(4, const MRMetalWorldContactDispatchGPU, contactDispatch, kContactDispatch) \
    X(5, MRTaskEvidenceStateGPU, evidence, kTaskEvidenceState) \
    X(6, uint, resetMasks, kResetMasks) \
    X(7, float, q, qState) \
    X(8, float, v, vState) \
    X(9, const MRBodyStateGPU, initialScene, kInitialSceneBodies) \
    X(10, float, criticHistory, kTaskCriticHistory) \
    X(11, MRBodyStateGPU, scene, sceneState) \
    X(12, const float, defaultQ, kTaskDefaultQ) \
    X(13, MRTaskStateGPU, state, kTaskState) \
    X(14, float, actionHistory, kTaskActionHistory) \
    X(15, float, actorHistory, kTaskActorHistory) \
    X(16, float, cleanHistory, kTaskCleanHistory) \
    X(17, float, actor, kTaskActorObservations) \
    X(18, float, critic, kTaskCriticObservations) \
    X(19, const float, actions, kTaskActions) \
    X(20, float, effort, kWorkingEffort) \
    X(21, MRBodyStateGPU, bodies, kCurrentBodies) \
    X(22, MRContactConstraintGPU, contacts, kContacts) \
    X(23, MRMetalWorldContactStatusGPU, contactStatus, kContactStatuses) \
    X(24, MRMetalWorldStatusGPU, status, kEnvironmentStatuses) \
    X(25, const MRDofPropertiesGPU, dofs, kDofs) \
    X(26, MRTaskTransitionGPU, transitions, kTaskTransitions) \
    X(27, float, acceleration, kCandidateAcceleration) \
    X(28, float, observations, kObservations) \
    X(29, float, accelerations, kAccelerationTrajectory) \
    X(30, MRMetalWorldStatusGPU, statuses, kPublicStatuses) \
    X(31, MRMetalWorldContactStatusGPU, contactStatuses, kPublicContactStatuses)
