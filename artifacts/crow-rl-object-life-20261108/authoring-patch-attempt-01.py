from pathlib import Path
r=Path(__file__).parent
p=r/'c_api.cpp';s=p.read_text()
s=s.replace('checkpoint::require(h.visualRuntime && h.visualRuntime->retinal && h.resourceRuntime &&','const bool resourceFreeObject = !h.resourceRuntime && h.taskId=="american_crow_object_push_v14";\n    checkpoint::require(h.visualRuntime && h.visualRuntime->retinal && (h.resourceRuntime || resourceFreeObject) &&')
s=s.replace('"Life checkpoint requires the native v13 retinal feeding runtime and an exact policy"','"Life checkpoint requires native retinal feeding or object push and an exact policy"')
s=s.replace('h.stepConfig.basePolicyProgram.fingerprint(),h.resourceRuntime->fingerprint(),h.stepConfig.taskSeed,','h.stepConfig.basePolicyProgram.fingerprint(),h.resourceRuntime ? h.resourceRuntime->fingerprint() : std::uint64_t{0},h.stepConfig.taskSeed,')
s=s.replace('const auto retina=handle->visualRuntime->retinal->checkpointState();const auto resource=handle->resourceRuntime->checkpointState();','const auto retina=handle->visualRuntime->retinal->checkpointState();\n        const auto resource=handle->resourceRuntime ? handle->resourceRuntime->checkpointState() : checkpoint::Bytes{};')
s=s.replace('const auto world=reader.blob(),retina=reader.blob(),resource=reader.blob();reader.end();','const auto world=reader.blob(),retina=reader.blob(),resource=reader.blob();reader.end();\n        checkpoint::require(bool(handle->resourceRuntime)==!resource.empty(),"Life checkpoint resource component contract mismatch");')
s=s.replace('auto food=std::make_unique<FiniteResourceContext>(handle->resourceRuntime->deviceView().config);','std::unique_ptr<FiniteResourceContext> food;\n        if(handle->resourceRuntime) food=std::make_unique<FiniteResourceContext>(handle->resourceRuntime->deviceView().config);')
s=s.replace('visual->retinal->restoreState(retina);food->restoreState(resource);','visual->retinal->restoreState(retina);\n        if(food) food->restoreState(resource);')
s=s.replace('handle->stepConfig.devicePhysicsProgram=handle->resourceRuntime->deviceProgram();\n        handle->submittedControlSteps=steps;', 'handle->stepConfig.devicePhysicsProgram=handle->resourceRuntime ? handle->resourceRuntime->deviceProgram() : MetalWorldDevicePhysicsProgram{};\n        handle->submittedControlSteps=steps;')
mark='int mr_task_rollout_copy_resource_accounting(\n'
assert s.count(mark)==1
new='''int mr_task_rollout_copy_lifetime_ticks(
    const MRTaskRolloutHandle* handle, uint64_t* output, const size_t output_count
) {
    if (!requireTaskRolloutHandle(handle)) return -1;
    return translateErrors([&] {
        using namespace metalrobo;
        checkpoint::require(output && output_count==handle->environmentCount,
            "Lifetime tick output has the wrong size");
        checkpoint::require(!handle->rejectedRetinalBatch &&
            (handle->residentState.valid() || !handle->pendingLifeWorld.empty()) &&
            handle->submittedControlSteps &&
            handle->submittedControlSteps<=std::numeric_limits<std::uint64_t>::max()/handle->environmentCount &&
            handle->completedEnvironmentSteps==handle->submittedControlSteps*handle->environmentCount,
            "Lifetime ticks require a successful accepted or restored native life");
        std::vector<std::uint64_t> ticks(output_count);
        if(handle->resourceRuntime) {
            const auto states=handle->resourceRuntime->snapshot();
            checkpoint::require(states.size()==output_count,"Resource lifetime lane count mismatch");
            for(std::size_t lane=0;lane<states.size();++lane)
                ticks[lane]=(std::uint64_t(states[lane].clock.y)<<32u)|states[lane].clock.x;
        } else {
            // Retinal batches fail closed: any rejected lane invalidates the
            // handle before these counters publish. The resource-free object
            // task therefore has one successful stream clock shared by all
            // lanes, preserved across episode resets and exact life restore.
            checkpoint::require(handle->taskId=="american_crow_object_push_v14" &&
                handle->visualRuntime && handle->visualRuntime->retinal,
                "Resource-free lifetime clock is not admitted for this task");
            std::fill(ticks.begin(),ticks.end(),handle->submittedControlSteps);
        }
        checkpoint::require(std::all_of(ticks.begin(),ticks.end(),[&](auto tick){
            return tick==handle->submittedControlSteps;
        }),"Native lifetime clocks disagree with accepted host accounting");
        std::copy(ticks.begin(),ticks.end(),output);
    });
}

'''
s=s.replace(mark,new+mark);p.write_text(s)
p=r/'c_api.h';s=p.read_text();mark='// Quiescent accepted resource ledger, one record per environment.';assert mark in s
s=s.replace(mark,'''// Accepted lifetime ticks, independent of episode resets. Supports native
// finite-resource tasks and the resource-free retinal object-push task, after
// a successful submission or atomic life restore. Rejected/unadvanced handles
// fail without modifying output. This is publication-only evidence readback.
MR_API int mr_task_rollout_copy_lifetime_ticks(
    const MRTaskRolloutHandle* handle,
    uint64_t* output,
    size_t output_count
);
'''+mark);p.write_text(s)
p=r/'MetalRoboTaskRollout.swift';s=p.read_text();mark='    public func finiteResourceAccounting() throws -> [MRTaskResourceAccountingC] {';assert mark in s
s=s.replace(mark,'''    public func acceptedLifetimeControlTicks() throws -> [UInt64] {
        var ticks = [UInt64](repeating: 0, count: layout.environmentCount)
        let status = ticks.withUnsafeMutableBufferPointer {
            mr_task_rollout_copy_lifetime_ticks(handle, $0.baseAddress, $0.count)
        }
        guard status == 0 else {
            throw MetalRoboTaskRolloutError.native(Self.lastError())
        }
        return ticks
    }

'''+mark);p.write_text(s)
p=r/'TrainingLifeCheckpoint.swift';s=p.read_text();s=s.replace('let ticks = try context.finiteResourceAccounting().map { $0.accepted_control_ticks }','let ticks = try context.acceptedLifetimeControlTicks()');p.write_text(s)
p=r/'task_train.swift';s=p.read_text();s=s.replace('guard birdFlowAmericanCrowJourney, birdFlowJourneyVariant == .v13RetinalFeeding,','guard birdFlowAmericanCrowJourney,\n                  (birdFlowJourneyVariant == .v13RetinalFeeding || birdFlowJourneyVariant == .v14ObjectPush),');s=s.replace('Life training requires native v13 without teacher','Life training requires native v13/v14 without teacher')
old='''                try context.restoreLifeCheckpoint(from: URL(fileURLWithPath: path).appendingPathComponent("life.native"))''';assert old in s;s=s.replace(old,old+'''
                let nativeTicks = try context.acceptedLifetimeControlTicks()
                try MetalRoboTrainingLifeCheckpoint.require(nativeTicks == options.trainingResume?.acceptedControlTicks,
                    "Training checkpoint accepted lifetime clock disagrees with native state.")''')
# Independent cue factory remains exclusive to the food/cue task.
s=s.replace('guard checkpointLife, visualObservationConfig != nil,','guard birdFlowJourneyVariant == .v13RetinalFeeding, checkpointLife, visualObservationConfig != nil,')
mark='''            guard JSONSerialization.isValidJSONObject(output) else {''';assert s.count(mark)==1
s=s.replace(mark,'''            if options.birdFlowAmericanCrowJourney &&
               (options.birdFlowJourneyVariant == .v13RetinalFeeding || options.birdFlowJourneyVariant == .v14ObjectPush) {
                output["accepted_lifetime_control_ticks_by_environment"] = try context.acceptedLifetimeControlTicks()
            }
'''+mark);p.write_text(s)
p=r/'task_rollout.swift';s=p.read_text();s=s.replace('birdFlowJourneyVariant == .v13RetinalFeeding,\n                  policyPack != nil,', '(birdFlowJourneyVariant == .v13RetinalFeeding || birdFlowJourneyVariant == .v14ObjectPush),\n                  policyPack != nil,');s=s.replace('Life checkpoints require v13,','Life checkpoints require v13/v14,')
mark='''            output["life_checkpoint_restored"] = options.resumeLifeCheckpoint != nil''';assert mark in s;s=s.replace(mark,'''            if options.birdFlowAmericanCrowJourney &&
               (options.birdFlowJourneyVariant == .v13RetinalFeeding || options.birdFlowJourneyVariant == .v14ObjectPush) {
                output["accepted_lifetime_control_ticks_by_environment"] = try context.acceptedLifetimeControlTicks()
            }
'''+mark);p.write_text(s)
