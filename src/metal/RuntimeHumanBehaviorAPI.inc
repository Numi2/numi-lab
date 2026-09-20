// Included once by NumanXRuntimeV1.mm after its private owner types/helpers.
// Privileged native diagnostic API; no Brain observation or physics mutation.
extern "C" bool mrnx_bridge_v1_runtime_behavior_attach(
    mrnx_runtime_v1* handle,const char* path,const char* expectedSHA,
    std::uint64_t initialCommittedTimestampNanoseconds) {
    if(handle==nullptr||handle->state==nullptr||path==nullptr||expectedSHA==nullptr)return false;
    auto runtime=handle->state;
    try {
        const std::lock_guard lock(runtime->mutex);
        if(runtime->behavior!=nullptr||runtime->active!=nullptr||runtime->beginInProgress||
           runtime->terminalQuarantine||runtime->behaviorTraceFinalized||
           runtime->publishedOnce||runtime->lastAttemptedControlStep!=0||
           (!runtime->exactClock && initialCommittedTimestampNanoseconds%1000ull!=0))return false;
        const std::string expected(expectedSHA);
        if(expected.size()!=64||!std::all_of(expected.begin(),expected.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))return false;
        const auto image=loadImmutablePayload(path,"Human behavior metric program");
        if(image.bytes.size()>65536)return false;
        std::array<unsigned char,32> hash{};CC_SHA256(image.bytes.data(),static_cast<CC_LONG>(image.bytes.size()),hash.data());
        constexpr char hex[]="0123456789abcdef";std::string actual;actual.reserve(64);
        for(auto x:hash){actual.push_back(hex[x>>4]);actual.push_back(hex[x&15]);}if(actual!=expected)return false;
        metalrobo::HumanBehaviorProgramSource source;metalrobo::CompiledHumanBehaviorProgram compiled;std::string error;
        if(!metalrobo::decodeHumanBehaviorProgram(std::as_bytes(std::span(image.bytes.data(),image.bytes.size())),source,error)||
           !metalrobo::compileHumanBehaviorProgram(source,runtime->behaviorBinding,compiled,error)){runtime->behaviorError=error;return false;}
        auto telemetry=std::make_unique<metalrobo::MetalHumanBehaviorTelemetry>((__bridge void*)runtime->device,
            compiled,runtime->behaviorMetallibPath,1,0,initialCommittedTimestampNanoseconds);
        runtime->behaviorInitialTimestampNanoseconds=initialCommittedTimestampNanoseconds;
        runtime->behaviorMetricSHA256=actual;
        runtime->behavior=std::move(telemetry);runtime->behaviorError.clear();return true;
    }catch(const std::exception& e){try{const std::lock_guard lock(runtime->mutex);runtime->behaviorError=e.what();}catch(...){}return false;}catch(...){return false;}
}

extern "C" std::size_t mrnx_bridge_v1_runtime_behavior_flush_json(
    mrnx_runtime_v1* handle,char* output,std::size_t capacity) {
    if(handle==nullptr||handle->state==nullptr)return 0;auto runtime=handle->state;
    try {
        const std::lock_guard lock(runtime->mutex);
        if(runtime->behavior==nullptr||runtime->active!=nullptr||runtime->beginInProgress||runtime->terminalQuarantine||!runtime->behaviorError.empty())return 0;
        const auto encode=[](void* raw,void* commandBuffer)noexcept{
            auto* r=static_cast<RuntimeState*>(raw);return r->behavior->encodeFlush(commandBuffer,r->behaviorError);};
        if(!runtime->owner->flushReadOnlyObserver(runtime.get(),encode,runtime->behaviorError))return 0;
        const auto snapshot=runtime->behavior->snapshot();if(snapshot.reduction.size()!=1||snapshot.reduction[0].status!=0)return 0;
        const auto& r=snapshot.reduction[0];const auto& p=snapshot.release[0];
        if(r.publicationSerial!=p.publicationSerial||r.publicationSerial!=runtime->behavior->completedAttempts())return 0;
        NSMutableArray* audit=[NSMutableArray arrayWithCapacity:8];for(auto x:r.auditViolations)[audit addObject:@(x)];
        auto pair=[](float a,float b){return @[@(a),@(b)];};
        NSDictionary* document=@{
            @"schema":@"numi.human.accepted-metric-snapshot.v1",
            @"metric_program_sha256":[NSString stringWithUTF8String:runtime->behaviorMetricSHA256.c_str()],
            @"program_fingerprint":@(r.programFingerprint),@"initial_physics_generation":@(r.initialPhysicsGeneration),
            @"initial_timestamp_ns":@(r.initialTimestampNanoseconds),@"end_ns":@(r.endNanoseconds),
            @"step_ns":@(runtime->behaviorBinding.timestepNanoseconds),
            @"accepted_root_count":@(r.acceptedRootCount),@"rejected_attempt_count":@(r.rejectedAttemptCount),
            @"completed_attempt_count":@(r.publicationSerial),@"metric_sample_count":@(r.metricSampleCount),
            @"audit_covered_root_count":@(r.auditCoveredRootCount),@"audit_covered_attempt_count":@(r.auditCoveredAttemptCount),
            @"posture_violation_count":@(r.postureViolationCount),@"settled_suffix_steps":@(r.settledSuffixSteps),
            @"speed_error_sample_count":@(r.speedErrorSampleCount),@"audit_violation_counts":audit,
            @"minimum_root_height_m_hi_lo":pair(r.extremaHigh.x,r.extremaLow.x),
            @"maximum_trunk_tilt_rad_hi_lo":pair(r.extremaHigh.y,r.extremaLow.y),
            @"maximum_planar_speed_mps_hi_lo":pair(r.extremaHigh.z,r.extremaLow.z),
            @"speed_squared_error_sum_hi_lo":pair(r.extremaHigh.w,r.extremaLow.w),
            @"last_transaction_fingerprint":@(r.lastTransactionFingerprint),@"last_joint_fence_fingerprint":@(r.lastJointFenceFingerprint),
            @"initial_posture_valid":r.initialPostureValid==2u?[NSNull null]:@(r.initialPostureValid!=0u),
            @"initial_settled":r.initialSettled==2u?[NSNull null]:@(r.initialSettled!=0u),
            @"forbidden_contact_coverage":@"unavailable",@"native_audit_coverage":@"unavailable",
            @"generic_taskpack_lowering":@"source_bound_metric_program",@"full_behavior_qualified":@NO,
            @"accepted_root_proof_sha256": [NSNull null],@"finalized":@YES};
        NSError* jsonError=nil;NSData* data=[NSJSONSerialization dataWithJSONObject:document options:NSJSONWritingSortedKeys error:&jsonError];
        if(data==nil)return 0;const std::size_t required=data.length+1;
        if(output!=nullptr&&capacity>=required){std::memcpy(output,data.bytes,data.length);output[data.length]='\0';}
        return required;
    }catch(const std::exception& e){try{const std::lock_guard lock(runtime->mutex);runtime->behaviorError=e.what();}catch(...){}return 0;}catch(...){return 0;}
}

extern "C" bool mrnx_bridge_v1_runtime_behavior_trace_attach(
    mrnx_runtime_v1* handle,
    const mrnx_behavior_trace_config_v1* config) {
    if (handle == nullptr || handle->state == nullptr || config == nullptr)
        return false;
    auto runtime = handle->state;
    try {
        const std::lock_guard lock(runtime->mutex);
        if (!runtime->exactClock ||
            runtime->clockQuantumNanoseconds !=
                MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS ||
            runtime->behavior == nullptr ||
            runtime->behavior->traceAttached() ||
            runtime->behaviorTraceFinalized ||
            runtime->behaviorTraceTerminalCached ||
            runtime->active != nullptr || runtime->beginInProgress ||
            runtime->terminalQuarantine || runtime->publishedOnce ||
            runtime->lastAttemptedControlStep != 0u ||
            runtime->behaviorMetricSHA256.size() != 64u ||
            runtime->assets.sourceFingerprint == 0u ||
            runtime->matter == nullptr) {
            return false;
        }
        const std::uint64_t proofProgramFingerprint =
            runtime->matter->acceptedStateProofProgramFingerprintV2();
        if (proofProgramFingerprint == 0u ||
            proofProgramFingerprint != runtime->info.
                accepted_state_proof_program_fingerprint) {
            return false;
        }

        metalrobo::HumanBehaviorTraceBinding binding{};
        const auto nibble = [](const char character,
                               std::uint8_t& value) noexcept {
            if (character >= '0' && character <= '9') {
                value = static_cast<std::uint8_t>(character - '0');
                return true;
            }
            if (character >= 'a' && character <= 'f') {
                value = static_cast<std::uint8_t>(character - 'a' + 10);
                return true;
            }
            return false;
        };
        for (std::size_t index = 0u;
             index < binding.metricProgramSHA256.size(); ++index) {
            std::uint8_t high = 0u;
            std::uint8_t low = 0u;
            if (!nibble(runtime->behaviorMetricSHA256[index * 2u], high) ||
                !nibble(runtime->behaviorMetricSHA256[index * 2u + 1u], low)) {
                return false;
            }
            binding.metricProgramSHA256[index] =
                static_cast<std::uint8_t>((high << 4u) | low);
        }
        binding.modelSourceFingerprint = runtime->assets.sourceFingerprint;
        binding.acceptedStateProofProgramFingerprint =
            proofProgramFingerprint;
        binding.clockDomain =
            MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
        binding.clockQuantumNanoseconds =
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
        const auto initialSensorState =
            metalrobo::humanBehaviorUnpublishedSensorStateSHA256();
        const auto initialPublicationState =
            metalrobo::humanBehaviorUnpublishedPublicationStateSHA256();
        if (!metalrobo::humanBehaviorStateDigestPresent(
                initialSensorState) ||
            !metalrobo::humanBehaviorStateDigestPresent(
                initialPublicationState)) {
            runtime->behaviorTraceError =
                "initial behavior state-component digest failed";
            return false;
        }
        const bool attached = runtime->behavior->traceAttach(
            *config, binding, runtime->behaviorTraceError);
        if (attached) {
            runtime->behaviorAcceptedSensorStateSHA256 =
                initialSensorState;
            runtime->behaviorAcceptedPublicationStateSHA256 =
                initialPublicationState;
            runtime->behaviorStateComponentsInitialized = true;
            runtime->behaviorTraceError.clear();
        }
        return attached;
    } catch (const std::exception& exception) {
        try {
            const std::lock_guard lock(runtime->mutex);
            runtime->behaviorTraceError = exception.what();
        } catch (...) {}
        return false;
    } catch (...) {
        return false;
    }
}

extern "C" bool mrnx_bridge_v1_runtime_behavior_trace_drain(
    mrnx_runtime_v1* handle,
    mrnx_behavior_trace_chunk_v1* chunk,
    mrnx_behavior_trace_record_v1* records,
    const std::uint32_t recordCapacity) {
    if (handle == nullptr || handle->state == nullptr || chunk == nullptr ||
        (recordCapacity != 0u && records == nullptr)) {
        return false;
    }
    auto runtime = handle->state;
    try {
        const std::lock_guard lock(runtime->mutex);
        if (!runtime->exactClock || runtime->behavior == nullptr ||
            !runtime->behavior->traceAttached() ||
            runtime->behavior->traceFinalized() ||
            runtime->behaviorTraceFinalized ||
            runtime->behaviorTraceTerminalCached ||
            runtime->active != nullptr || runtime->beginInProgress) {
            return false;
        }
        const auto encode = [](void* raw, void* commandBuffer) noexcept {
            auto* state = static_cast<RuntimeState*>(raw);
            return state->behavior->encodeFlush(
                commandBuffer, state->behaviorError);
        };
        if (!runtime->owner->flushReadOnlyObserver(
                runtime.get(), encode, runtime->behaviorError)) {
            return false;
        }
        mrnx_behavior_trace_chunk_v1 result{};
        if (!runtime->behavior->traceDrain(
                result,
                std::span<mrnx_behavior_trace_record_v1>(
                    records, recordCapacity),
                runtime->behaviorTraceError)) {
            return false;
        }
        *chunk = result;
        runtime->behaviorTraceError.clear();
        return true;
    } catch (const std::exception& exception) {
        try {
            const std::lock_guard lock(runtime->mutex);
            runtime->behaviorTraceError = exception.what();
        } catch (...) {}
        return false;
    } catch (...) {
        return false;
    }
}

extern "C" bool
mrnx_bridge_v1_runtime_behavior_trace_state_component_drain(
    mrnx_runtime_v1* handle,
    mrnx_behavior_trace_chunk_v1* chunk,
    mrnx_behavior_trace_record_v1* records,
    mrnx_behavior_trace_state_component_record_v1* stateComponents,
    const std::uint32_t recordCapacity) {
    if (handle == nullptr || handle->state == nullptr || chunk == nullptr ||
        (recordCapacity != 0u &&
         (records == nullptr || stateComponents == nullptr))) {
        return false;
    }
    auto runtime = handle->state;
    try {
        const std::lock_guard lock(runtime->mutex);
        if (!runtime->exactClock || runtime->behavior == nullptr ||
            !runtime->behavior->traceAttached() ||
            runtime->behavior->traceFinalized() ||
            runtime->behaviorTraceFinalized ||
            runtime->behaviorTraceTerminalCached ||
            runtime->active != nullptr || runtime->beginInProgress ||
            !runtime->behaviorStateComponentsInitialized) {
            return false;
        }
        const auto encode = [](void* raw, void* commandBuffer) noexcept {
            auto* state = static_cast<RuntimeState*>(raw);
            return state->behavior->encodeFlush(
                commandBuffer, state->behaviorError);
        };
        if (!runtime->owner->flushReadOnlyObserver(
                runtime.get(), encode, runtime->behaviorError)) {
            return false;
        }
        mrnx_behavior_trace_chunk_v1 result{};
        if (!runtime->behavior->traceDrain(
                result,
                std::span<mrnx_behavior_trace_record_v1>(
                    records, recordCapacity),
                std::span<mrnx_behavior_trace_state_component_record_v1>(
                    stateComponents, recordCapacity),
                runtime->behaviorTraceError)) {
            return false;
        }
        *chunk = result;
        runtime->behaviorTraceError.clear();
        return true;
    } catch (const std::exception& exception) {
        try {
            const std::lock_guard lock(runtime->mutex);
            runtime->behaviorTraceError = exception.what();
        } catch (...) {}
        return false;
    } catch (...) {
        return false;
    }
}

extern "C" bool
mrnx_bridge_v1_runtime_copy_accepted_physical_state_digest(
    const mrnx_runtime_v1* handle,
    mrnx_behavior_physical_state_digest_v1* output) {
    if (output == nullptr || output->abi_version !=
            MRNX_BEHAVIOR_PHYSICAL_STATE_DIGEST_ABI_V1 ||
        output->struct_size != sizeof(*output)) {
        return false;
    }
    *output = {};
    if (handle == nullptr || handle->state == nullptr ||
        !handle->state->exactClock) {
        return false;
    }
    struct Read {
        RuntimeState* state;
        mrnx_behavior_physical_state_digest_v1* output;
    } read{handle->state.get(), output};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        handle->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            auto& runtime = *context.state;
            if (runtime.terminalQuarantine || !runtime.publishedOnce ||
                runtime.matter == nullptr ||
                !runtime.publishedPhysicalStateIdentity.has_value()) {
                return false;
            }
            const auto& source =
                *runtime.publishedPhysicalStateIdentity;
            if (!validRuntimePhysicalStateDigest(
                    source,
                    runtime.publishedTimestampNanoseconds,
                    runtime.publishedPhysicsGeneration,
                    runtime.matter->sourcePhysicsFingerprint(),
                    runtime.matter->deviceProgramFingerprint()) ||
                runtime.exactAggregate.publication_epoch == 0u ||
                runtime.exactAggregate.publication_epoch !=
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        runtime.domain)) {
                return false;
            }
            auto& result = *context.output;
            result.abi_version =
                MRNX_BEHAVIOR_PHYSICAL_STATE_DIGEST_ABI_V1;
            result.struct_size = sizeof(result);
            result.status = MRNX_BEHAVIOR_PHYSICAL_STATE_DIGEST_READY_V1;
            result.environment = source.environment;
            result.schema_version = source.schemaVersion;
            result.manifest_version = source.manifestVersion;
            result.source_count = source.sourceCount;
            result.publication_epoch =
                runtime.exactAggregate.publication_epoch;
            result.accepted_timestamp_nanoseconds =
                source.acceptedTimestampNanoseconds;
            result.physics_generation = source.physicsGeneration;
            result.matter_source_physics_fingerprint =
                source.matterSourcePhysicsFingerprint;
            result.matter_device_program_fingerprint =
                source.matterDeviceProgramFingerprint;
            std::memcpy(
                result.human_sha256,
                source.humanSHA256.bytes,
                sizeof(result.human_sha256));
            std::memcpy(
                result.matter_sha256,
                source.matterSHA256.bytes,
                sizeof(result.matter_sha256));
            std::memcpy(
                result.physical_sha256,
                source.physicalSHA256.bytes,
                sizeof(result.physical_sha256));
            return true;
        });
}

extern "C" bool mrnx_bridge_v1_runtime_behavior_trace_finalize(
    mrnx_runtime_v1* handle,
    const mrnx_behavior_trace_terminal_request_v1* request,
    mrnx_behavior_trace_terminal_v1* terminal) {
    if (handle == nullptr || handle->state == nullptr || request == nullptr ||
        terminal == nullptr ||
        request->abi_version != MRNX_BEHAVIOR_TRACE_ABI_V1 ||
        request->struct_size != sizeof(*request) ||
        request->reason < MRNX_BEHAVIOR_TRACE_TERMINAL_COMPLETED_V1 ||
        request->reason > MRNX_BEHAVIOR_TRACE_TERMINAL_FAILED_V1 ||
        request->reserved0 != 0u || request->reserved1 != 0u) {
        return false;
    }
    auto runtime = handle->state;
    try {
        const mrnx_behavior_trace_terminal_request_v1 copiedRequest =
            *request;
        const std::lock_guard lock(runtime->mutex);
        if (runtime->behaviorTraceTerminalCached) {
            if (std::memcmp(
                    &copiedRequest, &runtime->behaviorTraceTerminalRequest,
                    sizeof(copiedRequest)) != 0) {
                return false;
            }
            *terminal = runtime->behaviorTraceTerminal;
            return true;
        }
        if (!runtime->exactClock || runtime->behavior == nullptr ||
            !runtime->behavior->traceAttached() ||
            runtime->behavior->traceFinalized() ||
            runtime->behaviorTraceFinalized ||
            runtime->active != nullptr || runtime->beginInProgress) {
            return false;
        }

        const bool priorBehaviorFailure = !runtime->behaviorError.empty();
        const auto encode = [](void* raw, void* commandBuffer) noexcept {
            auto* state = static_cast<RuntimeState*>(raw);
            return state->behavior->encodeFlush(
                commandBuffer, state->behaviorError);
        };
        if (!runtime->owner->flushReadOnlyObserver(
                runtime.get(), encode, runtime->behaviorError)) {
            return false;
        }

        metalrobo::HumanBehaviorTraceFinalContext finalContext{};
        finalContext.quiescent = true;
        finalContext.terminalQuarantine =
            runtime->terminalQuarantine || priorBehaviorFailure ||
            !runtime->behaviorError.empty();
        finalContext.publicationEpoch =
            runtime->exactAggregate.publication_epoch;
        finalContext.physicsGeneration =
            runtime->exactAggregate.physics_generation;
        finalContext.brainGeneration =
            runtime->exactAggregate.brain_generation;
        finalContext.sensorGeneration =
            runtime->exactAggregate.sensor_generation;
        finalContext.timestampNanoseconds = runtime->publishedOnce
            ? runtime->publishedTimestampNanoseconds
            : runtime->behaviorInitialTimestampNanoseconds;
        finalContext.acceptedTokenFingerprint = runtime->exactAggregate.
            sensor_packet.accepted_physics_token_fingerprint;
        finalContext.publicationFingerprint = runtime->exactAggregate.
            publication.publication_fingerprint;

        mrnx_behavior_trace_terminal_v1 result{};
        if (!runtime->behavior->traceFinalize(
                copiedRequest, finalContext, result,
                runtime->behaviorTraceError)) {
            return false;
        }
        runtime->behaviorTraceTerminalRequest = copiedRequest;
        runtime->behaviorTraceTerminal = result;
        runtime->behaviorTraceTerminalCached = true;
        runtime->behaviorTraceFinalized = true;
        runtime->behaviorTraceError.clear();
        *terminal = result;
        return true;
    } catch (const std::exception& exception) {
        try {
            const std::lock_guard lock(runtime->mutex);
            runtime->behaviorTraceError = exception.what();
        } catch (...) {}
        return false;
    } catch (...) {
        return false;
    }
}
