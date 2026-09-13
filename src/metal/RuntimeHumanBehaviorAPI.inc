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
           runtime->terminalQuarantine||runtime->publishedOnce||runtime->lastAttemptedControlStep!=0||
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
