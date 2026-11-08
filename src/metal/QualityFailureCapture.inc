// Bounded readback after completion. The protected kernel record never feeds
// state back into the physical solve, task, actor, or learner.
namespace {
void captureQualityFailure(const detail::MetalWorldContextState& context,
                           const MetalWorldResult& result) noexcept {
    const char* filename=std::getenv("METALROBO_QUALITY_FAILURE_CAPTURE");
    if (!filename || !*filename || !context.captureQualityFailure) return;
    const auto found=std::find_if(result.statuses.begin(),result.statuses.end(),
        [](const auto& s){return s.code==MR_STEP_DID_NOT_CONVERGE;});
    if (found==result.statuses.end()) return;
    try {@autoreleasepool {
        const auto& dispatch=context.boundQualityDispatch;
        const std::size_t bytes=mrUnifiedQualityFailureRecordBytes(
            dispatch.generalizedVelocityCount,dispatch.rowCount,dispatch.blockCount);
        const std::size_t offset=std::size_t(found->environment)*bytes;
        id<MTLBuffer> source=context.buffers[kQualityFailureSnapshot];
        if (bytes<sizeof(MRUnifiedQualityFailureHeaderGPU) ||
            bytes>MR_UNIFIED_QUALITY_FAILURE_MAX_BYTES || source==nil ||
            offset>source.length || bytes>source.length-offset)
            throw std::runtime_error("quality witness range invalid");
        id<MTLBuffer> readback=[context.device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[context.queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
        if (readback==nil || command==nil || encoder==nil)
            throw std::runtime_error("quality witness readback unavailable");
        [encoder copyFromBuffer:source sourceOffset:offset toBuffer:readback
            destinationOffset:0u size:bytes];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)
            throw std::runtime_error("quality witness readback failed");
        MRUnifiedQualityFailureHeaderGPU saved{};
        std::memcpy(&saved,readback.contents,sizeof(saved));
        const auto contact=std::find_if(result.contactStatuses.begin(),result.contactStatuses.end(),
            [&](const auto& s){return s.environment==found->environment &&
                s.controlStep==found->controlStep && s.physicsSubstep==found->failingSubstep;});
        if(saved.valid!=MR_UNIFIED_QUALITY_FAILURE_MAGIC ||
           saved.version!=MR_UNIFIED_QUALITY_FAILURE_VERSION || saved.recordBytes!=bytes ||
           saved.environment!=found->environment || saved.controlStep!=found->controlStep ||
           saved.physicsSubstep!=found->failingSubstep ||
           saved.status.problemIndex!=found->environment ||
           (saved.status.code!=MR_UNIFIED_QUALITY_DID_NOT_CONVERGE &&
            saved.status.code!=MR_UNIFIED_QUALITY_LINE_SEARCH_FAILED) ||
           saved.packet.identity.x!=found->environment ||
           std::memcmp(&saved.dispatch,&dispatch,sizeof(dispatch))!=0 ||
           contact==result.contactStatuses.end() || contact->code!=found->code ||
           contact->qualityNewtonIterations!=saved.status.newtonIterations ||
           contact->qualityPCGIterations!=saved.status.pcgIterations ||
           contact->qualityLineSearchBacktracks!=saved.status.lineSearchBacktracks ||
           std::memcmp(&contact->qualityCertificates,&saved.status.certificates0,
               sizeof(saved.status.certificates0))!=0)
            throw std::runtime_error("protected quality witness does not match public failure");
        const int fd=::open(filename,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if(fd<0) throw std::runtime_error("quality witness destination must be a new file");
        const auto* data=static_cast<const char*>(readback.contents);std::size_t written=0u;
        while(written<bytes){const auto n=::write(fd,data+written,bytes-written);
            if(n<0 && errno==EINTR)continue;if(n<=0)break;written+=std::size_t(n);}
        const bool flushed=written==bytes && ::fsync(fd)==0;const bool closed=::close(fd)==0;
        if(!flushed || !closed)throw std::runtime_error("quality witness incomplete; partial file preserved");
        std::fprintf(stderr,"MetalWorld protected quality failure capture: %zu bytes, control %u, substep %u\n",
            bytes,saved.controlStep,saved.physicsSubstep);
    }}catch(const std::exception& e){
        std::fprintf(stderr,"MetalWorld quality failure capture unavailable: %s\n",e.what());
    }
}
} // namespace
