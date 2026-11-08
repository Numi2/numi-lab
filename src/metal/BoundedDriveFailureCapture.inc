// Bounded diagnostic readback after completion; never feeds the native solve.
namespace {
void captureBoundedDriveFailure(const detail::MetalWorldContextState& context,
                               const MetalWorldResult& result) noexcept {
    const char* filename=std::getenv("METALROBO_BOUNDED_FAILURE_CAPTURE");
    if (!filename || !*filename ||
        !context.useBoundedDrives) return;
    const auto found=std::find_if(result.statuses.begin(),result.statuses.end(),
        [](const auto& s){return s.code==MR_STEP_DID_NOT_CONVERGE && s.diagnostics.y==256.0f;});
    if (found==result.statuses.end()) return;
    try {@autoreleasepool {
        const std::size_t offset=std::size_t(found->environment)*sizeof(MRBoundedDriveFailureGPU);
        id<MTLBuffer> source=context.buffers[kDriveFailureSnapshot];
        if (source==nil || offset>source.length || sizeof(MRBoundedDriveFailureGPU)>source.length-offset)
            throw std::runtime_error("bounded witness range invalid");
        id<MTLBuffer> readback=[context.device newBufferWithLength:sizeof(MRBoundedDriveFailureGPU)
            options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[context.queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
        if (readback==nil || command==nil || encoder==nil)
            throw std::runtime_error("bounded witness readback unavailable");
        [encoder copyFromBuffer:source sourceOffset:offset toBuffer:readback destinationOffset:0
            size:sizeof(MRBoundedDriveFailureGPU)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status!=MTLCommandBufferStatusCompleted)
            throw std::runtime_error("bounded witness readback failed");
        MRBoundedDriveFailureGPU saved{};std::memcpy(&saved,readback.contents,sizeof(saved));
        if (saved.valid!=0x42444632u || saved.environment!=found->environment ||
            saved.controlStep!=found->controlStep || saved.physicsSubstep!=found->failingSubstep ||
            saved.code!=found->code || saved.nv>MR_ARTICULATED_ABA_MAX_DOFS)
            throw std::runtime_error("protected bounded witness does not match public failure");
        const int fd=::open(filename,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if (fd<0) throw std::runtime_error("bounded witness destination must be a new file");
        const auto* data=reinterpret_cast<const char*>(&saved);std::size_t written=0u;
        while (written<sizeof(saved)) {
            const auto amount=::write(fd,data+written,sizeof(saved)-written);
            if (amount<0 && errno==EINTR) continue;
            if (amount<=0) break;
            written+=std::size_t(amount);
        }
        const bool flushed=written==sizeof(saved) && ::fsync(fd)==0;
        const bool closed=::close(fd)==0;
        if (!flushed || !closed) throw std::runtime_error("bounded witness incomplete; partial file preserved");
        std::fprintf(stderr,"MetalWorld bounded motor failure capture: %zu bytes\n",sizeof(saved));
    }} catch(const std::exception& e) {
        std::fprintf(stderr,"MetalWorld bounded motor failure capture unavailable: %s\n",e.what());
    }
}
} // namespace
