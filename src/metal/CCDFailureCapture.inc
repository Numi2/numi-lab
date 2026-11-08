// Optional, bounded inspection after a failed submission has completed. It
// never supplies state to physics, tasks, actors or the learner. A captured
// post-submission sweep is causal evidence only if replay matches its stored
// CCD record and the public failure boundary.
namespace {
void captureCCDFailure(
    const detail::MetalWorldContextState& context,
    const MetalWorldResult& result
) noexcept {
    const char* filename = std::getenv("METALROBO_CCD_FAILURE_CAPTURE");
    if (filename == nullptr || filename[0] == '\0' ||
        context.boundContactDispatch.ccdMode != MR_WORLD_CCD_HYBRID) {
        return;
    }
    const auto found = std::find_if(
        result.contactStatuses.begin(), result.contactStatuses.end(),
        [](const MRMetalWorldContactStatusGPU& status) {
            return status.code == MR_STEP_DID_NOT_CONVERGE &&
                status.firstFailingPair != MR_INVALID_INDEX;
        }
    );
    if (found == result.contactStatuses.end()) return;
    try {
        const auto& status = *found;
        const auto& dispatch = context.boundContactDispatch;
        if (status.environment >= dispatch.environmentCount ||
            status.firstFailingPair >= dispatch.eligiblePairCount ||
            dispatch.shapeCount > 4096u ||
            dispatch.eligiblePairCount > 4096u ||
            dispatch.ccdCandidateCapacity > 4096u) {
            throw std::runtime_error("CCD capture exceeds inspection envelope");
        }
        struct Segment {
            std::array<char, 8> tag{};
            std::vector<std::byte> bytes;
            id<MTLBuffer> readback = nil;
        };
        std::vector<Segment> segments;
        constexpr std::size_t maximumBytes = 1024u * 1024u;
        std::size_t total = 32u;
        const auto add = [&](const char* tag, const std::size_t size) -> Segment& {
            if (size == 0u || size > maximumBytes - 16u ||
                total > maximumBytes - 16u - size) {
                throw std::runtime_error("CCD capture byte bound exceeded");
            }
            total += 16u + size;
            auto& segment = segments.emplace_back();
            std::memcpy(segment.tag.data(), tag, std::min(std::strlen(tag), 8ul));
            segment.bytes.resize(size);
            return segment;
        };
        auto& configuration = add("dispatch", sizeof(dispatch));
        std::memcpy(configuration.bytes.data(), &dispatch, sizeof(dispatch));
        auto& failure = add("status", sizeof(status));
        std::memcpy(failure.bytes.data(), &status, sizeof(status));
        @autoreleasepool {
            id<MTLCommandBuffer> command = [context.queue commandBuffer];
            id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
            if (command == nil || encoder == nil) {
                throw std::runtime_error("CCD capture command unavailable");
            }
            const auto copy = [&](const char* tag, const std::size_t index,
                                  const std::size_t offset, const std::size_t size) {
                auto& segment = add(tag, size);
                id<MTLBuffer> source = context.buffers[index];
                if (source == nil || offset > source.length ||
                    size > source.length - offset) {
                    throw std::runtime_error("CCD capture source range invalid");
                }
                segment.readback = [context.device newBufferWithLength:size
                    options:MTLResourceStorageModeShared];
                if (segment.readback == nil) {
                    throw std::runtime_error("CCD capture allocation failed");
                }
                [encoder copyFromBuffer:source sourceOffset:offset
                    toBuffer:segment.readback destinationOffset:0 size:size];
            };
            copy("shapes", kShapes, 0u,
                dispatch.shapeCount * sizeof(MRShapeGPU));
            copy("pairs", kEligiblePairs, 0u,
                dispatch.eligiblePairCount * sizeof(MRCompiledCollisionPairGPU));
            const std::size_t projected = dispatch.shapeCount * sizeof(MRProjectedColliderGPU);
            copy("current", kProjectedColliders, status.environment * projected, projected);
            copy("future", kFutureProjectedColliders, status.environment * projected, projected);
            const std::size_t candidates = dispatch.ccdCandidateCapacity * sizeof(MRCCDPairGPU);
            copy("ccdPairs", kCCDPairs, status.environment * candidates, candidates);
            copy("eventA", kCCDEventStatesA,
                status.environment * sizeof(MRCCDEventStateGPU), sizeof(MRCCDEventStateGPU));
            copy("eventB", kCCDEventStatesB,
                status.environment * sizeof(MRCCDEventStateGPU), sizeof(MRCCDEventStateGPU));
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("CCD capture readback failed");
            }
            for (auto& segment : segments) {
                if (segment.readback != nil) {
                    std::memcpy(segment.bytes.data(), segment.readback.contents, segment.bytes.size());
                }
            }
        }
        std::vector<std::byte> image;
        image.reserve(total);
        const auto append = [&](const void* source, const std::size_t size) {
            const auto* first = static_cast<const std::byte*>(source);
            image.insert(image.end(), first, first + size);
        };
        constexpr char magic[8] = {'C','C','D','S','N','P','0','1'};
        append(magic, sizeof(magic));
        const std::uint32_t header[] = {1u, status.environment, status.controlStep,
            status.physicsSubstep, status.firstFailingPair, std::uint32_t(segments.size())};
        append(header, sizeof(header));
        for (const auto& segment : segments) {
            append(segment.tag.data(), segment.tag.size());
            const std::uint64_t size = segment.bytes.size();
            append(&size, sizeof(size));
            append(segment.bytes.data(), segment.bytes.size());
        }
        if (image.size() != total) throw std::runtime_error("CCD capture envelope mismatch");
        const int descriptor = ::open(filename, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (descriptor < 0) throw std::runtime_error("CCD capture destination must be a new file");
        std::size_t written = 0u;
        while (written < image.size()) {
            const ssize_t count = ::write(descriptor, image.data() + written, image.size() - written);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) break;
            written += std::size_t(count);
        }
        const bool flushed = written == image.size() && ::fsync(descriptor) == 0;
        const bool closed = ::close(descriptor) == 0;
        if (!flushed || !closed) throw std::runtime_error("CCD capture write incomplete; partial file preserved");
        std::fprintf(stderr, "MetalWorld CCD failure capture: %zu bytes\n", image.size());
    } catch (const std::exception& exception) {
        // Preserve the original failure and rollback diagnostics even when an
        // inspection cannot be written. Never weaken physics admission.
        std::fprintf(stderr, "MetalWorld CCD failure capture unavailable: %s\n", exception.what());
    }
}
} // namespace
