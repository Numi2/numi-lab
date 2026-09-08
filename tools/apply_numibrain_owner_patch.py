#!/usr/bin/env python3
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise SystemExit(f"anchor not unique in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


mm = "src/metal/MetalWorld.mm"

# Retain the authoritative logical byte ranges that were already validated by
# validateAndBuildLayout. Capacity is an allocator concern and must never become
# part of physical-state identity.
replace_once(
    mm,
    """    std::array<std::size_t, kRawBufferCount> capacities{};
    std::array<std::size_t, kRawBufferCount> uploadCapacities{};""",
    """    std::array<std::size_t, kRawBufferCount> capacities{};
    std::array<std::size_t, kRawBufferCount> logicalBytes{};
    std::array<std::size_t, kRawBufferCount> uploadCapacities{};""",
)

replace_once(
    mm,
    """    context.stats.retainedBufferBytes = retainedBytes;
    context.stats.usingPrivateHeaps =""",
    """    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        const std::size_t logical =
            requirements.entries[index].logicalBytes;
        if (logical > context.capacities[index]) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::internalFailure,
                std::string(requirements.entries[index].label) +
                    \" logical byte range exceeds the retained arena\"
            );
        }
        context.logicalBytes[index] = logical;
    }
    context.stats.retainedBufferBytes = retainedBytes;
    context.stats.usingPrivateHeaps =""",
)

replace_once(
    mm,
    """        id<MTLBuffer> source = context->buffers[index];
        if (source == nil || source.length == 0u ||
            source.storageMode != MTLStorageModePrivate) {
            continue;
        }
        id<MTLBuffer> copy = [context->device
            newBufferWithLength:source.length
                       options:MTLResourceStorageModeShared];
        if (copy == nil || copy.contents == nullptr) {
            [blit endEncoding];
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint could not allocate readback storage\");
        }
        staged[index] = copy;
        [blit copyFromBuffer:source
                sourceOffset:0u
                    toBuffer:copy
           destinationOffset:0u
                        size:source.length];""",
    """        id<MTLBuffer> source = context->buffers[index];
        const std::size_t bytes = context->logicalBytes[index];
        if (bytes == 0u) {
            continue;
        }
        if (source == nil || source.length < bytes) {
            [blit endEncoding];
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint logical range exceeds retained storage\");
        }
        if (source.storageMode != MTLStorageModePrivate) {
            continue;
        }
        id<MTLBuffer> copy = [context->device
            newBufferWithLength:static_cast<NSUInteger>(bytes)
                       options:MTLResourceStorageModeShared];
        if (copy == nil || copy.contents == nullptr || copy.length < bytes) {
            [blit endEncoding];
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint could not allocate readback storage\");
        }
        staged[index] = copy;
        [blit copyFromBuffer:source
                sourceOffset:0u
                    toBuffer:copy
           destinationOffset:0u
                        size:static_cast<NSUInteger>(bytes)];""",
)

replace_once(
    mm,
    """        id<MTLBuffer> source = context->buffers[index];
        const std::uint64_t index64 = static_cast<std::uint64_t>(index);
        scalar(index64);
        const std::uint64_t length = source == nil
            ? 0u : static_cast<std::uint64_t>(source.length);
        scalar(length);
        if (length == 0u) {
            continue;
        }
        id<MTLBuffer> readable = source.storageMode == MTLStorageModePrivate
            ? staged[index] : source;
        if (readable == nil || readable.contents == nullptr ||
            readable.length < source.length) {
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint encountered unreadable persistent storage\");
        }
        append(readable.contents, static_cast<std::size_t>(source.length));""",
    """        id<MTLBuffer> source = context->buffers[index];
        const std::uint64_t index64 = static_cast<std::uint64_t>(index);
        scalar(index64);
        const std::size_t bytes = context->logicalBytes[index];
        const std::uint64_t length = static_cast<std::uint64_t>(bytes);
        scalar(length);
        if (bytes == 0u) {
            continue;
        }
        if (source == nil || source.length < bytes) {
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint logical range exceeds retained storage\");
        }
        id<MTLBuffer> readable = source.storageMode == MTLStorageModePrivate
            ? staged[index] : source;
        if (readable == nil || readable.contents == nullptr ||
            readable.length < bytes) {
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                \"resident-state fingerprint encountered unreadable persistent storage\");
        }
        append(readable.contents, bytes);""",
)

# Tighten public wording to match the new semantic digest exactly.
for path in ("include/metalrobo/MetalWorld.hpp", "include/metalrobo/c_api.h"):
    p = Path(path)
    text = p.read_text()
    text = text.replace(
        "includes every persistent arena buffer plus resident metadata.",
        "includes every persistent buffer's validated logical bytes plus resident metadata.",
    ).replace(
        "The digest covers the complete persistent continuation arena.",
        "The digest covers validated logical bytes of the persistent continuation state.",
    )
    p.write_text(text)

# Source-level regression checks: no allocator length may re-enter the digest.
text = Path(mm).read_text()
start = text.index("MetalWorldDiagnostics MetalWorldContext::residentStateFingerprint(")
end = text.index("const char* metalWorldHostStatusName(", start)
fingerprint = text[start:end]
required = [
    "context->logicalBytes[index]",
    "append(readable.contents, bytes);",
    "source.length < bytes",
]
for token in required:
    if token not in fingerprint:
        raise SystemExit(f"fingerprint hardening token missing: {token}")
for forbidden in ("newBufferWithLength:source.length", "size:source.length", "append(readable.contents, static_cast<std::size_t>(source.length))"):
    if forbidden in fingerprint:
        raise SystemExit(f"allocator capacity leaked into resident fingerprint: {forbidden}")
print("NumiBrain owner fingerprint now hashes validated logical continuation bytes only")
