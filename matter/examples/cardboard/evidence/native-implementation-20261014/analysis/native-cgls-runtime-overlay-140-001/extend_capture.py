from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/native-cgls-runtime-overlay-140-001'
P = O / 'candidate/matter/src/runtime.mm'
before = P.read_text()
text = before

def replace(old, new):
    global text
    assert text.count(old) == 1, old[:160]
    text = text.replace(old, new)

replace('''            if (femBaseCauchyFallbackRequested) {
                const auto rejectBaseCauchy''', '''            std::uint32_t femBaseCGLSCaptureIteration = NM_INVALID_INDEX;
            std::string femBaseCGLSCapturePath;
            const char* cglsCapturePath = std::getenv("NM_FEM_BASE_CGLS_CAPTURE_PATH");
            const char* cglsCaptureNewton = std::getenv("NM_FEM_BASE_CGLS_CAPTURE_NEWTON");
            if (cglsCapturePath != nullptr || cglsCaptureNewton != nullptr) {
                char* end = nullptr;
                const unsigned long parsed = cglsCaptureNewton == nullptr ? 257ul :
                    std::strtoul(cglsCaptureNewton, &end, 10);
                if (femBaseCGLSIterations == 0u || cglsCapturePath == nullptr ||
                    *cglsCapturePath == '\\0' || cglsCaptureNewton == nullptr ||
                    end == cglsCaptureNewton || *end != '\\0' || parsed > 256ul ||
                    std::to_string(parsed) != cglsCaptureNewton) {
                    diagnostics.message = "bounded CGLS capture requires path and canonical Newton index0..256";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
                femBaseCGLSCaptureIteration = static_cast<std::uint32_t>(parsed);
                femBaseCGLSCapturePath = cglsCapturePath;
            }
            if (femBaseCauchyFallbackRequested) {
                const auto rejectBaseCauchy''')

start = text.index('            const auto encodeBaseCauchyDirectionTrace =')
end = text.index('            const auto encodeBaseCauchyDirection =', start)
part = text[start:end]
old = '''                [copy endEncoding];
                const auto capturedRoot'''
assert part.count(old) == 1
new = '''                // Optional bounded read-only snapshot of generated scratch.
                // Header uint32[8]: magic CGLS, version, E, N, U, cap, segmentCount,0.
                // Twelve aligned segments: x,r,s,p,Ax,b,mask,envMap,scalars,control,errors,selected.
                id<MTLBuffer> cglsSnapshot = nil;
                NSUInteger cglsSnapshotBytes = 0u;
                const bool captureCGLS = femBaseCGLSIterations != 0u &&
                    nonlinearIteration == femBaseCGLSCaptureIteration &&
                    !femBaseCGLSCapturePath.empty();
                if (captureCGLS) {
                    const std::uint32_t unknowns = state.fgmresLayout.unknownCount;
                    const std::array<id<MTLBuffer>, 12> sourceBuffers = {
                        state.femBaseCGLSX, state.femBaseCGLSR, state.femBaseCGLSS,
                        state.femBaseCGLSP, state.femBaseCGLSQ,
                        state.femBaseCauchyRightHandSide, state.femBaseCGLSMask,
                        state.femBaseCGLSEnvironment, state.femBaseCGLSScalars,
                        state.femBaseCGLSControl, state.femBaseCGLSErrors,
                        state.femBaseCGLSSelected};
                    const NSUInteger vectorBytes = unknowns * sizeof(nm_float4);
                    const std::array<NSUInteger, 12> sizes = {
                        vectorBytes, vectorBytes, vectorBytes, vectorBytes,
                        vectorBytes, vectorBytes, vectorBytes,
                        unknowns * sizeof(std::uint32_t),
                        environments * sizeof(nm_float4), environments * sizeof(nm_uint4),
                        environments * sizeof(std::uint32_t), environments * sizeof(std::uint32_t)};
                    std::array<NSUInteger, 12> offsets{};
                    cglsSnapshotBytes = 8u * sizeof(std::uint32_t);
                    for (std::size_t i = 0u; i < sizes.size(); ++i) {
                        cglsSnapshotBytes = (cglsSnapshotBytes + 15u) & ~NSUInteger(15u);
                        offsets[i] = cglsSnapshotBytes;
                        cglsSnapshotBytes += sizes[i];
                    }
                    cglsSnapshot = [state.device newBufferWithLength:cglsSnapshotBytes
                        options:MTLResourceStorageModeShared];
                    if (cglsSnapshot == nil) {
                        [copy endEncoding];
                        diagnostics.message = "CGLS scratch snapshot allocation failed";
                        return false;
                    }
                    std::memset(cglsSnapshot.contents, 0, cglsSnapshotBytes);
                    const std::array<std::uint32_t, 8> header = {
                        0x534c4743u, 1u, static_cast<std::uint32_t>(environments),
                        state.dispatch.femNodeCount, unknowns, femBaseCGLSIterations, 12u, 0u};
                    std::memcpy(cglsSnapshot.contents, header.data(), sizeof(header));
                    for (std::size_t i = 0u; i < sizes.size(); ++i)
                        [copy copyFromBuffer:sourceBuffers[i] sourceOffset:0u
                            toBuffer:cglsSnapshot destinationOffset:offsets[i] size:sizes[i]];
                }
                [copy endEncoding];
                const auto capturedCGLSPath = captureCGLS ? femBaseCGLSCapturePath : std::string();
                const auto capturedRoot'''
part = part.replace(old, new)
old = '''                    const auto* values = static_cast<const nm_float4*>(snapshot.contents);'''
new = '''                    if (cglsSnapshot != nil) {
                        const int fd = ::open(capturedCGLSPath.c_str(),
                            O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
                        bool complete = fd >= 0;
                        std::size_t written = 0u;
                        const auto* raw = static_cast<const unsigned char*>(cglsSnapshot.contents);
                        while (complete && written < cglsSnapshotBytes) {
                            const ssize_t count = ::write(fd, raw + written, cglsSnapshotBytes - written);
                            if (count < 0 && errno == EINTR) continue;
                            if (count <= 0) complete = false;
                            else written += static_cast<std::size_t>(count);
                        }
                        if (fd >= 0 && ::close(fd) != 0) complete = false;
                        std::fprintf(stderr,
                            "fem_base_cgls_capture={\\\"schema\\\":1,\\\"root\\\":%u,\\\"newton_iteration\\\":%u,\\\"bytes\\\":%zu,\\\"expected_bytes\\\":%zu,\\\"complete\\\":%s,\\\"physical_state_changed\\\":false}\\n",
                            capturedRoot, capturedIteration, written,
                            static_cast<std::size_t>(cglsSnapshotBytes), complete ? "true" : "false");
                    }
                    const auto* values = static_cast<const nm_float4*>(snapshot.contents);'''
assert part.count(old) == 1
part = part.replace(old, new)
text = text[:start] + part + text[end:]
P.write_text(text)
(O / 'capture.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), text.splitlines(True), fromfile='a/matter/src/runtime.mm',
    tofile='b/matter/src/runtime.mm')))
(O / 'extend_capture.py').write_bytes(Path(__file__).read_bytes())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
(O / 'capture-preflight.json').write_text(json.dumps({
    'before_runtime_sha256': sha(before.encode()), 'after_runtime_sha256': sha(text.encode()),
    'capture_is_default_off_readonly_and_one_selected_newton': True,
    'physical_steps': 0, 'status': 'AWAITING_NATIVE_PARITY_AND_CAPTURE',
    'binary_layout': 'Little-endian uint32[8] header magic0x534c4743/version1/E/N/U/cap/12/0; each subsequent segment starts at16-byte aligned offset. x,r,s,p,Ax,b,mask=float4[U];envMap=uint32[U];scalars=float4[E];control=uint4[E];errors=uint32[E];selected=uint32[E].'
}, indent=2) + '\n')
print('extended capture140')
