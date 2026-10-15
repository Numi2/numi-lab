#!/usr/bin/env python3
from __future__ import annotations
import difflib, hashlib, json, pathlib, shutil, sys

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-primary-policy-143')
OUT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = BASE / 'matter/src/runtime.mm'
EXPECTED = '01162665b98a7f88d4d6341e2bc6ea50afb9f5d831b04fe446ce27d34e7ad5ec'
raw = SOURCE.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED:
    raise SystemExit(f'frozen source hash mismatch: {actual}')
s = raw.decode('utf-8')

struct_insert = r'''
struct BaseCGLSStateCaptureArray {
    std::string name;
    std::string dtype;
    std::string semantic;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0u;
    std::uint64_t byteCount = 0u;
    std::uint64_t elementBytes = 0u;
};

struct BaseCGLSStateCaptureMetadata {
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t microtick = 0u;
    std::uint32_t microtickCount = 0u;
    std::uint32_t newtonIteration = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t objectCount = 0u;
    std::uint32_t materialCount = 0u;
    std::uint32_t parameterCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t materialStateStride = 0u;
    std::uint32_t unknownCount = 0u;
    std::uint32_t materialStateCandidateIsCurrentProjection = 0u;
    std::uint32_t tangentCacheEnabled = 0u;
    std::uint32_t tangentCacheForceRebuild = 0u;
    std::uint32_t tangentCacheValidCount = 0u;
    std::uint32_t mixedSolverPreconditioner = 0u;
    std::uint32_t cglsIterations = 0u;
    std::uint32_t microFlags = 0u;
    std::uint32_t dispatchABI = 0u;
    std::uint32_t dispatchFlags = 0u;
    std::uint32_t rigidGeneralizedCapacity = 0u;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t executionFingerprint = 0u;
    std::uint64_t deviceRegistryID = 0u;
    std::string deviceName;
    std::string cglsCaptureFile;
    float dtSeconds = 0.0f;
    float inverseDtSeconds = 0.0f;
};

void writeBaseCGLSStateCapture(
    const void* payload,
    const std::uint64_t payloadBytes,
    const std::vector<BaseCGLSStateCaptureArray>& arrays,
    const BaseCGLSStateCaptureMetadata& metadata,
    const std::string& basePath
) {
    const auto* bytes = static_cast<const unsigned char*>(payload);
    const std::string payloadHash = sha256Hex(
        bytes, static_cast<std::size_t>(payloadBytes));
    const std::string binaryPath = basePath + ".bin";
    const std::string manifestPath = basePath + ".json";
    const std::string payloadName =
        std::filesystem::path(binaryPath).filename().string();
    std::ostringstream manifest;
    manifest.imbue(std::locale::classic());
    manifest << "{\"schema\":\"numi.matter.base_cgls_state_capture.v1\","
        << "\"complete\":true,"
        << "\"byte_order\":\"little_endian\","
        << "\"capture_point\":\"after_newton_correction_before_cgls_direction_construction_or_installation\","
        << "\"physical_state_changed\":false,"
        << "\"root\":" << metadata.controlStep
        << ",\"control_step\":" << metadata.controlStep
        << ",\"physics_substep\":" << metadata.physicsSubstep
        << ",\"microtick\":" << metadata.microtick
        << ",\"microtick_count\":" << metadata.microtickCount
        << ",\"newton_iteration\":" << metadata.newtonIteration
        << ",\"environment_count\":" << metadata.environmentCount
        << ",\"object_count\":" << metadata.objectCount
        << ",\"material_count\":" << metadata.materialCount
        << ",\"parameter_count\":" << metadata.parameterCount
        << ",\"node_count\":" << metadata.nodeCount
        << ",\"tetrahedron_count\":" << metadata.tetrahedronCount
        << ",\"material_state_stride\":" << metadata.materialStateStride
        << ",\"unknown_count\":" << metadata.unknownCount
        << ",\"cgls_iterations\":" << metadata.cglsIterations
        << ",\"mixed_solver_preconditioner\":" << metadata.mixedSolverPreconditioner
        << ",\"array_count\":" << arrays.size()
        << ",\"cgls_capture_file\":" << jsonQuoted(metadata.cglsCaptureFile)
        << ",\"microstep_flags\":" << metadata.microFlags
        << ",\"dispatch_abi_version\":" << metadata.dispatchABI
        << ",\"dispatch_flags\":" << metadata.dispatchFlags
        << ",\"rigid_generalized_capacity\":" << metadata.rigidGeneralizedCapacity
        << ",\"layout\":{\"unknown_count\":" << metadata.unknownCount
        << ",\"mechanical_rows\":\"first E*N float4 entries\",\"field_backing_rows\":\"second E*N float4 entries; CGLS requires exact pure-FEM layout and preserves captured bits\"},"
        << "\"dt_seconds\":" << jsonFloat32Token(metadata.dtSeconds)
        << ",\"inverse_dt_seconds\":" << jsonFloat32Token(metadata.inverseDtSeconds)
        << ",\"source_physics_fingerprint_u64_hex\":"
        << jsonQuoted(hexU64(metadata.sourcePhysicsFingerprint))
        << ",\"world_fingerprint_u64_hex\":"
        << jsonQuoted(hexU64(metadata.worldFingerprint))
        << ",\"device_program_fingerprint_u64_hex\":"
        << jsonQuoted(hexU64(metadata.executionFingerprint))
        << ",\"device\":{\"name\":" << jsonQuoted(metadata.deviceName)
        << ",\"registry_id\":" << metadata.deviceRegistryID << "},"
        << "\"tangent_cache\":{\"enabled_by_solver_policy\":"
        << (metadata.tangentCacheEnabled != 0u ? "true" : "false")
        << ",\"force_rebuild_requested\":"
        << (metadata.tangentCacheForceRebuild != 0u ? "true" : "false")
        << ",\"valid_entry_count\":" << metadata.tangentCacheValidCount
        << ",\"generation_claim\":\"raw selected-point cache bytes; reader must adjudicate against captured state and production cache reuse rules\"},"
        << "\"material_history_projection\":{\"current_newton_status\":\"unavailable_not_persisted\","
        << "\"candidate_buffer_name\":\"candidate_material_state_buffer\","
        << "\"candidate_buffer_is_current_projection\":false,"
        << "\"reason\":\"nm_fem_build_algorithmic_tangent_cache projects state in local temporaries and stores tangent columns only; nm_fem_validate writes the candidate arena later in the nonlinear transaction\"},"
        << "\"payload\":{\"file\":" << jsonQuoted(payloadName)
        << ",\"byte_count\":" << payloadBytes
        << ",\"sha256\":" << jsonQuoted(payloadHash) << "},\"arrays\":[";
    for (std::size_t index = 0u; index < arrays.size(); ++index) {
        const auto& array = arrays[index];
        if (index != 0u) manifest << ',';
        manifest << "{\"name\":" << jsonQuoted(array.name)
            << ",\"dtype\":" << jsonQuoted(array.dtype)
            << ",\"semantic\":" << jsonQuoted(array.semantic)
            << ",\"shape\":[";
        for (std::size_t axis = 0u; axis < array.shape.size(); ++axis) {
            if (axis != 0u) manifest << ',';
            manifest << array.shape[axis];
        }
        manifest << "],\"offset\":" << array.offset
            << ",\"byte_count\":" << array.byteCount
            << ",\"element_bytes\":" << array.elementBytes
            << ",\"sha256\":"
            << jsonQuoted(sha256Hex(bytes + array.offset,
                static_cast<std::size_t>(array.byteCount))) << '}';
    }
    manifest << "]}\n";
    const std::string manifestText = manifest.str();

    const int binaryFD = ::open(binaryPath.c_str(),
        O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (binaryFD < 0) {
        std::fprintf(stderr,
            "fem_base_cgls_state_capture={\"schema\":1,\"root\":%u,\"complete\":false,\"reason\":\"binary_exclusive_create_failed\",\"errno\":%d}\n",
            metadata.controlStep, errno);
        return;
    }
    const bool binaryWritten = writeAllBytes(binaryFD, bytes,
        static_cast<std::size_t>(payloadBytes));
    const bool binarySynced = binaryWritten && ::fsync(binaryFD) == 0;
    const int binaryClose = ::close(binaryFD);
    if (!binaryWritten || !binarySynced || binaryClose != 0) {
        ::unlink(binaryPath.c_str());
        std::fprintf(stderr,
            "fem_base_cgls_state_capture={\"schema\":1,\"root\":%u,\"complete\":false,\"reason\":\"binary_write_sync_or_close_failed\",\"errno\":%d}\n",
            metadata.controlStep, errno);
        return;
    }
    const int manifestFD = ::open(manifestPath.c_str(),
        O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (manifestFD < 0) {
        const int savedErrno = errno;
        ::unlink(binaryPath.c_str());
        std::fprintf(stderr,
            "fem_base_cgls_state_capture={\"schema\":1,\"root\":%u,\"complete\":false,\"reason\":\"manifest_exclusive_create_failed\",\"errno\":%d}\n",
            metadata.controlStep, savedErrno);
        return;
    }
    const bool manifestWritten = writeAllBytes(manifestFD,
        manifestText.data(), manifestText.size());
    const bool manifestSynced = manifestWritten && ::fsync(manifestFD) == 0;
    const int manifestClose = ::close(manifestFD);
    if (!manifestWritten || !manifestSynced || manifestClose != 0) {
        ::unlink(binaryPath.c_str());
        ::unlink(manifestPath.c_str());
        std::fprintf(stderr,
            "fem_base_cgls_state_capture={\"schema\":1,\"root\":%u,\"complete\":false,\"reason\":\"manifest_write_sync_or_close_failed\",\"errno\":%d}\n",
            metadata.controlStep, errno);
        return;
    }
    std::fprintf(stderr,
        "fem_base_cgls_state_capture={\"schema\":1,\"root\":%u,\"newton_iteration\":%u,\"bytes\":%llu,\"array_count\":%llu,\"complete\":true,\"manifest\":%s}\n",
        metadata.controlStep, metadata.newtonIteration,
        static_cast<unsigned long long>(payloadBytes),
        static_cast<unsigned long long>(arrays.size()),
        jsonQuoted(manifestPath).c_str());
}
'''
marker = '\nclass UploadPlan {'
if s.count(marker) != 1: raise SystemExit('writer insertion anchor not unique')
s = s.replace(marker, struct_insert + marker)

parse_anchor = '''                femBaseCGLSCaptureIteration = static_cast<std::uint32_t>(parsed);
                femBaseCGLSCapturePath = cglsCapturePath;
            }
'''
parse_replacement = parse_anchor + '''            bool femBaseCGLSStateCaptureRequested = false;
            bool femBaseCGLSStateCaptureCaptured = false;
            const char* cglsStateCaptureText = std::getenv(
                "NM_FEM_BASE_CGLS_STATE_CAPTURE");
            if (cglsStateCaptureText != nullptr) {
                if (std::strcmp(cglsStateCaptureText, "0") != 0 &&
                    std::strcmp(cglsStateCaptureText, "1") != 0) {
                    diagnostics.message =
                        "CGLS state capture must be exactly 0 or 1";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
                femBaseCGLSStateCaptureRequested =
                    std::strcmp(cglsStateCaptureText, "1") == 0;
                if (femBaseCGLSStateCaptureRequested &&
                    (femBaseCGLSIterations == 0u ||
                     femBaseCGLSCapturePath.empty() ||
                     femBaseCGLSCaptureIteration == NM_INVALID_INDEX ||
                     !state.femBaseCGLSScratchEnabled)) {
                    diagnostics.message =
                        "CGLS state capture requires the existing selected CGLS capture path, Newton selector, and admitted scratch";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
            }
'''
if s.count(parse_anchor) != 1: raise SystemExit('parse anchor not unique')
s = s.replace(parse_anchor, parse_replacement)

lambda_anchor = '''            const auto encodeBaseCauchyDirectionTrace = [&](
                bool primaryPolicy = false) -> bool {
'''
lambda_insert = r'''            const auto encodeBaseCGLSStateCapture = [&]() -> bool {
                if (!femBaseCGLSStateCaptureRequested ||
                    femBaseCGLSStateCaptureCaptured ||
                    request.controlStep == NM_INVALID_INDEX ||
                    nonlinearIteration != femBaseCGLSCaptureIteration) {
                    return true;
                }
                femBaseCGLSStateCaptureCaptured = true;
                const auto skip = [&](const char* reason) {
                    std::fprintf(stderr,
                        "fem_base_cgls_state_capture_skipped={\"schema\":1,\"root\":%u,\"newton_iteration\":%u,\"reason\":\"%s\",\"physical_state_changed\":false}\n",
                        request.controlStep, nonlinearIteration, reason);
                    return true;
                };
                const std::uint64_t envCount = state.dispatch.environmentCount;
                const std::uint64_t nodeCount = state.dispatch.femNodeCount;
                const std::uint64_t tetCount = state.dispatch.tetrahedronCount;
                const std::uint64_t stride = state.dispatch.materialStateStride;
                const std::uint64_t unknownCount = state.fgmresLayout.unknownCount;
                constexpr std::uint64_t maximumCaptureBytes =
                    96u * 1024u * 1024u;
                if (envCount == 0u || envCount > 2u || nodeCount == 0u ||
                    nodeCount > 4096u || tetCount == 0u || tetCount > 32768u ||
                    stride == 0u || stride > NM_MAX_MATERIAL_STATE ||
                    unknownCount != 2u * envCount * nodeCount) {
                    return skip("unsupported_dispatch_extent");
                }
                if (state.femAlgorithmicTangentCache == nil ||
                    state.femAlgorithmicTangentCacheValid == nil) {
                    return skip("algorithmic_tangent_cache_unavailable");
                }
                std::vector<BaseCGLSStateCaptureArray> arrays;
                std::array<id<MTLBuffer>, 24> sourceBuffers{};
                std::uint64_t cursor = 0u;
                const auto append = [&](const char* name, const char* dtype,
                    const char* semantic, const std::uint64_t elementBytes,
                    const std::vector<std::uint64_t>& shape,
                    const std::uint64_t byteCount, id<MTLBuffer> source) -> bool {
                    if (arrays.size() >= sourceBuffers.size() || elementBytes == 0u ||
                        cursor > maximumCaptureBytes - 15u ||
                        byteCount > std::numeric_limits<std::uint64_t>::max() - 15u)
                        return false;
                    std::uint64_t elementCount = 1u;
                    for (const auto axis : shape) {
                        if (axis != 0u && elementCount >
                            std::numeric_limits<std::uint64_t>::max() / axis)
                            return false;
                        elementCount *= axis;
                    }
                    if (elementCount >
                        std::numeric_limits<std::uint64_t>::max() / elementBytes ||
                        elementCount * elementBytes != byteCount)
                        return false;
                    const std::uint64_t aligned = (cursor + 15u) & ~std::uint64_t(15u);
                    if (aligned > maximumCaptureBytes ||
                        byteCount > maximumCaptureBytes - aligned ||
                        (byteCount != 0u && (source == nil || source.length < byteCount)))
                        return false;
                    BaseCGLSStateCaptureArray entry;
                    entry.name = name;
                    entry.dtype = dtype;
                    entry.semantic = semantic;
                    entry.shape = shape;
                    entry.offset = aligned;
                    entry.byteCount = byteCount;
                    entry.elementBytes = elementBytes;
                    arrays.push_back(std::move(entry));
                    sourceBuffers[arrays.size() - 1u] = source;
                    cursor = aligned + byteCount;
                    return true;
                };
                const auto checkedBytes = [](const std::uint64_t count,
                    const std::uint64_t elementBytes, std::uint64_t& result) {
                    if (elementBytes == 0u || count >
                        std::numeric_limits<std::uint64_t>::max() / elementBytes)
                        return false;
                    result = count * elementBytes;
                    return true;
                };
                const std::uint64_t envNodes = envCount * nodeCount;
                const std::uint64_t envTets = envCount * tetCount;
                const std::uint64_t envStateScalars = envTets * stride;
                const std::uint64_t tangentScalars = envTets * 81u;
                std::uint64_t nodeBytes = 0u, tetBytes = 0u, historyBytes = 0u;
                std::uint64_t correctionBytes = 0u, tangentBytes = 0u;
                std::uint64_t validityBytes = 0u;
                if (!checkedBytes(envNodes, sizeof(NMFEMNodeStateGPU), nodeBytes) ||
                    !checkedBytes(envTets, sizeof(NMTetrahedronGPU), tetBytes) ||
                    !checkedBytes(envStateScalars, sizeof(float), historyBytes) ||
                    !checkedBytes(unknownCount, sizeof(nm_float4), correctionBytes) ||
                    !checkedBytes(tangentScalars, sizeof(float), tangentBytes) ||
                    !checkedBytes(envTets, sizeof(std::uint32_t), validityBytes))
                    return skip("checked_extent_overflow");
                const auto appendStorageBuffer = [&](const char* name,
                    const char* dtype, const char* semantic,
                    const std::uint64_t elementBytes, id<MTLBuffer> source) {
                    if (source == nil || source.length % elementBytes != 0u)
                        return false;
                    const std::uint64_t count = source.length / elementBytes;
                    return append(name, dtype, semantic, elementBytes, {count},
                        source.length, source);
                };
                if (!append("accepted_nodes", "record:NMFEMNodeStateGPU",
                        "accepted physical FEM nodes, environment-major",
                        sizeof(NMFEMNodeStateGPU), {envCount, nodeCount},
                        nodeBytes, state.femAccepted) ||
                    !append("candidate_nodes", "record:NMFEMNodeStateGPU",
                        "exact candidate FEM node arena at selected Newton center",
                        sizeof(NMFEMNodeStateGPU), {envCount, nodeCount},
                        nodeBytes, state.femCandidate) ||
                    !append("accepted_tetrahedra", "record:NMTetrahedronGPU",
                        "accepted topology and material-frame records, environment-major",
                        sizeof(NMTetrahedronGPU), {envCount, tetCount},
                        tetBytes, state.femTetrahedraAccepted) ||
                    !append("candidate_tetrahedra", "record:NMTetrahedronGPU",
                        "exact candidate topology and material-frame records at selected Newton center",
                        sizeof(NMTetrahedronGPU), {envCount, tetCount},
                        tetBytes, state.femTetrahedraCandidate) ||
                    !append("accepted_material_history", "float32",
                        "accepted internal material history, environment/tetrahedron/state order",
                        sizeof(float), {envCount, tetCount, stride},
                        historyBytes, state.femMaterialStateAccepted) ||
                    !append("candidate_material_state_buffer", "float32",
                        "raw candidate history arena; not asserted to be projected for this Newton linearization",
                        sizeof(float), {envCount, tetCount, stride},
                        historyBytes, state.femMaterialStateCandidate) ||
                    !appendStorageBuffer("objects", "record:NMContinuumObjectGPU",
                        "exact resident object descriptors; storage extent, includes typed sentinel if empty",
                        sizeof(NMContinuumObjectGPU), state.objects) ||
                    !appendStorageBuffer("materials", "record:NMMaterialGPU",
                        "exact resident material descriptors; storage extent, includes typed sentinel if empty",
                        sizeof(NMMaterialGPU), state.materials) ||
                    !appendStorageBuffer("scalar_programs", "record:NMScalarProgramGPU",
                        "exact resident scalar programs; storage extent, includes typed sentinel if empty",
                        sizeof(NMScalarProgramGPU), state.scalarPrograms) ||
                    !appendStorageBuffer("expression_instructions", "record:NMExpressionInstructionGPU",
                        "exact resident expression instructions; storage extent, includes typed sentinel if empty",
                        sizeof(NMExpressionInstructionGPU), state.instructions) ||
                    !append("environment_parameters", "float32",
                        "environment-major exact parameter arena",
                        sizeof(float), {envCount, state.dispatch.parameterCount},
                        envCount * state.dispatch.parameterCount * sizeof(float),
                        state.environmentParameters) ||
                    !append("schedulers", "record:NMSchedulerStateGPU",
                        "exact environment/object scheduler state",
                        sizeof(NMSchedulerStateGPU), {envCount, state.dispatch.objectCount},
                        envCount * state.dispatch.objectCount * sizeof(NMSchedulerStateGPU),
                        state.schedulers) ||
                    !append("adaptive_state", "record:NMAdaptiveStateGPU",
                        "exact environment/object adaptive state",
                        sizeof(NMAdaptiveStateGPU), {envCount, state.dispatch.objectCount},
                        envCount * state.dispatch.objectCount * sizeof(NMAdaptiveStateGPU),
                        state.adaptive) ||
                    !append("original_newton_correction", "float32x4",
                        "original FGMRES Newton correction before CGLS replacement; layout U=2*E*N",
                        sizeof(nm_float4), {unknownCount}, correctionBytes,
                        state.femSolution) ||
                    !append("base_residual_rhs", "float32x4",
                        "unmodified base residual right-hand side consumed by CGLS layout preparation",
                        sizeof(nm_float4), {unknownCount}, correctionBytes,
                        state.femBaseCauchyRightHandSide) ||
                    !append("algorithmic_tangent_cache", "float32",
                        "raw cached dP/dF entries in environment/tet/row/column order",
                        sizeof(float), {envCount, tetCount, 9u, 9u},
                        tangentBytes, state.femAlgorithmicTangentCache) ||
                    !append("algorithmic_tangent_cache_valid", "uint32",
                        "raw cache-valid flags in environment/tetrahedron order",
                        sizeof(std::uint32_t), {envCount, tetCount},
                        validityBytes, state.femAlgorithmicTangentCacheValid) ||
                    !append("fgmres_state", "record:NMFGMRESStateGPU",
                        "per-environment Newton/FGMRES certificate state at capture point",
                        sizeof(NMFGMRESStateGPU), {envCount},
                        envCount * sizeof(NMFGMRESStateGPU), state.fgmresStates) ||
                    !append("mixed_solver", "record:NMMixedSolverGPU",
                        "exact compiled solver tolerances and execution policy",
                        sizeof(NMMixedSolverGPU), {1u}, sizeof(NMMixedSolverGPU),
                        state.mixedSolver) ||
                    !append("armijo_metrics", "float32x4",
                        "current Armijo base metrics and CGLS eligibility marker",
                        sizeof(nm_float4), {envCount},
                        envCount * sizeof(nm_float4), state.femArmijoMetrics) ||
                    !append("armijo_statuses", "record:NMMatterStatusGPU",
                        "per-environment Armijo status at CGLS entry",
                        sizeof(NMMatterStatusGPU), {envCount},
                        envCount * sizeof(NMMatterStatusGPU), state.femArmijoStatuses) ||
                    !append("production_statuses", "record:NMMatterStatusGPU",
                        "per-environment production status at CGLS entry",
                        sizeof(NMMatterStatusGPU), {envCount},
                        envCount * sizeof(NMMatterStatusGPU), state.statuses) ||
                    !append("fem_line_search", "float32x4",
                        "full environment/object FEM line-search records at CGLS entry",
                        sizeof(nm_float4), {envCount, state.dispatch.objectCount},
                        envCount * state.dispatch.objectCount * sizeof(nm_float4),
                        state.femLineSearch) ||
                    !append("environment_line_search", "float32",
                        "per-environment accepted trial fraction at CGLS entry",
                        sizeof(float), {envCount},
                        envCount * sizeof(float), state.environmentLineSearch)) {
                    return skip("required_input_buffer_missing_or_extent_overflow");
                }
                const std::uint64_t payloadBytes = cursor;
                if (payloadBytes == 0u || payloadBytes > maximumCaptureBytes ||
                    payloadBytes > state.device.maxBufferLength) {
                    return skip("aggregate_capture_cap_or_device_buffer_limit");
                }
                id<MTLBuffer> snapshot = [state.device newBufferWithLength:
                    static_cast<NSUInteger>(payloadBytes)
                    options:MTLResourceStorageModeShared];
                if (snapshot == nil) return skip("snapshot_allocation_failed");
                std::memset(snapshot.contents, 0,
                    static_cast<std::size_t>(payloadBytes));
                [encoder endEncoding]; encoder = nil;
                id<MTLBlitCommandEncoder> copy = [commandBuffer blitCommandEncoder];
                if (copy == nil) {
                    std::fprintf(stderr,
                        "fem_base_cgls_state_capture_skipped={\"schema\":1,\"root\":%u,\"newton_iteration\":%u,\"reason\":\"blit_encoder_unavailable\",\"physical_state_changed\":false}\n",
                        request.controlStep, nonlinearIteration);
                    encoder = [commandBuffer computeCommandEncoder];
                    return encoder != nil;
                }
                for (std::size_t index = 0u; index < arrays.size(); ++index) {
                    if (arrays[index].byteCount == 0u) continue;
                    [copy copyFromBuffer:sourceBuffers[index] sourceOffset:0u
                        toBuffer:snapshot destinationOffset:
                            static_cast<NSUInteger>(arrays[index].offset)
                        size:static_cast<NSUInteger>(arrays[index].byteCount)];
                }
                [copy endEncoding];
                BaseCGLSStateCaptureMetadata metadata;
                metadata.controlStep = request.controlStep;
                metadata.physicsSubstep = request.physicsSubstep;
                metadata.microtick = micro.microtick;
                metadata.microtickCount = micro.microtickCount;
                metadata.newtonIteration = nonlinearIteration;
                metadata.environmentCount = state.dispatch.environmentCount;
                metadata.objectCount = state.dispatch.objectCount;
                metadata.materialCount = state.dispatch.materialCount;
                metadata.parameterCount = state.dispatch.parameterCount;
                metadata.nodeCount = state.dispatch.femNodeCount;
                metadata.tetrahedronCount = state.dispatch.tetrahedronCount;
                metadata.materialStateStride = state.dispatch.materialStateStride;
                metadata.unknownCount = state.fgmresLayout.unknownCount;
                metadata.tangentCacheEnabled =
                    state.mixedSolverValue.executionBudgets.w == 1u ? 1u : 0u;
                metadata.tangentCacheForceRebuild =
                    state.forceCompletedTangentCacheRebuild ? 1u : 0u;
                metadata.mixedSolverPreconditioner =
                    state.mixedSolverValue.executionBudgets.z;
                metadata.cglsIterations = femBaseCGLSIterations;
                metadata.microFlags = micro.flags;
                metadata.dispatchABI = state.dispatch.abiVersion;
                metadata.dispatchFlags = state.dispatch.flags;
                metadata.rigidGeneralizedCapacity =
                    state.dispatch.rigidGeneralizedCapacity;
                metadata.sourcePhysicsFingerprint =
                    state.sourcePhysicsFingerprint;
                metadata.worldFingerprint = state.worldFingerprint;
                metadata.executionFingerprint = state.executionFingerprint;
                metadata.deviceRegistryID =
                    static_cast<std::uint64_t>(state.device.registryID);
                metadata.deviceName = nsString(state.device.name);
                metadata.cglsCaptureFile =
                    std::filesystem::path(femBaseCGLSCapturePath).filename().string();
                metadata.dtSeconds = micro.time.x;
                metadata.inverseDtSeconds = micro.time.y;
                // Read the exact validity words only in the asynchronous callback;
                // the manifest keeps the count as an explicit post-read check.
                const auto capturedArrays = arrays;
                const auto capturedMetadata = metadata;
                const auto capturedPath = femBaseCGLSCapturePath + ".state";
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status != MTLCommandBufferStatusCompleted) {
                        std::fprintf(stderr,
                            "fem_base_cgls_state_capture_skipped={\"schema\":1,\"root\":%u,\"newton_iteration\":%u,\"reason\":\"command_buffer_not_completed\",\"command_buffer_status\":%u,\"physical_state_changed\":false}\n",
                            capturedMetadata.controlStep,
                            capturedMetadata.newtonIteration,
                            static_cast<unsigned>(completed.status));
                        return;
                    }
                    const auto* raw = static_cast<const unsigned char*>(snapshot.contents);
                    std::uint32_t validCount = 0u;
                    for (const auto& array : capturedArrays) {
                        if (array.name != "algorithmic_tangent_cache_valid") continue;
                        const auto* valid = reinterpret_cast<const std::uint32_t*>(
                            raw + array.offset);
                        const std::size_t count = static_cast<std::size_t>(
                            array.byteCount / sizeof(std::uint32_t));
                        for (std::size_t i = 0u; i < count; ++i)
                            validCount += valid[i] != 0u ? 1u : 0u;
                    }
                    auto finalMetadata = capturedMetadata;
                    finalMetadata.tangentCacheValidCount = validCount;
                    writeBaseCGLSStateCapture(snapshot.contents, payloadBytes,
                        capturedArrays, finalMetadata, capturedPath);
                }];
                encoder = [commandBuffer computeCommandEncoder];
                if (encoder == nil) {
                    diagnostics.message =
                        "CGLS state capture continuation encoder unavailable";
                    return false;
                }
                [encoder setLabel:@"Numi Matter CGLS exact input state capture continuation"];
                return true;
            };

'''
if s.count(lambda_anchor) != 1: raise SystemExit('lambda anchor not unique')
s = s.replace(lambda_anchor, lambda_insert + lambda_anchor)

primary_call = '''                    encodeBaseCauchyDirection();
                    if (!encodeBaseCauchyDirectionTrace(true)) {
'''
primary_repl = '''                    if (!encodeBaseCGLSStateCapture()) {
                        ownership->preDynamicsOpen = false;
                        return diagnostics;
                    }
                    encodeBaseCauchyDirection();
                    if (!encodeBaseCauchyDirectionTrace(true)) {
'''
if s.count(primary_call) != 1: raise SystemExit('primary call anchor not unique')
s = s.replace(primary_call, primary_repl)
fallback_call = '''                if (directionPass == 1u) {
                    encodeBaseCauchyDirection();
                    if (!encodeBaseCauchyDirectionTrace()) {
'''
fallback_repl = '''                if (directionPass == 1u) {
                    if (!encodeBaseCGLSStateCapture()) {
                        ownership->preDynamicsOpen = false;
                        return diagnostics;
                    }
                    encodeBaseCauchyDirection();
                    if (!encodeBaseCauchyDirectionTrace()) {
'''
if s.count(fallback_call) != 1: raise SystemExit('fallback call anchor not unique')
s = s.replace(fallback_call, fallback_repl)

candidate = OUT / 'candidate/matter/src/runtime.mm'
candidate.parent.mkdir(parents=True, exist_ok=True)
candidate.write_bytes(s.encode('utf-8'))
old_lines = raw.decode('utf-8').splitlines(keepends=True)
new_lines = s.splitlines(keepends=True)
patch = ''.join(difflib.unified_diff(old_lines, new_lines,
    fromfile='a/matter/src/runtime.mm', tofile='b/matter/src/runtime.mm'))
(OUT / 'overlay.patch').write_text(patch)
manifest = {
  'schema':'numi.matter.source_overlay.v1',
  'base_snapshot':str(BASE),
  'base_file':'matter/src/runtime.mm',
  'base_sha256':EXPECTED,
  'candidate_file':'candidate/matter/src/runtime.mm',
  'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
  'patch_file':'overlay.patch',
  'patch_sha256':hashlib.sha256((OUT/'overlay.patch').read_bytes()).hexdigest(),
  'scope':'diagnostic-only opt-in exact state sidecar for selected CGLS Newton capture; no shader or physical-state writes',
  'gpu_or_build_run':False,
}
(OUT/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(manifest,indent=2,sort_keys=True))
