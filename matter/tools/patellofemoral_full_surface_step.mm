#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif
#ifndef NUMI_MATTER_FIXTURE_MATERIAL
#define NUMI_MATTER_FIXTURE_MATERIAL ""
#endif

namespace {

void require(bool okay, const std::string& message) {
    if (!okay) throw std::runtime_error(message);
}

template <typename T>
void readExact(std::ifstream& stream, T* data, std::size_t count,
               const char* label) {
    stream.read(reinterpret_cast<char*>(data),
                static_cast<std::streamsize>(count * sizeof(T)));
    require(stream.gcount() == static_cast<std::streamsize>(count * sizeof(T)),
            std::string("short source input: ") + label);
}

struct Input {
    std::uint32_t side = 0u;
    std::array<std::uint64_t, 4> referenceIdentity{};
    std::array<double, 3> translation{};
    std::vector<std::array<double, 3>> ptcReference;
    std::vector<std::array<double, 3>> ptcCurrent;
    std::vector<std::array<std::uint32_t, 4>> ptcTetrahedra;
    std::vector<std::array<double, 3>> fmcReference;
    std::vector<std::array<std::uint32_t, 4>> fmcTetrahedra;
    std::vector<std::uint32_t> ptcFixedNodes;
};

std::vector<std::uint32_t> readFixedNodes(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(stream.good(), "cannot open PTC fixed-node set");
    const auto bytes = static_cast<std::streamoff>(stream.tellg());
    require(bytes > 0 && bytes % sizeof(std::uint32_t) == 0 &&
            bytes <= static_cast<std::streamoff>(26121u * sizeof(std::uint32_t)),
            "PTC fixed-node set has invalid length");
    stream.seekg(0);
    std::vector<std::uint32_t> nodes(
        static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
    readExact(stream, nodes.data(), nodes.size(), "PTC fixed-node set");
    require(nodes.front() < 26121u && nodes.back() < 26121u &&
            std::adjacent_find(nodes.begin(), nodes.end(),
                [](const std::uint32_t a, const std::uint32_t b) {
                    return a >= b;
                }) == nodes.end(),
            "PTC fixed-node indices must be unique, sorted and in range");
    return nodes;
}

std::vector<std::array<double, 3>> readPositions(
        std::ifstream& stream, std::uint32_t count) {
    std::vector<float> packed(static_cast<std::size_t>(count) * 3u);
    readExact(stream, packed.data(), packed.size(), "positions");
    std::vector<std::array<double, 3>> positions(count);
    for (std::uint32_t index = 0u; index < count; ++index)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const float value = packed[3u * index + axis];
            require(std::isfinite(value), "nonfinite source position");
            positions[index][axis] = value;
        }
    return positions;
}

std::vector<std::array<std::uint32_t, 4>> readTetrahedra(
        std::ifstream& stream, std::uint32_t count, std::uint32_t nodes) {
    std::vector<std::array<std::uint32_t, 4>> tetrahedra(count);
    readExact(stream, tetrahedra.data(), count, "tetrahedra");
    for (const auto& cell : tetrahedra)
        for (const auto index : cell)
            require(index < nodes, "source tetrahedron has an invalid node");
    return tetrahedra;
}

Input readInput(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open source cartilage input");
    char magic[8];
    std::uint32_t meta[7];
    std::uint8_t hashes[64];
    Input input;
    readExact(stream, magic, 8u, "magic");
    readExact(stream, meta, 7u, "header");
    readExact(stream, hashes, 64u, "source identities");
    readExact(stream, input.translation.data(), 3u, "pose translation");
    require(std::memcmp(magic, "NHCAR1\0\0", 8u) == 0 &&
            meta[0] == 1u && meta[1] <= 1u && meta[6] == 0u &&
            meta[2] == 26121u && meta[3] == 121105u &&
            meta[4] == 24870u && meta[5] == 87072u,
            "unexpected source cartilage input header");
    input.side = meta[1];
    std::memcpy(input.referenceIdentity.data(), hashes, 32u);
    for (double value : input.translation)
        require(std::isfinite(value) && std::abs(value) < 1.0e-3,
                "invalid supplied pose translation");
    input.ptcReference = readPositions(stream, meta[2]);
    input.ptcCurrent = readPositions(stream, meta[2]);
    input.ptcTetrahedra = readTetrahedra(stream, meta[3], meta[2]);
    input.fmcReference = readPositions(stream, meta[4]);
    input.fmcTetrahedra = readTetrahedra(stream, meta[5], meta[4]);
    require(stream.peek() == std::char_traits<char>::eof(),
            "source cartilage input has trailing bytes");
    return input;
}

numi::matter::CompiledWorld cook(const Input& input, bool baseline,
                                double contactSlop, double approachSpeed,
                                bool disableContact) {
    auto material = numi::matter::parseMatterFile(NUMI_MATTER_FIXTURE_MATERIAL);
    require(material.succeeded(), "synthetic preflight material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-6;
    source.gravity = {0.0, 0.0, 0.0};
    source.contactSlop = contactSlop;
    source.mixedSolver.newtonIterations = 8u;
    source.materials.push_back(std::move(material.material));
    for (std::uint32_t side = 0u; side < 2u; ++side) {
        numi::matter::ObjectSource object;
        object.name = side == 0u ? "source_patellar_cartilage" :
                                  "source_femoral_cartilage";
        object.representation = numi::matter::Representation::fem;
        object.materialIndex = 0u;
        object.mixedFEM = false;
        object.deformableContact = !disableContact;
        object.deformableSelfContact = false;
        object.characteristicLength = 0.001;
        object.femCapacity.deformableContacts = 65536u;
        if (side == 0u) {
            object.femNodes = baseline ? input.ptcReference : input.ptcCurrent;
            object.femReferenceNodes = input.ptcReference;
            object.femReferenceSourceIdentity = input.referenceIdentity;
            object.femFixedNodes = input.ptcFixedNodes;
            const double length = std::sqrt(
                input.translation[0] * input.translation[0] +
                input.translation[1] * input.translation[1] +
                input.translation[2] * input.translation[2]);
            require(length > 0.0, "source pose has no approach direction");
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                object.femInitialVelocity[axis] =
                    -approachSpeed * input.translation[axis] / length;
            for (const auto& cell : input.ptcTetrahedra)
                object.tetrahedra.push_back({cell});
        } else {
            object.femNodes = input.fmcReference;
            for (const auto& cell : input.fmcTetrahedra)
                object.tetrahedra.push_back({cell});
        }
        source.objects.push_back(std::move(object));
    }
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    auto cooked = numi::matter::compileWorld(source, options);
    std::string diagnostic = "full source cartilage world did not compile";
    for (const auto& row : cooked.diagnostics)
        diagnostic += "; " + row.message;
    require(cooked.succeeded(), diagnostic);
    require(cooked.world.fem.nodes.size() == 50991u &&
            cooked.world.fem.tetrahedra.size() == 208177u,
            "full source cartilage dimensions drifted");
    return std::move(cooked.world);
}

void run(const numi::matter::CompiledWorld& world,
         const char* outputPath, bool baseline, std::uint32_t side,
         double contactSlop, double approachSpeed,
         bool disableContact,
         const std::vector<std::uint32_t>& ptcFixedNodes) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MRMetalWorldStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_SUCCESS;
        id<MTLBuffer> statuses = [device
            newBufferWithBytes:&initialStatus length:sizeof(initialStatus)
            options:MTLResourceStorageModeShared];
        require(queue != nil && statuses != nil, "Metal setup failed");
        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(world, {
            .metallib = NUMI_MATTER_METALLIB,
            .environmentCount = 1u,
            .captureEvents = false,
            .captureDiagnostics = true,
            .automaticIdentification = false,
            .adaptiveTransfer = false,
        });
        require(initialized.encoded && runtime.valid(),
                "runtime initialization: " + initialized.message);
        const auto before = runtime.snapshot();
        require(before.available && before.femNodes.size() == 50991u,
                "initial snapshot: " + before.message);
        id<MTLBuffer> rawConstraintReactions = (__bridge id<MTLBuffer>)
            runtime.femConstraintReactionBuffer();
        const NSUInteger reactionBytes = static_cast<NSUInteger>(
            world.fem.nodes.size() * sizeof(nm_float4));
        id<MTLBuffer> reactionSnapshot = rawConstraintReactions == nil
            ? nil
            : [device newBufferWithLength:reactionBytes
                options:MTLResourceStorageModeShared];
        require(rawConstraintReactions != nil && reactionSnapshot != nil &&
                rawConstraintReactions.length >= reactionBytes,
                "FEM constraint-reaction readback is unavailable");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.controlStep = 0u;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = numi::matter::EncodePhase::preDynamics;
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "preDynamics: " + encoded.message);
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "FEM reaction copy encoder is unavailable");
        [blit copyFromBuffer:rawConstraintReactions sourceOffset:0u
                  toBuffer:reactionSnapshot destinationOffset:0u
                      size:reactionBytes];
        [blit endEncoding];
        request.phase = numi::matter::EncodePhase::postCommit;
        encoded = runtime.encode(request);
        require(encoded.encoded, "postCommit: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Metal command did not complete: " +
                std::string(command.error == nil ? "unknown" :
                    [[command.error localizedDescription] UTF8String]));
        const auto after = runtime.snapshot();
        require(after.available && after.statuses.size() == 1u &&
                after.femNodes.size() == before.femNodes.size(),
                "completion snapshot failed");
        std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
        require(output.good(), "cannot create accepted position buffer");
        bool rolledBack = true;
        double maximumMovement = 0.0;
        double initialKineticEnergy = 0.0;
        double acceptedKineticEnergy = 0.0;
        double maximumFixedNodeMovement = 0.0;
        std::size_t movedFixedNodes = 0u;
        std::array<double, 3> momentumChange{};
        for (std::size_t index = 0u; index < after.femNodes.size(); ++index) {
            const auto& a = before.femNodes[index];
            const auto& b = after.femNodes[index];
            rolledBack &= std::memcmp(&a, &b, sizeof(a)) == 0;
            const auto& p = b.positionAndMass;
            const std::array<float, 3> xyz{p.x, p.y, p.z};
            output.write(reinterpret_cast<const char*>(xyz.data()),
                         3u * sizeof(float));
            const double dx = double(p.x) - a.positionAndMass.x;
            const double dy = double(p.y) - a.positionAndMass.y;
            const double dz = double(p.z) - a.positionAndMass.z;
            maximumMovement = std::max(maximumMovement,
                std::sqrt(dx * dx + dy * dy + dz * dz));
            const auto kinetic = [](const NMFEMNodeStateGPU& node) {
                const auto& v = node.velocityAndInverseMass;
                return 0.5 * double(node.positionAndMass.w) *
                    (double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
            };
            initialKineticEnergy += kinetic(a);
            acceptedKineticEnergy += kinetic(b);
            const auto& beforeVelocity = a.velocityAndInverseMass;
            const auto& afterVelocity = b.velocityAndInverseMass;
            const std::array<double, 3> beforeVelocityXYZ{
                beforeVelocity.x, beforeVelocity.y, beforeVelocity.z};
            const std::array<double, 3> afterVelocityXYZ{
                afterVelocity.x, afterVelocity.y, afterVelocity.z};
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                momentumChange[axis] += double(a.positionAndMass.w) *
                    (afterVelocityXYZ[axis] - beforeVelocityXYZ[axis]);
        }
        std::array<double, 3> fixedReaction{};
        std::array<double, 3> fixedCentroid{};
        std::array<double, 3> fixedMomentAboutOrigin{};
        double fixedReactionL1 = 0.0;
        double maximumFixedReaction = 0.0;
        const auto* reactions = static_cast<const nm_float4*>(
            reactionSnapshot.contents);
        for (const std::uint32_t index : ptcFixedNodes) {
            const auto& a = before.femNodes[index].positionAndMass;
            const auto& b = after.femNodes[index].positionAndMass;
            const double movement = std::sqrt(
                std::pow(double(b.x) - a.x, 2) +
                std::pow(double(b.y) - a.y, 2) +
                std::pow(double(b.z) - a.z, 2));
            maximumFixedNodeMovement = std::max(
                maximumFixedNodeMovement, movement);
            movedFixedNodes += movement != 0.0;
            fixedCentroid[0] += a.x;
            fixedCentroid[1] += a.y;
            fixedCentroid[2] += a.z;
            if (after.statuses[0].code == NM_STATUS_SUCCESS) {
                const nm_float4 reaction = reactions[index];
                const double magnitude = std::sqrt(
                    double(reaction.x) * reaction.x +
                    double(reaction.y) * reaction.y +
                    double(reaction.z) * reaction.z);
                require(std::isfinite(magnitude),
                        "accepted FEM bone-tie reaction is non-finite");
                fixedReaction[0] += reaction.x;
                fixedReaction[1] += reaction.y;
                fixedReaction[2] += reaction.z;
                fixedMomentAboutOrigin[0] +=
                    double(a.y) * reaction.z - double(a.z) * reaction.y;
                fixedMomentAboutOrigin[1] +=
                    double(a.z) * reaction.x - double(a.x) * reaction.z;
                fixedMomentAboutOrigin[2] +=
                    double(a.x) * reaction.y - double(a.y) * reaction.x;
                fixedReactionL1 += magnitude;
                maximumFixedReaction = std::max(maximumFixedReaction, magnitude);
            }
        }
        if (!ptcFixedNodes.empty())
            for (double& component : fixedCentroid)
                component /= static_cast<double>(ptcFixedNodes.size());
        const std::array<double, 3> fixedMoment{
            fixedMomentAboutOrigin[0] -
                (fixedCentroid[1] * fixedReaction[2] -
                 fixedCentroid[2] * fixedReaction[1]),
            fixedMomentAboutOrigin[1] -
                (fixedCentroid[2] * fixedReaction[0] -
                 fixedCentroid[0] * fixedReaction[2]),
            fixedMomentAboutOrigin[2] -
                (fixedCentroid[0] * fixedReaction[1] -
                 fixedCentroid[1] * fixedReaction[0]),
        };
        const double fixedMomentMagnitude = std::sqrt(
            fixedMoment[0] * fixedMoment[0] +
            fixedMoment[1] * fixedMoment[1] +
            fixedMoment[2] * fixedMoment[2]);
        const double fixedReactionResultant = std::sqrt(
            fixedReaction[0] * fixedReaction[0] +
            fixedReaction[1] * fixedReaction[1] +
            fixedReaction[2] * fixedReaction[2]);
        const double momentumRateResidual = std::sqrt(
            std::pow(momentumChange[0] / world.dispatch.gravityAndTimestep.w +
                     fixedReaction[0], 2) +
            std::pow(momentumChange[1] / world.dispatch.gravityAndTimestep.w +
                     fixedReaction[1], 2) +
            std::pow(momentumChange[2] / world.dispatch.gravityAndTimestep.w +
                     fixedReaction[2], 2));
        require(output.good(), "accepted position write failed");
        std::size_t activeHistories = 0u;
        double barrierImpulseMagnitude = 0.0;
        for (const auto& history : after.deformableContactHistories)
            if (history.laggedTangentAndFriction.w > 0.5f) {
                ++activeHistories;
                barrierImpulseMagnitude += std::abs(
                    double(history.normalAndBarrier.w));
            }
        const auto& status = after.statuses[0];
        const std::string reactionPath =
            std::string(outputPath) + ".reactions.f32le";
        if (status.code == NM_STATUS_SUCCESS) {
            std::ofstream reactionsOutput(reactionPath,
                std::ios::binary | std::ios::trunc);
            require(reactionsOutput.good(),
                    "cannot create accepted FEM reaction stream");
            reactionsOutput.write(
                static_cast<const char*>(reactionSnapshot.contents),
                static_cast<std::streamsize>(reactionBytes));
            require(reactionsOutput.good(),
                    "accepted FEM reaction stream write failed");
        } else {
            std::remove(reactionPath.c_str());
        }
        std::printf("{\"device\":\"%s\",\"abi\":%u,\"side\":%u,"
                    "\"baseline\":%s,\"source_nodes\":%zu,\"source_tetrahedra\":%zu,"
                    "\"contact_slop_m\":%.9g,\"approach_speed_mps\":%.9g,"
                    "\"contact_disabled\":%s,"
                    "\"ptc_fixed_node_count\":%zu,"
                    "\"ptc_fixed_nodes_moved\":%zu,"
                    "\"maximum_fixed_node_movement_m\":%.9g,"
                    "\"fixed_tie_reaction_l1_n\":%.12g,"
                    "\"fixed_tie_reaction_resultant_n\":%.12g,"
                    "\"fixed_tie_reaction_max_node_n\":%.12g,"
                    "\"fixed_tie_reaction_xyz_n\":[%.12g,%.12g,%.12g],"
                    "\"fixed_tie_centroid_m\":[%.12g,%.12g,%.12g],"
                    "\"fixed_tie_moment_about_centroid_nm\":[%.12g,%.12g,%.12g],"
                    "\"fixed_tie_moment_magnitude_nm\":%.12g,"
                    "\"momentum_reaction_residual_n\":%.12g,"
                    "\"accepted_reaction_stream_bytes\":%zu,"
                    "\"surface_faces\":%zu,\"status_code\":%u,"
                    "\"completed_microsteps\":%u,\"failing_index\":%u,"
                    "\"active_deformable_histories\":%zu,"
                    "\"rollback_bitwise\":%s,\"maximum_movement_m\":%.9g,"
                    "\"initial_kinetic_energy_j\":%.12g,"
                    "\"accepted_kinetic_energy_j\":%.12g,"
                    "\"accepted_barrier_impulse_magnitude_raw\":%.12g,"
                    "\"diagnostics\":[%.9g,%.9g,%.9g,%.9g]}\n",
                    [[device name] UTF8String], NM_MATTER_ABI_VERSION, side,
                    baseline ? "true" : "false", world.fem.nodes.size(),
                    world.fem.tetrahedra.size(), contactSlop, approachSpeed,
                    disableContact ? "true" : "false",
                    ptcFixedNodes.size(), movedFixedNodes,
                    maximumFixedNodeMovement,
                    fixedReactionL1, fixedReactionResultant,
                    maximumFixedReaction,
                    fixedReaction[0], fixedReaction[1], fixedReaction[2],
                    fixedCentroid[0], fixedCentroid[1], fixedCentroid[2],
                    fixedMoment[0], fixedMoment[1], fixedMoment[2],
                    fixedMomentMagnitude,
                    momentumRateResidual,
                    status.code == NM_STATUS_SUCCESS
                        ? static_cast<std::size_t>(reactionBytes) : 0u,
                    world.fem.surfaceFaces.size(),
                    status.code, status.completedMicrosteps, status.failingIndex,
                    activeHistories, rolledBack ? "true" : "false",
                    maximumMovement, initialKineticEnergy,
                    acceptedKineticEnergy, barrierImpulseMagnitude,
                    double(status.diagnostics.x), double(status.diagnostics.y),
                    double(status.diagnostics.z), double(status.diagnostics.w));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 3,
                "usage: probe source.nhcar accepted-positions.f32le "
                "[--baseline] [--contact-slop-m value] "
                "[--approach-speed-mps value] [--disable-contact] "
                "[--ptc-fixed-nodes sorted-indices.u32le]");
        bool baseline = false;
        bool disableContact = false;
        double contactSlop = 1.0e-5;
        double approachSpeed = 0.0;
        const char* fixedNodePath = nullptr;
        for (int index = 3; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--baseline")
                baseline = true;
            else if (option == "--disable-contact")
                disableContact = true;
            else if (option == "--contact-slop-m" && index + 1 < argc)
                contactSlop = std::strtod(argv[++index], nullptr);
            else if (option == "--approach-speed-mps" && index + 1 < argc)
                approachSpeed = std::strtod(argv[++index], nullptr);
            else if (option == "--ptc-fixed-nodes" && index + 1 < argc)
                fixedNodePath = argv[++index];
            else
                throw std::runtime_error("unknown native step option");
        }
        require(std::isfinite(contactSlop) && contactSlop > 0.0 &&
                contactSlop <= 1.0e-3, "invalid diagnostic contact slop");
        require(std::isfinite(approachSpeed) && approachSpeed >= 0.0 &&
                approachSpeed <= 1.0, "invalid diagnostic approach speed");
        auto input = readInput(argv[1]);
        if (fixedNodePath != nullptr)
            input.ptcFixedNodes = readFixedNodes(fixedNodePath);
        const auto world = cook(input, baseline, contactSlop, approachSpeed,
                                disableContact);
        run(world, argv[2], baseline, input.side, contactSlop, approachSpeed,
            disableContact, input.ptcFixedNodes);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "patellofemoral full source step: %s\n", error.what());
        return 1;
    }
}
