#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "NumiHumanSeparationForce.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string readFile(const char* path) {
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "missing source input");
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

static std::string extractStructure(const std::string& header,
                                    const char* name) {
    const std::string terminator = std::string("} ") + name + ";";
    const std::size_t end = header.find(terminator);
    check(end != std::string::npos, "missing ABI structure");
    const std::size_t begin = header.rfind("typedef struct ", end);
    check(begin != std::string::npos, "missing ABI structure declaration");
    return header.substr(begin, end + terminator.size() - begin);
}

static bool crossing(const std::uint32_t muscle) {
    return muscle == 1u || muscle == 3u;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            check(argc == 3, "usage: test_separation_forces KERNEL METALROBO_INCLUDE");
            constexpr std::uint32_t environments = 2u;
            constexpr std::uint32_t muscles = 4u;
            constexpr std::uint32_t endpoints = 2u * muscles;
            constexpr std::uint32_t dofs = 3u;
            constexpr std::uint32_t releaseStep = 100u;
            constexpr std::uint32_t forceOffset = environments * muscles * dofs;
            constexpr std::uint32_t forceStride = dofs;
            const std::size_t forceElements = forceOffset + environments * forceStride;
            const std::size_t transferElements = environments * endpoints;

            const std::string tendonHeader = readFile(argv[2]);
            std::string source =
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "#define MR_ALIGN16 alignas(16)\n"
                "using mr_u32 = uint;\n"
                "using mr_float4 = float4;\n";
            source += extractStructure(tendonHeader,
                "MRNumiHumanTendonBindingGPU");
            source += "\n";
            source += extractStructure(tendonHeader,
                "MRNumiHumanTendonTransferResultGPU");
            source += "\n" + readFile(argv[1]);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            check(device != nil, "Metal device unavailable");
            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions =
                MTLMathFloatingPointFunctionsPrecise;
            options.languageVersion = MTLLanguageVersion4_0;
            id<MTLLibrary> library = [device
                newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                             options:options error:&error];
            if (library == nil) {
                const std::string detail = error.localizedDescription == nil
                    ? "Metal source compile failed"
                    : error.localizedDescription.UTF8String;
                throw std::runtime_error(detail);
            }

            const std::array<std::uint32_t, 2u> crossingMuscles{{1u, 3u}};
            metalrobo::NumiHumanSeparationForceAdapter adapter;
            check(adapter.initialize(library, crossingMuscles, releaseStep,
                                     0x5345504152415445ull),
                  "force-release adapter initialization failed");
            const auto program = adapter.program();
            check(program.valid(), "force-release callback program is invalid");

            std::vector<float> originalForces(forceElements, 0.0f);
            for (std::uint32_t environment = 0u;
                 environment < environments; ++environment) {
                for (std::uint32_t muscle = 0u; muscle < muscles; ++muscle) {
                    for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                        const std::size_t row =
                            (environment * muscles + muscle) * dofs + dof;
                        originalForces[row] = 10.0f * (muscle + 1u) +
                            100.0f * environment + static_cast<float>(dof);
                    }
                }
                for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                    float reduced = 1000.0f * (environment + 1u) + dof;
                    for (std::uint32_t muscle = 0u; muscle < muscles; ++muscle) {
                        reduced += originalForces[
                            (environment * muscles + muscle) * dofs + dof];
                    }
                    originalForces[forceOffset + environment * forceStride + dof] =
                        reduced;
                }
            }

            std::vector<MRNumiHumanTendonBindingGPU> originalBindings(endpoints);
            std::vector<MRNumiHumanTendonTransferResultGPU>
                originalTransfers(transferElements);
            std::vector<float> originalCorrections(
                transferElements * dofs, 0.0f);
            for (std::uint32_t endpoint = 0u; endpoint < endpoints; ++endpoint) {
                auto& binding = originalBindings[endpoint];
                binding.muscleIndex = endpoint / 2u;
                binding.endpointOrdinal = endpoint % 2u;
                binding.mode = MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
                binding.envelopeIndex = MR_INVALID_INDEX;
                binding.sourceLocalPoint = {0.0f, 0.0f, 0.0f, 0.0f};
                for (std::uint32_t environment = 0u;
                     environment < environments; ++environment) {
                    const std::size_t transferIndex =
                        environment * endpoints + endpoint;
                    auto& transfer = originalTransfers[transferIndex];
                    transfer.status = endpoint == 6u
                        ? MR_NUMI_HUMAN_TENDON_TRANSFER_NONFINITE_RESULT
                        : MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
                    transfer.environment = environment;
                    transfer.bindingIndex = endpoint;
                    transfer.envelopeIndex = MR_INVALID_INDEX;
                    transfer.terminalWorldForce = {
                        1.0f + endpoint, 2.0f + environment,
                        3.0f + endpoint, 0.0f};
                    for (std::uint32_t node = 0u; node < 4u; ++node) {
                        transfer.nodalWorldForces[node] = {
                            4.0f + endpoint, 5.0f + node,
                            6.0f + environment, 0.0f};
                    }
                    transfer.residualsAndForce = {
                        0.01f + endpoint, 0.02f + environment,
                        0.03f + endpoint, 80.0f + endpoint};
                    for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                        originalCorrections[transferIndex * dofs + dof] =
                            100.0f * environment + 10.0f * endpoint + dof + 0.5f;
                    }
                }
            }

            const auto allocate = [&](const void* bytes, const std::size_t count) {
                return [device newBufferWithBytes:bytes length:count
                    options:MTLResourceStorageModeShared];
            };
            id<MTLBuffer> forceBuffer = allocate(
                originalForces.data(), originalForces.size() * sizeof(float));
            id<MTLBuffer> bindingBuffer = allocate(originalBindings.data(),
                originalBindings.size() * sizeof(originalBindings.front()));
            id<MTLBuffer> transferBuffer = allocate(originalTransfers.data(),
                originalTransfers.size() * sizeof(originalTransfers.front()));
            id<MTLBuffer> correctionBuffer = allocate(originalCorrections.data(),
                originalCorrections.size() * sizeof(float));
            check(forceBuffer != nil && bindingBuffer != nil &&
                  transferBuffer != nil && correctionBuffer != nil,
                  "Metal test buffers unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            check(queue != nil, "Metal command queue unavailable");

            metalrobo::MetalNumiHumanTendonLoadPass pass{};
            pass.bindings = (__bridge void*)bindingBuffer;
            pass.transfers = (__bridge void*)transferBuffer;
            pass.generalizedCorrections = (__bridge void*)correctionBuffer;
            pass.generalizedForces = (__bridge void*)forceBuffer;
            pass.environmentCount = environments;
            pass.endpointCount = endpoints;
            pass.muscleCount = muscles;
            pass.dofCount = dofs;
            pass.generalizedForceStride = forceStride;
            pass.generalizedForceOffset = forceOffset;

            const auto execute = [&](const std::uint32_t step,
                                     const bool expectEncode) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                check(command != nil, "Metal command buffer unavailable");
                pass.commandBuffer = (__bridge void*)command;
                pass.stepIndex = step;
                const bool encoded = program.encodePreDynamics(
                    program.context, pass);
                check(encoded == expectEncode, "unexpected callback encode result");
                if (encoded) {
                    check(program.encodePostValidation(program.context, pass),
                          "post-validation no-op callback rejected");
                }
                [command commit];
                [command waitUntilCompleted];
                check(command.status == MTLCommandBufferStatusCompleted,
                      "Metal callback command failed");
            };

            execute(releaseStep - 1u, true);
            check(std::memcmp(forceBuffer.contents, originalForces.data(),
                              originalForces.size() * sizeof(float)) == 0,
                  "pre-release step changed generalized forces");
            check(std::memcmp(transferBuffer.contents, originalTransfers.data(),
                              originalTransfers.size() *
                                  sizeof(originalTransfers.front())) == 0,
                  "pre-release step changed tendon transfers");
            check(std::memcmp(correctionBuffer.contents,
                              originalCorrections.data(),
                              originalCorrections.size() * sizeof(float)) == 0,
                  "pre-release step changed endpoint corrections");

            // Host shape/buffer validation must reject a shifted field boundary
            // and a truncated endpoint-transfer allocation before encoding.
            pass.stepIndex = releaseStep;
            pass.generalizedForceOffset = forceOffset + 1u;
            id<MTLCommandBuffer> invalidCommand = [queue commandBuffer];
            pass.commandBuffer = (__bridge void*)invalidCommand;
            check(!program.encodePreDynamics(program.context, pass),
                  "invalid reduced-force offset was accepted");
            pass.generalizedForceOffset = forceOffset;
            id<MTLBuffer> shortTransfers = [device
                newBufferWithLength:originalTransfers.size() *
                    sizeof(originalTransfers.front()) - 1u
                options:MTLResourceStorageModeShared];
            pass.transfers = (__bridge void*)shortTransfers;
            check(!program.encodePreDynamics(program.context, pass),
                  "truncated transfer buffer was accepted");
            pass.transfers = (__bridge void*)transferBuffer;

            const auto restore = [&] {
                std::memcpy(forceBuffer.contents, originalForces.data(),
                            originalForces.size() * sizeof(float));
                std::memcpy(transferBuffer.contents, originalTransfers.data(),
                            originalTransfers.size() *
                                sizeof(originalTransfers.front()));
                std::memcpy(correctionBuffer.contents, originalCorrections.data(),
                            originalCorrections.size() * sizeof(float));
            };
            const auto verifyReleased = [&] {
                const auto* forces = static_cast<const float*>(forceBuffer.contents);
                for (std::uint32_t environment = 0u;
                     environment < environments; ++environment) {
                    for (std::uint32_t muscle = 0u; muscle < muscles; ++muscle) {
                        for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                            const std::size_t row =
                                (environment * muscles + muscle) * dofs + dof;
                            check(forces[row] == (crossing(muscle)
                                ? 0.0f : originalForces[row]),
                                "released or unrelated muscle row is incorrect");
                        }
                    }
                    for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                        float expected = originalForces[
                            forceOffset + environment * forceStride + dof];
                        for (std::uint32_t muscle = 0u; muscle < muscles; ++muscle) {
                            if (crossing(muscle)) {
                                expected -= originalForces[
                                    (environment * muscles + muscle) * dofs + dof];
                            }
                        }
                        check(forces[forceOffset + environment * forceStride + dof]
                                  == expected,
                              "reduced force did not subtract released rows once");
                    }
                }

                const auto* transfers = static_cast<
                    const MRNumiHumanTendonTransferResultGPU*>(
                        transferBuffer.contents);
                const auto* corrections = static_cast<const float*>(
                    correctionBuffer.contents);
                for (std::uint32_t environment = 0u;
                     environment < environments; ++environment) {
                    for (std::uint32_t endpoint = 0u;
                         endpoint < endpoints; ++endpoint) {
                        const std::size_t transferIndex =
                            environment * endpoints + endpoint;
                        const bool released = crossing(endpoint / 2u);
                        const auto& actual = transfers[transferIndex];
                        const auto& original = originalTransfers[transferIndex];
                        check(actual.status == original.status,
                              "physical tendon transfer status was changed");
                        check(actual.residualsAndForce.x == original.residualsAndForce.x &&
                              actual.residualsAndForce.y == original.residualsAndForce.y &&
                              actual.residualsAndForce.z == original.residualsAndForce.z,
                              "geometric tendon residual diagnostics changed");
                        check(actual.residualsAndForce.w == (released ? 0.0f
                                  : original.residualsAndForce.w),
                              "represented delivered-force audit is inconsistent");
                        if (released) {
                            check(actual.terminalWorldForce.x == 0.0f &&
                                  actual.terminalWorldForce.y == 0.0f &&
                                  actual.terminalWorldForce.z == 0.0f &&
                                  actual.terminalWorldForce.w == 0.0f,
                                  "released terminal transfer was not zeroed");
                            for (const auto& force : actual.nodalWorldForces) {
                                check(force.x == 0.0f && force.y == 0.0f &&
                                      force.z == 0.0f && force.w == 0.0f,
                                      "released nodal transfer was not zeroed");
                            }
                        } else {
                            check(std::memcmp(&actual, &original, sizeof(actual)) == 0,
                                  "unrelated tendon transfer was changed");
                        }
                        for (std::uint32_t dof = 0u; dof < dofs; ++dof) {
                            const std::size_t index = transferIndex * dofs + dof;
                            check(corrections[index] == (released ? 0.0f
                                : originalCorrections[index]),
                                "endpoint correction release/isolation failed");
                        }
                    }
                }
            };

            restore();
            execute(releaseStep, true);
            verifyReleased();
            std::vector<float> firstForces(forceElements);
            std::vector<MRNumiHumanTendonTransferResultGPU> firstTransfers(
                transferElements);
            std::vector<float> firstCorrections(transferElements * dofs);
            std::memcpy(firstForces.data(), forceBuffer.contents,
                        firstForces.size() * sizeof(float));
            std::memcpy(firstTransfers.data(), transferBuffer.contents,
                        firstTransfers.size() * sizeof(firstTransfers.front()));
            std::memcpy(firstCorrections.data(), correctionBuffer.contents,
                        firstCorrections.size() * sizeof(float));

            // Same-buffer re-application is idempotent. A rejected-step retry
            // is represented by restoring the owner-generated source arena and
            // endpoint data before the same authoritative callback runs again.
            execute(releaseStep, true);
            check(std::memcmp(forceBuffer.contents, firstForces.data(),
                              firstForces.size() * sizeof(float)) == 0,
                  "release callback was not idempotent");
            restore();
            execute(releaseStep, true);
            verifyReleased();
            check(std::memcmp(forceBuffer.contents, firstForces.data(),
                              firstForces.size() * sizeof(float)) == 0 &&
                  std::memcmp(transferBuffer.contents, firstTransfers.data(),
                              firstTransfers.size() * sizeof(firstTransfers.front())) == 0 &&
                  std::memcmp(correctionBuffer.contents, firstCorrections.data(),
                              firstCorrections.size() * sizeof(float)) == 0,
                  "regenerated rejected-step source arena did not replay identically");

            std::cout << "PASS " << device.name.UTF8String
                      << ": release-step boundary, exact row subtraction, endpoint-force cancellation, "
                         "unrelated-row isolation, preserved status/residuals, consistent force audit, and regenerated retry\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << exception.what() << "\n";
            return 1;
        }
    }
}
