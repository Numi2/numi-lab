#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef NUMI_MATTER_STATEFUL_MATERIAL
#define NUMI_MATTER_STATEFUL_MATERIAL ""
#endif

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}


[[nodiscard]] float evaluateProgram(
    const numi::matter::CompiledWorld& world,
    const std::uint32_t programIndex,
    const std::span<const float> parameters,
    const std::span<const float> state,
    const std::array<float, 9>& deformation,
    const std::array<float, 9>& direction,
    const std::array<float, 9>& rate,
    const float timestep,
    const float temperature
) {
    require(
        programIndex < world.scalarPrograms.size(),
        "scalar program index is outside the compiled arena"
    );
    const NMScalarProgramGPU program = world.scalarPrograms[programIndex];
    require(
        program.maximumStack <= NM_EXPRESSION_STACK_CAPACITY &&
            program.firstInstruction <= world.instructions.size() &&
            program.instructionCount <=
                world.instructions.size() - program.firstInstruction,
        "scalar program range is invalid"
    );
    std::array<float, NM_EXPRESSION_STACK_CAPACITY> stack{};
    std::size_t stackSize = 0u;
    const auto push = [&](const float value) {
        require(
            stackSize < stack.size() && std::isfinite(value),
            "scalar bytecode produced an invalid value"
        );
        stack[stackSize++] = value;
    };
    const auto pop = [&]() {
        require(stackSize != 0u, "scalar bytecode stack underflow");
        return stack[--stackSize];
    };
    for (std::uint32_t local = 0u;
         local < program.instructionCount;
         ++local) {
        const NMExpressionInstructionGPU instruction = world.instructions[
            program.firstInstruction + local
        ];
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT:
            push(instruction.immediate.x);
            break;
        case NM_EXPR_PARAMETER:
            require(
                instruction.index < parameters.size(),
                "parameter bytecode index is invalid"
            );
            push(parameters[instruction.index]);
            break;
        case NM_EXPR_F:
            require(instruction.index < deformation.size(), "F index is invalid");
            push(deformation[instruction.index]);
            break;
        case NM_EXPR_DF:
            require(instruction.index < direction.size(), "dF index is invalid");
            push(direction[instruction.index]);
            break;
        case NM_EXPR_RATE:
            require(instruction.index < rate.size(), "D index is invalid");
            push(rate[instruction.index]);
            break;
        case NM_EXPR_STATE:
            require(instruction.index < state.size(), "state index is invalid");
            push(state[instruction.index]);
            break;
        case NM_EXPR_DT:
            push(timestep);
            break;
        case NM_EXPR_TEMPERATURE:
            push(temperature);
            break;
        case NM_EXPR_ADD: {
            const float right = pop();
            const float left = pop();
            push(left + right);
            break;
        }
        case NM_EXPR_SUBTRACT: {
            const float right = pop();
            const float left = pop();
            push(left - right);
            break;
        }
        case NM_EXPR_MULTIPLY: {
            const float right = pop();
            const float left = pop();
            push(left * right);
            break;
        }
        case NM_EXPR_DIVIDE: {
            const float right = pop();
            const float left = pop();
            require(std::abs(right) > 1.0e-20f, "bytecode division by zero");
            push(left / right);
            break;
        }
        case NM_EXPR_NEGATE:
            push(-pop());
            break;
        case NM_EXPR_LOG: {
            const float value = pop();
            require(value > 0.0f, "bytecode logarithm domain error");
            push(std::log(value));
            break;
        }
        case NM_EXPR_EXP:
            push(std::exp(pop()));
            break;
        case NM_EXPR_SQRT: {
            const float value = pop();
            require(value >= 0.0f, "bytecode square-root domain error");
            push(std::sqrt(value));
            break;
        }
        case NM_EXPR_ABS:
            push(std::abs(pop()));
            break;
        case NM_EXPR_MIN: {
            const float right = pop();
            const float left = pop();
            push(std::min(left, right));
            break;
        }
        case NM_EXPR_MAX: {
            const float right = pop();
            const float left = pop();
            push(std::max(left, right));
            break;
        }
        case NM_EXPR_POW_INTEGER: {
            const float value = pop();
            push(std::pow(value, static_cast<float>(instruction.integer)));
            break;
        }
        case NM_EXPR_CLAMP: {
            const float upper = pop();
            const float lower = pop();
            const float value = pop();
            require(upper >= lower, "bytecode clamp range is invalid");
            push(std::clamp(value, lower, upper));
            break;
        }
        default:
            throw std::runtime_error("unknown scalar bytecode opcode");
        }
    }
    require(stackSize == 1u, "scalar bytecode did not terminate with one value");
    return stack[0];
}

void verifyConstitutiveSemantics(
    const numi::matter::CompiledWorld& world
) {
    require(world.materials.size() == 1u, "semantic check requires one material");
    const NMMaterialGPU& material = world.materials.front();
    std::vector<float> parameters;
    parameters.reserve(world.parameters.size());
    for (const NMParameterRangeGPU parameter : world.parameters) {
        parameters.push_back(parameter.valueAndBounds.x);
    }
    const std::array<float, 9> deformation{
        1.30f, 0.0f, 0.0f,
        0.0f, 0.85f, 0.0f,
        0.0f, 0.0f, 0.85f,
    };
    const std::array<float, 9> zero{};
    const std::array<float, 9> rate{
        2.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, -1.0f,
    };
    constexpr float timestep = 0.01f;
    constexpr float temperature = 293.15f;
    const std::array<float, 2> undamaged{0.0f, 0.0f};
    const std::array<float, 2> halfDamaged{0.5f, 0.0f};

    const float dissipation = evaluateProgram(
        world,
        material.dissipationProgram,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    require(
        std::abs(dissipation - 4500.0f) < 0.5f,
        "compiled dissipation potential has the wrong value"
    );

    const float viscousXX = evaluateProgram(
        world,
        material.viscousStressProgramOffset,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    const float viscousYY = evaluateProgram(
        world,
        material.viscousStressProgramOffset + 4u,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    require(
        std::abs(viscousXX - 3000.0f) < 0.5f &&
            std::abs(viscousYY + 1500.0f) < 0.5f,
        "dissipation derivative did not produce the expected viscous stress"
    );

    std::array<float, 9> unitRateDirection{};
    unitRateDirection[0] = 1.0f;
    const float viscousTangentXX = evaluateProgram(
        world,
        material.viscousTangentProgramOffset,
        parameters,
        undamaged,
        deformation,
        unitRateDirection,
        rate,
        timestep,
        temperature
    );
    require(
        std::abs(viscousTangentXX - 1500.0f) < 0.5f,
        "compiled viscous tangent has the wrong directional response"
    );

    const float elasticUndamaged = evaluateProgram(
        world,
        material.stressProgramOffset,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    const float elasticHalfDamaged = evaluateProgram(
        world,
        material.stressProgramOffset,
        parameters,
        halfDamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    require(
        std::abs(elasticHalfDamaged - 0.5f * elasticUndamaged) <=
            2.0e-4f * std::max(std::abs(elasticUndamaged), 1.0f),
        "state-dependent stored energy did not degrade elastic stress"
    );

    const float nextDamage = evaluateProgram(
        world,
        material.stateUpdateProgramOffset,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    const float nextAccumulatedStrain = evaluateProgram(
        world,
        material.stateUpdateProgramOffset + 1u,
        parameters,
        undamaged,
        deformation,
        zero,
        rate,
        timestep,
        temperature
    );
    require(
        nextDamage > 0.0f && nextDamage < 0.95f &&
            std::abs(nextAccumulatedStrain -
                timestep * std::sqrt(6.0f)) < 1.0e-5f,
        "compiled state evolution has the wrong update semantics"
    );
}

void verifyAdaptiveLayout() {
    const auto parsed = numi::matter::parseMatterFile(
        NUMI_MATTER_STATEFUL_MATERIAL
    );
    require(parsed.succeeded(), "adaptive validation material did not parse");

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 480.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.femPCGIterations = 4u;
    source.materials.push_back(parsed.material);

    numi::matter::RigidProxySource fallback;
    fallback.shape = NM_RIGID_SPHERE;
    fallback.bodyIndex = 0u;
    fallback.sceneBodyIndex = 0u;
    fallback.materialIndex = 0u;
    fallback.radiusOrOffset = 0.01;
    fallback.dynamic = true;
    source.rigidProxies.push_back(fallback);

    numi::matter::ObjectSource object;
    object.name = "adaptive_stateful_mpm";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::mpm;
    object.adaptive = true;
    object.rigidBinding = 0u;
    object.characteristicLength = 0.01;
    object.mpmGridMinimum = {-0.02, -0.02, -0.02};
    object.mpmGridMaximum = {0.02, 0.02, 0.02};
    constexpr double spacing = 0.005;
    constexpr double volume = spacing * spacing * spacing;
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                object.particles.push_back({
                    .position = {
                        -0.005 + spacing * x,
                        -0.005 + spacing * y,
                        -0.005 + spacing * z,
                    },
                    .velocity = {0.0, 0.0, 0.0},
                    .mass = 1100.0 * volume,
                    .referenceVolume = volume,
                });
            }
        }
    }
    source.objects.push_back(std::move(object));

    auto compiled = numi::matter::compileWorld(source);
    require(compiled.succeeded(), "adaptive stateful world did not compile");
    std::string validationError;
    require(
        numi::matter::validateCompiledWorldLayout(
            compiled.world,
            &validationError
        ),
        validationError
    );
}

[[nodiscard]] numi::matter::CompiledWorld compileStatefulWorld() {
    const auto parsed = numi::matter::parseMatterFile(
        NUMI_MATTER_STATEFUL_MATERIAL
    );
    require(parsed.succeeded(), "stateful material did not parse");
    require(
        parsed.material.internalState.size() == 2u &&
            parsed.material.stateUpdateRoots.size() == 2u,
        "stateful material did not retain both authored state variables"
    );
    require(
        parsed.material.dissipationRoot != NM_INVALID_INDEX,
        "stateful material did not retain its dissipation potential"
    );

    numi::matter::WorldSource source;
    source.environmentCount = 2u;
    source.frameTimestep = 1.0 / 480.0;
    source.gravity = {0.0, 0.0, -9.81};
    source.femPCGIterations = 4u;
    source.materials.push_back(parsed.material);

    numi::matter::RigidProxySource plane;
    plane.shape = NM_RIGID_PLANE;
    plane.materialIndex = 0u;
    plane.localCenter = {0.0, 0.0, 1.0};
    plane.radiusOrOffset = 0.0;
    source.rigidProxies.push_back(plane);

    numi::matter::ObjectSource mpm;
    mpm.name = "stateful_mpm";
    mpm.materialIndex = 0u;
    mpm.representation = numi::matter::Representation::mpm;
    mpm.characteristicLength = 0.01;
    mpm.mpmGridMinimum = {-0.03, -0.02, -0.01};
    mpm.mpmGridMaximum = {0.01, 0.02, 0.06};
    constexpr double spacing = 0.005;
    constexpr double volume = spacing * spacing * spacing;
    for (int particleIndex = 0; particleIndex < 2; ++particleIndex) {
        numi::matter::ParticleSource particle;
        particle.position = {
            -0.02 + spacing * static_cast<double>(particleIndex),
            0.0,
            0.025,
        };
        particle.velocity = {
            particleIndex == 0 ? -0.20 : 0.20,
            0.0,
            -0.10,
        };
        particle.mass = 1100.0 * volume;
        particle.referenceVolume = volume;
        mpm.particles.push_back(particle);
    }
    source.objects.push_back(std::move(mpm));

    numi::matter::ObjectSource fem;
    fem.name = "stateful_fem";
    fem.materialIndex = 0u;
    fem.representation = numi::matter::Representation::fem;
    fem.characteristicLength = 0.02;
    fem.femInitialVelocity = {0.15, 0.0, -0.10};
    fem.femNodes = {
        {0.015, -0.01, 0.025},
        {0.035, -0.01, 0.025},
        {0.015,  0.01, 0.025},
        {0.015, -0.01, 0.045},
    };
    fem.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    source.objects.push_back(std::move(fem));

    auto unsupportedRestart = source;
    unsupportedRestart.mixedSolver.fgmresRestart =
        NM_MIXED_FGMRES_RESTART + 1u;
    unsupportedRestart.mixedSolver.fgmresIterations =
        NM_MIXED_FGMRES_RESTART + 1u;
    const auto rejectedRestart =
        numi::matter::compileWorld(unsupportedRestart);
    require(
        !rejectedRestart.succeeded() &&
            std::ranges::any_of(
                rejectedRestart.diagnostics,
                [](const numi::matter::Diagnostic& diagnostic) {
                    return diagnostic.message.find(
                        "FGMRES restart exceeds the compiled basis capacity"
                    ) != std::string::npos;
                }
            ),
        "compiler accepted an unsupported FGMRES restart depth"
    );

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 2u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "stateful mixed world did not compile");
    require(
        compiled.world.dispatch.materialStateStride == 2u &&
            compiled.world.dispatch.stateInitialCount == 2u,
        "compiled state arena does not match authored state width"
    );
    require(
        compiled.world.dispatch.particleCount == 2u &&
            compiled.world.dispatch.tetrahedronCount == 1u,
        "stateful compiler check did not retain both representations"
    );
    require(
        compiled.world.stateInitials.size() == 2u &&
            std::ranges::all_of(
                compiled.world.stateInitials,
                [](const float value) {
                    return value == 0.0f;
                }
            ),
        "authored initial state was not compiled exactly"
    );

    const NMMaterialGPU& material = compiled.world.materials.front();
    require(
        (material.flags & NM_MATERIAL_HAS_STATE) != 0u &&
            (material.flags & NM_MATERIAL_HAS_DISSIPATION) != 0u,
        "compiled material flags omit state or dissipation"
    );
    require(
        material.stateInitialOffset == 0u &&
            material.stateUpdateProgramOffset != NM_INVALID_INDEX &&
            material.viscousStressProgramOffset != NM_INVALID_INDEX &&
            material.viscousTangentProgramOffset != NM_INVALID_INDEX &&
            material.dissipationProgram != NM_INVALID_INDEX,
        "compiled material program ranges are incomplete"
    );

    bool sawState = false;
    bool sawRate = false;
    bool sawTimestep = false;
    for (const NMExpressionInstructionGPU& instruction :
         compiled.world.instructions) {
        sawState = sawState || instruction.opcode == NM_EXPR_STATE;
        sawRate = sawRate || instruction.opcode == NM_EXPR_RATE;
        sawTimestep = sawTimestep || instruction.opcode == NM_EXPR_DT;
    }
    require(
        sawState && sawRate && sawTimestep,
        "compiled bytecode omits a required stateful operator"
    );
    require(
        compiled.world.fingerprint ==
            numi::matter::compiledWorldFingerprint(compiled.world),
        "compiled stateful world does not use the canonical fingerprint"
    );

    verifyConstitutiveSemantics(compiled.world);

    const auto package = std::filesystem::temp_directory_path() /
        ("numi-matter-stateful-" +
         std::to_string(compiled.world.fingerprint) + ".nmatterpack");
    std::string error;
    require(
        numi::matter::writePackage(compiled, package, &error),
        "stateful package write failed"
    );
    numi::matter::CompiledWorld roundTrip;
    std::string generated;
    require(
        numi::matter::readPackage(
            package,
            roundTrip,
            &generated,
            &error
        ),
        "stateful package readback failed"
    );
    std::error_code removeError;
    std::filesystem::remove(package, removeError);
    require(
        roundTrip.fingerprint == compiled.world.fingerprint &&
            roundTrip.stateInitials == compiled.world.stateInitials &&
            roundTrip.dispatch.materialStateStride == 2u,
        "stateful package roundtrip changed executable identity"
    );

    auto mutated = roundTrip;
    mutated.stateInitials.front() = 0.125f;
    require(
        numi::matter::compiledWorldFingerprint(mutated) !=
            roundTrip.fingerprint,
        "canonical fingerprint ignores initial material state"
    );

    const auto requireRejected = [&](
        numi::matter::CompiledWorld candidate,
        const std::string_view expected
    ) {
        candidate.fingerprint =
            numi::matter::compiledWorldFingerprint(candidate);
        std::string validationError;
        require(
            !numi::matter::validateCompiledWorldLayout(
                candidate,
                &validationError
            ),
            "canonical layout validator accepted corrupted topology"
        );
        require(
            validationError.find(expected) != std::string::npos,
            "canonical layout validator returned the wrong diagnostic"
        );
    };

    {
        auto candidate = roundTrip;
        candidate.materials.front().stateInitialOffset += 1u;
        requireRejected(std::move(candidate), "initial-state arena");
    }
    {
        auto candidate = roundTrip;
        candidate.scalarPrograms.at(1u).firstInstruction += 1u;
        requireRejected(std::move(candidate), "overlap or contain a gap");
    }
    {
        auto candidate = roundTrip;
        require(!candidate.mpm.blockLookup.empty(), "MPM block lookup is empty");
        candidate.mpm.blockLookup.front() = NM_INVALID_INDEX;
        requireRejected(std::move(candidate), "outside its grid block range");
    }
    {
        auto candidate = roundTrip;
        require(!candidate.fem.tetrahedra.empty(), "FEM topology is empty");
        candidate.fem.tetrahedra.front().nodes.x = NM_INVALID_INDEX;
        requireRejected(std::move(candidate), "outside the object or repeated");
    }
    {
        auto candidate = roundTrip;
        require(
            !candidate.contact.nodeIncidence.empty(),
            "contact incidence is empty"
        );
        candidate.contact.nodeIncidence.front() =
            static_cast<std::uint32_t>(candidate.contact.pairs.size());
        requireRejected(std::move(candidate), "source indices are invalid");
    }
    {
        auto candidate = roundTrip;
        require(
            !candidate.identification.empty(),
            "identification program is empty"
        );
        candidate.identification.front().identity.z = NM_INVALID_INDEX;
        requireRejected(std::move(candidate), "parameter ownership");
    }

    return roundTrip;
}

void verifyLearnedMaterialRoundTrip() {
    numi::matter::LearnedMaterialSource source;
    source.invariantCount = 5u;
    source.softplusBeta = 2.5f;
    source.determinantFloor = 0.04f;
    source.growthCoefficient = 0.02f;
    numi::matter::LearnedLayerSource hidden;
    hidden.inputWidth = 5u;
    hidden.outputWidth = 2u;
    hidden.inputWeights.assign(10u, 0.125f);
    hidden.biases = {-0.25f, 0.5f};
    source.layers.push_back(hidden);
    numi::matter::LearnedLayerSource output;
    output.inputWidth = 5u;
    output.outputWidth = 1u;
    output.inputWeights.assign(5u, 0.0625f);
    output.recurrentWeights = {0.5f, 0.75f};
    output.biases = {0.125f};
    source.layers.push_back(output);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "numi-matter-stateful-check.nmatnet";
    std::string error;
    require(
        numi::matter::writeLearnedMaterial(source, path, &error), error
    );
    numi::matter::LearnedMaterialSource decoded;
    require(
        numi::matter::readLearnedMaterial(path, decoded, &error), error
    );
    require(
        decoded.invariantCount == source.invariantCount &&
        decoded.layers.size() == source.layers.size() &&
        decoded.layers[0].inputWeights == source.layers[0].inputWeights &&
        decoded.layers[1].recurrentWeights ==
            source.layers[1].recurrentWeights,
        "canonical learned-material roundtrip changed trained weights"
    );
}

} // namespace

int main() {
    try {
        verifyAdaptiveLayout();
        verifyLearnedMaterialRoundTrip();
        const auto world = compileStatefulWorld();
        std::cout
            << "{\"schema\":\"numi.matter.stateful-compiler.v1\""
            << ",\"fingerprint\":" << world.fingerprint
            << ",\"state_stride\":"
            << world.dispatch.materialStateStride
            << ",\"particles\":" << world.dispatch.particleCount
            << ",\"tetrahedra\":" << world.dispatch.tetrahedronCount
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi-matter-stateful-check: " << error.what() << '\n';
        return 1;
    }
}
