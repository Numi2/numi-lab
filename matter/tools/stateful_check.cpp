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
#ifndef NUMI_MATTER_BLOOD_MATERIAL
#define NUMI_MATTER_BLOOD_MATERIAL ""
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
        case NM_EXPR_EXPM1_MINUS_X: {
            const float value = pop();
            push(std::expm1(value) - value);
            break;
        }
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

    source.mixedSolver.fieldSmootherPasses = 2u;
    source.mixedSolver.transportTolerance = 2.0e-4;
    source.mixedSolver.minimumContactSeparationRatio = 2.0e-3;

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

    auto overflowingIterationBudget = source;
    overflowingIterationBudget.mixedSolver.fgmresIterations =
        std::numeric_limits<std::uint32_t>::max();
    const auto rejectedIterationBudget =
        numi::matter::compileWorld(overflowingIterationBudget);
    require(
        !rejectedIterationBudget.succeeded() &&
            std::ranges::any_of(
                rejectedIterationBudget.diagnostics,
                [](const numi::matter::Diagnostic& diagnostic) {
                    return diagnostic.message.find(
                        "FGMRES iteration budget overflows restart-cycle arithmetic"
                    ) != std::string::npos;
                }
            ),
        "compiler accepted an overflowing FGMRES iteration budget"
    );

    auto excessiveIterationBudget = source;
    excessiveIterationBudget.mixedSolver.fgmresIterations =
        NM_MIXED_FGMRES_MAX_ITERATIONS + 1u;
    const auto rejectedExcessiveIterationBudget =
        numi::matter::compileWorld(excessiveIterationBudget);
    require(
        !rejectedExcessiveIterationBudget.succeeded() &&
            std::ranges::any_of(
                rejectedExcessiveIterationBudget.diagnostics,
                [](const numi::matter::Diagnostic& diagnostic) {
                    return diagnostic.message.find(
                        "FGMRES iteration budget exceeds the operational encoding capacity"
                    ) != std::string::npos;
                }
            ),
        "compiler accepted an operationally unbounded FGMRES budget"
    );

    for (const double invalidResidual : {
             std::numeric_limits<double>::denorm_min(),
             std::numeric_limits<double>::max()}) {
        auto unrepresentableResidual = source;
        unrepresentableResidual.mixedSolver.relativeResidual =
            invalidResidual;
        const auto rejectedResidual =
            numi::matter::compileWorld(unrepresentableResidual);
        require(
            !rejectedResidual.succeeded() &&
                std::ranges::any_of(
                    rejectedResidual.diagnostics,
                    [](const numi::matter::Diagnostic& diagnostic) {
                        return diagnostic.message.find(
                            "relative residual is not positive finite FP32 policy"
                        ) != std::string::npos;
                    }
                ),
            "compiler accepted a relative residual that is not executable FP32 policy"
        );
    }

    auto unsupportedSmoother = source;
    unsupportedSmoother.mixedSolver.fieldSmootherPasses =
        NM_MIXED_FIELD_SMOOTHER_MAX_PASSES + 1u;
    const auto rejectedSmoother =
        numi::matter::compileWorld(unsupportedSmoother);
    require(
        !rejectedSmoother.succeeded() &&
            std::ranges::any_of(
                rejectedSmoother.diagnostics,
                [](const numi::matter::Diagnostic& diagnostic) {
                    return diagnostic.message.find(
                        "field smoother exceeds the compiled fixed-pass capacity"
                    ) != std::string::npos;
                }
            ),
        "compiler accepted an unsupported field-smoother pass count"
    );

    auto unsupportedSeparation = source;
    unsupportedSeparation.mixedSolver.minimumContactSeparationRatio = 1.0;
    const auto rejectedSeparation =
        numi::matter::compileWorld(unsupportedSeparation);
    require(
        !rejectedSeparation.succeeded(),
        "compiler accepted a contact separation ratio at the activation edge"
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
    const NMMixedSolverGPU& mixedSolver = compiled.world.mixedSolver;
    require(
        mixedSolver.executionBudgets.x == 2u &&
            mixedSolver.executionBudgets.y ==
                source.mixedSolver.mutationRestarts &&
            mixedSolver.executionBudgets.z == 0u &&
            mixedSolver.executionBudgets.w == 0u,
        "compiled solver budgets do not encode smoother and restart ownership"
    );
    require(
        std::abs(
            mixedSolver.residualTolerances.w -
            static_cast<float>(source.mixedSolver.transportTolerance)
        ) <= 1.0e-8f &&
            std::abs(
                mixedSolver.contactAcceptance.x -
                static_cast<float>(
                    source.mixedSolver.minimumContactSeparationRatio
                )
            ) <= 1.0e-8f &&
            mixedSolver.contactAcceptance.y == 0.0f &&
            mixedSolver.contactAcceptance.z == 0.0f &&
            mixedSolver.contactAcceptance.w == 0.0f,
        "compiled residual and contact-acceptance controls changed meaning"
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
            std::string("canonical layout validator expected '") +
                std::string(expected) + "' but returned '" + validationError + "'"
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
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.residualTolerances.x = -1.0f;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.residualTolerances.x = 0.0f;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.nonlinearIterations.y =
            NM_MIXED_FGMRES_RESTART + 1u;
        candidate.mixedSolver.nonlinearIterations.z =
            NM_MIXED_FGMRES_RESTART + 1u;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.nonlinearIterations.y =
            NM_MIXED_FGMRES_RESTART;
        candidate.mixedSolver.nonlinearIterations.z =
            NM_MIXED_FGMRES_MAX_ITERATIONS + 1u;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.contactAcceptance.x = 1.0f;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        // Selector 5 is a defined graph-Schwarz mode, but this mixed
        // MPM/FEM world is outside its immutable, contact-free FEM scope.
        candidate.mixedSolver.executionBudgets.z = 5u;
        requireRejected(std::move(candidate), "graph Schwarz preconditioner");
    }
    {
        auto candidate = roundTrip;
        // Selector values above the defined range remain a mixed-policy error.
        candidate.mixedSolver.executionBudgets.z = 6u;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.regularization.y = -1.0f;
        requireRejected(std::move(candidate), "mixed solver policy");
    }
    {
        auto candidate = roundTrip;
        candidate.mixedSolver.globalization.w = 1.01f;
        requireRejected(std::move(candidate), "mixed solver policy");
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


void verifyDormantMPMCapacitySlot() {
    using namespace numi::matter;
    const auto parsed = parseMatterFile(NUMI_MATTER_STATEFUL_MATERIAL);
    require(parsed.succeeded(), "dormant-slot material did not parse");

    constexpr double spacing = 0.005;
    constexpr double volume = spacing * spacing * spacing;
    constexpr double mass = 1100.0 * volume;
    ParticleSource legacyAggregate{
        {-0.020, 0.0, 0.025},
        {0.0, 0.0, 0.0},
        mass,
        volume,
    };
    require(
        legacyAggregate.active,
        "four-field ParticleSource aggregate no longer defaults active"
    );

    ParticleSource dormant = legacyAggregate;
    dormant.position[0] += spacing;
    dormant.active = false;

    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 480.0;
    source.gravity = {0.0, 0.0, -9.81};
    source.materials.push_back(parsed.material);

    ObjectSource object;
    object.name = "dormant_capacity_check";
    object.materialIndex = 0u;
    object.representation = Representation::mpm;
    object.characteristicLength = 0.01;
    object.mpmGridMinimum = {-0.03, -0.02, -0.01};
    object.mpmGridMaximum = {0.01, 0.02, 0.06};
    object.deformableContact = false;
    object.particles = {legacyAggregate, dormant};
    source.objects.push_back(std::move(object));

    const auto compiled = compileWorld(source);
    require(compiled.succeeded(), "dormant MPM capacity world did not compile");
    std::string validationError;
    require(
        validateCompiledWorldLayout(compiled.world, &validationError),
        "dormant MPM capacity world failed cooked validation: " + validationError
    );
    require(
        compiled.world.dispatch.particleCount == 2u &&
            compiled.world.objects.front().stateCount == 2u,
        "dormant slot was not retained as bounded compiled capacity"
    );

    const NMParticleStateGPU& activeState = compiled.world.mpm.particles[0];
    const NMParticleStateGPU& dormantState = compiled.world.mpm.particles[1];
    require(
        (activeState.identity.w & NM_OBJECT_ACTIVE) != 0u &&
            (dormantState.identity.w & NM_OBJECT_ACTIVE) == 0u &&
            dormantState.positionAndMass.w > 0.0f &&
            dormantState.velocityAndReferenceVolume.w > 0.0f,
        "dormant slot lacks finite positive capacity data or phase bit"
    );
    require(
        compiled.world.adaptive.front().massAndError.x ==
                activeState.positionAndMass.w &&
            compiled.world.adaptive.front().massAndError.x <
                activeState.positionAndMass.w + dormantState.positionAndMass.w,
        "dormant capacity mass leaked into the physical object mass tally"
    );

    auto incorrectlyActivated = compiled.world;
    incorrectlyActivated.mpm.particles[1].identity.w = NM_OBJECT_ACTIVE;
    incorrectlyActivated.fingerprint =
        compiledWorldFingerprint(incorrectlyActivated);
    require(
        incorrectlyActivated.fingerprint != compiled.world.fingerprint,
        "compiled identity ignores the authored active phase bit"
    );
    validationError.clear();
    require(
        !validateCompiledWorldLayout(incorrectlyActivated, &validationError) &&
            validationError.find("MPM active mass/center tally") !=
                std::string::npos,
        "cooked validation accepted dormant capacity mass as physical mass"
    );

    WorldSource allDormantSource = source;
    allDormantSource.objects.front().particles.front().active = false;
    const auto allDormant = compileWorld(allDormantSource);
    require(
        allDormant.succeeded(),
        "all-dormant MPM receiver capacity did not compile"
    );
    validationError.clear();
    require(
        validateCompiledWorldLayout(allDormant.world, &validationError),
        "all-dormant MPM receiver capacity failed validation: " +
            validationError
    );
    const NMAdaptiveStateGPU& emptyMass = allDormant.world.adaptive.front();
    require(
        emptyMass.massAndError.x == 0.0f &&
            emptyMass.referenceCenter.x == 0.0f &&
            emptyMass.referenceCenter.y == 0.0f &&
            emptyMass.referenceCenter.z == 0.0f &&
            emptyMass.centerAndRadius.x == 0.0f &&
            emptyMass.centerAndRadius.y == 0.0f &&
            emptyMass.centerAndRadius.z == 0.0f,
        "all-dormant receiver slots contributed physical mass or center"
    );
}

using Matrix3 = std::array<float, 9>;

[[nodiscard]] float determinant3(const Matrix3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

[[nodiscard]] Matrix3 inverse3(const Matrix3& a) {
    const float j = determinant3(a);
    require(std::abs(j) > 1.0e-6f, "test deformation is singular");
    return {
        (a[4] * a[8] - a[5] * a[7]) / j,
        (a[2] * a[7] - a[1] * a[8]) / j,
        (a[1] * a[5] - a[2] * a[4]) / j,
        (a[5] * a[6] - a[3] * a[8]) / j,
        (a[0] * a[8] - a[2] * a[6]) / j,
        (a[2] * a[3] - a[0] * a[5]) / j,
        (a[3] * a[7] - a[4] * a[6]) / j,
        (a[1] * a[6] - a[0] * a[7]) / j,
        (a[0] * a[4] - a[1] * a[3]) / j,
    };
}

[[nodiscard]] Matrix3 multiply3(const Matrix3& a, const Matrix3& b) {
    Matrix3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            for (std::size_t k = 0u; k < 3u; ++k) {
                result[3u * row + column] +=
                    a[3u * row + k] * b[3u * k + column];
            }
        }
    }
    return result;
}

[[nodiscard]] Matrix3 transpose3(const Matrix3& a) {
    Matrix3 result{};
    for (std::size_t row = 0u; row < 3u; ++row)
        for (std::size_t column = 0u; column < 3u; ++column)
            result[3u * row + column] = a[3u * column + row];
    return result;
}

[[nodiscard]] Matrix3 expectedNewtonianPK1(
    const Matrix3& f,
    const Matrix3& fdot,
    const float eta
) {
    const float j = determinant3(f);
    const Matrix3 finv = inverse3(f);
    Matrix3 adj = finv;
    for (float& value : adj) value *= j;
    const Matrix3 q = multiply3(fdot, adj);
    // q is Fdot*adj(F); dev(sym(q))/J is the spatial deviatoric rate.
    Matrix3 ddev{};
    const float trace = q[0] + q[4] + q[8];
    for (std::size_t i = 0; i < 9u; ++i)
        ddev[i] = 0.5f * (q[i] + q[(i % 3u) * 3u + i / 3u]);
    const float mean = trace / 3.0f;
    ddev[0] -= mean;
    ddev[4] -= mean;
    ddev[8] -= mean;
    const Matrix3 finvT = transpose3(finv);
    Matrix3 result = multiply3(ddev, finvT);
    const float scale = 2.0f * eta;
    for (float& value : result) value *= scale;
    return result;
}

[[nodiscard]] float expectedRayleighPotential(
    const Matrix3& f,
    const Matrix3& fdot,
    const float eta
) {
    const float j = determinant3(f);
    const Matrix3 finv = inverse3(f);
    Matrix3 adj = finv;
    for (float& value : adj) value *= j;
    const Matrix3 q = multiply3(fdot, adj);
    Matrix3 ddev{};
    const float trace = q[0] + q[4] + q[8];
    for (std::size_t i = 0; i < 9u; ++i)
        ddev[i] = 0.5f * (q[i] + q[(i % 3u) * 3u + i / 3u]);
    const float mean = trace / 3.0f;
    ddev[0] -= mean;
    ddev[4] -= mean;
    ddev[8] -= mean;
    float norm2 = 0.0f;
    for (float value : ddev) norm2 += value * value;
    return eta * norm2 / j;
}

void verifyBloodNewtonianBytecode() {
    using namespace numi::matter;
    const auto parsed = parseMatterFile(NUMI_MATTER_BLOOD_MATERIAL);
    if (!parsed.succeeded()) {
        for (const auto& diagnostic : parsed.diagnostics)
            std::cerr << "blood material parse: " << diagnostic.line << ":"
                << diagnostic.column << " " << diagnostic.message << '\n';
        throw std::runtime_error("blood proxy material did not parse");
    }
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 480.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials.push_back(parsed.material);
    ObjectSource object;
    object.name = "blood_receiver_constitutive_cpu_fixture";
    object.materialIndex = 0u;
    object.representation = Representation::mpm;
    object.characteristicLength = 0.005;
    object.mpmGridMinimum = {0.0, 0.0, 0.0};
    object.mpmGridMaximum = {0.01, 0.01, 0.01};
    object.deformableContact = false;
    object.deformableSelfContact = false;
    ParticleSource particle;
    particle.position = {0.0025, 0.0025, 0.0025};
    particle.mass = 1050.6 * 1.25e-7;
    particle.referenceVolume = 1.25e-7;
    object.particles.push_back(particle);
    source.objects.push_back(object);
    CompileOptions options;
    options.cfl = 0.35;
    options.maximumRateExponent = 8u;
    options.emitSpecializedMetal = false;
    const CompileResult compiled = compileWorld(source, options);
    if (!compiled.succeeded()) {
        for (const auto& diagnostic : compiled.diagnostics)
            std::cerr << "blood material compile: " << diagnostic.line << ":"
                << diagnostic.column << " " << diagnostic.message << '\n';
        throw std::runtime_error("blood proxy material did not compile");
    }
    std::string validationError;
    require(
        validateCompiledWorldLayout(compiled.world, &validationError),
        "blood proxy world failed cooked validation: " + validationError
    );
    require(compiled.world.materials.size() == 1u, "blood proxy world has wrong material count");
    const NMMaterialGPU& material = compiled.world.materials.front();
    require(
        material.stressProgramOffset != NM_INVALID_INDEX &&
            material.viscousStressProgramOffset != NM_INVALID_INDEX &&
            material.viscousTangentProgramOffset != NM_INVALID_INDEX &&
            material.dissipationProgram != NM_INVALID_INDEX,
        "blood proxy lacks compiled stored-energy or viscous bytecode"
    );
    require(
        compiled.world.objects.front().solver.x == 7u,
        "blood proxy CFL exponent does not reflect the 1-5 MPa numerical bulk bound"
    );
    std::vector<float> parameters;
    parameters.reserve(compiled.world.parameters.size());
    for (const NMParameterRangeGPU parameter : compiled.world.parameters)
        parameters.push_back(parameter.valueAndBounds.x);
    constexpr float eta = 0.004f;
    const std::array<float, 2> noState{0.0f, 0.0f};
    const std::array<float, 9> zero{};
    require(parameters.size() == 3u, "blood proxy parameter layout changed");
    const auto evalAt = [&](const std::vector<float>& values, const std::uint32_t program,
                            const Matrix3& f, const Matrix3& d) {
        return evaluateProgram(
            compiled.world, program, values, noState, f, zero, d, 1.0e-4f, 310.15f
        );
    };
    const auto eval = [&](const std::uint32_t program, const Matrix3& f, const Matrix3& d) {
        return evalAt(parameters, program, f, d);
    };
    const auto evalStressAt = [&](const std::vector<float>& values, const std::uint32_t first,
                                  const Matrix3& f, const Matrix3& d) {
        Matrix3 result{};
        for (std::uint32_t i = 0u; i < 9u; ++i)
            result[i] = evalAt(values, first + i, f, d);
        return result;
    };
    const auto evalStress = [&](const std::uint32_t first, const Matrix3& f, const Matrix3& d) {
        return evalStressAt(parameters, first, f, d);
    };
    const Matrix3 identity{1,0,0, 0,1,0, 0,0,1};
    const Matrix3 simpleShearRate{0,2,0, 0,0,0, 0,0,0};
    const Matrix3 simpleShearF{1,0.4f,0, 0,1,0, 0,0,1};
    const Matrix3 shearStress = evalStress(material.viscousStressProgramOffset, identity, simpleShearRate);
    require(
        std::abs(shearStress[1] - eta * 2.0f) < 2.0e-6f &&
            std::abs(shearStress[3] - eta * 2.0f) < 2.0e-6f,
        "simple-shear PK1 stress does not equal mu*shear-rate"
    );
    const float shearPhi = eval(material.dissipationProgram, identity, simpleShearRate);
    require(
        std::abs(shearPhi - eta * 2.0f) < 2.0e-6f,
        "simple-shear Rayleigh potential has the wrong value"
    );
    const Matrix3 staticShearStress = evalStress(material.stressProgramOffset, simpleShearF, zero);
    for (float value : staticShearStress)
        require(std::abs(value) < 2.0e-3f, "volumetric-only energy added equilibrium shear stress");

    const float angle = 0.73f;
    const float omega = 3.0f;
    const float c = std::cos(angle), s = std::sin(angle);
    const Matrix3 rotation{c,-s,0, s,c,0, 0,0,1};
    const Matrix3 spin{0,-omega,0, omega,0,0, 0,0,0};
    const Matrix3 rigidRotationRate = multiply3(spin, rotation);
    const float rotationPhi = eval(material.dissipationProgram, rotation, rigidRotationRate);
    const Matrix3 rotationStress = evalStress(
        material.viscousStressProgramOffset, rotation, rigidRotationRate
    );
    require(std::abs(rotationPhi) < 1.0e-6f, "rigid rotation generated viscous dissipation");
    for (float value : rotationStress)
        require(std::abs(value) < 1.0e-5f, "rigid rotation generated objective viscous stress");

    const Matrix3 generalF{1.1f,0.12f,-0.08f, 0.04f,0.93f,0.17f, -0.03f,0.09f,1.04f};
    const Matrix3 generalRate{0.7f,-0.4f,0.2f, 0.1f,-0.3f,0.6f, -0.5f,0.25f,0.15f};
    const Matrix3 actualGeneral = evalStress(
        material.viscousStressProgramOffset, generalF, generalRate
    );
    const Matrix3 expectedGeneral = expectedNewtonianPK1(generalF, generalRate, eta);
    const float expectedPhi = expectedRayleighPotential(generalF, generalRate, eta);
    const float actualPhi = eval(material.dissipationProgram, generalF, generalRate);
    require(expectedPhi >= 0.0f && actualPhi >= 0.0f, "Newtonian dissipation is negative");
    require(
        std::abs(actualPhi - expectedPhi) <= 3.0e-5f * std::max(expectedPhi, 1.0e-8f),
        "compiled Rayleigh potential differs from objective Newtonian reference"
    );
    for (std::size_t i = 0u; i < 9u; ++i)
        require(
            std::abs(actualGeneral[i] - expectedGeneral[i]) <=
                3.0e-4f * std::max(std::abs(expectedGeneral[i]), 1.0e-5f),
            "compiled PK1 viscous stress differs from objective Newtonian reference"
        );
    float power = 0.0f;
    for (std::size_t i = 0u; i < 9u; ++i)
        power += actualGeneral[i] * generalRate[i];
    require(power >= 0.0f, "Newtonian viscous power is negative");
    require(
        std::abs(power - 2.0f * actualPhi) <= 8.0e-5f * std::max(power, 1.0e-6f),
        "quadratic Rayleigh potential power is not twice the potential"
    );

    const Matrix3 compression{1,0,0, 0,1,0, 0,0,0.99f};
    const Matrix3 elasticCompression = evalStress(material.stressProgramOffset, compression, zero);
    require(
        elasticCompression[8] < 0.0f &&
            std::abs(elasticCompression[8] + 25000.0f) < 100.0f,
        "numerical bulk proxy has wrong compression pressure sign or magnitude"
    );

    // Exact hydrostatic EOS sensitivity for this W=K/2*(J-1)^2 penalty:
    // p=K*(1-J), so rho/rho0=1/J. This is a numerical weak-compressibility
    // bound, not a measured blood equation of state.
    constexpr float pressureBound = 30000.0f;
    constexpr std::array<float, 3> bulkCases{1.0e6f, 2.5e6f, 5.0e6f};
    constexpr std::array<float, 3> expectedDensityRise{0.0309278f, 0.0121457f, 0.0060362f};
    for (std::size_t i = 0u; i < bulkCases.size(); ++i) {
        std::vector<float> caseParameters = parameters;
        caseParameters[2] = bulkCases[i];
        const float j = 1.0f - pressureBound / bulkCases[i];
        const Matrix3 pressureF{1,0,0, 0,1,0, 0,0,j};
        const Matrix3 stress = evalStressAt(caseParameters, material.stressProgramOffset, pressureF, zero);
        const float densityRise = 1.0f / j - 1.0f;
        require(
            std::abs(stress[8] + pressureBound) < 2.0f &&
                std::abs(densityRise - expectedDensityRise[i]) < 2.0e-5f,
            "weak-compressibility pressure/density sensitivity changed"
        );
    }

    const float rhoLower = compiled.world.parameters[0].valueAndBounds.y;
    const float kUpper = compiled.world.parameters[2].valueAndBounds.z;
    const float waveSpeedBound = std::sqrt(kUpper / rhoLower);
    const float stableDt = options.cfl * 0.005f / waveSpeedBound;
    const float frameTicks = static_cast<float>(source.frameTimestep) / stableDt;
    const unsigned cflExponent = static_cast<unsigned>(std::ceil(std::log2(frameTicks)));
    require(
        cflExponent == 7u && compiled.world.objects.front().solver.x == cflExponent,
        "numerical-bulk upper bound no longer drives the expected MPM CFL subdivision"
    );

    std::cout
        << "{\"schema\":\"numi.matter.blood_newtonian_bytecode_cpu_check.v1\""
        << ",\"result\":\"pass\""
        << ",\"viscosity_Pa_s\":" << eta
        << ",\"density_rise_at_30kPa\":["
        << (1.0f / (1.0f - pressureBound / bulkCases[0]) - 1.0f) << ","
        << (1.0f / (1.0f - pressureBound / bulkCases[1]) - 1.0f) << ","
        << (1.0f / (1.0f - pressureBound / bulkCases[2]) - 1.0f) << "]"
        << ",\"numerical_bulk_proxy_Pa\":[" << bulkCases[0] << ","
        << bulkCases[1] << "," << bulkCases[2] << "]"
        << ",\"mpm_cfl_exponent\":" << cflExponent
        << ",\"scope\":\"CPU constitutive bytecode only; no GPU liquid transfer or rollback test\"}\n";
}

} // namespace

int main() {
    try {
        verifyAdaptiveLayout();
        verifyLearnedMaterialRoundTrip();
        verifyDormantMPMCapacitySlot();
        verifyBloodNewtonianBytecode();
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
