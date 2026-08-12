#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace numi::matter {
namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;

[[nodiscard]] nm_float4 f4(
    const double x = 0.0,
    const double y = 0.0,
    const double z = 0.0,
    const double w = 0.0
) noexcept {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

[[nodiscard]] bool finite(const double value) noexcept {
    return std::isfinite(value);
}

[[nodiscard]] bool finite(const Vec3& value) noexcept {
    return std::ranges::all_of(value, [](const double component) {
        return finite(component);
    });
}

[[nodiscard]] Vec3 subtract(const Vec3& left, const Vec3& right) noexcept {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

[[nodiscard]] double determinant(const Mat3& matrix) noexcept {
    return
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

[[nodiscard]] bool inverse(const Mat3& matrix, Mat3& output) noexcept {
    const double det = determinant(matrix);
    double scale = 1.0;
    for (const double value : matrix) {
        scale = std::max(scale, std::abs(value));
    }
    if (!(std::abs(det) > 128.0 * std::numeric_limits<double>::epsilon() *
          scale * scale * scale) || !finite(det)) {
        return false;
    }
    const double reciprocal = 1.0 / det;
    output = {
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * reciprocal,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * reciprocal,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * reciprocal,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * reciprocal,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * reciprocal,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * reciprocal,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * reciprocal,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * reciprocal,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * reciprocal,
    };
    return std::ranges::all_of(output, [](const double value) {
        return finite(value);
    });
}

[[nodiscard]] int floorDiv(const int value, const int divisor) noexcept {
    const int quotient = value / divisor;
    const int remainder = value % divisor;
    return remainder < 0 ? quotient - 1 : quotient;
}

[[nodiscard]] std::uint64_t spreadMortonBits(
    std::uint32_t value
) noexcept {
    std::uint64_t bits = value & 0x1fffffu;
    bits = (bits | (bits << 32u)) & 0x1f00000000ffffull;
    bits = (bits | (bits << 16u)) & 0x1f0000ff0000ffull;
    bits = (bits | (bits << 8u)) & 0x100f00f00f00f00full;
    bits = (bits | (bits << 4u)) & 0x10c30c30c30c30c3ull;
    bits = (bits | (bits << 2u)) & 0x1249249249249249ull;
    return bits;
}

[[nodiscard]] std::uint64_t morton3D(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z
) noexcept {
    return spreadMortonBits(x) |
        (spreadMortonBits(y) << 1u) |
        (spreadMortonBits(z) << 2u);
}

[[nodiscard]] std::optional<std::uint32_t> checkedU32(
    const std::size_t value,
    std::vector<Diagnostic>& diagnostics,
    const std::string_view role
) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            std::string(role) + " exceeds the 32-bit cooked ABI",
        });
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] bool supports(
    const MaterialProgram& material,
    const Representation representation
) {
    return std::ranges::find(
        material.supportedRepresentations,
        representation
    ) != material.supportedRepresentations.end();
}

[[nodiscard]] double parameterValue(
    const MaterialProgram& material,
    const std::string_view name,
    const double fallback
) {
    const auto iterator = std::ranges::find_if(
        material.parameters,
        [&](const Parameter& parameter) {
            return parameter.name == name;
        }
    );
    return iterator == material.parameters.end()
        ? fallback
        : iterator->defaultValue;
}

[[nodiscard]] double density(const MaterialProgram& material) {
    return parameterValue(material, "density", 0.0);
}

[[nodiscard]] double stiffnessScale(const MaterialProgram& material) {
    double result = 0.0;
    for (const Parameter& parameter : material.parameters) {
        if (parameter.dimension == kPressure) {
            result = std::max(result, std::abs(parameter.upper));
        }
    }
    return result;
}

[[nodiscard]] double mixedBulkModulus(const MaterialProgram& material) {
    if (material.mixed.bulkModulus > 0.0 &&
        finite(material.mixed.bulkModulus)) {
        return material.mixed.bulkModulus;
    }
    const double mu = parameterValue(material, "mu", 0.0);
    const double lambda = parameterValue(material, "lambda", 0.0);
    const double inferred = lambda + (2.0 / 3.0) * mu;
    if (inferred > 0.0 && finite(inferred)) {
        return inferred;
    }
    double fallback = 0.0;
    for (const Parameter& parameter : material.parameters) {
        if (parameter.dimension == kPressure &&
            parameter.defaultValue > 0.0 &&
            finite(parameter.defaultValue)) {
            fallback = std::max(fallback, parameter.defaultValue);
        }
    }
    return fallback;
}

[[nodiscard]] bool finiteMixedMaterial(const MixedMaterialSource& value) {
    const std::array<double, 20> scalars{
        value.bulkModulus,
        value.thermalExpansion,
        value.biotCoefficient,
        value.referenceTemperature,
        value.heatCapacity,
        value.thermalConductivity,
        value.heatSource,
        value.jouleHeatFraction,
        value.poreStorage,
        value.poreMobility,
        value.poreSource,
        value.electricalConductivity,
        value.activationDiffusivity,
        value.activationOnRate,
        value.activationOffRate,
        value.maximumActiveTension,
        value.activationThreshold,
        value.activationSlope,
        value.cohesiveStrength,
        value.fractureEnergy,
    };
    return std::ranges::all_of(scalars, [](const double scalar) {
        return finite(scalar);
    }) && finite(value.fibreDirection);
}

[[nodiscard]] NMMixedSolverGPU cookMixedSolver(
    const MixedSolverSource& source
) {
    NMMixedSolverGPU result{};
    result.nonlinearIterations = {
        source.newtonIterations,
        source.fgmresRestart,
        source.fgmresIterations,
        source.lineSearchSteps,
    };
    result.blockIterations = {
        source.velocityPCGIterations,
        source.pressurePCGIterations,
        source.fieldPCGIterations,
        source.mutationRestarts,
    };
    result.nonlinearTolerances = f4(
        source.relativeResidual,
        source.relativeCorrection,
        source.volumeTolerance,
        source.pressureTolerance
    );
    result.contactTolerances = f4(
        source.naturalResidualTolerance,
        source.coneTolerance,
        source.complementarityTolerance,
        source.energyTolerance
    );
    result.regularization = f4(
        source.diagonalFloor,
        source.initialLMShift,
        source.maximumLMShift,
        source.curvatureTolerance
    );
    result.globalization = f4(
        source.armijo,
        source.minimumTemperature,
        source.activationEpsilon,
        source.pressureStabilization
    );
    return result;
}

[[nodiscard]] NMMixedMaterialGPU cookMixedMaterial(
    const MaterialProgram& material
) {
    const MixedMaterialSource& source = material.mixed;
    NMMixedMaterialGPU result{};
    result.mechanics = f4(
        mixedBulkModulus(material),
        source.thermalExpansion,
        source.biotCoefficient,
        source.referenceTemperature
    );
    result.thermal = f4(
        source.heatCapacity,
        source.thermalConductivity,
        source.heatSource,
        source.jouleHeatFraction
    );
    result.porous = f4(
        source.poreStorage,
        source.poreMobility,
        source.poreSource,
        0.0
    );
    result.electrical = f4(
        source.electricalConductivity,
        source.activationDiffusivity,
        source.activationOnRate,
        source.activationOffRate
    );
    result.fibre = f4(
        source.fibreDirection[0],
        source.fibreDirection[1],
        source.fibreDirection[2],
        source.maximumActiveTension
    );
    result.coupling = f4(
        source.activationThreshold,
        source.activationSlope,
        source.cohesiveStrength,
        source.fractureEnergy
    );
    return result;
}

[[nodiscard]] std::uint32_t rateExponent(
    const ObjectSource& object,
    const MaterialProgram& material,
    const WorldSource& world,
    const CompileOptions& options
) {
    const double rho = density(material);
    const double modulus = stiffnessScale(material);
    if (!(rho > 0.0) || !(modulus > 0.0) ||
        !(object.characteristicLength > 0.0)) {
        return 0u;
    }
    const double waveSpeed = std::sqrt(modulus / rho);
    const double stable = options.cfl * object.characteristicLength / waveSpeed;
    if (!(stable > 0.0) || stable >= world.frameTimestep) {
        return 0u;
    }
    const double ratio = world.frameTimestep / stable;
    const double raw = std::ceil(std::log2(std::max(ratio, 1.0)));
    return std::min(
        options.maximumRateExponent,
        static_cast<std::uint32_t>(std::max(raw, 0.0))
    );
}

[[nodiscard]] std::uint32_t representationCode(
    const Representation representation
) noexcept {
    switch (representation) {
    case Representation::rigid: return NM_REPRESENTATION_RIGID;
    case Representation::mpm: return NM_REPRESENTATION_MPM;
    case Representation::fem: return NM_REPRESENTATION_FEM;
    }
    return NM_REPRESENTATION_RIGID;
}

[[nodiscard]] Representation selectRepresentation(
    const ObjectSource& object,
    const MaterialProgram& material,
    std::vector<Diagnostic>& diagnostics
) {
    Representation selected = object.representation;
    if (object.automaticRepresentation) {
        if (!object.tetrahedra.empty() && supports(material, Representation::fem)) {
            selected = Representation::fem;
        } else if (!object.particles.empty() && supports(material, Representation::mpm)) {
            selected = Representation::mpm;
        } else if (supports(material, Representation::rigid)) {
            selected = Representation::rigid;
        }
    }
    if (!supports(material, selected)) {
        diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "object '" + object.name + "' selects a representation not supported by material '" +
                material.name + "'",
        });
    }
    return selected;
}

struct IncidenceEntry {
    std::uint32_t owner = 0u;
    std::uint32_t source = 0u;

};

void buildIncidence(
    std::vector<IncidenceEntry> entries,
    const std::size_t ownerCount,
    const std::span<const std::uint32_t> ownerObjects,
    std::vector<std::uint32_t>& incidence,
    std::vector<NMIncidenceRangeGPU>& ranges
) {
    std::ranges::sort(
        entries,
        [](const IncidenceEntry& left, const IncidenceEntry& right) {
            return std::tie(left.owner, left.source) <
                std::tie(right.owner, right.source);
        }
    );
    incidence.clear();
    ranges.assign(ownerCount, {});
    incidence.reserve(entries.size());
    std::size_t cursor = 0u;
    for (std::size_t owner = 0u; owner < ownerCount; ++owner) {
        const std::size_t first = incidence.size();
        while (cursor < entries.size() && entries[cursor].owner == owner) {
            incidence.push_back(entries[cursor].source);
            ++cursor;
        }
        NMIncidenceRangeGPU range{};
        range.first = static_cast<nm_u32>(first);
        range.count = static_cast<nm_u32>(incidence.size() - first);
        range.objectIndex = owner < ownerObjects.size()
            ? ownerObjects[owner]
            : NM_INVALID_INDEX;
        ranges[owner] = range;
    }
}

void appendProgram(
    const ScalarBytecode& bytecode,
    std::vector<NMExpressionInstructionGPU>& instructions,
    std::vector<NMScalarProgramGPU>& programs
) {
    NMScalarProgramGPU descriptor{};
    descriptor.firstInstruction = static_cast<nm_u32>(instructions.size());
    descriptor.instructionCount = static_cast<nm_u32>(bytecode.instructions.size());
    descriptor.maximumStack = bytecode.maximumStack;
    programs.push_back(descriptor);
    instructions.insert(
        instructions.end(),
        bytecode.instructions.begin(),
        bytecode.instructions.end()
    );
}

void validateWorld(
    const WorldSource& source,
    const CompileOptions& options,
    std::vector<Diagnostic>& diagnostics
) {
    if (!(source.frameTimestep > 0.0) || !finite(source.frameTimestep)) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "world frame timestep must be finite and positive",
        });
    }
    if (!finite(source.gravity) || source.environmentCount == 0u) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "world gravity must be finite and environment count nonzero",
        });
    }
    if (!(source.contactSlop >= 0.0) || !finite(source.contactSlop) ||
        !(source.maximumDepenetrationSpeed > 0.0) ||
        !finite(source.maximumDepenetrationSpeed)) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "world contact slop must be finite and nonnegative and maximum "
            "depenetration speed must be finite and positive",
        });
    }
    if (!(options.cfl > 0.0) || !(options.cfl <= 1.0) ||
        options.maximumRateExponent > NM_MAX_RATE_EXPONENT ||
        options.maximumExpressionStack == 0u ||
        options.maximumExpressionStack > NM_EXPRESSION_STACK_CAPACITY) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "matter compiler options exceed the versioned ABI contract",
        });
    }
    if (source.materials.empty()) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "matter world requires at least one material",
        });
    }
    if (source.identificationCandidates != 0u &&
        ((source.identificationCandidates & 1u) != 0u ||
         source.identificationCandidates > source.environmentCount)) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "identification candidate count must be even and no larger than environment count",
        });
    }
    const MixedSolverSource& solver = source.mixedSolver;
    const std::array<std::uint32_t, 8> budgets{
        solver.newtonIterations,
        solver.fgmresRestart,
        solver.fgmresIterations,
        solver.lineSearchSteps,
        solver.velocityPCGIterations,
        solver.pressurePCGIterations,
        solver.fieldPCGIterations,
        solver.mutationRestarts,
    };
    const std::array<double, 16> tolerances{
        solver.relativeResidual,
        solver.relativeCorrection,
        solver.volumeTolerance,
        solver.pressureTolerance,
        solver.naturalResidualTolerance,
        solver.coneTolerance,
        solver.complementarityTolerance,
        solver.energyTolerance,
        solver.diagonalFloor,
        solver.initialLMShift,
        solver.maximumLMShift,
        solver.curvatureTolerance,
        solver.armijo,
        solver.minimumTemperature,
        solver.activationEpsilon,
        solver.pressureStabilization,
    };
    if (solver.fgmresRestart > NM_MIXED_FGMRES_RESTART) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "mixed solver FGMRES restart exceeds the compiled basis capacity",
        });
    }
    if (std::ranges::any_of(budgets, [](const std::uint32_t value) {
            return value == 0u;
        }) || solver.fgmresRestart > solver.fgmresIterations ||
        std::ranges::any_of(tolerances, [](const double value) {
            return !finite(value) || value < 0.0;
        }) || !(solver.maximumLMShift >= solver.initialLMShift) ||
        !(solver.minimumTemperature > 0.0) ||
        !(solver.activationEpsilon > 0.0 && solver.activationEpsilon < 0.5)) {
        diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "mixed solver policy is finite, positive, and bounded",
        });
    }
}

} // namespace

CompileResult compileWorld(
    const WorldSource& source,
    const CompileOptions& options
) {
    CompileResult result;
    validateWorld(source, options, result.diagnostics);
    CompiledWorld& world = result.world;
    world.mixedSolver = cookMixedSolver(source.mixedSolver);

    if (!result.succeeded()) {
        return result;
    }

    world.constitutive.reserve(source.materials.size());
    world.materials.reserve(source.materials.size());
    world.mixedMaterials.reserve(source.materials.size());
    for (const MaterialProgram& material : source.materials) {
        if (!finiteMixedMaterial(material.mixed) ||
            !(mixedBulkModulus(material) > 0.0) ||
            material.mixed.thermalExpansion < 0.0 ||
            material.mixed.biotCoefficient < 0.0 ||
            material.mixed.biotCoefficient > 1.0 ||
            material.mixed.heatCapacity < 0.0 ||
            material.mixed.thermalConductivity < 0.0 ||
            material.mixed.poreStorage < 0.0 ||
            material.mixed.poreMobility < 0.0 ||
            material.mixed.electricalConductivity < 0.0 ||
            material.mixed.activationDiffusivity < 0.0 ||
            material.mixed.activationOnRate < 0.0 ||
            material.mixed.activationOffRate < 0.0) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "material '" + material.name +
                    "' has an invalid mixed or multiphysics contract",
            });
            continue;
        }
        detail::ConstitutiveCompileResult compiled =
            detail::compileConstitutive(material, options.maximumExpressionStack);
        result.diagnostics.insert(
            result.diagnostics.end(),
            compiled.diagnostics.begin(),
            compiled.diagnostics.end()
        );
        if (!compiled.succeeded()) {
            continue;
        }
        ConstitutiveProgram program = std::move(compiled.program);
        program.gpu.parameterOffset =
            static_cast<nm_u32>(world.parameters.size());
        program.gpu.stateInitialOffset =
            static_cast<nm_u32>(world.stateInitials.size());
        for (const InternalState& state : program.material.internalState) {
            world.stateInitials.push_back(
                static_cast<float>(state.initialValue)
            );
        }

        program.gpu.stressProgramOffset =
            static_cast<nm_u32>(world.scalarPrograms.size());
        for (const ScalarBytecode& component : program.stress) {
            appendProgram(component, world.instructions, world.scalarPrograms);
        }
        program.gpu.tangentProgramOffset =
            static_cast<nm_u32>(world.scalarPrograms.size());
        for (const ScalarBytecode& component : program.tangentVector) {
            appendProgram(component, world.instructions, world.scalarPrograms);
        }

        if (program.dissipation.has_value()) {
            program.gpu.viscousStressProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& component : program.viscousStress) {
                appendProgram(
                    component,
                    world.instructions,
                    world.scalarPrograms
                );
            }
            program.gpu.viscousTangentProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& component :
                 program.viscousTangentVector) {
                appendProgram(
                    component,
                    world.instructions,
                    world.scalarPrograms
                );
            }
        } else {
            program.gpu.viscousStressProgramOffset = NM_INVALID_INDEX;
            program.gpu.viscousTangentProgramOffset = NM_INVALID_INDEX;
        }

        if (!program.stateUpdates.empty()) {
            program.gpu.stateUpdateProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& update : program.stateUpdates) {
                appendProgram(update, world.instructions, world.scalarPrograms);
            }
        } else {
            program.gpu.stateUpdateProgramOffset = NM_INVALID_INDEX;
        }
        if (!program.implicitResiduals.empty()) {
            program.gpu.implicitResidualProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& residual : program.implicitResiduals) {
                appendProgram(residual, world.instructions, world.scalarPrograms);
            }
            program.gpu.implicitJacobianProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& jacobian : program.implicitJacobians) {
                appendProgram(jacobian, world.instructions, world.scalarPrograms);
            }
            program.gpu.implicitDeformationProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& direction :
                 program.implicitDeformationDirections) {
                appendProgram(direction, world.instructions, world.scalarPrograms);
            }
            program.gpu.stressStateDerivativeProgramOffset =
                static_cast<nm_u32>(world.scalarPrograms.size());
            for (const ScalarBytecode& derivative :
                 program.stressStateDerivatives) {
                appendProgram(derivative, world.instructions, world.scalarPrograms);
            }
        } else {
            program.gpu.implicitResidualProgramOffset = NM_INVALID_INDEX;
            program.gpu.implicitJacobianProgramOffset = NM_INVALID_INDEX;
            program.gpu.implicitDeformationProgramOffset = NM_INVALID_INDEX;
            program.gpu.stressStateDerivativeProgramOffset = NM_INVALID_INDEX;
        }
        if (program.dissipation.has_value()) {
            program.gpu.dissipationProgram =
                static_cast<nm_u32>(world.scalarPrograms.size());
            appendProgram(
                *program.dissipation,
                world.instructions,
                world.scalarPrograms
            );
        } else {
            program.gpu.dissipationProgram = NM_INVALID_INDEX;
        }
        if (program.validity.has_value()) {
            program.gpu.validityProgram =
                static_cast<nm_u32>(world.scalarPrograms.size());
            appendProgram(
                *program.validity,
                world.instructions,
                world.scalarPrograms
            );
        } else {
            program.gpu.validityProgram = NM_INVALID_INDEX;
        }
        world.parameters.insert(
            world.parameters.end(),
            program.parameters.begin(),
            program.parameters.end()
        );
        world.materials.push_back(program.gpu);
        world.mixedMaterials.push_back(cookMixedMaterial(material));
        world.constitutive.push_back(std::move(program));
        if (material.learned.has_value()) {
            const LearnedMaterialSource& learned = *material.learned;
            bool learnedValid =
                learned.invariantCount >= 4u &&
                learned.invariantCount <= NM_LEARNED_MAX_INVARIANTS &&
                learned.softplusBeta > 0.0f &&
                learned.determinantFloor > 0.0f &&
                learned.growthCoefficient >= 0.0f &&
                !learned.layers.empty() &&
                learned.layers.size() <= NM_LEARNED_MAX_LAYERS;
            const std::size_t firstLayer = world.learnedLayers.size();
            const std::size_t firstWeight = world.learnedWeights.size();
            std::uint32_t previousWidth = 0u;
            for (std::size_t layerIndex = 0u;
                 layerIndex < learned.layers.size();
                 ++layerIndex) {
                const LearnedLayerSource& layer = learned.layers[layerIndex];
                const std::uint64_t inputCount =
                    static_cast<std::uint64_t>(layer.inputWidth) *
                    layer.outputWidth;
                const std::uint64_t recurrentCount =
                    static_cast<std::uint64_t>(previousWidth) *
                    layer.outputWidth;
                learnedValid = learnedValid &&
                    layer.inputWidth == learned.invariantCount &&
                    layer.outputWidth > 0u &&
                    layer.outputWidth <= NM_LEARNED_MAX_WIDTH &&
                    inputCount == layer.inputWeights.size() &&
                    recurrentCount == layer.recurrentWeights.size() &&
                    layer.biases.size() == layer.outputWidth &&
                    std::ranges::all_of(
                        layer.inputWeights,
                        [](const float value) {
                            return std::isfinite(value) && value >= 0.0f;
                        }
                    ) &&
                    std::ranges::all_of(
                        layer.recurrentWeights,
                        [](const float value) {
                            return std::isfinite(value) && value >= 0.0f;
                        }
                    ) &&
                    std::ranges::all_of(
                        layer.biases,
                        [](const float value) { return std::isfinite(value); }
                    );
                NMLearnedLayerGPU cooked{};
                cooked.layout = {
                    layer.inputWidth,
                    layer.outputWidth,
                    static_cast<nm_u32>(world.learnedWeights.size()),
                    0u,
                };
                world.learnedWeights.insert(
                    world.learnedWeights.end(),
                    layer.inputWeights.begin(),
                    layer.inputWeights.end()
                );
                cooked.routing.x = static_cast<nm_u32>(
                    world.learnedWeights.size()
                );
                world.learnedWeights.insert(
                    world.learnedWeights.end(),
                    layer.recurrentWeights.begin(),
                    layer.recurrentWeights.end()
                );
                cooked.layout.w = static_cast<nm_u32>(
                    world.learnedWeights.size()
                );
                world.learnedWeights.insert(
                    world.learnedWeights.end(),
                    layer.biases.begin(),
                    layer.biases.end()
                );
                world.learnedLayers.push_back(cooked);
                previousWidth = layer.outputWidth;
            }
            learnedValid = learnedValid && previousWidth == 1u &&
                world.learnedLayers.size() <=
                    std::numeric_limits<std::uint32_t>::max() &&
                world.learnedWeights.size() <=
                    std::numeric_limits<std::uint32_t>::max();
            if (!learnedValid) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "learned material '" + material.name +
                        "' violates the canonical polyconvex ICNN contract",
                });
                world.learnedLayers.resize(firstLayer);
                world.learnedWeights.resize(firstWeight);
                continue;
            }
            std::uint64_t fingerprint = learned.fingerprint;
            if (fingerprint == 0u) {
                fingerprint = detail::hashBytes(
                    world.learnedWeights.data() + firstWeight,
                    (world.learnedWeights.size() - firstWeight) * sizeof(float)
                );
            }
            NMLearnedMaterialGPU descriptor{};
            descriptor.layout = {
                static_cast<nm_u32>(firstLayer),
                static_cast<nm_u32>(learned.layers.size()),
                static_cast<nm_u32>(firstWeight),
                static_cast<nm_u32>(world.learnedWeights.size() - firstWeight),
            };
            descriptor.identity = {
                learned.invariantCount,
                static_cast<nm_u32>(world.materials.size() - 1u),
                NM_LEARNED_SOFTPLUS,
                0u,
            };
            descriptor.policy = f4(
                learned.softplusBeta,
                learned.determinantFloor,
                learned.growthCoefficient,
                0.0
            );
            descriptor.fingerprint = {
                static_cast<nm_u32>(fingerprint),
                static_cast<nm_u32>(fingerprint >> 32u),
                0u,
                0u,
            };
            world.learnedMaterials.push_back(descriptor);
            world.materials.back().constitutiveKind =
                NM_CONSTITUTIVE_POLYCONVEX_ICNN;
        }
    }

    if (!result.succeeded() || world.materials.size() != source.materials.size()) {
        return result;
    }

    world.contact.rigidProxies.reserve(source.rigidProxies.size());
    std::map<std::uint32_t, std::uint32_t> freeBodyIndices;
    for (const RigidProxySource& proxy : source.rigidProxies) {
        if (!finite(proxy.localCenter) || !finite(proxy.localExtent) ||
            !std::ranges::all_of(proxy.localOrientation, [](const double value) {
                return finite(value);
            }) || !finite(proxy.radiusOrOffset) ||
            proxy.materialIndex >= source.materials.size() ||
            (proxy.articulated && proxy.bodyIndex == NM_INVALID_INDEX) ||
            (proxy.dynamic &&
             (proxy.articulated ||
              proxy.bodyIndex == NM_INVALID_INDEX ||
              proxy.sceneBodyIndex == NM_INVALID_INDEX)) ||
            (!proxy.dynamic && proxy.sceneBodyIndex != NM_INVALID_INDEX)) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "rigid proxy contains invalid geometry or material binding",
            });
            continue;
        }
        NMRigidProxyGPU cooked{};
        cooked.shapeKind = static_cast<nm_u32>(proxy.shape);
        cooked.bodyIndex = proxy.bodyIndex;
        cooked.sceneBodyIndex = proxy.sceneBodyIndex;
        cooked.materialIndex = proxy.materialIndex;
        cooked.flags =
            (proxy.articulated ? NM_RIGID_ARTICULATED : 0u) |
            (proxy.dynamic ? NM_RIGID_DYNAMIC : 0u);
        cooked.adaptiveObjectIndex = NM_INVALID_INDEX;
        cooked.generalizedFreeBodyIndex = NM_INVALID_INDEX;
        if (proxy.dynamic) {
            const std::uint32_t nextIndex =
                static_cast<std::uint32_t>(freeBodyIndices.size());
            cooked.generalizedFreeBodyIndex =
                freeBodyIndices.try_emplace(proxy.bodyIndex, nextIndex)
                    .first->second;
        }
        cooked.localCenterAndRadius = f4(
            proxy.localCenter[0],
            proxy.localCenter[1],
            proxy.localCenter[2],
            proxy.radiusOrOffset
        );
        cooked.localExtent = f4(
            proxy.localExtent[0],
            proxy.localExtent[1],
            proxy.localExtent[2]
        );
        cooked.localOrientation = f4(
            proxy.localOrientation[0],
            proxy.localOrientation[1],
            proxy.localOrientation[2],
            proxy.localOrientation[3]
        );
        world.contact.rigidProxies.push_back(cooked);
    }

    std::vector<std::uint32_t> mpmNodeObjects;
    std::vector<std::uint32_t> femNodeObjects;
    std::vector<bool> femNodeContactEligible;
    std::vector<IncidenceEntry> mpmIncidence;
    std::vector<IncidenceEntry> femIncidence;
    std::set<std::uint32_t> adaptiveBindings;
    std::set<std::uint32_t> adaptiveBodyBindings;
    std::set<std::uint32_t> adaptiveSceneBindings;
    std::map<std::uint32_t, std::uint32_t> adaptiveBodyOwners;

    world.objects.reserve(source.objects.size());
    world.adaptive.reserve(source.objects.size());
    world.schedulers.reserve(source.objects.size());

    for (std::size_t objectIndexSize = 0u;
         objectIndexSize < source.objects.size();
         ++objectIndexSize) {
        const auto objectIndexValue = checkedU32(
            objectIndexSize, result.diagnostics, "object index"
        );
        if (!objectIndexValue.has_value()) {
            break;
        }
        const std::uint32_t objectIndex = *objectIndexValue;
        if (objectIndex > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max()
            )) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "object index exceeds the signed sparse-grid ABI",
            });
            return result;
        }
        const ObjectSource& object = source.objects[objectIndexSize];
        if (object.materialIndex >= source.materials.size() ||
            object.name.empty() ||
            !(object.characteristicLength > 0.0) ||
            !finite(object.characteristicLength) ||
            !std::isfinite(object.mutationPolicy.punctureImpulseThreshold) ||
            object.mutationPolicy.punctureImpulseThreshold < 0.0 ||
            object.mutationPolicy.punctureImpulseThreshold >
                std::numeric_limits<float>::max()) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "continuum object has invalid name, material, or characteristic length",
            });
            continue;
        }
        const MaterialProgram& material = source.materials[object.materialIndex];
        const Representation representation = selectRepresentation(
            object, material, result.diagnostics
        );
        const std::uint32_t exponent = rateExponent(
            object, material, source, options
        );

        NMContinuumObjectGPU descriptor{};
        descriptor.representation = representationCode(representation);
        descriptor.materialIndex = object.materialIndex;
        descriptor.flags = NM_OBJECT_ACTIVE |
            (object.twoWayCoupling ? NM_OBJECT_TWO_WAY_COUPLED : 0u) |
            (object.adaptive ? NM_OBJECT_ADAPTIVE : 0u) |
            (object.identifiable ? NM_OBJECT_IDENTIFIABLE : 0u) |
            (representation == Representation::fem && object.mixedFEM
                ? NM_OBJECT_MIXED_FEM : 0u) |
            (representation == Representation::fem && object.multiphysics.enabled
                ? NM_OBJECT_MULTIPHYSICS : 0u) |
            (representation == Representation::fem && object.mutationPolicy.enabled
                ? NM_OBJECT_MUTABLE_TOPOLOGY : 0u);
        descriptor.schedulerIndex = objectIndex;
        descriptor.rigidBinding = object.rigidBinding;
        descriptor.topologyGeneration = 1u;
        descriptor.solver = {
            exponent,
            options.maximumRateExponent,
            source.femPCGIterations,
            0u,
        };
        descriptor.fidelity = f4(
            object.characteristicLength,
            object.rigidTolerance,
            object.promotionStrain,
            object.demotionStrain
        );

        NMFEMCapacityGPU cookedFEMCapacity{};
        if (representation == Representation::mpm) {
            if (object.particles.empty()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM object '" + object.name + "' has no particles",
                });
                return result;
            }
            if (std::ranges::any_of(
                    object.particles,
                    [](const ParticleSource& particle) {
                        return
                            !finite(particle.position) ||
                            !finite(particle.velocity) ||
                            !(particle.mass > 0.0) ||
                            !(particle.referenceVolume > 0.0) ||
                            !finite(particle.mass) ||
                            !finite(particle.referenceVolume);
                    }
                )) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM object '" + object.name +
                        "' contains nonfinite or nonpositive particle state",
                });
                return result;
            }
            const double h = object.characteristicLength;
            if (!finite(object.mpmGridMinimum) ||
                !finite(object.mpmGridMaximum) ||
                !(object.mpmGridMinimum[0] < object.mpmGridMaximum[0]) ||
                !(object.mpmGridMinimum[1] < object.mpmGridMaximum[1]) ||
                !(object.mpmGridMinimum[2] < object.mpmGridMaximum[2])) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM object '" + object.name +
                        "' requires finite increasing fixed-grid bounds",
                });
                return result;
            }
            const auto gridCoordinate = [&](
                const double value
            ) -> std::optional<int> {
                const double scaled = value / h;
                if (!finite(scaled) ||
                    scaled < static_cast<double>(
                        std::numeric_limits<int>::min() + 4
                    ) ||
                    scaled > static_cast<double>(
                        std::numeric_limits<int>::max() - 4
                    )) {
                    return std::nullopt;
                }
                return static_cast<int>(std::floor(scaled));
            };
            const auto authoredMinimumX =
                gridCoordinate(object.mpmGridMinimum[0]);
            const auto authoredMinimumY =
                gridCoordinate(object.mpmGridMinimum[1]);
            const auto authoredMinimumZ =
                gridCoordinate(object.mpmGridMinimum[2]);
            const auto authoredMaximumX =
                gridCoordinate(object.mpmGridMaximum[0]);
            const auto authoredMaximumY =
                gridCoordinate(object.mpmGridMaximum[1]);
            const auto authoredMaximumZ =
                gridCoordinate(object.mpmGridMaximum[2]);
            if (!authoredMinimumX.has_value() ||
                !authoredMinimumY.has_value() ||
                !authoredMinimumZ.has_value() ||
                !authoredMaximumX.has_value() ||
                !authoredMaximumY.has_value() ||
                !authoredMaximumZ.has_value()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM fixed-grid bounds exceed the cooked integer grid domain",
                });
                return result;
            }

            const std::array<int, 3> nodeMinimum{
                *authoredMinimumX - 1,
                *authoredMinimumY - 1,
                *authoredMinimumZ - 1,
            };
            const std::array<int, 3> nodeMaximum{
                *authoredMaximumX + 2,
                *authoredMaximumY + 2,
                *authoredMaximumZ + 2,
            };
            const std::array<std::uint32_t, 3> nodeDimensions{
                static_cast<std::uint32_t>(
                    nodeMaximum[0] - nodeMinimum[0] + 1
                ),
                static_cast<std::uint32_t>(
                    nodeMaximum[1] - nodeMinimum[1] + 1
                ),
                static_cast<std::uint32_t>(
                    nodeMaximum[2] - nodeMinimum[2] + 1
                ),
            };
            const std::uint64_t nodeCount64 =
                static_cast<std::uint64_t>(nodeDimensions[0]) *
                nodeDimensions[1] * nodeDimensions[2];
            if (nodeCount64 == 0u ||
                nodeCount64 > std::numeric_limits<std::uint32_t>::max() ||
                world.mpm.nodes.size() >
                    std::numeric_limits<std::uint32_t>::max() - nodeCount64) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM grid node domain exceeds the 32-bit cooked ABI",
                });
                return result;
            }

            const std::array<int, 3> blockMinimum{
                floorDiv(nodeMinimum[0], int(NM_MPM_BLOCK_EDGE)),
                floorDiv(nodeMinimum[1], int(NM_MPM_BLOCK_EDGE)),
                floorDiv(nodeMinimum[2], int(NM_MPM_BLOCK_EDGE)),
            };
            const std::array<int, 3> blockMaximum{
                floorDiv(nodeMaximum[0], int(NM_MPM_BLOCK_EDGE)),
                floorDiv(nodeMaximum[1], int(NM_MPM_BLOCK_EDGE)),
                floorDiv(nodeMaximum[2], int(NM_MPM_BLOCK_EDGE)),
            };
            const std::array<std::uint32_t, 3> blockDimensions{
                static_cast<std::uint32_t>(
                    blockMaximum[0] - blockMinimum[0] + 1
                ),
                static_cast<std::uint32_t>(
                    blockMaximum[1] - blockMinimum[1] + 1
                ),
                static_cast<std::uint32_t>(
                    blockMaximum[2] - blockMinimum[2] + 1
                ),
            };
            const std::uint64_t blockCount64 =
                static_cast<std::uint64_t>(blockDimensions[0]) *
                blockDimensions[1] * blockDimensions[2];
            if (blockCount64 == 0u ||
                blockCount64 >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max()
                    ) ||
                world.mpm.blocks.size() >
                    static_cast<std::size_t>(
                        std::numeric_limits<std::int32_t>::max()
                    ) ||
                world.mpm.blocks.size() >
                    std::numeric_limits<std::uint32_t>::max() - blockCount64 ||
                world.mpm.blockLookup.size() >
                    std::numeric_limits<std::uint32_t>::max() - blockCount64) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM sparse block domain exceeds the 32-bit cooked ABI",
                });
                return result;
            }

            descriptor.stateOffset =
                static_cast<nm_u32>(world.mpm.particles.size());
            descriptor.stateCount = 0u;
            descriptor.elementOffset =
                static_cast<nm_u32>(world.mpm.grids.size());
            descriptor.elementCount = 1u;
            descriptor.auxiliaryOffset =
                static_cast<nm_u32>(world.mpm.nodes.size());
            descriptor.auxiliaryCount = static_cast<nm_u32>(nodeCount64);

            NMMPMGridGPU grid{};
            grid.nodeMinimumAndObject = {
                nodeMinimum[0],
                nodeMinimum[1],
                nodeMinimum[2],
                static_cast<nm_i32>(objectIndex),
            };
            grid.nodeDimensionsAndOffset = {
                nodeDimensions[0],
                nodeDimensions[1],
                nodeDimensions[2],
                descriptor.auxiliaryOffset,
            };
            grid.blockMinimumAndOffset = {
                blockMinimum[0],
                blockMinimum[1],
                blockMinimum[2],
                static_cast<nm_i32>(world.mpm.blocks.size()),
            };
            grid.blockDimensionsAndLookup = {
                blockDimensions[0],
                blockDimensions[1],
                blockDimensions[2],
                static_cast<nm_u32>(world.mpm.blockLookup.size()),
            };
            grid.metrics = f4(h, 1.0 / h, 1.5, 0.0);
            const std::uint32_t gridIndex =
                static_cast<std::uint32_t>(world.mpm.grids.size());
            world.mpm.grids.push_back(grid);

            for (int z = nodeMinimum[2]; z <= nodeMaximum[2]; ++z) {
                for (int y = nodeMinimum[1]; y <= nodeMaximum[1]; ++y) {
                    for (int x = nodeMinimum[0]; x <= nodeMaximum[0]; ++x) {
                        NMGridNodeStateGPU node{};
                        node.positionAndMass = f4(
                            static_cast<double>(x) * h,
                            static_cast<double>(y) * h,
                            static_cast<double>(z) * h,
                            0.0
                        );
                        world.mpm.nodes.push_back(node);
                        mpmNodeObjects.push_back(objectIndex);
                    }
                }
            }

            struct PendingBlock {
                std::uint64_t morton = 0u;
                std::uint32_t localLookup = 0u;
                std::array<int, 3> coordinate{};
            };
            std::vector<PendingBlock> pendingBlocks;
            pendingBlocks.reserve(static_cast<std::size_t>(blockCount64));
            for (std::uint32_t localZ = 0u;
                 localZ < blockDimensions[2];
                 ++localZ) {
                for (std::uint32_t localY = 0u;
                     localY < blockDimensions[1];
                     ++localY) {
                    for (std::uint32_t localX = 0u;
                         localX < blockDimensions[0];
                         ++localX) {
                        const std::uint32_t localLookup =
                            (localZ * blockDimensions[1] + localY) *
                                blockDimensions[0] +
                            localX;
                        pendingBlocks.push_back({
                            morton3D(localX, localY, localZ),
                            localLookup,
                            {
                                blockMinimum[0] + int(localX),
                                blockMinimum[1] + int(localY),
                                blockMinimum[2] + int(localZ),
                            },
                        });
                    }
                }
            }
            std::ranges::sort(
                pendingBlocks,
                [](const PendingBlock& left, const PendingBlock& right) {
                    return std::tie(left.morton, left.localLookup) <
                        std::tie(right.morton, right.localLookup);
                }
            );
            const std::size_t lookupOffset = world.mpm.blockLookup.size();
            world.mpm.blockLookup.resize(
                lookupOffset + static_cast<std::size_t>(blockCount64),
                NM_INVALID_INDEX
            );
            for (const PendingBlock& pending : pendingBlocks) {
                const std::uint32_t globalBlock =
                    static_cast<std::uint32_t>(world.mpm.blocks.size());
                NMMPMBlockGPU block{};
                block.identity = {
                    static_cast<nm_u32>(pending.morton),
                    static_cast<nm_u32>(pending.morton >> 32u),
                    gridIndex,
                    objectIndex,
                };
                block.coordinateAndLookup = {
                    pending.coordinate[0],
                    pending.coordinate[1],
                    pending.coordinate[2],
                    static_cast<nm_i32>(pending.localLookup),
                };
                world.mpm.blocks.push_back(block);
                world.mpm.blockLookup[
                    lookupOffset + pending.localLookup
                ] = globalBlock;
            }

            for (const ParticleSource& sourceParticle : object.particles) {
                const std::array<int, 3> particleNode{
                    static_cast<int>(std::floor(sourceParticle.position[0] / h)),
                    static_cast<int>(std::floor(sourceParticle.position[1] / h)),
                    static_cast<int>(std::floor(sourceParticle.position[2] / h)),
                };
                if (particleNode[0] < nodeMinimum[0] + 1 ||
                    particleNode[0] > nodeMaximum[0] - 2 ||
                    particleNode[1] < nodeMinimum[1] + 1 ||
                    particleNode[1] > nodeMaximum[1] - 2 ||
                    particleNode[2] < nodeMinimum[2] + 1 ||
                    particleNode[2] > nodeMaximum[2] - 2) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "MPM particle begins outside the cooked quadratic-support domain",
                    });
                    return result;
                }
                NMParticleStateGPU particle{};
                particle.positionAndMass = f4(
                    sourceParticle.position[0],
                    sourceParticle.position[1],
                    sourceParticle.position[2],
                    sourceParticle.mass
                );
                particle.velocityAndReferenceVolume = f4(
                    sourceParticle.velocity[0],
                    sourceParticle.velocity[1],
                    sourceParticle.velocity[2],
                    sourceParticle.referenceVolume
                );
                particle.deformationRow0 = f4(1.0, 0.0, 0.0);
                particle.deformationRow1 = f4(0.0, 1.0, 0.0);
                particle.deformationRow2 = f4(0.0, 0.0, 1.0);
                particle.affineRow0 = f4();
                particle.affineRow1 = f4();
                particle.affineRow2 = f4();
                particle.referenceAndTemperature = f4(
                    sourceParticle.position[0],
                    sourceParticle.position[1],
                    sourceParticle.position[2],
                    293.15
                );
                particle.identity = {
                    objectIndex,
                    object.materialIndex,
                    1u,
                    NM_OBJECT_ACTIVE,
                };
                world.mpm.particles.push_back(particle);
            }
            descriptor.stateCount =
                static_cast<nm_u32>(world.mpm.particles.size()) -
                descriptor.stateOffset;
        } else if (representation == Representation::fem) {
            if (object.femNodes.empty() || object.tetrahedra.empty()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name + "' requires nodes and tetrahedra",
                });
                return result;
            }
            if (!finite(object.femInitialVelocity) ||
                std::ranges::any_of(
                    object.femNodes,
                    [](const Vec3& node) { return !finite(node); }
                )) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name +
                        "' contains a nonfinite node or initial velocity",
                });
                return result;
            }
            std::vector<bool> fixedNodes(object.femNodes.size(), false);
            for (const std::uint32_t localNode : object.femFixedNodes) {
                if (localNode >= object.femNodes.size() || fixedNodes[localNode]) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM object '" + object.name +
                            "' contains an invalid or duplicate fixed node",
                    });
                    return result;
                }
                fixedNodes[localNode] = true;
            }
            const std::size_t nodeCapacity = object.femCapacity.nodes == 0u
                ? object.femNodes.size()
                : object.femCapacity.nodes;
            const std::size_t tetrahedronCapacity =
                object.femCapacity.tetrahedra == 0u
                    ? object.tetrahedra.size()
                    : object.femCapacity.tetrahedra;
            if (nodeCapacity < object.femNodes.size() ||
                tetrahedronCapacity < object.tetrahedra.size() ||
                nodeCapacity > std::numeric_limits<std::uint32_t>::max() ||
                tetrahedronCapacity > std::numeric_limits<std::uint32_t>::max()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name +
                        "' has topology capacity below authored topology",
                });
                return result;
            }
            std::vector<bool> contactEligible(nodeCapacity,
                object.femContactNodes.empty());
            if (!object.femContactNodes.empty()) {
                for (const std::uint32_t localNode : object.femContactNodes) {
                    if (localNode >= object.femNodes.size() ||
                        contactEligible[localNode]) {
                        result.diagnostics.push_back({
                            Diagnostic::Severity::error, 0u, 0u,
                            "FEM object '" + object.name +
                                "' contains an invalid or duplicate contact node",
                        });
                        return result;
                    }
                    contactEligible[localNode] = true;
                }
            }
            descriptor.stateOffset = static_cast<nm_u32>(world.fem.nodes.size());
            descriptor.stateCount = static_cast<nm_u32>(nodeCapacity);
            descriptor.elementOffset = static_cast<nm_u32>(world.fem.tetrahedra.size());
            std::uint32_t sourceNodeIndex = 0u;
            for (const Vec3& sourceNode : object.femNodes) {
                const bool fixed = fixedNodes[sourceNodeIndex];
                NMFEMNodeStateGPU node{};
                node.positionAndMass = f4(sourceNode[0], sourceNode[1], sourceNode[2], 0.0);
                node.velocityAndInverseMass = f4(
                    fixed ? 0.0 : object.femInitialVelocity[0],
                    fixed ? 0.0 : object.femInitialVelocity[1],
                    fixed ? 0.0 : object.femInitialVelocity[2],
                    0.0
                );
                node.restAndFixed = f4(
                    sourceNode[0], sourceNode[1], sourceNode[2],
                    fixed ? 1.0 : 0.0
                );
                world.fem.nodes.push_back(node);
                femNodeObjects.push_back(objectIndex);
                femNodeContactEligible.push_back(
                    contactEligible[sourceNodeIndex]
                );
                NMFEMTopologyNodeGPU topologyNode{};
                topologyNode.identity = {
                    sourceNodeIndex++,
                    objectIndex,
                    1u,
                    NM_TOPOLOGY_ACTIVE,
                };
                world.fem.topologyNodes.push_back(topologyNode);
                NMFEMFieldStateGPU field{};
                field.primary = f4(
                    object.multiphysics.initialMechanicalPressure,
                    object.multiphysics.initialTemperature,
                    object.multiphysics.initialPorePressure,
                    object.multiphysics.initialElectricPotential
                );
                field.secondary = f4(
                    object.multiphysics.initialActivation,
                    0.0,
                    0.0,
                    object.multiphysics.enabled ? 1.0 : 0.0
                );
                world.fem.fields.push_back(field);
            }
            for (std::size_t local = object.femNodes.size();
                 local < nodeCapacity;
                 ++local) {
                world.fem.nodes.push_back({});
                femNodeObjects.push_back(objectIndex);
                // Dormant mutable-topology slots must already own analytic
                // rigid-proxy pairs. A split can activate the slot inside a
                // fixed-capacity command buffer, where host-side pair cooking
                // is intentionally unavailable.
                femNodeContactEligible.push_back(
                    object.mutationPolicy.enabled);
                NMFEMTopologyNodeGPU topologyNode{};
                topologyNode.identity = {
                    static_cast<nm_u32>(local),
                    objectIndex,
                    1u,
                    0u,
                };
                world.fem.topologyNodes.push_back(topologyNode);
                NMFEMFieldStateGPU field{};
                field.primary.y = static_cast<float>(
                    object.multiphysics.initialTemperature
                );
                world.fem.fields.push_back(field);
            }
            for (const FieldBoundarySource& boundary : object.fieldBoundaries) {
                if (boundary.node >= object.femNodes.size() ||
                    (boundary.flags & ~(NM_FIELD_DIRICHLET_TEMPERATURE |
                        NM_FIELD_DIRICHLET_PORE_PRESSURE |
                        NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL |
                        NM_FIELD_DIRICHLET_ACTIVATION |
                        NM_FIELD_NEUMANN_TEMPERATURE |
                        NM_FIELD_NEUMANN_PORE_PRESSURE |
                        NM_FIELD_NEUMANN_ELECTRIC_CURRENT)) != 0u ||
                    !std::ranges::all_of(
                        boundary.value,
                        [](const double value) { return finite(value); }
                    ) ||
                    !finite(boundary.flux)) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM object '" + object.name +
                            "' contains an invalid field boundary",
                    });
                    return result;
                }
                NMFieldBoundaryGPU cooked{};
                cooked.identity = {
                    descriptor.stateOffset + boundary.node,
                    objectIndex,
                    boundary.flags,
                    boundary.stableIdentifier,
                };
                cooked.value = f4(
                    boundary.value[0], boundary.value[1],
                    boundary.value[2], boundary.value[3]
                );
                cooked.flux = f4(
                    boundary.flux[0], boundary.flux[1], boundary.flux[2], 0.0
                );
                world.fem.fieldBoundaries.push_back(cooked);
            }
            if (object.multiphysics.enabled &&
                material.mixed.electricalConductivity > 0.0) {
                std::vector<std::uint32_t> parent(object.femNodes.size());
                std::iota(parent.begin(), parent.end(), 0u);
                const auto root = [&](std::uint32_t node) {
                    while (parent[node] != node) {
                        parent[node] = parent[parent[node]];
                        node = parent[node];
                    }
                    return node;
                };
                for (const TetrahedronSource& tetrahedron : object.tetrahedra) {
                    if (std::ranges::any_of(
                            tetrahedron.nodes,
                            [&](const std::uint32_t node) {
                                return node >= object.femNodes.size();
                            })) continue;
                    const std::uint32_t first = root(tetrahedron.nodes[0]);
                    for (std::uint32_t slot = 1u; slot < 4u; ++slot) {
                        const std::uint32_t other = root(tetrahedron.nodes[slot]);
                        if (other != first) parent[other] = first;
                    }
                }
                std::set<std::uint32_t> conductiveComponents;
                for (const TetrahedronSource& tetrahedron : object.tetrahedra)
                    if (tetrahedron.nodes[0] < object.femNodes.size())
                        conductiveComponents.insert(root(tetrahedron.nodes[0]));
                std::set<std::uint32_t> groundedComponents;
                for (const FieldBoundarySource& boundary : object.fieldBoundaries)
                    if ((boundary.flags &
                         NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL) != 0u)
                        groundedComponents.insert(root(boundary.node));
                for (const std::uint32_t component : conductiveComponents) {
                    if (!groundedComponents.contains(component)) {
                        result.diagnostics.push_back({
                            Diagnostic::Severity::error, 0u, 0u,
                            "conductive FEM component in object '" +
                                object.name +
                                "' has no electric Dirichlet/ground boundary",
                        });
                        return result;
                    }
                }
            }
            const double rho = density(material);
            std::vector<double> localMass(nodeCapacity, 0.0);
            for (const TetrahedronSource& sourceTet : object.tetrahedra) {
                if (std::ranges::any_of(sourceTet.nodes, [&](const std::uint32_t node) {
                    return node >= object.femNodes.size();
                })) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM tetrahedron references an invalid local node",
                    });
                    continue;
                }
                const Vec3& x0 = object.femNodes[sourceTet.nodes[0]];
                const Vec3& x1 = object.femNodes[sourceTet.nodes[1]];
                const Vec3& x2 = object.femNodes[sourceTet.nodes[2]];
                const Vec3& x3 = object.femNodes[sourceTet.nodes[3]];
                const Vec3 e1 = subtract(x1, x0);
                const Vec3 e2 = subtract(x2, x0);
                const Vec3 e3 = subtract(x3, x0);
                const Mat3 rest{
                    e1[0], e2[0], e3[0],
                    e1[1], e2[1], e3[1],
                    e1[2], e2[2], e3[2],
                };
                Mat3 inverseRest{};
                const double signedVolume = determinant(rest) / 6.0;
                if (!(signedVolume > 1.0e-18) || !inverse(rest, inverseRest)) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM tetrahedron is inverted or degenerate in the rest state",
                    });
                    continue;
                }
                NMTetrahedronGPU tetrahedron{};
                tetrahedron.nodes = {
                    descriptor.stateOffset + sourceTet.nodes[0],
                    descriptor.stateOffset + sourceTet.nodes[1],
                    descriptor.stateOffset + sourceTet.nodes[2],
                    descriptor.stateOffset + sourceTet.nodes[3],
                };
                tetrahedron.inverseRestRow0 = f4(
                    inverseRest[0], inverseRest[1], inverseRest[2], signedVolume
                );
                tetrahedron.inverseRestRow1 = f4(
                    inverseRest[3], inverseRest[4], inverseRest[5]
                );
                tetrahedron.inverseRestRow2 = f4(
                    inverseRest[6], inverseRest[7], inverseRest[8]
                );
                tetrahedron.identity = {
                    object.materialIndex,
                    objectIndex,
                    1u,
                    NM_OBJECT_ACTIVE,
                };
                const std::uint32_t tetrahedronIndex =
                    static_cast<nm_u32>(world.fem.tetrahedra.size());
                world.fem.tetrahedra.push_back(tetrahedron);
                const double nodalMass = rho * signedVolume * 0.25;
                for (const std::uint32_t localNode : sourceTet.nodes) {
                    localMass[localNode] += nodalMass;
                    femIncidence.push_back({
                        descriptor.stateOffset + localNode,
                        tetrahedronIndex,
                    });
                }
            }
            for (std::size_t localNode = 0u; localNode < localMass.size(); ++localNode) {
                const std::size_t globalNode =
                    static_cast<std::size_t>(descriptor.stateOffset) + localNode;
                world.fem.nodes[globalNode].positionAndMass.w =
                    static_cast<float>(localMass[localNode]);
                world.fem.nodes[globalNode].velocityAndInverseMass.w =
                    localMass[localNode] > 0.0 && !fixedNodes[localNode]
                        ? static_cast<float>(1.0 / localMass[localNode])
                        : 0.0f;
            }
            const std::size_t authoredTetrahedronCount =
                world.fem.tetrahedra.size() - descriptor.elementOffset;
            for (std::size_t local = authoredTetrahedronCount;
                 local < tetrahedronCapacity; ++local) {
                NMTetrahedronGPU inactive{};
                inactive.identity = {
                    object.materialIndex,
                    objectIndex,
                    1u,
                    0u,
                };
                world.fem.tetrahedra.push_back(inactive);
            }
            // The object descriptor owns the complete private arena. Active
            // topology is carried exclusively by identity.w/counts, so later
            // GPU transactions can claim dormant slots without changing the
            // command-buffer resource layout.
            descriptor.elementCount = static_cast<nm_u32>(tetrahedronCapacity);

            using FaceKey = std::array<std::uint32_t, 3>;
            struct FaceSide {
                std::uint32_t tetrahedron = NM_INVALID_INDEX;
                std::uint32_t oppositeCorner = NM_INVALID_INDEX;
            };
            std::map<FaceKey, std::vector<FaceSide>> faceAdjacency;
            constexpr std::array<std::array<std::uint32_t, 3>, 4> kFaces{{
                {{1u, 2u, 3u}},
                {{0u, 3u, 2u}},
                {{0u, 1u, 3u}},
                {{0u, 2u, 1u}},
            }};
            for (std::uint32_t localTet = 0u;
                 localTet < object.tetrahedra.size();
                 ++localTet) {
                const TetrahedronSource& sourceTet = object.tetrahedra[localTet];
                for (std::uint32_t opposite = 0u; opposite < kFaces.size();
                     ++opposite) {
                    const auto& localFace = kFaces[opposite];
                    FaceKey key{
                        sourceTet.nodes[localFace[0]],
                        sourceTet.nodes[localFace[1]],
                        sourceTet.nodes[localFace[2]],
                    };
                    std::ranges::sort(key);
                    faceAdjacency[key].push_back({
                        descriptor.elementOffset + localTet, opposite
                    });
                }
            }
            std::size_t internalFaceCount = 0u;
            for (const auto& [key, adjacent] : faceAdjacency) {
                (void)key;
                if (adjacent.empty() || adjacent.size() > 2u) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM object '" + object.name +
                            "' has a non-manifold tetrahedral face",
                    });
                    return result;
                }
                if (adjacent.size() == 2u) {
                    ++internalFaceCount;
                }
            }
            std::uint32_t stableSurfaceFace = 0u;
            for (const auto& [key, adjacent] : faceAdjacency) {
                (void)key;
                NMFEMSurfaceFaceGPU surface{};
                surface.adjacency = {
                    adjacent[0].tetrahedron,
                    adjacent.size() == 2u
                        ? adjacent[1].tetrahedron : NM_INVALID_INDEX,
                    objectIndex,
                    NM_TOPOLOGY_ACTIVE,
                };
                surface.sides = {
                    adjacent[0].oppositeCorner,
                    adjacent.size() == 2u
                        ? adjacent[1].oppositeCorner : NM_INVALID_INDEX,
                    stableSurfaceFace++,
                    0u,
                };
                world.fem.surfaceFaces.push_back(surface);
            }
            const std::size_t cohesiveCapacity =
                object.femCapacity.cohesiveFaces == 0u
                    ? (object.mutationPolicy.cohesiveFracture
                        ? internalFaceCount : 0u)
                    : object.femCapacity.cohesiveFaces;
            if (cohesiveCapacity < internalFaceCount &&
                object.mutationPolicy.cohesiveFracture) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name +
                        "' has insufficient cohesive-face capacity",
                });
                return result;
            }
            const std::size_t firstCohesive = world.fem.cohesiveFaces.size();
            if (object.mutationPolicy.cohesiveFracture) {
                std::uint32_t stableFace = 0u;
                for (const auto& [key, adjacent] : faceAdjacency) {
                    if (adjacent.size() != 2u) {
                        continue;
                    }
                    const Vec3& x0 = object.femNodes[key[0]];
                    const Vec3& x1 = object.femNodes[key[1]];
                    const Vec3& x2 = object.femNodes[key[2]];
                    const Vec3 e0 = subtract(x1, x0);
                    const Vec3 e1 = subtract(x2, x0);
                    const Vec3 cross{
                        e0[1] * e1[2] - e0[2] * e1[1],
                        e0[2] * e1[0] - e0[0] * e1[2],
                        e0[0] * e1[1] - e0[1] * e1[0],
                    };
                    const double twiceArea = std::sqrt(
                        cross[0] * cross[0] + cross[1] * cross[1] +
                        cross[2] * cross[2]
                    );
                    NMCohesiveFaceGPU face{};
                    face.nodesAndFirst = {
                        descriptor.stateOffset + key[0],
                        descriptor.stateOffset + key[1],
                        descriptor.stateOffset + key[2],
                        adjacent[0].tetrahedron,
                    };
                    face.adjacency = {
                        adjacent[1].tetrahedron, objectIndex, stableFace++,
                        NM_TOPOLOGY_ACTIVE | NM_TOPOLOGY_COHESIVE,
                    };
                    if (twiceArea > 0.0) {
                        face.geometry = f4(
                            cross[0] / twiceArea,
                            cross[1] / twiceArea,
                            cross[2] / twiceArea,
                            0.5 * twiceArea
                        );
                    }
                    world.fem.cohesiveFaces.push_back(face);
                }
            }
            world.fem.cohesiveFaces.resize(
                firstCohesive + cohesiveCapacity
            );

            const std::size_t firstMutation = world.fem.mutationCommands.size();
            const std::size_t mutationCapacity =
                object.femCapacity.mutationCommands == 0u
                    ? object.mutationCommands.size()
                    : object.femCapacity.mutationCommands;
            if (mutationCapacity < object.mutationCommands.size()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name +
                        "' has insufficient mutation-command capacity",
                });
                return result;
            }
            for (const MutationCommandSource& command : object.mutationCommands) {
                NMMutationCommandGPU cooked{};
                cooked.identity = {
                    static_cast<nm_u32>(command.kind), objectIndex,
                    command.stableIdentifier, NM_MUTATION_ACTIVE,
                };
                cooked.schedule = {
                    command.controlStep, command.target, command.priority, 0u,
                };
                cooked.geometry0 = f4(
                    command.geometry0[0], command.geometry0[1],
                    command.geometry0[2], command.geometry0[3]
                );
                cooked.geometry1 = f4(
                    command.geometry1[0], command.geometry1[1],
                    command.geometry1[2], command.geometry1[3]
                );
                world.fem.mutationCommands.push_back(cooked);
            }
            std::ranges::sort(
                world.fem.mutationCommands.begin() + firstMutation,
                world.fem.mutationCommands.end(),
                [](const NMMutationCommandGPU& left,
                   const NMMutationCommandGPU& right) {
                    return std::tuple{
                        left.schedule.x, left.schedule.z, left.identity.z,
                        left.schedule.y
                    } < std::tuple{
                        right.schedule.x, right.schedule.z, right.identity.z,
                        right.schedule.y
                    };
                }
            );
            world.fem.mutationCommands.resize(firstMutation + mutationCapacity);

            const std::size_t firstChannel = world.fem.punctureChannels.size();
            world.fem.punctureChannels.resize(
                firstChannel + object.femCapacity.punctureChannels
            );
            for (std::size_t channel = firstChannel;
                 channel < world.fem.punctureChannels.size(); ++channel) {
                world.fem.punctureChannels[channel].identity.x = objectIndex;
            }
            cookedFEMCapacity.topology = {
                static_cast<nm_u32>(nodeCapacity),
                static_cast<nm_u32>(tetrahedronCapacity),
                static_cast<nm_u32>(cohesiveCapacity),
                object.femCapacity.punctureChannels,
            };
            const std::uint64_t contactCapacity =
                static_cast<std::uint64_t>(nodeCapacity) *
                world.contact.rigidProxies.size();
            if (contactCapacity > std::numeric_limits<nm_u32>::max()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM contact capacity exceeds the 32-bit ABI",
                });
                return result;
            }
            const std::uint64_t activeContactCapacity =
                object.femCapacity.activeContacts == 0u
                ? contactCapacity
                : object.femCapacity.activeContacts;
            if (activeContactCapacity > contactCapacity) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name +
                        "' active contact capacity exceeds eligible contact rows",
                });
                return result;
            }
            cookedFEMCapacity.work = {
                static_cast<nm_u32>(mutationCapacity),
                static_cast<nm_u32>(nodeCapacity * 4u),
                static_cast<nm_u32>(activeContactCapacity),
                std::bit_cast<nm_u32>(static_cast<float>(
                    object.mutationPolicy.punctureImpulseThreshold
                )),
            };
        }

        NMAdaptiveStateGPU adaptive{};
        adaptive.activeRepresentation = descriptor.representation;
        adaptive.requestedRepresentation = descriptor.representation;
        adaptive.angularVelocityAndMinimumJ.w = 1.0f;
        adaptive.orientation = f4(0.0, 0.0, 0.0, 1.0);

        // The rest-frame centre is immutable transfer metadata. Promotion
        // reconstructs every continuum point as C_rigid + R_rigid (X_rest -
        // C_rest), so current translation never contaminates deformation.
        double restMass = 0.0;
        Vec3 restMoment{};
        if (representation == Representation::mpm) {
            for (std::uint32_t local = 0u; local < descriptor.stateCount; ++local) {
                const NMParticleStateGPU& particle =
                    world.mpm.particles[descriptor.stateOffset + local];
                const double mass = particle.positionAndMass.w;
                restMass += mass;
                restMoment[0] += mass * particle.referenceAndTemperature.x;
                restMoment[1] += mass * particle.referenceAndTemperature.y;
                restMoment[2] += mass * particle.referenceAndTemperature.z;
            }
        } else if (representation == Representation::fem) {
            for (std::uint32_t local = 0u; local < descriptor.stateCount; ++local) {
                const NMFEMNodeStateGPU& node =
                    world.fem.nodes[descriptor.stateOffset + local];
                const double mass = node.positionAndMass.w;
                restMass += mass;
                restMoment[0] += mass * node.restAndFixed.x;
                restMoment[1] += mass * node.restAndFixed.y;
                restMoment[2] += mass * node.restAndFixed.z;
            }
        }
        if (restMass > 0.0 && finite(restMass)) {
            adaptive.referenceCenter = f4(
                restMoment[0] / restMass,
                restMoment[1] / restMass,
                restMoment[2] / restMass,
                0.0
            );
            adaptive.centerAndRadius = adaptive.referenceCenter;
            adaptive.massAndError.x = static_cast<float>(restMass);
        }
        if (object.adaptive) {
            if (representation == Representation::rigid ||
                descriptor.stateCount == 0u ||
                !(restMass > 0.0) ||
                !finite(restMass)) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "adaptive object '" + object.name +
                        "' requires a positive-mass MPM or FEM representation",
                });
            }
            if (object.rigidBinding >= source.rigidProxies.size()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "adaptive object '" + object.name + "' has no valid rigid proxy binding",
                });
            } else {
                const RigidProxySource& binding = source.rigidProxies[object.rigidBinding];
                if (!binding.dynamic || binding.articulated ||
                    binding.bodyIndex == NM_INVALID_INDEX ||
                    binding.sceneBodyIndex == NM_INVALID_INDEX ||
                    !adaptiveBindings.insert(object.rigidBinding).second ||
                    !adaptiveBodyBindings.insert(binding.bodyIndex).second ||
                    !adaptiveSceneBindings.insert(binding.sceneBodyIndex).second) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "adaptive rigid bindings must own unique free-dynamic proxy and scene-body targets",
                    });
                } else {
                    adaptiveBodyOwners.emplace(
                        binding.bodyIndex,
                        objectIndex
                    );
                }
            }
        }
        world.adaptive.push_back(adaptive);

        NMSchedulerStateGPU scheduler{};
        scheduler.baseExponent = exponent;
        scheduler.activeExponent = exponent;
        scheduler.requestedExponent = exponent;
        scheduler.numerical.w = representation == Representation::rigid ? 0.0f : 1.0f;
        scheduler.thresholds = f4(
            0.05,
            1.0e-4,
            object.promotionStrain,
            1.0e-3
        );
        world.schedulers.push_back(scheduler);
        world.fem.capacities.push_back(cookedFEMCapacity);
        world.objects.push_back(descriptor);
    }

    // Every collision proxy attached to an adaptive fallback body shares the
    // same representation owner. This disables the complete rigid shape set
    // while continuum owns the object, rather than only the one proxy named by
    // the object's transfer binding.
    for (NMRigidProxyGPU& proxy : world.contact.rigidProxies) {
        const auto owner = adaptiveBodyOwners.find(proxy.bodyIndex);
        if (owner != adaptiveBodyOwners.end()) {
            proxy.adaptiveObjectIndex = owner->second;
        }
    }

    buildIncidence(
        std::move(mpmIncidence),
        world.mpm.nodes.size(),
        mpmNodeObjects,
        world.mpm.nodeIncidence,
        world.mpm.nodeRanges
    );
    buildIncidence(
        std::move(femIncidence),
        world.fem.nodes.size(),
        femNodeObjects,
        world.fem.nodeIncidence,
        world.fem.nodeRanges
    );

    std::vector<std::uint32_t> unifiedNodeObjects;
    std::vector<bool> unifiedNodeContactEligible;
    unifiedNodeObjects.reserve(mpmNodeObjects.size() + femNodeObjects.size());
    unifiedNodeObjects.insert(
        unifiedNodeObjects.end(), mpmNodeObjects.begin(), mpmNodeObjects.end()
    );
    unifiedNodeObjects.insert(
        unifiedNodeObjects.end(), femNodeObjects.begin(), femNodeObjects.end()
    );
    unifiedNodeContactEligible.insert(
        unifiedNodeContactEligible.end(), mpmNodeObjects.size(), true
    );
    unifiedNodeContactEligible.insert(
        unifiedNodeContactEligible.end(),
        femNodeContactEligible.begin(), femNodeContactEligible.end()
    );
    std::vector<IncidenceEntry> contactNodeIncidence;
    std::vector<IncidenceEntry> rigidIncidence;
    const std::size_t mpmNodeCount = world.mpm.nodes.size();
    for (std::size_t nodeSize = 0u;
         nodeSize < unifiedNodeObjects.size();
         ++nodeSize) {
        if (!unifiedNodeContactEligible[nodeSize]) continue;
        const std::uint32_t node = static_cast<std::uint32_t>(nodeSize);
        for (std::size_t proxySize = 0u;
             proxySize < world.contact.rigidProxies.size();
             ++proxySize) {
            const std::uint32_t proxy = static_cast<std::uint32_t>(proxySize);
            const std::uint32_t objectIndex = unifiedNodeObjects[nodeSize];
            // Every proxy on an adaptive fallback body represents the same
            // matter. Generating any continuum↔fallback pair would apply an
            // unphysical self-contact while the continuum representation is
            // active, including additional shapes on the same rigid body.
            if (objectIndex < world.objects.size() &&
                world.contact.rigidProxies[proxySize]
                    .adaptiveObjectIndex == objectIndex) {
                continue;
            }
            NMContactPairGPU pair{};
            pair.continuumNode = node;
            pair.rigidProxy = proxy;
            pair.objectIndex = objectIndex;
            pair.materialInterface = world.contact.rigidProxies[proxySize].materialIndex;
            const std::uint32_t pairIndex =
                static_cast<std::uint32_t>(world.contact.pairs.size());
            world.contact.pairs.push_back(pair);
            contactNodeIncidence.push_back({node, pairIndex});
            rigidIncidence.push_back({proxy, pairIndex});
        }
    }
    (void)mpmNodeCount;
    buildIncidence(
        std::move(contactNodeIncidence),
        unifiedNodeObjects.size(),
        unifiedNodeObjects,
        world.contact.nodeIncidence,
        world.contact.nodeRanges
    );
    std::vector<std::uint32_t> noObject(world.contact.rigidProxies.size(), NM_INVALID_INDEX);
    buildIncidence(
        std::move(rigidIncidence),
        world.contact.rigidProxies.size(),
        noObject,
        world.contact.rigidIncidence,
        world.contact.rigidRanges
    );

    for (std::size_t materialIndex = 0u;
         materialIndex < source.materials.size();
         ++materialIndex) {
        const MaterialProgram& material = source.materials[materialIndex];
        for (std::size_t localParameter = 0u;
             localParameter < material.parameters.size();
             ++localParameter) {
            const Parameter& parameter = material.parameters[localParameter];
            if (!parameter.identifiable) {
                continue;
            }
            NMIdentificationDistributionGPU distribution{};
            const std::uint32_t globalParameter =
                world.materials[materialIndex].parameterOffset +
                static_cast<std::uint32_t>(localParameter);
            distribution.identity = {
                static_cast<nm_u32>(materialIndex),
                static_cast<nm_u32>(localParameter),
                globalParameter,
                parameter.logarithmic ? 1u : 0u,
            };
            const double mean = parameter.logarithmic
                ? std::log(parameter.defaultValue)
                : parameter.defaultValue;
            const double lower = parameter.logarithmic
                ? std::log(parameter.lower)
                : parameter.lower;
            const double upper = parameter.logarithmic
                ? std::log(parameter.upper)
                : parameter.upper;
            const double standardDeviation = std::max(
                0.05 * std::abs(upper - lower),
                1.0e-6 * std::max(std::abs(mean), 1.0)
            );
            distribution.momentsAndBounds = f4(
                mean, standardDeviation, lower, upper
            );
            distribution.update = f4(
                0.2,
                1.0,
                1.0e-5 * std::max(std::abs(mean), 1.0),
                parameter.logarithmic ? 1.0 : 0.0
            );
            world.identification.push_back(distribution);
        }
    }

    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion = NM_MATTER_ABI_VERSION;
    dispatch.flags =
        (source.deterministic ? NM_MATTER_DETERMINISTIC : 0u) |
        NM_MATTER_IPC |
        (!world.contact.pairs.empty() ? NM_MATTER_CONTACT : 0u) |
        (std::ranges::any_of(source.objects, [](const ObjectSource& object) {
            return object.adaptive;
        }) ? NM_MATTER_ADAPTIVE : 0u) |
        (!world.identification.empty() ? NM_MATTER_IDENTIFICATION : 0u);
    dispatch.flags |=
        (std::ranges::any_of(world.objects, [](const NMContinuumObjectGPU& object) {
            return (object.flags & NM_OBJECT_MIXED_FEM) != 0u;
        }) ? NM_MATTER_MIXED_FEM : 0u) |
        (std::ranges::any_of(world.objects, [](const NMContinuumObjectGPU& object) {
            return (object.flags & NM_OBJECT_MULTIPHYSICS) != 0u;
        }) ? NM_MATTER_MULTIPHYSICS : 0u) |
        (std::ranges::any_of(world.objects, [](const NMContinuumObjectGPU& object) {
            return (object.flags & NM_OBJECT_MUTABLE_TOPOLOGY) != 0u;
        }) ? NM_MATTER_MUTATION : 0u) |
        (!world.learnedMaterials.empty() ? NM_MATTER_LEARNED_MATERIAL : 0u);
    dispatch.environmentCount = source.environmentCount;
    dispatch.objectCount = static_cast<nm_u32>(world.objects.size());
    dispatch.materialCount = static_cast<nm_u32>(world.materials.size());
    dispatch.parameterCount = static_cast<nm_u32>(world.parameters.size());
    dispatch.particleCount = static_cast<nm_u32>(world.mpm.particles.size());
    dispatch.gridNodeCount = static_cast<nm_u32>(world.mpm.nodes.size());
    dispatch.femNodeCount = static_cast<nm_u32>(world.fem.nodes.size());
    dispatch.tetrahedronCount = static_cast<nm_u32>(world.fem.tetrahedra.size());
    dispatch.rigidProxyCount = static_cast<nm_u32>(world.contact.rigidProxies.size());
    dispatch.contactPairCount = static_cast<nm_u32>(world.contact.pairs.size());
    std::uint64_t activeMPMNodeCapacity = 0u;
    for (const NMContinuumObjectGPU& object : world.objects) {
        if (object.representation != NM_REPRESENTATION_MPM) continue;
        const std::uint64_t supportBound =
            static_cast<std::uint64_t>(object.stateCount) *
            NM_MPM_STENCIL_WIDTH;
        activeMPMNodeCapacity += std::min<std::uint64_t>(
            object.auxiliaryCount, supportBound);
    }
    if (activeMPMNodeCapacity > dispatch.gridNodeCount ||
        activeMPMNodeCapacity > std::numeric_limits<nm_u32>::max()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "active MPM node capacity exceeds the cooked grid arena"
        });
        return result;
    }
    dispatch.mpmActiveNodeCapacity =
        static_cast<nm_u32>(activeMPMNodeCapacity);
    bool hasArticulatedProxy = false;
    for (const NMRigidProxyGPU& proxy : world.contact.rigidProxies) {
        hasArticulatedProxy = hasArticulatedProxy ||
            (proxy.flags & NM_RIGID_ARTICULATED) != 0u;
    }
    const std::uint64_t rigidGeneralizedCapacity =
        (hasArticulatedProxy ? NM_MATTER_MAX_ARTICULATED_DOFS : 0u) +
        static_cast<std::uint64_t>(freeBodyIndices.size()) * 6u;
    const std::uint64_t rigidQCapacity =
        hasArticulatedProxy ? NM_MATTER_MAX_ARTICULATED_Q : 0u;
    if (rigidGeneralizedCapacity > std::numeric_limits<nm_u32>::max() ||
        rigidQCapacity > std::numeric_limits<nm_u32>::max()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "rigid generalized candidate capacity exceeds 32-bit ABI"
        });
        return result;
    }
    dispatch.rigidGeneralizedCapacity =
        static_cast<nm_u32>(rigidGeneralizedCapacity);
    dispatch.rigidQCapacity = static_cast<nm_u32>(rigidQCapacity);
    std::uint64_t activeContactCapacity = 0u;
    for (const NMFEMCapacityGPU& capacity : world.fem.capacities)
        activeContactCapacity += capacity.work.z;
    dispatch.reservedMixed1 = static_cast<nm_u32>(std::min<std::uint64_t>(
        activeContactCapacity == 0u
            ? dispatch.contactPairCount : activeContactCapacity,
        dispatch.contactPairCount
    ));
    dispatch.maximumRateExponent = options.maximumRateExponent;
    dispatch.femPCGIterations = source.femPCGIterations;
    dispatch.identificationCandidateCount = source.identificationCandidates;
    const std::uint64_t eventStride =
        static_cast<std::uint64_t>(NM_EVENT_CLASS_COUNT) *
        dispatch.objectCount;
    if (eventStride > std::numeric_limits<std::uint32_t>::max()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "event-token stride exceeds the 32-bit cooked ABI",
        });
        return result;
    }
    dispatch.eventStride = static_cast<nm_u32>(eventStride);
    dispatch.mpmGridCount = static_cast<nm_u32>(world.mpm.grids.size());
    dispatch.mpmBlockCount = static_cast<nm_u32>(world.mpm.blocks.size());
    dispatch.mpmBlockLookupCount =
        static_cast<nm_u32>(world.mpm.blockLookup.size());
    dispatch.maximumParticlesPerBlock =
        NM_MPM_MAX_PARTICLES_PER_BLOCK;
    dispatch.materialStateStride = 0u;
    for (const NMMaterialGPU& material : world.materials) {
        dispatch.materialStateStride = std::max(
            dispatch.materialStateStride,
            material.stateCount
        );
    }
    dispatch.stateInitialCount =
        static_cast<nm_u32>(world.stateInitials.size());
    dispatch.mixedMaterialCount = static_cast<nm_u32>(world.mixedMaterials.size());
    dispatch.fieldBoundaryCount =
        static_cast<nm_u32>(world.fem.fieldBoundaries.size());
    dispatch.cohesiveFaceCount =
        static_cast<nm_u32>(world.fem.cohesiveFaces.size());
    dispatch.mutationCommandCount =
        static_cast<nm_u32>(world.fem.mutationCommands.size());
    dispatch.learnedMaterialCount =
        static_cast<nm_u32>(world.learnedMaterials.size());
    dispatch.learnedLayerCount =
        static_cast<nm_u32>(world.learnedLayers.size());
    dispatch.learnedWeightCount =
        static_cast<nm_u32>(world.learnedWeights.size());
    dispatch.topologyNodeCapacity =
        static_cast<nm_u32>(world.fem.topologyNodes.size());
    dispatch.punctureChannelCount =
        static_cast<nm_u32>(world.fem.punctureChannels.size());
    dispatch.femCapacityCount =
        static_cast<nm_u32>(world.fem.capacities.size());
    dispatch.surfaceFaceCount =
        static_cast<nm_u32>(world.fem.surfaceFaces.size());
    std::uint64_t deformableContactCapacity = 0u;
    for (const ObjectSource& object : source.objects) {
        if (object.femCapacity.deformableContacts != 0u) {
            deformableContactCapacity +=
                object.femCapacity.deformableContacts;
        } else if (object.representation == Representation::fem) {
            // A closed tetrahedral surface has O(elements) boundary faces;
            // reserve a linear active manifold by default rather than the
            // quadratic broadphase cross product.
            deformableContactCapacity += 8u * object.tetrahedra.size();
        } else if (object.representation == Representation::mpm) {
            // Compact active-grid points form the MPM contact surface. Their
            // active manifold remains linear in material-point count even
            // though the background grid is fixed capacity.
            deformableContactCapacity += 8u * object.particles.size();
        }
    }
    dispatch.deformableContactCapacity = static_cast<nm_u32>(
        std::min<std::uint64_t>(
            deformableContactCapacity,
            std::numeric_limits<std::uint32_t>::max()));
    std::uint64_t topologyTetCapacity = 0u;
    for (const NMFEMCapacityGPU capacity : world.fem.capacities) {
        topologyTetCapacity += capacity.topology.y;
    }
    if (topologyTetCapacity > std::numeric_limits<std::uint32_t>::max()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "FEM topology tetrahedron capacity exceeds the 32-bit ABI",
        });
        return result;
    }
    dispatch.topologyTetrahedronCapacity =
        static_cast<nm_u32>(topologyTetCapacity);
    dispatch.gravityAndTimestep = f4(
        source.gravity[0], source.gravity[1], source.gravity[2], source.frameTimestep
    );
    dispatch.numericalLimits = f4(
        source.contactSlop,
        source.maximumDepenetrationSpeed,
        1.0e-4,
        1.0e20
    );
    world.dispatch = dispatch;

    if (!result.succeeded()) {
        return result;
    }

    world.fingerprint = compiledWorldFingerprint(world);
    const bool hasAllocationOverrides = std::ranges::any_of(
        source.objects,
        [](const ObjectSource& object) {
            const FEMCapacitySource& capacity = object.femCapacity;
            return capacity.nodes != 0u ||
                capacity.tetrahedra != 0u ||
                capacity.cohesiveFaces != 0u ||
                capacity.punctureChannels != 0u ||
                capacity.mutationCommands != 0u ||
                capacity.activeContacts != 0u ||
                capacity.deformableContacts != 0u;
        }
    );
    if (hasAllocationOverrides) {
        WorldSource canonicalSource = source;
        for (ObjectSource& object : canonicalSource.objects) {
            object.femCapacity = {};
        }
        CompileOptions canonicalOptions = options;
        canonicalOptions.emitSpecializedMetal = false;
        CompileResult canonical = compileWorld(canonicalSource, canonicalOptions);
        if (!canonical.succeeded()) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error,
                0u,
                0u,
                "failed to derive allocation-independent Matter physics fingerprint",
            });
            return result;
        }
        world.physicsFingerprint = canonical.world.fingerprint;
    } else {
        world.physicsFingerprint = world.fingerprint;
    }
    std::string layoutError;
    if (!validateCompiledWorldLayout(world, &layoutError)) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "compiled Matter layout is invalid: " + layoutError,
        });
        return result;
    }
    result.generatedMetal = options.emitSpecializedMetal
        ? emitSpecializedMetal(world.constitutive)
        : std::string{};
    return result;
}

} // namespace numi::matter
