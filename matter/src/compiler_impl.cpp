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

[[nodiscard]] Vec3 cross(const Vec3& left, const Vec3& right) noexcept {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

[[nodiscard]] double dot(const Vec3& left, const Vec3& right) noexcept {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
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

struct GridKey {
    std::uint32_t object = 0u;
    int x = 0;
    int y = 0;
    int z = 0;

    friend bool operator<(const GridKey& left, const GridKey& right) noexcept {
        return std::tie(left.object, left.x, left.y, left.z) <
            std::tie(right.object, right.x, right.y, right.z);
    }
};

struct IncidenceEntry {
    std::uint32_t owner = 0u;
    std::uint32_t source = 0u;

    friend bool operator<(const IncidenceEntry& left, const IncidenceEntry& right) noexcept {
        return std::tie(left.owner, left.source) < std::tie(right.owner, right.source);
    }
};

void buildIncidence(
    std::vector<IncidenceEntry> entries,
    const std::size_t ownerCount,
    const std::span<const std::uint32_t> ownerObjects,
    std::vector<std::uint32_t>& incidence,
    std::vector<NMIncidenceRangeGPU>& ranges
) {
    std::ranges::sort(entries);
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

[[nodiscard]] std::array<double, 3> quadraticWeights(const double coordinate) {
    return {
        0.5 * (1.5 - coordinate) * (1.5 - coordinate),
        0.75 - (coordinate - 1.0) * (coordinate - 1.0),
        0.5 * (coordinate - 0.5) * (coordinate - 0.5),
    };
}

[[nodiscard]] std::array<double, 3> quadraticDerivatives(const double coordinate) {
    return {
        coordinate - 1.5,
        -2.0 * (coordinate - 1.0),
        coordinate - 0.5,
    };
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
}

} // namespace

CompileResult compileWorld(
    const WorldSource& source,
    const CompileOptions& options
) {
    CompileResult result;
    validateWorld(source, options, result.diagnostics);
    CompiledWorld& world = result.world;

    if (!result.succeeded()) {
        return result;
    }

    world.constitutive.reserve(source.materials.size());
    world.materials.reserve(source.materials.size());
    for (const MaterialProgram& material : source.materials) {
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
        program.gpu.parameterOffset = static_cast<nm_u32>(world.parameters.size());
        program.gpu.stressProgramOffset = static_cast<nm_u32>(world.scalarPrograms.size());
        for (const ScalarBytecode& component : program.stress) {
            appendProgram(component, world.instructions, world.scalarPrograms);
        }
        program.gpu.tangentProgramOffset = static_cast<nm_u32>(world.scalarPrograms.size());
        for (const ScalarBytecode& component : program.tangentVector) {
            appendProgram(component, world.instructions, world.scalarPrograms);
        }
        if (program.validity.has_value()) {
            program.gpu.validityProgram = static_cast<nm_u32>(world.scalarPrograms.size());
            appendProgram(*program.validity, world.instructions, world.scalarPrograms);
        } else {
            program.gpu.validityProgram = NM_INVALID_INDEX;
        }
        world.parameters.insert(
            world.parameters.end(),
            program.parameters.begin(),
            program.parameters.end()
        );
        world.materials.push_back(program.gpu);
        world.constitutive.push_back(std::move(program));
    }

    if (!result.succeeded() || world.materials.size() != source.materials.size()) {
        return result;
    }

    world.contact.rigidProxies.reserve(source.rigidProxies.size());
    for (const RigidProxySource& proxy : source.rigidProxies) {
        if (!finite(proxy.localCenter) || !finite(proxy.localExtent) ||
            !std::ranges::all_of(proxy.localOrientation, [](const double value) {
                return finite(value);
            }) || !finite(proxy.radiusOrOffset) ||
            proxy.materialIndex >= source.materials.size()) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "rigid proxy contains invalid geometry or material binding",
            });
            continue;
        }
        NMRigidProxyGPU cooked{};
        cooked.shapeKind = static_cast<nm_u32>(proxy.shape);
        cooked.bodyIndex = proxy.bodyIndex;
        cooked.materialIndex = proxy.materialIndex;
        cooked.flags =
            (proxy.articulated ? NM_RIGID_ARTICULATED : 0u) |
            (proxy.dynamic ? NM_RIGID_DYNAMIC : 0u);
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
    std::vector<IncidenceEntry> mpmIncidence;
    std::vector<IncidenceEntry> femIncidence;
    std::map<GridKey, std::uint32_t> gridNodes;
    std::set<std::uint32_t> adaptiveBindings;

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
        const ObjectSource& object = source.objects[objectIndexSize];
        if (object.materialIndex >= source.materials.size() ||
            object.name.empty() ||
            !(object.characteristicLength > 0.0) ||
            !finite(object.characteristicLength)) {
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
            (object.identifiable ? NM_OBJECT_IDENTIFIABLE : 0u);
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

        if (representation == Representation::mpm) {
            if (object.particles.empty()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "MPM object '" + object.name + "' has no particles",
                });
            }
            descriptor.stateOffset = static_cast<nm_u32>(world.mpm.particles.size());
            descriptor.stateCount = static_cast<nm_u32>(object.particles.size());
            descriptor.elementOffset = static_cast<nm_u32>(world.mpm.stencils.size());
            const double h = object.characteristicLength;
            for (std::size_t localParticle = 0u;
                 localParticle < object.particles.size();
                 ++localParticle) {
                const ParticleSource& sourceParticle = object.particles[localParticle];
                if (!finite(sourceParticle.position) || !finite(sourceParticle.velocity) ||
                    !(sourceParticle.mass > 0.0) ||
                    !(sourceParticle.referenceVolume > 0.0) ||
                    !finite(sourceParticle.mass) ||
                    !finite(sourceParticle.referenceVolume)) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "MPM particle contains nonfinite or nonpositive state",
                    });
                    continue;
                }
                NMParticleStateGPU particle{};
                particle.positionAndMass = f4(
                    sourceParticle.position[0], sourceParticle.position[1],
                    sourceParticle.position[2], sourceParticle.mass
                );
                particle.velocityAndReferenceVolume = f4(
                    sourceParticle.velocity[0], sourceParticle.velocity[1],
                    sourceParticle.velocity[2], sourceParticle.referenceVolume
                );
                particle.deformationRow0 = f4(1.0, 0.0, 0.0);
                particle.deformationRow1 = f4(0.0, 1.0, 0.0);
                particle.deformationRow2 = f4(0.0, 0.0, 1.0);
                particle.referenceAndTemperature = f4(
                    sourceParticle.position[0], sourceParticle.position[1],
                    sourceParticle.position[2], 293.15
                );
                particle.identity = {
                    objectIndex,
                    object.materialIndex,
                    1u,
                    NM_OBJECT_ACTIVE,
                };
                const std::uint32_t particleIndex =
                    static_cast<nm_u32>(world.mpm.particles.size());
                world.mpm.particles.push_back(particle);

                const Vec3 cell{
                    sourceParticle.position[0] / h,
                    sourceParticle.position[1] / h,
                    sourceParticle.position[2] / h,
                };
                const std::array<int, 3> base{
                    static_cast<int>(std::floor(cell[0] - 0.5)),
                    static_cast<int>(std::floor(cell[1] - 0.5)),
                    static_cast<int>(std::floor(cell[2] - 0.5)),
                };
                const Vec3 fractional{
                    cell[0] - static_cast<double>(base[0]),
                    cell[1] - static_cast<double>(base[1]),
                    cell[2] - static_cast<double>(base[2]),
                };
                const auto wx = quadraticWeights(fractional[0]);
                const auto wy = quadraticWeights(fractional[1]);
                const auto wz = quadraticWeights(fractional[2]);
                const auto dx = quadraticDerivatives(fractional[0]);
                const auto dy = quadraticDerivatives(fractional[1]);
                const auto dz = quadraticDerivatives(fractional[2]);
                for (int z = 0; z < 3; ++z) {
                    for (int y = 0; y < 3; ++y) {
                        for (int x = 0; x < 3; ++x) {
                            const GridKey key{
                                objectIndex,
                                base[0] + x,
                                base[1] + y,
                                base[2] + z,
                            };
                            auto [iterator, inserted] = gridNodes.try_emplace(
                                key,
                                static_cast<std::uint32_t>(world.mpm.nodes.size())
                            );
                            if (inserted) {
                                NMGridNodeStateGPU node{};
                                node.positionAndMass = f4(
                                    static_cast<double>(key.x) * h,
                                    static_cast<double>(key.y) * h,
                                    static_cast<double>(key.z) * h,
                                    0.0
                                );
                                world.mpm.nodes.push_back(node);
                                mpmNodeObjects.push_back(objectIndex);
                            }
                            const double weight = wx[static_cast<std::size_t>(x)] *
                                wy[static_cast<std::size_t>(y)] *
                                wz[static_cast<std::size_t>(z)];
                            const Vec3 gradient{
                                dx[static_cast<std::size_t>(x)] *
                                    wy[static_cast<std::size_t>(y)] *
                                    wz[static_cast<std::size_t>(z)] / h,
                                wx[static_cast<std::size_t>(x)] *
                                    dy[static_cast<std::size_t>(y)] *
                                    wz[static_cast<std::size_t>(z)] / h,
                                wx[static_cast<std::size_t>(x)] *
                                    wy[static_cast<std::size_t>(y)] *
                                    dz[static_cast<std::size_t>(z)] / h,
                            };
                            NMMPMStencilGPU stencil{};
                            stencil.particleIndex = particleIndex;
                            stencil.nodeIndex = iterator->second;
                            stencil.localSlot = static_cast<nm_u32>(9 * z + 3 * y + x);
                            stencil.gradientAndWeight = f4(
                                gradient[0], gradient[1], gradient[2], weight
                            );
                            const std::uint32_t stencilIndex =
                                static_cast<nm_u32>(world.mpm.stencils.size());
                            world.mpm.stencils.push_back(stencil);
                            mpmIncidence.push_back({iterator->second, stencilIndex});
                        }
                    }
                }
            }
            descriptor.elementCount =
                static_cast<nm_u32>(world.mpm.stencils.size()) - descriptor.elementOffset;
        } else if (representation == Representation::fem) {
            if (object.femNodes.empty() || object.tetrahedra.empty()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "FEM object '" + object.name + "' requires nodes and tetrahedra",
                });
            }
            descriptor.stateOffset = static_cast<nm_u32>(world.fem.nodes.size());
            descriptor.stateCount = static_cast<nm_u32>(object.femNodes.size());
            descriptor.elementOffset = static_cast<nm_u32>(world.fem.tetrahedra.size());
            for (const Vec3& sourceNode : object.femNodes) {
                if (!finite(sourceNode)) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "FEM node is nonfinite",
                    });
                }
                NMFEMNodeStateGPU node{};
                node.positionAndMass = f4(sourceNode[0], sourceNode[1], sourceNode[2], 0.0);
                node.velocityAndInverseMass = f4();
                node.restAndFixed = f4(sourceNode[0], sourceNode[1], sourceNode[2], 0.0);
                world.fem.nodes.push_back(node);
                femNodeObjects.push_back(objectIndex);
            }
            const double rho = density(material);
            std::vector<double> localMass(object.femNodes.size(), 0.0);
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
                    localMass[localNode] > 0.0
                        ? static_cast<float>(1.0 / localMass[localNode])
                        : 0.0f;
            }
            descriptor.elementCount =
                static_cast<nm_u32>(world.fem.tetrahedra.size()) - descriptor.elementOffset;
        }

        NMAdaptiveStateGPU adaptive{};
        adaptive.activeRepresentation = descriptor.representation;
        adaptive.requestedRepresentation = descriptor.representation;
        adaptive.angularVelocityAndMinimumJ.w = 1.0f;
        adaptive.orientation = f4(0.0, 0.0, 0.0, 1.0);
        if (object.adaptive) {
            if (object.rigidBinding >= source.rigidProxies.size()) {
                result.diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "adaptive object '" + object.name + "' has no valid rigid proxy binding",
                });
            } else {
                const RigidProxySource& binding = source.rigidProxies[object.rigidBinding];
                if (binding.bodyIndex == NM_INVALID_INDEX ||
                    !adaptiveBindings.insert(object.rigidBinding).second) {
                    result.diagnostics.push_back({
                        Diagnostic::Severity::error, 0u, 0u,
                        "adaptive rigid bindings must be body-backed and unique",
                    });
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
        world.objects.push_back(descriptor);
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
    unifiedNodeObjects.reserve(mpmNodeObjects.size() + femNodeObjects.size());
    unifiedNodeObjects.insert(
        unifiedNodeObjects.end(), mpmNodeObjects.begin(), mpmNodeObjects.end()
    );
    unifiedNodeObjects.insert(
        unifiedNodeObjects.end(), femNodeObjects.begin(), femNodeObjects.end()
    );
    std::vector<IncidenceEntry> contactNodeIncidence;
    std::vector<IncidenceEntry> rigidIncidence;
    const std::size_t mpmNodeCount = world.mpm.nodes.size();
    for (std::size_t nodeSize = 0u;
         nodeSize < unifiedNodeObjects.size();
         ++nodeSize) {
        const std::uint32_t node = static_cast<std::uint32_t>(nodeSize);
        for (std::size_t proxySize = 0u;
             proxySize < world.contact.rigidProxies.size();
             ++proxySize) {
            const std::uint32_t proxy = static_cast<std::uint32_t>(proxySize);
            NMContactPairGPU pair{};
            pair.continuumNode = node;
            pair.rigidProxy = proxy;
            pair.objectIndex = unifiedNodeObjects[nodeSize];
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
        (!world.contact.pairs.empty() ? NM_MATTER_CONTACT : 0u) |
        (std::ranges::any_of(source.objects, [](const ObjectSource& object) {
            return object.adaptive;
        }) ? NM_MATTER_ADAPTIVE : 0u) |
        (!world.identification.empty() ? NM_MATTER_IDENTIFICATION : 0u);
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
    dispatch.maximumRateExponent = options.maximumRateExponent;
    dispatch.femPCGIterations = source.femPCGIterations;
    dispatch.identificationCandidateCount = source.identificationCandidates;
    dispatch.eventStride = NM_EVENT_CLASS_COUNT * dispatch.objectCount;
    dispatch.gravityAndTimestep = f4(
        source.gravity[0], source.gravity[1], source.gravity[2], source.frameTimestep
    );
    dispatch.numericalLimits = f4(
        1.0e-5,
        5.0,
        1.0e-4,
        1.0e20
    );
    world.dispatch = dispatch;

    if (!result.succeeded()) {
        return result;
    }

    std::uint64_t fingerprint = detail::hashBytes(&dispatch, sizeof(dispatch));
    const auto hashVector = [&](const auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        fingerprint = detail::hashBytes(
            values.data(), values.size() * sizeof(Value), fingerprint
        );
    };
    hashVector(world.materials);
    hashVector(world.parameters);
    hashVector(world.instructions);
    hashVector(world.scalarPrograms);
    hashVector(world.objects);
    hashVector(world.mpm.particles);
    hashVector(world.mpm.nodes);
    hashVector(world.mpm.stencils);
    hashVector(world.fem.nodes);
    hashVector(world.fem.tetrahedra);
    hashVector(world.contact.rigidProxies);
    hashVector(world.contact.pairs);
    hashVector(world.identification);
    world.fingerprint = fingerprint == 0u ? 1u : fingerprint;
    result.generatedMetal = options.emitSpecializedMetal
        ? emitSpecializedMetal(world.constitutive)
        : std::string{};
    return result;
}

} // namespace numi::matter
