#include "cardboard_damage_builder.hpp"
#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace numi::matter;
using namespace numi::cardboard;
using namespace numi::cardboard::damage;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double parameterValue(const MaterialProgram& program, const std::string& name) {
    for (const auto& parameter : program.parameters)
        if (parameter.name == name) return parameter.defaultValue;
    throw std::runtime_error("generated program lacks parameter " + name);
}

using Matrix9 = std::array<double, 9>;
using State27 = std::array<double, 27>;

double evaluateExpressionNode(const MaterialProgram& material,
                              std::uint32_t root,
                              const Matrix9& deformation,
                              const State27& accepted,
                              const State27& candidate) {
    std::vector<double> cache(material.expressions.nodes.size(),
                              std::numeric_limits<double>::quiet_NaN());
    std::function<double(std::uint32_t)> visit = [&](std::uint32_t index) -> double {
        require(index < material.expressions.nodes.size(),
                "energy expression node reference is out of range");
        if (std::isfinite(cache[index])) return cache[index];
        const auto& expression = material.expressions.nodes[index];
        const auto arg = [&](std::size_t slot) {
            return visit(expression.arguments.at(slot));
        };
        double value = 0.0;
        switch (expression.kind) {
        case ExprKind::constant: value = expression.constant; break;
        case ExprKind::parameter:
            value = material.parameters.at(expression.index).defaultValue; break;
        case ExprKind::internalState: value = accepted.at(expression.index); break;
        case ExprKind::candidateState: value = candidate.at(expression.index); break;
        case ExprKind::deformation: value = deformation.at(expression.index); break;
        case ExprKind::deformationDirection:
        case ExprKind::deformationRate: value = 0.0; break;
        case ExprKind::timeStep: value = 0.005; break;
        case ExprKind::temperature: value = 293.15; break;
        case ExprKind::add: value = arg(0) + arg(1); break;
        case ExprKind::subtract: value = arg(0) - arg(1); break;
        case ExprKind::multiply: value = arg(0) * arg(1); break;
        case ExprKind::divide: value = arg(0) / arg(1); break;
        case ExprKind::negate: value = -arg(0); break;
        case ExprKind::logarithm: value = std::log(arg(0)); break;
        case ExprKind::exponential: value = std::exp(arg(0)); break;
        case ExprKind::squareRoot: value = std::sqrt(arg(0)); break;
        case ExprKind::absolute: value = std::abs(arg(0)); break;
        case ExprKind::minimum: value = std::min(arg(0), arg(1)); break;
        case ExprKind::maximum: value = std::max(arg(0), arg(1)); break;
        case ExprKind::integerPower: value = std::pow(arg(0), expression.integer); break;
        case ExprKind::clamp: value = std::clamp(arg(0), arg(1), arg(2)); break;
        case ExprKind::expm1MinusArgument: {
            const double x = arg(0); value = std::expm1(x) - x; break;
        }
        case ExprKind::selectNonnegative:
            value = arg(0) >= 0.0 ? arg(1) : arg(2); break;
        case ExprKind::fiberExpLinear:
            throw std::runtime_error("fiber opcode is not part of this source material");
        }
        require(std::isfinite(value), "source free-energy expression was nonfinite");
        cache[index] = value;
        return value;
    };
    return visit(root);
}

bool hasParameter(const MaterialProgram& program, const std::string& name) {
    for (const auto& parameter : program.parameters)
        if (parameter.name == name) return true;
    return false;
}

ElementDamageRequest requestFor(std::uint32_t elementIndex,
                                double sideM,
                                FractureVec3 offset) {
    ElementDamageRequest request;
    request.elementIndex = elementIndex;
    const std::array<FractureVec3, 4> unit{{
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    for (std::size_t i = 0; i < unit.size(); ++i) {
        request.referenceVerticesM[i] = {
            offset.x + sideM * unit[i].x,
            offset.y + sideM * unit[i].y,
            offset.z + sideM * unit[i].z};
    }
    request.selectedPlanes[0].normal = {1.0, 0.0, 0.0};
    request.selectedPlanes[1].normal = {0.0, 1.0, 0.0};
    request.selectedPlanes[2].normal = {0.0, 0.0, 1.0};
    request.selectedPlanes[3].normal = {1.0, 1.0, 1.0};
    request.fractureEnergyJPerM2 = {100.0, 120.0, 140.0, 160.0};
    request.strengthHardening =
        numi_cardboard::robertsson2023::publishedStrengthHardening();
    request.strengthHardeningBinding = publishedStrengthHardeningBinding();
    request.fractureEnergyBinding = ExplicitSetBinding{
        "cardboard_damage_builder_synthetic_Gc_fixture_v1",
        "b2d2a98a6d6d86edcdfad55a90fc440d5dfce9ebff058d030214ee9b82985691",
        "synthetic_test_only_not_physical_calibration"};
    request.energyConvention = EnergyConvention::paper_absolute;
    request.damageCap = 0.96;
    return request;
}

bool near(double a, double b, double relative = 1.0e-12) {
    return std::abs(a - b) <= relative * std::max({1.0, std::abs(a), std::abs(b)});
}

void expectInvalid(const std::function<void()>& action,
                   const std::string& message) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

CompiledWorld compileTwoElementCallable(const std::vector<ElementDamageProgram>& built) {
    require(built.size() == 2u, "callable fixture needs two per-element programs");
    WorldSource source;
    source.environmentCount = 1u;
    source.gravity = {0.0f, 0.0f, 0.0f};
    source.frameTimestep = 1.0e-4;
    source.materials = {built[0].program, built[1].program};

    ObjectSource object;
    object.name = "source_bound_per_element_damage_callable_check";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.mixedFEM = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.characteristicLength = 0.002;
    // Regional material maps require a stable authored source identity even
    // in a synthetic compiler-admission fixture.
    object.femMaterialSourceIdentity = {
        0x63617264626f6172ull, 0x645f64616d616765ull,
        0x6275696c6465725full, 0x6370755f76310000ull};
    const auto appendTet = [&](const ElementDamageRequest& request) {
        const std::uint32_t nodeBase =
            static_cast<std::uint32_t>(object.femNodes.size());
        for (const FractureVec3 p : request.referenceVerticesM)
            object.femNodes.push_back({p.x, p.y, p.z});
        object.tetrahedra.push_back(
            {nodeBase, nodeBase + 1u, nodeBase + 2u, nodeBase + 3u});
        object.femMaterialIndices.push_back(request.elementIndex);
    };
    const auto first = requestFor(0u, 0.001, {0.0, 0.0, 0.0});
    const auto second = requestFor(1u, 0.002, {0.01, 0.0, 0.0});
    appendTet(first);
    appendTet(second);
    for (std::uint32_t node = 0u;
         node < static_cast<std::uint32_t>(object.femNodes.size()); ++node)
        object.femFixedNodes.push_back(node);
    source.objects = {object};

    CompileOptions options;
    options.maximumRateExponent = 0;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const auto result = compileWorld(source, options);
    if (!result.succeeded()) {
        std::string messages;
        for (const auto& diagnostic : result.diagnostics)
            messages += diagnostic.message + "; ";
        throw std::runtime_error("per-element damage callable compile failed: " + messages);
    }
    return result.world;
}

void testActualElementAreasAndLengths() {
    auto firstRequest = requestFor(7u, 0.001, {0.0, 0.0, 0.0});
    auto secondRequest = requestFor(8u, 0.002, {0.01, 0.0, 0.0});
    const auto first = buildElementDamageProgram(firstRequest);
    const auto second = buildElementDamageProgram(secondRequest);

    require(first.mappingIdentity ==
                "Robertsson2023_tetra_selected_plane_crack_band_candidate" &&
            first.program.name.find("tet_selected_plane_candidate") != std::string::npos &&
            first.program.name.find("paper_absolute") != std::string::npos &&
            first.program.name.find("_hard_f3f35358_gc_b2d2a98a") != std::string::npos &&
            first.program.name.find("_tet7") == std::string::npos,
            "candidate program identity does not bind its selected-plane and energy conventions");
    require(first.program.internalState.size() == 27u &&
            first.program.stateImplicitRoots.size() == 27u,
            "candidate must preserve all 27 source states and implicit updates");

    for (std::size_t mode = 0; mode < 4u; ++mode) {
        const std::string suffix = "nu" + std::to_string(9u + mode);
        const double ratio1 = parameterValue(
            first.program, "area_ratio_" + suffix);
        const double ratio2 = parameterValue(
            second.program, "area_ratio_" + suffix);
        const double gc = first.fractureEnergyJPerM2[mode];
        const double length1 = first.modeLengthM[mode];
        const double length2 = second.modeLengthM[mode];
        const double softening1 = parameterValue(
            first.program, "fracture_energy_" + suffix + "_J_m2") / length1;
        const double softening2 = parameterValue(
            second.program, "fracture_energy_" + suffix + "_J_m2") / length2;
        require(near(ratio2 / ratio1, 4.0),
                "selected-plane area ratio did not follow the actual 4x area change");
        require(near(length2 / length1, 2.0),
                "per-element characteristic length did not follow geometric scale");
        require(near(softening1, gc / length1) && near(softening2, gc / length2),
                "candidate did not derive local softening work from Gc/(V/A)");
        require(near(first.selectedPlaneGeometry[mode].referenceVolumeM3,
                     first.referenceVolumeM3) &&
                    first.selectedPlaneGeometry[mode].referenceAreaM2 > 0.0,
                "selected mode did not retain its actual element cut geometry");
        require(hasParameter(first.program,
                    "initial_strength_nu" + std::to_string(mode + 1u) + "_MPa") &&
                hasParameter(first.program,
                    "hardening_amplitude_nu" + std::to_string(mode + 1u) + "_MPa") &&
                hasParameter(first.program,
                    "hardening_rate_nu" + std::to_string(mode + 1u)),
                "candidate callable omitted an explicit strength/hardening parameter");
    }
    require(parameterValue(first.program, "area_ratio_nu9") !=
                parameterValue(second.program, "area_ratio_nu9") &&
            parameterValue(first.program, "selected_plane_length_nu9_m") !=
                parameterValue(second.program, "selected_plane_length_nu9_m"),
            "per-element material programs shared the same selected-plane parameters");

    const auto source = numi_cardboard::robertsson2023::makePublishedReferenceMaterial(
        EnergyConvention::paper_absolute);
    const auto legacy = numi_cardboard::robertsson2023::makeLegacyRecenteredMaterial();
    require(source.name.find("paper_absolute") != std::string::npos &&
            source.name.find("tet_selected_plane_candidate") == std::string::npos,
            "published source path was mislabeled as the tetrahedral candidate");
    require(legacy.name.find("recentered") != std::string::npos,
            "historical regression lacks its explicit recentered identity");
    for (std::size_t mode = 0; mode < 4u; ++mode)
        require(near(parameterValue(source, "area_ratio_nu" +
                                     std::to_string(9u + mode)), 1.0),
                "published source-reference area was not explicit and exact");

    const auto built = buildMeshDamagePrograms({firstRequest, secondRequest});
    const auto world = compileTwoElementCallable(built);
    require(world.materials.size() == 2u &&
            world.materials[0].stateCount == 27u &&
            world.materials[1].stateCount == 27u &&
            world.dispatch.tetrahedronCount == 2u,
            "compiler did not emit distinct per-element 27-state callables");
    require(!world.instructions.empty() && !world.scalarPrograms.empty(),
            "candidate material compiled without callable instructions");

    auto exactDuplicate = requestFor(9u, 0.001, {0.0, 0.0, 0.0});
    const auto pool = buildMeshDamageProgramPool(
        {firstRequest, secondRequest, exactDuplicate});
    require(pool.materials.size() == 2u &&
                pool.materialIndices == std::vector<std::uint32_t>{0u, 1u, 0u},
            "pool failed to reuse exactly identical native FP32 programs");
    require(pool.elementMappings.size() == 3u &&
                pool.elementMappings[0].elementIndex == 7u &&
                pool.elementMappings[2].elementIndex == 9u &&
                pool.elementMappings[0].modeAreaMm2 ==
                    pool.elementMappings[2].modeAreaMm2 &&
                pool.elementMappings[0].fractureEnergyBinding.sha256 ==
                    pool.elementMappings[2].fractureEnergyBinding.sha256,
            "pooled material reuse dropped per-tet geometry or provenance");
}

void testFailClosedMissingInputs() {
    auto missing = requestFor(1u, 0.001, {0.0, 0.0, 0.0});
    missing.strengthHardening.reset();
    expectInvalid([&] { (void)buildElementDamageProgram(missing); },
                  "candidate admitted a missing 12-mode strength/hardening set");

    missing = requestFor(2u, 0.001, {0.0, 0.0, 0.0});
    missing.energyConvention.reset();
    expectInvalid([&] { (void)buildElementDamageProgram(missing); },
                  "candidate admitted an implicit energy convention");

    missing = requestFor(3u, 0.001, {0.0, 0.0, 0.0});
    missing.fractureEnergyJPerM2[2] = 0.0;
    expectInvalid([&] { (void)buildElementDamageProgram(missing); },
                  "candidate admitted an invented zero fracture energy");

    missing = requestFor(4u, 0.001, {0.0, 0.0, 0.0});
    missing.selectedPlanes[3].normal = {};
    expectInvalid([&] { (void)buildElementDamageProgram(missing); },
                  "candidate admitted a missing selected crack plane");

    missing = requestFor(5u, 0.001, {0.0, 0.0, 0.0});
    missing.fractureEnergyBinding->sha256 = "not-a-source-hash";
    expectInvalid([&] { (void)buildElementDamageProgram(missing); },
                  "candidate admitted an unbound fracture-energy set");

    auto first = requestFor(7u, 0.001, {0.0, 0.0, 0.0});
    auto duplicate = requestFor(7u, 0.002, {0.01, 0.0, 0.0});
    expectInvalid([&] { (void)buildMeshDamagePrograms({first, duplicate}); },
                  "mesh builder admitted duplicate element identities");
}

void testOwnGradeCompressionEnergyIdentity() {
    using Convention = numi_cardboard::robertsson2023::CompressionEnergyConvention;
    const auto absolute = numi_cardboard::robertsson2023::makePublishedReferenceMaterial(
        Convention::paper_absolute);
    const auto recentered = numi_cardboard::robertsson2023::makeLegacyRecenteredMaterial();
    State27 initial{};
    for (std::size_t i = 0; i < absolute.internalState.size(); ++i)
        initial[i] = absolute.internalState[i].initialValue;
    Matrix9 f{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const double psiAbsolute = evaluateExpressionNode(
        absolute, absolute.energyRoot, f, initial, initial);
    const double psiRecentered = evaluateExpressionNode(
        recentered, recentered.energyRoot, f, initial, initial);
    constexpr double expectedOffsetPa = -333260.8695652174;
    require(std::abs(psiAbsolute) < 1.0e-8 &&
                std::abs(psiRecentered) < 1.0e-8,
            "paper_absolute/recentered own-grade energy should be zero at the tension-side identity state");

    f[8] = 0.99;
    const double compressedAbsolute = evaluateExpressionNode(
        absolute, absolute.energyRoot, f, initial, initial);
    const double compressedRecentered = evaluateExpressionNode(
        recentered, recentered.energyRoot, f, initial, initial);
    require(std::abs((compressedAbsolute - compressedRecentered) - expectedOffsetPa) < 2.0e-6,
            "source conventions differ by more than the printed constant energy offset");
    std::cout << "source_energy_identity=paper_absolute:" << psiAbsolute
              << ",recentered:" << psiRecentered << " Pa\n"
              << "source_compression_offset_is_exposed_not_resolved=1\n";
}

} // namespace

int main() {
    try {
        testActualElementAreasAndLengths();
        testFailClosedMissingInputs();
        testOwnGradeCompressionEnergyIdentity();
        std::cout << "cardboard_damage_builder_cpu_passed\n"
                  << "callable_generation=two distinct per-element 27-state programs\n"
                  << "native_program_pool=exact_FP32_parameter_dedup_no_quantization\n"
                  << "candidate_law=Robertsson2023 architecture with tetra selected-plane Gc/l\n"
                  << "fixture_Gc_values_are_synthetic_test_inputs=1\n"
                  << "physical_calibration=not_claimed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
