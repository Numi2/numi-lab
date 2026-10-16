#include "numi/matter/detail.hpp"
#include "numi/matter/regional_galerkin_bending_basis.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace numi::matter;
unsigned checks = 0u;

std::uint32_t decodedRegion(const std::uint32_t flags) {
    return (flags & NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
        NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT;
}

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string diagnostics(const CompileResult& result) {
    std::string output;
    for (const Diagnostic& diagnostic : result.diagnostics)
        output += diagnostic.message + "; ";
    return output;
}

WorldSource fixture() {
    const auto materialPath = std::filesystem::path(__FILE__).parent_path()
        .parent_path() / "materials" / "damageable_silicone.nmatter";
    const auto parsed = parseMatterFile(materialPath);
    if (!parsed.succeeded())
        throw std::runtime_error("fixture material did not parse");

    WorldSource source;
    source.materials.push_back(parsed.material);
    source.gravity = {0.0, 0.0, 0.0};
    source.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::regionalGalerkin;

    ObjectSource object;
    object.name = "two-region-preconditioner-compiler-check";
    object.materialIndex = 0u;
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {
        {{0.0, 0.0, 0.0}}, {{0.01, 0.0, 0.0}},
        {{0.0, 0.01, 0.0}}, {{0.0, 0.0, 0.01}},
        {{0.0, 0.0, -0.01}},
    };
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}, {{0u, 2u, 1u, 4u}}};
    object.femPreconditionerRegionIds = {0u, 1u};
    source.objects.push_back(std::move(object));
    return source;
}

WorldSource contactFixture() {
    WorldSource source = fixture();
    // The regional-reference compiler path intentionally excludes parameter
    // identification. This fixture exercises source/layout admission only.
    for (Parameter& parameter : source.materials.front().parameters)
        parameter.identifiable = false;
    source.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::regionalGalerkinContact;
    ObjectSource& object = source.objects.front();
    object.name = "contact-regional-preconditioner-compiler-check";
    object.deformableContact = true;
    object.deformableSelfContact = true;
    object.femNodes.push_back(object.femNodes[0]);
    object.femNodes.push_back(object.femNodes[2]);
    object.femNodes.push_back(object.femNodes[1]);
    object.tetrahedra[1] = {{5u, 6u, 7u, 4u}};
    object.femPreconditionerRegionIds = {0u, 1u};
    object.femMaterialIndices = {0u, 0u};
    object.femMaterialSourceIdentity = {5u, 6u, 7u, 8u};
    object.femReferenceNodes = object.femNodes;
    object.femReferenceSourceIdentity = {1u, 2u, 3u, 4u};
    object.femReferenceDisplacementGradient = true;

    CohesiveMaterialSource cohesive;
    cohesive.name = "fixture-only-explicit-mixed-cohesive-law";
    cohesive.normalPenaltyPaPerM = 1.0e9;
    cohesive.shearPenaltyPaPerM = 1.0e9;
    cohesive.compressionPenaltyPaPerM = 2.0e9;
    cohesive.normalStrengthPa = 1.0e6;
    cohesive.shearStrengthPa = 1.0e6;
    cohesive.fractureEnergyModeIJPerM2 = 1000.0;
    cohesive.fractureEnergyModeIIJPerM2 = 1500.0;
    cohesive.bkExponent = 1.5;
    source.cohesiveMaterials.push_back(cohesive);

    CohesiveMortarPatchSource patch;
    patch.sideANodes = {0u, 2u, 1u};
    patch.sideBNodes = {5u, 6u, 7u};
    patch.sideAVertexBarycentric = {{{1.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0},
                                      {0.0, 0.0, 1.0}}};
    patch.sideBVertexBarycentric = patch.sideAVertexBarycentric;
    patch.firstTetrahedron = 0u;
    patch.secondTetrahedron = 1u;
    patch.materialIndex = 0u;
    patch.stableIdentifier = 77u;
    patch.cohesiveMaterialIndex = 0u;
    patch.activationGapToleranceM = 1.0e-6;
    object.femCohesiveMortarPatches.push_back(patch);

    RigidProxySource proxy;
    proxy.shape = NM_RIGID_PLANE;
    proxy.materialIndex = 0u;
    proxy.localCenter = {0.0, 0.0, 1.0};
    source.rigidProxies.push_back(proxy);
    return source;
}

void verifyAdditiveCoarseOperatorOracle() {
    // Independent dense algebra check of the defining selector-4 action:
    // bulk, contact, and cohesive directional blocks must all contribute to
    // A before forming P^T A P. This is a portable algebra oracle, not a
    // substitute for the native kernel/operator parity test.
    constexpr std::size_t n = 4u;
    constexpr std::size_t m = 2u;
    using Matrix = std::array<std::array<double, n>, n>;
    const Matrix bulk{{{{4.0, -1.0, 0.0, 0.0}},
                       {{-1.0, 3.0, -0.5, 0.0}},
                       {{0.0, -0.5, 2.0, -0.25}},
                       {{0.0, 0.0, -0.25, 1.0}}}};
    const Matrix contact{{{{1.5, -1.5, 0.0, 0.0}},
                          {{-1.5, 1.5, 0.0, 0.0}},
                          {{0.0, 0.0, 0.0, 0.0}},
                          {{0.0, 0.0, 0.0, 0.0}}}};
    const Matrix cohesive{{{{0.0, 0.0, 0.0, 0.0}},
                           {{0.0, 2.0, -2.0, 0.0}},
                           {{0.0, -2.0, 2.0, 0.0}},
                           {{0.0, 0.0, 0.0, 0.0}}}};
    const std::array<std::array<double, m>, n> P{{
        {{1.0, 0.0}}, {{0.5, 0.5}}, {{0.0, 1.0}}, {{-0.25, 0.25}},
    }};
    const auto multiply = [&](const Matrix& A) {
        std::array<std::array<double, m>, n> result{};
        for (std::size_t row = 0u; row < n; ++row)
            for (std::size_t col = 0u; col < m; ++col)
                for (std::size_t inner = 0u; inner < n; ++inner)
                    result[row][col] += A[row][inner] * P[inner][col];
        return result;
    };
    Matrix full{};
    for (std::size_t i = 0u; i < n; ++i)
        for (std::size_t j = 0u; j < n; ++j)
            full[i][j] = bulk[i][j] + contact[i][j] + cohesive[i][j];
    const auto AP = multiply(full);
    const auto bulkAP = multiply(bulk);
    const auto contactAP = multiply(contact);
    const auto cohesiveAP = multiply(cohesive);
    for (std::size_t i = 0u; i < n; ++i)
        for (std::size_t j = 0u; j < m; ++j)
            require(std::abs(AP[i][j] - bulkAP[i][j] - contactAP[i][j] -
                             cohesiveAP[i][j]) < 1.0e-14,
                "additive AP assembly omitted or double-counted an operator block");
    std::array<std::array<double, m>, m> rap{};
    for (std::size_t row = 0u; row < m; ++row)
        for (std::size_t col = 0u; col < m; ++col)
            for (std::size_t node = 0u; node < n; ++node)
                rap[row][col] += P[node][row] * AP[node][col];
    require(std::abs(rap[0][0] - rap[0][1]) > 1.0e-6,
        "coarse product did not retain the distinct contact/cohesive contributions");
}

void denied(const WorldSource& source, const char* label) {
    const CompileResult result = compileWorld(source);
    require(!result.succeeded(), std::string("invalid source admitted: ") + label);
    std::cout << "rejected=" << label << " diagnostic=" << diagnostics(result) << '\n';
}

} // namespace

int main() {
    try {
        const WorldSource source = fixture();
        const CompileResult compiled = compileWorld(source);
        require(compiled.succeeded(), "valid two-region source failed: " +
            diagnostics(compiled));
        const CompiledWorld& world = compiled.world;
        require(world.fem.tetrahedra.size() == 2u,
            "compiler changed authored tet count");
        require(decodedRegion(world.fem.tetrahedra[0].identity.w) == 0u &&
                decodedRegion(world.fem.tetrahedra[1].identity.w) == 1u,
            "compiled tet region IDs do not preserve source order");
        std::string layoutError;
        require(validateCompiledWorldLayout(world, &layoutError),
            "regional compiled layout rejected: " + layoutError);

        WorldSource cachedTangentSource = source;
        cachedTangentSource.mixedSolver.cacheFEMAlgorithmicTangent = true;
        const CompileResult cachedTangentCompiled =
            compileWorld(cachedTangentSource);
        require(cachedTangentCompiled.succeeded(),
            "valid pure-FEM tangent-cache source failed: " +
                diagnostics(cachedTangentCompiled));
        const CompiledWorld& cachedTangentWorld = cachedTangentCompiled.world;
        require(cachedTangentWorld.mixedSolver.executionBudgets.w == 1u &&
                cachedTangentWorld.fingerprint != world.fingerprint &&
                cachedTangentWorld.physicsFingerprint != world.physicsFingerprint,
            "tangent-cache policy was not cooked and fingerprint-bound");
        layoutError.clear();
        require(validateCompiledWorldLayout(cachedTangentWorld, &layoutError),
            "tangent-cache compiled layout rejected: " + layoutError);
        WorldSource fourEnvironmentTangentSource = cachedTangentSource;
        fourEnvironmentTangentSource.environmentCount = 4u;
        const CompileResult fourEnvironmentTangent =
            compileWorld(fourEnvironmentTangentSource);
        require(fourEnvironmentTangent.succeeded(),
            "tangent-cache four-environment boundary was rejected: " +
                diagnostics(fourEnvironmentTangent));

        WorldSource blockSource = source;
        blockSource.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::regionalGalerkinBlockJacobi;
        const CompileResult blockCompiled = compileWorld(blockSource);
        require(blockCompiled.succeeded(), "valid selector-3 source failed: " +
            diagnostics(blockCompiled));
        const CompiledWorld& blockWorld = blockCompiled.world;
        require(blockWorld.mixedSolver.executionBudgets.z == 3u &&
                blockWorld.fem.tetrahedra.size() == world.fem.tetrahedra.size() &&
                decodedRegion(blockWorld.fem.tetrahedra[0].identity.w) == 0u &&
                decodedRegion(blockWorld.fem.tetrahedra[1].identity.w) == 1u,
            "selector 3 did not preserve the same authored regions and tet order");
        require(blockWorld.fingerprint != world.fingerprint &&
                blockWorld.physicsFingerprint != world.physicsFingerprint,
            "selector 3 was not bound into compiled world identity");
        layoutError.clear();
        require(validateCompiledWorldLayout(blockWorld, &layoutError),
            "selector-3 compiled layout rejected: " + layoutError);

        const WorldSource contactSource = contactFixture();
        const CompileResult contactCompiled = compileWorld(contactSource);
        require(contactCompiled.succeeded(),
            "valid selector-4 contact/cohesive source failed: " +
                diagnostics(contactCompiled));
        const CompiledWorld& contactWorld = contactCompiled.world;
        require(contactWorld.mixedSolver.executionBudgets.z == 4u &&
                contactWorld.dispatch.environmentCount == 1u &&
                contactWorld.dispatch.femNodeCount == 8u &&
                contactWorld.dispatch.rigidGeneralizedCapacity == 0u &&
                contactWorld.dispatch.rigidProxyCount == 1u &&
                contactWorld.dispatch.contactPairCount > 0u &&
                contactWorld.dispatch.cohesiveFaceCount == 1u &&
                contactWorld.fem.tetrahedra.size() == 2u &&
                decodedRegion(contactWorld.fem.tetrahedra[0].identity.w) == 0u &&
                decodedRegion(contactWorld.fem.tetrahedra[1].identity.w) == 1u,
            "selector 4 did not preserve its bounded coupled source layout");
        require((contactWorld.contact.rigidProxies.front().flags &
                 (NM_RIGID_DYNAMIC | NM_RIGID_ARTICULATED |
                  NM_RIGID_SUTURE_STRAND)) == 0u,
            "selector 4 admitted a generalized or live rigid proxy");
        require((contactWorld.fem.cohesiveFaces.front().adjacency.w &
                 NM_TOPOLOGY_COHESIVE_MORTAR) != 0u,
            "selector 4 fixture did not cook its dormant mixed-mode mortar face");
        layoutError.clear();
        require(validateCompiledWorldLayout(contactWorld, &layoutError),
            "selector-4 compiled layout rejected: " + layoutError);
        require(contactWorld.fingerprint != blockWorld.fingerprint &&
                contactWorld.physicsFingerprint != blockWorld.physicsFingerprint,
            "selector 4 was not bound into compiled world identity");
        verifyAdditiveCoarseOperatorOracle();

        const auto contactPackagePath =
            std::filesystem::temp_directory_path() /
            ("numi-regional-contact-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".nmpkg");
        std::string contactPackageError;
        require(writePackage(contactCompiled, contactPackagePath,
                             &contactPackageError),
            "selector-4 package write failed: " + contactPackageError);
        CompiledWorld decodedContact;
        require(readPackage(contactPackagePath, decodedContact, nullptr,
                            &contactPackageError),
            "selector-4 package read failed: " + contactPackageError);
        std::filesystem::remove(contactPackagePath);
        require(decodedContact.fingerprint == contactWorld.fingerprint &&
                decodedContact.physicsFingerprint == contactWorld.physicsFingerprint &&
                decodedContact.mixedSolver.executionBudgets.z == 4u &&
                decodedContact.dispatch.contactPairCount ==
                    contactWorld.dispatch.contactPairCount &&
                decodedContact.fem.cohesiveFaces.size() == 1u &&
                decodedContact.fem.tetrahedra.size() == 2u,
            "selector-4 package round-trip changed contact/cohesive operator identity");

        WorldSource changedSource = source;
        changedSource.objects.front().femPreconditionerRegionIds = {1u, 0u};
        const CompileResult changed = compileWorld(changedSource);
        require(changed.succeeded(), "permuted regional IDs failed to compile");
        require(changed.world.fingerprint != world.fingerprint &&
                changed.world.physicsFingerprint != world.physicsFingerprint,
            "regional labels were not bound to both compiled fingerprints");

        const auto packagePath = std::filesystem::temp_directory_path() /
            ("numi-regional-galerkin-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".nmpkg");
        std::string packageError;
        require(writePackage(compiled, packagePath, &packageError),
            "regional package write failed: " + packageError);
        CompiledWorld decoded;
        require(readPackage(packagePath, decoded, nullptr, &packageError),
            "regional package read failed: " + packageError);
        std::filesystem::remove(packagePath);
        require(decoded.fingerprint == world.fingerprint &&
                decoded.physicsFingerprint == world.physicsFingerprint,
            "package round-trip changed regional world identity");
        require(decoded.mixedSolver.executionBudgets.z == 2u &&
                decoded.fem.tetrahedra.size() == world.fem.tetrahedra.size() &&
                decodedRegion(decoded.fem.tetrahedra[0].identity.w) == 0u &&
                decodedRegion(decoded.fem.tetrahedra[1].identity.w) == 1u,
            "package round-trip lost regional selector or tet labels");

        const auto cachedTangentPackagePath =
            std::filesystem::temp_directory_path() /
            ("numi-fem-tangent-cache-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".nmpkg");
        packageError.clear();
        require(writePackage(cachedTangentCompiled, cachedTangentPackagePath,
                             &packageError),
            "tangent-cache package write failed: " + packageError);
        CompiledWorld decodedCachedTangent;
        require(readPackage(cachedTangentPackagePath, decodedCachedTangent,
                            nullptr, &packageError),
            "tangent-cache package read failed: " + packageError);
        std::filesystem::remove(cachedTangentPackagePath);
        require(decodedCachedTangent.fingerprint == cachedTangentWorld.fingerprint &&
                decodedCachedTangent.physicsFingerprint ==
                    cachedTangentWorld.physicsFingerprint &&
                decodedCachedTangent.mixedSolver.executionBudgets.w == 1u,
            "package round-trip lost the tangent-cache policy identity");

        const auto blockPackagePath = std::filesystem::temp_directory_path() /
            ("numi-regional-block-jacobi-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".nmpkg");
        packageError.clear();
        require(writePackage(blockCompiled, blockPackagePath, &packageError),
            "selector-3 package write failed: " + packageError);
        CompiledWorld decodedBlock;
        require(readPackage(blockPackagePath, decodedBlock, nullptr,
                            &packageError),
            "selector-3 package read failed: " + packageError);
        std::filesystem::remove(blockPackagePath);
        require(decodedBlock.fingerprint == blockWorld.fingerprint &&
                decodedBlock.physicsFingerprint == blockWorld.physicsFingerprint &&
                decodedBlock.mixedSolver.executionBudgets.z == 3u &&
                decodedBlock.fem.tetrahedra.size() ==
                    blockWorld.fem.tetrahedra.size() &&
                decodedRegion(decodedBlock.fem.tetrahedra[0].identity.w) == 0u &&
                decodedRegion(decodedBlock.fem.tetrahedra[1].identity.w) == 1u,
            "selector-3 package round-trip changed its mode or source regions");

        WorldSource invalid = source;
        invalid.objects.front().femPreconditionerRegionIds.clear();
        denied(invalid, "missing-labels");
        invalid = source;
        invalid.objects.front().femPreconditionerRegionIds = {0u};
        denied(invalid, "label-count");
        invalid = source;
        invalid.objects.front().femPreconditionerRegionIds = {0u, 2u};
        denied(invalid, "noncompact-labels");
        invalid = source;
        invalid.objects.front().femPreconditionerRegionIds = {0u, 8u};
        denied(invalid, "out-of-range-label");
        invalid = source;
        invalid.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::scalarDiagonal;
        denied(invalid, "labels-without-selector");
        invalid = source;
        invalid.objects.front().deformableSelfContact = true;
        denied(invalid, "self-contact-scope");
        invalid = source;
        invalid.environmentCount = 5u;
        denied(invalid, "environment-bound");
        invalid = cachedTangentSource;
        invalid.environmentCount = 5u;
        denied(invalid, "tangent-cache-environment-bound");
        invalid = cachedTangentSource;
        invalid.objects.front().mixedFEM = true;
        denied(invalid, "tangent-cache-mixed-fem-scope");
        invalid = contactSource;
        invalid.mixedSolver.cacheFEMAlgorithmicTangent = true;
        denied(invalid, "tangent-cache-cohesive-contact-scope");
        invalid = blockSource;
        invalid.objects.front().femPreconditionerRegionIds.clear();
        denied(invalid, "selector-3-missing-labels");
        invalid = blockSource;
        invalid.objects.front().deformableContact = true;
        denied(invalid, "selector-3-contact-scope");

        invalid = contactSource;
        invalid.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::regionalGalerkin;
        denied(invalid, "selector-2-contact-cohesive-scope");
        invalid = contactSource;
        invalid.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::regionalGalerkinBlockJacobi;
        denied(invalid, "selector-3-contact-cohesive-scope");
        invalid = contactSource;
        invalid.rigidProxies.front().dynamic = true;
        invalid.rigidProxies.front().bodyIndex = 4u;
        invalid.rigidProxies.front().sceneBodyIndex = 0u;
        denied(invalid, "selector-4-dynamic-rigid-scope");
        invalid = contactSource;
        invalid.environmentCount = 2u;
        denied(invalid, "selector-4-environment-scope");
        invalid = contactSource;
        for (std::uint32_t i = 0u; i < 8185u; ++i) {
            invalid.objects.front().femNodes.push_back({
                {0.02 + 1.0e-6 * i, 0.03, 0.04}});
            invalid.objects.front().femReferenceNodes.push_back({
                {0.02 + 1.0e-6 * i, 0.03, 0.04}});
        }
        denied(invalid, "selector-4-node-bound-8193");

        CompiledWorld corrupt = world;
        corrupt.fem.tetrahedra[1].identity.w =
            (corrupt.fem.tetrahedra[1].identity.w &
             ~NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) |
            (2u << NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT);
        corrupt.fingerprint = compiledWorldFingerprint(corrupt);
        layoutError.clear();
        require(!validateCompiledWorldLayout(corrupt, &layoutError),
            "cooked noncompact labels were admitted");

        // Coarse trial fields cancel transverse shear before constraint
        // masking. Verify this independently using coordinate differences;
        // no material or solver acceptance is inferred from these checks.
        const auto bending = regionalBendingCoordinates(
            {-0.04, -0.01, -0.002}, {0.04, 0.01, 0.002});
        require(bending.available && bending.normalAxis == 2u &&
                bending.planeAxes == std::array<unsigned, 2>{0u, 1u},
            "plate trial coordinate ordering is incorrect");
        const std::array<double, 3> point{0.006, 0.002, 0.0006};
        constexpr double h = 1.0e-7;
        for (unsigned mode = 0u; mode < 8u; ++mode) {
            for (unsigned axis = 0u; axis < 2u; ++axis) {
                auto a = point, b = point, c = point, d = point;
                a[axis] += h; b[axis] -= h;
                c[2] += h; d[2] -= h;
                const auto va = regionalBendingTrialField(bending, a, mode);
                const auto vb = regionalBendingTrialField(bending, b, mode);
                const auto vc = regionalBendingTrialField(bending, c, mode);
                const auto vd = regionalBendingTrialField(bending, d, mode);
                const double shear = (va[2] - vb[2] + vc[axis] - vd[axis]) /
                    (2.0 * h);
                require(std::abs(shear) < 1.0e-7,
                    "bending coarse trial contains spurious transverse shear");
            }
        }
        const auto permuted = regionalBendingCoordinates(
            {-0.002, -0.04, -0.01}, {0.002, 0.04, 0.01});
        require(permuted.available && permuted.normalAxis == 0u &&
                permuted.planeAxes == std::array<unsigned, 2>{1u, 2u},
            "bending coarse trial does not respect coordinate permutation");
        require(!regionalBendingCoordinates({0.0, 0.0, 0.0},
                                            {0.01, 0.0, 0.0}).available,
            "degenerate plane was admitted for bending enrichment");

        std::cout << "PASS fem_regional_galerkin_compiler checks=" << checks
                  << " physical_steps=0 selectors=2,3,4 regions=2 "
                     "selector4_contact_cohesive=1 tangent_cache=1 "
                     "package_roundtrip=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL fem_regional_galerkin_compiler checks=" << checks
                  << " error=" << error.what() << '\n';
        return 1;
    }
}
