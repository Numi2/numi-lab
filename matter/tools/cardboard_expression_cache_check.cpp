#include "numi/matter/detail.hpp"
#include "cardboard_expression_cache_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
using namespace numi_cardboard_expression_cache;
using Matrix = std::array<double, 9>;
unsigned checks = 0;

struct InvalidScalarDomain final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void require(const bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string out;
    for (const auto& value : values) out += value.message + "; ";
    return out;
}

double evaluate(
    const ScalarBytecode& bytecode,
    const MaterialProgram& material,
    const Matrix& f,
    const Matrix& h,
    std::array<double, NM_EXPRESSION_CACHE_CAPACITY>* sharedCache = nullptr
) {
    std::array<double, NM_EXPRESSION_CACHE_CAPACITY> localCache{};
    auto& cache = sharedCache != nullptr ? *sharedCache : localCache;
    std::vector<double> stack;
    const auto pop = [&]() {
        require(!stack.empty(), "compiled cache fixture stack underflow");
        const double value = stack.back();
        stack.pop_back();
        return value;
    };
    for (const auto& instruction : bytecode.instructions) {
        double value = 0.0;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER:
            value = material.parameters.at(instruction.index).defaultValue;
            break;
        case NM_EXPR_F: value = f.at(instruction.index); break;
        case NM_EXPR_DF: value = h.at(instruction.index); break;
        case NM_EXPR_CACHE_STORE:
            require(instruction.index < cache.size() && !stack.empty(),
                    "compiled cache store out of range or underflow");
            cache[instruction.index] = stack.back();
            continue;
        case NM_EXPR_CACHE_LOAD:
            require(instruction.index < cache.size(), "compiled cache load out of range");
            stack.push_back(cache[instruction.index]);
            continue;
        case NM_EXPR_ADD: { const double b = pop(); value = pop() + b; break; }
        case NM_EXPR_SUBTRACT: { const double b = pop(); value = pop() - b; break; }
        case NM_EXPR_MULTIPLY: { const double b = pop(); value = pop() * b; break; }
        case NM_EXPR_DIVIDE: {
            const double denominator = pop();
            const double numerator = pop();
            if (!(std::abs(denominator) > 1.0e-12))
                throw InvalidScalarDomain("division denominator is too small");
            value = numerator / denominator;
            break;
        }
        case NM_EXPR_NEGATE: value = -pop(); break;
        case NM_EXPR_SQRT: {
            const double argument = pop();
            if (argument < 0.0)
                throw InvalidScalarDomain("square-root argument is negative");
            value = std::sqrt(argument);
            break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const double negative = pop();
            const double nonnegative = pop();
            const double condition = pop();
            value = condition >= 0.0 ? nonnegative : negative;
            break;
        }
        default:
            throw std::runtime_error("unexpected opcode in cache fixture: " +
                std::to_string(instruction.opcode));
        }
        require(std::isfinite(value), "compiled cache fixture produced a nonfinite value");
        stack.push_back(value);
    }
    require(stack.size() == 1u, "compiled cache bytecode did not reduce to one value");
    return stack.back();
}

std::pair<std::uint32_t, std::uint32_t> cacheStoresAndLoads(
    const ScalarBytecode& program, std::vector<std::uint32_t>* storeSlots = nullptr,
    std::vector<std::uint32_t>* loadSlots = nullptr) {
    std::uint32_t stores = 0u, loads = 0u;
    for (const auto& instruction : program.instructions) {
        if (instruction.opcode == NM_EXPR_CACHE_STORE) {
            ++stores;
            if (storeSlots) storeSlots->push_back(instruction.index);
        }
        if (instruction.opcode == NM_EXPR_CACHE_LOAD) {
            ++loads;
            if (loadSlots) loadSlots->push_back(instruction.index);
        }
    }
    return {stores, loads};
}

CompileResult cook(const MaterialProgram& material) {
    auto source = makeWorld(material);
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    auto result = compileWorld(source, options);
    require(result.succeeded(), "cache fixture compile: " + diagnostics(result.diagnostics));
    std::string error;
    require(validateCompiledWorldLayout(result.world, &error),
        "valid cache fixture failed layout admission: " + error);
    return result;
}

void checkProgramResponses() {
    const auto cached = makeMaterial(true);
    const auto uncached = makeMaterial(false);
    const auto c = detail::compileConstitutive(cached, NM_EXPRESSION_STACK_CAPACITY);
    const auto u = detail::compileConstitutive(uncached, NM_EXPRESSION_STACK_CAPACITY);
    require(c.succeeded(), "cached constitutive compiler rejected DAG: " + diagnostics(c.diagnostics));
    require(u.succeeded(), "uncached constitutive compiler rejected DAG: " + diagnostics(u.diagnostics));

    std::vector<std::uint32_t> stores, loads;
    const auto counts = cacheStoresAndLoads(c.program.stress[0], &stores, &loads);
    require(counts.first >= 2u && counts.second >= 2u,
        "stress bytecode did not retain the repeated shared DAG");
    require(stores.size() >= 2u && loads.size() >= 2u,
        "shared DAG cache slot trace is incomplete");
    bool reusedSlot = false;
    for (std::size_t a = 0; a < stores.size(); ++a)
        for (std::size_t b = a + 1u; b < stores.size(); ++b)
            reusedSlot = reusedSlot || stores[a] == stores[b];
    require(reusedSlot, "disjoint shared-DAG lifetimes did not reuse a live cache slot");
    const auto branchCount = std::ranges::count_if(c.program.stress[0].instructions,
        [](const auto& i) { return i.opcode == NM_EXPR_SELECT_NONNEGATIVE; });
    require(branchCount >= 1, "stress bytecode omitted selectNonnegative");

    const Matrix h{0.17, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (const double sign : {-1.0, 1.0}) {
        Matrix f{1.0 + sign * 0.025, 0.13, 0.0, 0.0, 1.0, -0.07,
            0.0, 0.0, 1.0};
        const auto cP00 = evaluate(c.program.stress[0], cached, f, h);
        const auto uP00 = evaluate(u.program.stress[0], uncached, f, h);
        const auto cP01 = evaluate(c.program.stress[1], cached, f, h);
        const auto uP01 = evaluate(u.program.stress[1], uncached, f, h);
        const auto cDP00 = evaluate(c.program.tangentVector[0], cached, f, h);
        const auto uDP00 = evaluate(u.program.tangentVector[0], uncached, f, h);
        const auto cDP01 = evaluate(c.program.tangentVector[1], cached, f, h);
        const auto uDP01 = evaluate(u.program.tangentVector[1], uncached, f, h);
        const double k = sign > 0.0 ? 4.0e6 : 7.0e6;
        const double expectedP00 = k * (f[0] - 1.0);
        const double expectedP01 = 4.0 * 2.5e5 * (f[1] + 0.2);
        const double expectedDP00 = k * h[0];
        const double expectedDP01 = 4.0 * 2.5e5 * h[1];
        const auto close = [](double a, double b, double rel) {
            return std::abs(a - b) <= rel * std::max({1.0, std::abs(a), std::abs(b)});
        };
        require(close(cP00, uP00, 2e-6) && close(cP01, uP01, 2e-6) &&
                close(cDP00, uDP00, 2e-6) && close(cDP01, uDP01, 2e-6),
            "cached and uncached bytecode disagree on a signed branch");
        require(close(cP00, expectedP00, 2e-6) && close(cP01, expectedP01, 2e-6),
            "bytecode response chose the wrong active branch or DAG value");
        require(close(cDP00, expectedDP00, 2e-6) && close(cDP01, expectedDP01, 2e-6),
            "bytecode tangent chose the wrong active branch or DAG value");

        constexpr double epsilon = 1.0e-4;
        Matrix plus = f, minus = f;
        for (unsigned i = 0; i < 9u; ++i) {
            plus[i] += epsilon * h[i];
            minus[i] -= epsilon * h[i];
        }
        const double fd00 = (evaluate(c.program.stress[0], cached, plus, h) -
            evaluate(c.program.stress[0], cached, minus, h)) / (2.0 * epsilon);
        const double fd01 = (evaluate(c.program.stress[1], cached, plus, h) -
            evaluate(c.program.stress[1], cached, minus, h)) / (2.0 * epsilon);
        require(close(fd00, cDP00, 2e-5) && close(fd01, cDP01, 2e-5),
            "compiled branch tangent disagrees with centered stress differences");
    }
    std::cout << "cache_bytecode_shared_stress_stores=" << counts.first
              << " loads=" << counts.second << " reused_slot=true"
              << " select_nonnegative_positive_negative=true" << '\n';
}

void checkWholeTangentBundle() {
    const MaterialProgram cachedMaterial = makeBundleMaterial(true);
    const MaterialProgram uncachedMaterial = makeBundleMaterial(false);
    const auto cached = detail::compileConstitutive(
        cachedMaterial, NM_EXPRESSION_STACK_CAPACITY);
    const auto uncached = detail::compileConstitutive(
        uncachedMaterial, NM_EXPRESSION_STACK_CAPACITY);
    require(cached.succeeded(), "bundle material compile failed: " +
        diagnostics(cached.diagnostics));
    require(uncached.succeeded(), "uncached bundle reference compile failed: " +
        diagnostics(uncached.diagnostics));
    require(!cached.tangentBundle.programs.empty() &&
            cached.tangentBundle.preludeCount > 0u &&
            cached.tangentBundle.preludeCount <= NM_TANGENT_BUNDLE_CACHE_CAPACITY &&
            cached.tangentBundle.programs.size() ==
                cached.tangentBundle.preludeCount + NM_TANGENT_BUNDLE_OUTPUT_COUNT,
        "eligible scalar-CSE material did not compile an ordered tangent bundle");
    require(uncached.tangentBundle.programs.empty() &&
            (uncached.program.gpu.flags & NM_MATERIAL_HAS_TANGENT_BUNDLE) == 0u,
        "cacheExpressions=false material unexpectedly compiled a bundle");

    std::uint32_t preludeStoreCount = 0u;
    for (std::uint32_t prelude = 0u;
         prelude < cached.tangentBundle.preludeCount;
         ++prelude) {
        const ScalarBytecode& bytecode = cached.tangentBundle.programs[prelude];
        std::uint32_t stores = 0u;
        for (const auto& instruction : bytecode.instructions) {
            require(instruction.opcode != NM_EXPR_DF &&
                    instruction.opcode != NM_EXPR_STATE &&
                    instruction.opcode != NM_EXPR_NEXT_STATE,
                "bundle prelude depends on tangent direction or state");
            if (instruction.opcode == NM_EXPR_CACHE_STORE) {
                ++stores;
                require(instruction.index == prelude,
                    "bundle prelude slot assignment is not canonical");
            }
        }
        require(stores == 1u, "bundle prelude must publish exactly one value");
        preludeStoreCount += stores;
    }
    require(preludeStoreCount == cached.tangentBundle.preludeCount,
        "bundle prelude store census is incomplete");
    std::size_t legacyInstructionCount = 0u;
    std::size_t bundleInstructionCount = 0u;
    for (const ScalarBytecode& bytecode : cached.program.tangentVector)
        legacyInstructionCount += bytecode.instructions.size();
    for (const ScalarBytecode& bytecode : cached.tangentBundle.programs)
        bundleInstructionCount += bytecode.instructions.size();
    std::cout << "tangent_bundle_bytecode_instructions legacy="
              << legacyInstructionCount
              << " bundled=" << bundleInstructionCount
              << " prelude=" << cached.tangentBundle.preludeCount
              << " outputs=" << NM_TANGENT_BUNDLE_OUTPUT_COUNT << '\n';

    const auto runBundle = [&](const Matrix& f, const Matrix& h) {
        std::array<double, NM_EXPRESSION_CACHE_CAPACITY> shared{};
        for (std::uint32_t prelude = 0u;
             prelude < cached.tangentBundle.preludeCount;
             ++prelude) {
            (void)evaluate(cached.tangentBundle.programs[prelude],
                cachedMaterial, f, h, &shared);
        }
        std::array<double, NM_TANGENT_BUNDLE_OUTPUT_COUNT> result{};
        for (std::uint32_t component = 0u;
             component < NM_TANGENT_BUNDLE_OUTPUT_COUNT;
             ++component) {
            const ScalarBytecode& bytecode =
                cached.tangentBundle.programs[cached.tangentBundle.preludeCount +
                                              component];
            require(std::ranges::none_of(bytecode.instructions,
                [](const auto& instruction) {
                    return instruction.opcode == NM_EXPR_CACHE_STORE;
                }), "bundle tangent output writes a shared prelude slot");
            result[component] = evaluate(
                bytecode, cachedMaterial, f, h, &shared);
        }
        return result;
    };

    const std::array<Matrix, 5> deformationSamples{
        Matrix{0.91, 0.08, -0.03, -0.02, 1.07, 0.04, 0.01, -0.06, 0.98},
        Matrix{1.13, -0.11, 0.02, 0.07, 0.89, -0.04, -0.03, 0.05, 1.04},
        Matrix{0.77, 0.03, 0.09, -0.08, 1.21, 0.02, 0.05, -0.01, 0.93},
        Matrix{1.02, 0.17, -0.12, -0.09, 0.96, 0.08, 0.04, -0.06, 1.09},
        Matrix{0.88, -0.16, 0.06, 0.12, 1.14, -0.07, -0.02, 0.09, 0.81},
    };
    for (const Matrix& f : deformationSamples) {
        for (std::uint32_t column = 0u; column < 9u; ++column) {
            Matrix h{};
            h[column] = 1.0;
            const auto bundled = runBundle(f, h);
            for (std::uint32_t component = 0u; component < 9u; ++component) {
                const double reference = evaluate(
                    cached.program.tangentVector[component],
                    cachedMaterial, f, h);
                require(bundled[component] == reference,
                    "bundle basis action differs from the legacy scalar tangent bytecode");
            }
        }
        for (const Matrix& h : std::array<Matrix, 2>{
                Matrix{0.21, -0.37, 0.14, -0.09, 0.42, 0.31, 0.05, -0.18, 0.27},
                Matrix{-0.44, 0.07, 0.33, 0.16, -0.22, 0.51, -0.19, 0.29, -0.08}}) {
            const auto bundled = runBundle(f, h);
            for (std::uint32_t component = 0u; component < 9u; ++component) {
                const double reference = evaluate(
                    cached.program.tangentVector[component],
                    cachedMaterial, f, h);
                require(bundled[component] == reference,
                    "bundle random-direction action differs from scalar tangent bytecode");
            }
        }
    }

    const auto cooked = cook(cachedMaterial);
    const NMMaterialGPU& gpu = cooked.world.materials[0];
    require((gpu.flags & NM_MATERIAL_HAS_TANGENT_BUNDLE) != 0u,
        "compiled world dropped the tangent bundle availability tag");
    const std::uint32_t first = gpu.tangentProgramOffset +
        NM_TANGENT_BUNDLE_OUTPUT_COUNT;
    for (std::uint32_t local = 0u;
         local < cached.tangentBundle.programs.size();
         ++local) {
        const std::uint32_t expected =
            NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_CACHE |
            (local < cached.tangentBundle.preludeCount
                ? NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE : 0u);
        require(cooked.world.scalarPrograms[first + local].flags == expected,
            "compiled world bundle descriptor order/flags are not canonical");
    }
    auto invalidBundleLoad = cooked.world;
    bool changedBundleLoad = false;
    const std::uint32_t outputFirst = first + cached.tangentBundle.preludeCount;
    for (std::uint32_t local = 0u;
         local < NM_TANGENT_BUNDLE_OUTPUT_COUNT && !changedBundleLoad;
         ++local) {
        const NMScalarProgramGPU& output = invalidBundleLoad.scalarPrograms[
            outputFirst + local];
        for (std::uint32_t instruction = 0u;
             instruction < output.instructionCount;
             ++instruction) {
            NMExpressionInstructionGPU& item = invalidBundleLoad.instructions[
                output.firstInstruction + instruction];
            if (item.opcode == NM_EXPR_CACHE_LOAD) {
                item.index = NM_TANGENT_BUNDLE_CACHE_CAPACITY;
                changedBundleLoad = true;
                break;
            }
        }
    }
    require(changedBundleLoad, "compiled bundle has no external cache load to poison");
    invalidBundleLoad.fingerprint = compiledWorldFingerprint(invalidBundleLoad);
    std::string validationError;
    require(!validateCompiledWorldLayout(invalidBundleLoad, &validationError) &&
            validationError.find("tangent bundle cache load") != std::string::npos,
        "layout admission accepted an out-of-range tangent bundle slot");

    MaterialProgram stateful = makeBundleMaterial(true);
    InternalState history;
    history.name = "history";
    stateful.internalState.push_back(history);
    const auto statefulCompiled = detail::compileConstitutive(
        stateful, NM_EXPRESSION_STACK_CAPACITY);
    require(statefulCompiled.succeeded() &&
            statefulCompiled.tangentBundle.programs.empty() &&
            (statefulCompiled.program.gpu.flags &
             NM_MATERIAL_HAS_TANGENT_BUNDLE) == 0u,
        "stateful material escaped the stateless tangent-bundle eligibility gate");
    std::cout << "tangent_bundle_preludes="
              << cached.tangentBundle.preludeCount
              << " basis_directions=9 random_directions=2 deformations=5"
              << " exact_legacy_action=true stateful_fallback=true\n";
}

void checkEagerSelectionDomain() {
    const Matrix h{};
    const Matrix positiveF{1.025, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const Matrix negativeF{0.975, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    for (const bool cached : {false, true}) {
        const auto material = makeMaterial(cached, true);
        const auto compiled = detail::compileConstitutive(
            material, NM_EXPRESSION_STACK_CAPACITY);
        require(compiled.succeeded(), "eager-select domain fixture compile: " +
            diagnostics(compiled.diagnostics));
        const auto& stress = compiled.program.stress[0];
        const auto select = std::ranges::find_if(stress.instructions,
            [](const auto& instruction) {
                return instruction.opcode == NM_EXPR_SELECT_NONNEGATIVE;
            });
        require(select != stress.instructions.end(),
            "eager-select domain fixture omitted selection opcode");
        require(std::ranges::any_of(stress.instructions,
            [](const auto& instruction) { return instruction.opcode == NM_EXPR_SQRT; }),
            "invalid inactive square root was not emitted into the stress program");

        bool rejectedInactiveArm = false;
        try {
            (void)evaluate(stress, material, positiveF, h);
        } catch (const InvalidScalarDomain&) {
            rejectedInactiveArm = true;
        }
        require(rejectedInactiveArm,
            std::string(cached ? "cached" : "uncached") +
            " finite selected arm did not reject invalid inactive arm");

        const double negativeResponse = evaluate(stress, material, negativeF, h);
        const double expectedNegativeResponse = -7.0e6 /
            (2.0 * std::sqrt(0.025));
        require(std::abs(negativeResponse - expectedNegativeResponse) < 1.0e-8 *
                std::max(1.0, std::abs(expectedNegativeResponse)),
            std::string(cached ? "cached" : "uncached") +
            " eager select chose the wrong valid negative arm");
    }
    std::cout << "select_nonnegative=eager inactive_invalid_rejected_cached_uncached=true"
              << " selected_valid_arm_preserved=true\n";
}

void checkCachedPackageAndAdmission() {
    auto result = cook(makeMaterial(true));
    const auto& world = result.world;
    require(world.materials.size() == 1u && world.materials[0].stateCount == 0u,
        "cache material layout unexpectedly contains history");
    require(std::ranges::any_of(world.scalarPrograms, [](const auto& p) { return p.flags == 1u; }),
        "cooked package did not mark cached scalar programs");

    auto badLoad = world;
    bool changed = false;
    for (auto& instruction : badLoad.instructions) {
        if (instruction.opcode == NM_EXPR_CACHE_LOAD) {
            instruction.index = NM_EXPRESSION_CACHE_CAPACITY;
            changed = true;
            break;
        }
    }
    require(changed, "cached package has no load instruction for validator negative case");
    badLoad.fingerprint = compiledWorldFingerprint(badLoad);
    std::string error;
    require(!validateCompiledWorldLayout(badLoad, &error) &&
            error.find("expression cache load") != std::string::npos,
        "layout admission accepted an out-of-range expression-cache load");

    auto badBranch = world;
    changed = false;
    for (auto& instruction : badBranch.instructions) {
        if (instruction.opcode == NM_EXPR_CONSTANT) {
            instruction.opcode = NM_EXPR_SELECT_NONNEGATIVE;
            changed = true;
            break;
        }
    }
    require(changed, "cached package has no constant for select underflow negative case");
    badBranch.fingerprint = compiledWorldFingerprint(badBranch);
    error.clear();
    require(!validateCompiledWorldLayout(badBranch, &error) &&
            error.find("active branch selection underflows") != std::string::npos,
        "layout admission accepted a selectNonnegative stack underflow");

    const auto package = std::filesystem::temp_directory_path() /
        "numi-cardboard-expression-cache-roundtrip.nmatterpack";
    error.clear();
    require(writePackage(result, package, &error), "cached package write failed: " + error);
    CompiledWorld roundTrip;
    require(readPackage(package, roundTrip, nullptr, &error),
        "cached package read failed: " + error);
    std::error_code ignored;
    std::filesystem::remove(package, ignored);
    require(!ignored && roundTrip.fingerprint == world.fingerprint &&
            roundTrip.instructions.size() == world.instructions.size() &&
            roundTrip.scalarPrograms.size() == world.scalarPrograms.size(),
        "cached bytecode package roundtrip changed executable identity");
}

void checkState31Masks() {
    const auto sum = cook(makeState32(InternalState::Transfer::sum));
    const auto maximum = cook(makeState32(InternalState::Transfer::maximum));
    constexpr std::uint64_t highField = 3ull << 62u;
    require(sum.world.materials[0].stateCount == 32u &&
            maximum.world.materials[0].stateCount == 32u,
        "ABI41 state capacity did not compile exactly 32 states");
    require((sum.world.materials[0].stateTransferMask & highField) == (2ull << 62u),
        "state31 sum transfer code was truncated or shifted incorrectly");
    require((maximum.world.materials[0].stateTransferMask & highField) == (1ull << 62u),
        "state31 maximum transfer code was truncated or shifted incorrectly");
    require(sum.world.fingerprint != maximum.world.fingerprint,
        "cooked identity did not distinguish state31 sum and maximum transfer");

    const auto roundTrip = [](const CompileResult& source) {
        const auto path = std::filesystem::temp_directory_path() /
            ("numi-cardboard-state31-" + std::to_string(source.world.fingerprint) + ".nmatterpack");
        std::string error;
        require(writePackage(source, path, &error), "state31 package write failed: " + error);
        CompiledWorld loaded;
        const bool ok = readPackage(path, loaded, nullptr, &error);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        require(ok && !ignored, "state31 package read failed: " + error);
        return loaded;
    };
    const auto loadedSum = roundTrip(sum);
    const auto loadedMax = roundTrip(maximum);
    require(loadedSum.materials[0].stateCount == 32u &&
            loadedSum.materials[0].stateTransferMask == (2ull << 62u),
        "state31 sum field did not survive package serialization");
    require(loadedMax.materials[0].stateCount == 32u &&
            loadedMax.materials[0].stateTransferMask == (1ull << 62u),
        "state31 maximum field did not survive package serialization");

    auto invalid = sum.world;
    invalid.materials[0].stateTransferMask =
        (invalid.materials[0].stateTransferMask & ~highField) | (3ull << 62u);
    invalid.fingerprint = compiledWorldFingerprint(invalid);
    std::string error;
    require(!validateCompiledWorldLayout(invalid, &error) &&
            error.find("state transfer policy is invalid") != std::string::npos,
        "layout admission did not inspect invalid u64 state31 policy");
    std::cout << "abi=" << NM_MATTER_ABI_VERSION << " state_count=32"
              << " state31_sum=" << (loadedSum.materials[0].stateTransferMask >> 62u)
              << " state31_max=" << (loadedMax.materials[0].stateTransferMask >> 62u)
              << " package_roundtrip=true invalid_state31_rejected=true" << '\n';
}
} // namespace

int main() {
    try {
        checkProgramResponses();
        checkWholeTangentBundle();
        checkEagerSelectionDomain();
        checkCachedPackageAndAdmission();
        checkState31Masks();
        std::cout << "cardboard_expression_cache_cpu_checks=" << checks
                  << " native_physics_steps=0 result=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard expression cache CPU check failed: " << error.what() << '\n';
        return 1;
    }
}
