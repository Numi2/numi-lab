#include "cardboard_retained_damage_fp64_oracle.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace numi::matter;
using namespace numi::cardboard::retained_damage::fp64;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

NMExpressionInstructionGPU instruction(std::uint32_t opcode,
                                       std::uint32_t index = 0u,
                                       float immediate = 0.0f) {
    NMExpressionInstructionGPU result{};
    result.opcode = opcode;
    result.index = index;
    result.immediate.x = immediate;
    return result;
}

ScalarBytecode code(std::initializer_list<NMExpressionInstructionGPU> instructions,
                    std::uint32_t maximumStack) {
    ScalarBytecode result;
    result.instructions.assign(instructions.begin(), instructions.end());
    result.maximumStack = maximumStack;
    return result;
}

void checkLinearImplicitSystem() {
    ConstitutiveProgram program;
    InternalState state;
    state.name = "z";
    state.initialValue = 0.0;
    program.material.internalState.push_back(state);

    program.stateUpdates.push_back(code({instruction(NM_EXPR_CONSTANT, 0u, 0.0f)}, 1u));
    program.implicitResiduals.push_back(code({
        instruction(NM_EXPR_NEXT_STATE, 0u),
        instruction(NM_EXPR_F, 0u),
        instruction(NM_EXPR_SUBTRACT)}, 2u));
    program.implicitJacobians.push_back(code({
        instruction(NM_EXPR_CONSTANT, 0u, 1.0f)}, 1u));
    program.implicitDeformationDirections.push_back(code({
        instruction(NM_EXPR_CONSTANT, 0u, -1.0f)}, 1u));

    program.stress[0] = code({
        instruction(NM_EXPR_CONSTANT, 0u, 2.0f),
        instruction(NM_EXPR_F, 0u),
        instruction(NM_EXPR_MULTIPLY),
        instruction(NM_EXPR_NEXT_STATE, 0u),
        instruction(NM_EXPR_ADD)}, 2u);
    for (std::size_t i = 1u; i < program.stress.size(); ++i)
        program.stress[i] = code({instruction(NM_EXPR_CONSTANT, 0u, 0.0f)}, 1u);
    program.tangentVector[0] = code({
        instruction(NM_EXPR_CONSTANT, 0u, 2.0f)}, 1u);
    for (std::size_t i = 1u; i < program.tangentVector.size(); ++i)
        program.tangentVector[i] = code({instruction(NM_EXPR_CONSTANT, 0u, 0.0f)}, 1u);
    program.stressStateDerivatives.push_back(code({
        instruction(NM_EXPR_CONSTANT, 0u, 1.0f)}, 1u));
    for (std::size_t i = 1u; i < 9u; ++i)
        program.stressStateDerivatives.push_back(code({
            instruction(NM_EXPR_CONSTANT, 0u, 0.0f)}, 1u));

    Matrix f{};
    Matrix direction{};
    State accepted{0.0};
    f[0] = 1.25;
    direction[0] = 1.0;
    const auto projected = project(program, f, accepted, 0.002);
    const auto p = stress(program, f, accepted, projected.state, 0.002);
    const auto tangent = tangentAction(program, f, accepted,
        projected.state, direction, 0.002);
    require(projected.iterations == 1u && projected.residualInf < 1.0e-14,
            "FP64 implicit projection did not solve the exact linear fixture");
    require(std::abs(projected.state[0] - 1.25) < 1.0e-14,
            "FP64 projection returned the wrong history state");
    require(std::abs(p[0] - 3.75) < 1.0e-14,
            "FP64 source stress evaluator returned the wrong stress");
    require(std::abs(tangent[0] - 3.0) < 1.0e-14,
            "FP64 implicit consistent tangent omitted its state contribution");
}

void checkFloatWordReconstruction() {
    Matrix high{};
    Matrix low{};
    high[0] = static_cast<float>(1.0038);
    low[0] = static_cast<float>(1.0038 - static_cast<double>(high[0]));
    high[4] = 1.0f;
    low[4] = std::ldexp(1.0f, -25);
    const auto combined = combineFloatWords(high, low);
    const double expected0 = static_cast<double>(static_cast<float>(high[0])) +
                             static_cast<double>(static_cast<float>(low[0]));
    const double expected4 = 1.0 + std::ldexp(1.0, -25);
    require(combined[0] == expected0 && combined[4] == expected4,
            "FP64 oracle did not reconstruct both binary32 deformation words");
    require(combined[4] != static_cast<double>(static_cast<float>(high[4])),
            "FP64 oracle silently discarded the deformation low word");
}

void checkCompensatedPerturbationSweep() {
    Matrix center{{1.0038, 0.12, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0}};
    Matrix direction{{1.0, 0.12, 0.0,
                      0.0, 0.15, 0.0,
                      0.0, 0.0, 0.0}};
    Matrix high{};
    Matrix low{};
    for (std::size_t i = 0; i < center.size(); ++i) {
        high[i] = static_cast<float>(center[i]);
        low[i] = static_cast<float>(center[i] - high[i]);
    }
    const Matrix base = combineFloatWords(high, low);
    const std::array<double, 8> epsilons{{1.0e-3, 3.0e-4, 1.0e-4,
        3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7}};
    for (const double epsilon : epsilons) {
        Matrix plusHigh{}, plusLow{}, minusHigh{}, minusLow{};
        for (std::size_t i = 0; i < center.size(); ++i) {
            const auto plus = perturbFloatWords(static_cast<float>(high[i]),
                static_cast<float>(low[i]), static_cast<float>(direction[i]),
                epsilon, 1.0);
            const auto minus = perturbFloatWords(static_cast<float>(high[i]),
                static_cast<float>(low[i]), static_cast<float>(direction[i]),
                epsilon, -1.0);
            plusHigh[i] = plus.high;
            plusLow[i] = plus.low;
            minusHigh[i] = minus.high;
            minusLow[i] = minus.low;
        }
        const Matrix plus = combineFloatWords(plusHigh, plusLow);
        const Matrix minus = combineFloatWords(minusHigh, minusLow);
        double realizedDifference2 = 0.0;
        double directionNorm2 = 0.0;
        double midpointError = 0.0;
        for (std::size_t i = 0; i < center.size(); ++i) {
            const double realized = (plus[i] - minus[i]) / (2.0 * epsilon);
            const double requested = static_cast<float>(direction[i]);
            realizedDifference2 += (realized - requested) * (realized - requested);
            directionNorm2 += requested * requested;
            midpointError = std::max(midpointError,
                std::abs(0.5 * (plus[i] + minus[i]) - base[i]));
            const double expectedPlus = base[i] + epsilon * requested;
            const double expectedMinus = base[i] - epsilon * requested;
            require(std::abs(plus[i] - expectedPlus) < 2.0e-15 &&
                    std::abs(minus[i] - expectedMinus) < 2.0e-15,
                    "two-word perturbation does not replay the requested FP64 pair");
        }
        const double realizedRelative = std::sqrt(realizedDifference2) /
            std::sqrt(directionNorm2);
        require(realizedRelative <= 1.0e-3,
                "two-word perturbation pair does not resolve the requested direction");
        require(midpointError < 2.0e-15,
                "symmetric two-word perturbation shifted its center");
    }
}
} // namespace

int main() {
    try {
        checkLinearImplicitSystem();
        checkFloatWordReconstruction();
        checkCompensatedPerturbationSweep();
        std::cout << "cardboard_retained_damage_fp64_oracle_check: CPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
