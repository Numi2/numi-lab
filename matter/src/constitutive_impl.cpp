#include "numi/matter/detail.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace numi::matter::detail {
namespace {

[[nodiscard]] nm_float4 float4(
    const float x = 0.0f,
    const float y = 0.0f,
    const float z = 0.0f,
    const float w = 0.0f
) noexcept {
    return {x, y, z, w};
}

[[nodiscard]] bool constantValue(
    const ExpressionGraph& graph,
    const std::uint32_t node,
    double& value
) {
    if (node == NM_INVALID_INDEX || node >= graph.nodes.size() ||
        graph.nodes[node].kind != ExprKind::constant) {
        return false;
    }
    value = graph.nodes[node].constant;
    return true;
}

class Symbolic {
public:
    explicit Symbolic(ExpressionGraph& graph) : graph_(graph) {}

    [[nodiscard]] std::uint32_t zero(const Dimension dimension = {}) {
        return graph_.constant(0.0, dimension);
    }

    [[nodiscard]] std::uint32_t one(const Dimension dimension = {}) {
        return graph_.constant(1.0, dimension);
    }

    [[nodiscard]] std::uint32_t unary(
        const ExprKind kind,
        const std::uint32_t argument,
        const Dimension dimension
    ) {
        double value = 0.0;
        if (constantValue(graph_, argument, value)) {
            switch (kind) {
            case ExprKind::negate: return graph_.constant(-value, dimension);
            case ExprKind::logarithm: return graph_.constant(std::log(value), dimension);
            case ExprKind::exponential: return graph_.constant(std::exp(value), dimension);
            case ExprKind::squareRoot: return graph_.constant(std::sqrt(value), dimension);
            case ExprKind::absolute: return graph_.constant(std::abs(value), dimension);
            default: break;
            }
        }
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = argument;
        return graph_.append(expression);
    }

    [[nodiscard]] std::uint32_t binary(
        const ExprKind kind,
        const std::uint32_t left,
        const std::uint32_t right,
        const Dimension dimension
    ) {
        double a = 0.0;
        double b = 0.0;
        const bool aConstant = constantValue(graph_, left, a);
        const bool bConstant = constantValue(graph_, right, b);
        if (aConstant && bConstant) {
            switch (kind) {
            case ExprKind::add: return graph_.constant(a + b, dimension);
            case ExprKind::subtract: return graph_.constant(a - b, dimension);
            case ExprKind::multiply: return graph_.constant(a * b, dimension);
            case ExprKind::divide: return graph_.constant(a / b, dimension);
            case ExprKind::minimum: return graph_.constant(std::min(a, b), dimension);
            case ExprKind::maximum: return graph_.constant(std::max(a, b), dimension);
            default: break;
            }
        }
        if (kind == ExprKind::add) {
            if (aConstant && a == 0.0) return right;
            if (bConstant && b == 0.0) return left;
        }
        if (kind == ExprKind::subtract && bConstant && b == 0.0) return left;
        if (kind == ExprKind::multiply) {
            if ((aConstant && a == 0.0) || (bConstant && b == 0.0)) return zero(dimension);
            if (aConstant && a == 1.0) return right;
            if (bConstant && b == 1.0) return left;
        }
        if (kind == ExprKind::divide) {
            if (aConstant && a == 0.0) return zero(dimension);
            if (bConstant && b == 1.0) return left;
        }
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = left;
        expression.arguments[1] = right;
        return graph_.append(expression);
    }

    [[nodiscard]] std::uint32_t power(
        const std::uint32_t argument,
        const int exponent,
        const Dimension dimension
    ) {
        double value = 0.0;
        if (constantValue(graph_, argument, value)) {
            return graph_.constant(std::pow(value, exponent), dimension);
        }
        if (exponent == 0) return one(dimension);
        if (exponent == 1) return argument;
        Expr expression;
        expression.kind = ExprKind::integerPower;
        expression.dimension = dimension;
        expression.integer = exponent;
        expression.arguments[0] = argument;
        return graph_.append(expression);
    }

    enum class VariableDomain : std::uint8_t {
        deformation = 0u,
        rate = 1u,
        candidateState = 2u,
    };

    [[nodiscard]] Dimension variableDimension(
        const VariableDomain domain,
        const std::uint32_t variable
    ) const noexcept {
        if (domain == VariableDomain::deformation) return kDimensionless;
        if (domain == VariableDomain::rate) return kRate;
        for (const Expr& expression : graph_.nodes) {
            if (expression.kind == ExprKind::candidateState &&
                expression.index == variable) {
                return expression.dimension;
            }
        }
        return {};
    }

    [[nodiscard]] std::uint32_t derivative(
        const std::uint32_t node,
        const std::uint32_t variable,
        const VariableDomain domain,
        std::vector<Diagnostic>& diagnostics
    ) {
        const std::uint64_t key =
            (static_cast<std::uint64_t>(domain) << 60u) |
            (static_cast<std::uint64_t>(variable) << 32u) |
            node;
        if (const auto iterator = derivatives_.find(key);
            iterator != derivatives_.end()) {
            return iterator->second;
        }
        if (node >= graph_.nodes.size()) {
            return NM_INVALID_INDEX;
        }
        const Expr source = graph_.nodes[node];
        const Dimension variableDim = variableDimension(domain, variable);
        const Dimension resultDimension = source.dimension - variableDim;
        const auto d = [&](const std::uint32_t argument) {
            return derivative(argument, variable, domain, diagnostics);
        };
        std::uint32_t result = NM_INVALID_INDEX;
        switch (source.kind) {
        case ExprKind::constant:
        case ExprKind::parameter:
        case ExprKind::internalState:
        case ExprKind::deformationDirection:
        case ExprKind::timeStep:
        case ExprKind::temperature:
            result = zero(resultDimension);
            break;
        case ExprKind::candidateState:
            result = graph_.constant(
                domain == VariableDomain::candidateState &&
                    source.index == variable ? 1.0 : 0.0,
                resultDimension
            );
            break;
        case ExprKind::deformation:
            result = graph_.constant(
                domain == VariableDomain::deformation &&
                    source.index == variable
                    ? 1.0
                    : 0.0,
                resultDimension
            );
            break;
        case ExprKind::deformationRate:
            result = graph_.constant(
                domain == VariableDomain::rate &&
                    source.index == variable
                    ? 1.0
                    : 0.0,
                resultDimension
            );
            break;
        case ExprKind::add:
        case ExprKind::subtract:
            result = binary(
                source.kind,
                d(source.arguments[0]),
                d(source.arguments[1]),
                resultDimension
            );
            break;
        case ExprKind::multiply: {
            const std::uint32_t left = binary(
                ExprKind::multiply,
                d(source.arguments[0]),
                source.arguments[1],
                resultDimension
            );
            const std::uint32_t right = binary(
                ExprKind::multiply,
                source.arguments[0],
                d(source.arguments[1]),
                resultDimension
            );
            result = binary(
                ExprKind::add,
                left,
                right,
                resultDimension
            );
            break;
        }
        case ExprKind::divide: {
            const Dimension numeratorDimension =
                source.dimension +
                graph_.nodes[source.arguments[1]].dimension * 2 -
                variableDim;
            const std::uint32_t first = binary(
                ExprKind::multiply,
                d(source.arguments[0]),
                source.arguments[1],
                numeratorDimension
            );
            const std::uint32_t second = binary(
                ExprKind::multiply,
                source.arguments[0],
                d(source.arguments[1]),
                numeratorDimension
            );
            const std::uint32_t numerator = binary(
                ExprKind::subtract,
                first,
                second,
                numeratorDimension
            );
            const std::uint32_t denominator = power(
                source.arguments[1],
                2,
                graph_.nodes[source.arguments[1]].dimension * 2
            );
            result = binary(
                ExprKind::divide,
                numerator,
                denominator,
                resultDimension
            );
            break;
        }
        case ExprKind::negate:
            result = unary(
                ExprKind::negate,
                d(source.arguments[0]),
                resultDimension
            );
            break;
        case ExprKind::logarithm:
            result = binary(
                ExprKind::divide,
                d(source.arguments[0]),
                source.arguments[0],
                resultDimension
            );
            break;
        case ExprKind::exponential:
            result = binary(
                ExprKind::multiply,
                node,
                d(source.arguments[0]),
                resultDimension
            );
            break;
        case ExprKind::squareRoot: {
            const std::uint32_t denominator = binary(
                ExprKind::multiply,
                graph_.constant(2.0),
                node,
                graph_.nodes[source.arguments[0]].dimension
            );
            result = binary(
                ExprKind::divide,
                d(source.arguments[0]),
                denominator,
                resultDimension
            );
            break;
        }
        case ExprKind::integerPower: {
            if (source.integer == 0) {
                result = zero(resultDimension);
            } else {
                const Dimension baseDimension =
                    graph_.nodes[source.arguments[0]].dimension;
                const std::uint32_t factor = graph_.constant(
                    static_cast<double>(source.integer)
                );
                const std::uint32_t reduced = power(
                    source.arguments[0],
                    source.integer - 1,
                    baseDimension * (source.integer - 1)
                );
                result = binary(
                    ExprKind::multiply,
                    binary(
                        ExprKind::multiply,
                        factor,
                        reduced,
                        baseDimension * (source.integer - 1)
                    ),
                    d(source.arguments[0]),
                    resultDimension
                );
            }
            break;
        }
        case ExprKind::absolute:
        case ExprKind::minimum:
        case ExprKind::maximum:
        case ExprKind::clamp:
            diagnostics.push_back({
                Diagnostic::Severity::error,
                0u,
                0u,
                domain == VariableDomain::deformation
                    ? "nonsmooth abs/min/max/clamp is not permitted in stored energy; use it in validity or state evolution"
                    : "nonsmooth abs/min/max/clamp is not permitted in the dissipation potential",
            });
            result = zero(resultDimension);
            break;
        }
        derivatives_.emplace(key, result);
        return result;
    }

    [[nodiscard]] std::uint32_t directional(
        const std::uint32_t node,
        const VariableDomain domain,
        std::vector<Diagnostic>& diagnostics
    ) {
        const std::uint64_t key =
            (static_cast<std::uint64_t>(domain) << 32u) | node;
        if (const auto iterator = directional_.find(key);
            iterator != directional_.end()) {
            return iterator->second;
        }
        if (node >= graph_.nodes.size()) {
            return NM_INVALID_INDEX;
        }
        const Expr source = graph_.nodes[node];
        const auto d = [&](const std::uint32_t argument) {
            return directional(argument, domain, diagnostics);
        };
        std::uint32_t result = NM_INVALID_INDEX;
        switch (source.kind) {
        case ExprKind::constant:
        case ExprKind::parameter:
        case ExprKind::internalState:
        case ExprKind::candidateState:
        case ExprKind::timeStep:
        case ExprKind::temperature:
            result = zero(source.dimension);
            break;
        case ExprKind::deformation:
        case ExprKind::deformationRate: {
            const bool selected =
                (source.kind == ExprKind::deformation &&
                 domain == VariableDomain::deformation) ||
                (source.kind == ExprKind::deformationRate &&
                 domain == VariableDomain::rate);
            if (!selected) {
                result = zero(source.dimension);
                break;
            }
            Expr direction;
            direction.kind = ExprKind::deformationDirection;
            direction.dimension = source.dimension;
            direction.index = source.index;
            result = graph_.append(direction);
            break;
        }
        case ExprKind::deformationDirection:
            result = zero(source.dimension);
            break;
        case ExprKind::add:
        case ExprKind::subtract:
            result = binary(
                source.kind,
                d(source.arguments[0]),
                d(source.arguments[1]),
                source.dimension
            );
            break;
        case ExprKind::multiply: {
            const std::uint32_t left = binary(
                ExprKind::multiply,
                d(source.arguments[0]),
                source.arguments[1],
                source.dimension
            );
            const std::uint32_t right = binary(
                ExprKind::multiply,
                source.arguments[0],
                d(source.arguments[1]),
                source.dimension
            );
            result = binary(
                ExprKind::add,
                left,
                right,
                source.dimension
            );
            break;
        }
        case ExprKind::divide: {
            const Dimension numeratorDimension = source.dimension +
                graph_.nodes[source.arguments[1]].dimension * 2;
            const std::uint32_t first = binary(
                ExprKind::multiply,
                d(source.arguments[0]),
                source.arguments[1],
                numeratorDimension
            );
            const std::uint32_t second = binary(
                ExprKind::multiply,
                source.arguments[0],
                d(source.arguments[1]),
                numeratorDimension
            );
            const std::uint32_t numerator = binary(
                ExprKind::subtract,
                first,
                second,
                numeratorDimension
            );
            const std::uint32_t denominator = power(
                source.arguments[1],
                2,
                graph_.nodes[source.arguments[1]].dimension * 2
            );
            result = binary(
                ExprKind::divide,
                numerator,
                denominator,
                source.dimension
            );
            break;
        }
        case ExprKind::negate:
            result = unary(
                ExprKind::negate,
                d(source.arguments[0]),
                source.dimension
            );
            break;
        case ExprKind::logarithm:
            result = binary(
                ExprKind::divide,
                d(source.arguments[0]),
                source.arguments[0],
                source.dimension
            );
            break;
        case ExprKind::exponential:
            result = binary(
                ExprKind::multiply,
                node,
                d(source.arguments[0]),
                source.dimension
            );
            break;
        case ExprKind::squareRoot: {
            const std::uint32_t denominator = binary(
                ExprKind::multiply,
                graph_.constant(2.0),
                node,
                graph_.nodes[source.arguments[0]].dimension
            );
            result = binary(
                ExprKind::divide,
                d(source.arguments[0]),
                denominator,
                source.dimension
            );
            break;
        }
        case ExprKind::integerPower: {
            const Dimension baseDimension =
                graph_.nodes[source.arguments[0]].dimension;
            const std::uint32_t factor = graph_.constant(
                static_cast<double>(source.integer)
            );
            const std::uint32_t reduced = power(
                source.arguments[0],
                source.integer - 1,
                baseDimension * (source.integer - 1)
            );
            result = binary(
                ExprKind::multiply,
                binary(
                    ExprKind::multiply,
                    factor,
                    reduced,
                    baseDimension * (source.integer - 1)
                ),
                d(source.arguments[0]),
                source.dimension
            );
            break;
        }
        case ExprKind::absolute:
        case ExprKind::minimum:
        case ExprKind::maximum:
        case ExprKind::clamp:
            diagnostics.push_back({
                Diagnostic::Severity::error,
                0u,
                0u,
                "nonsmooth expression reached a constitutive tangent",
            });
            result = zero(source.dimension);
            break;
        }
        directional_.emplace(key, result);
        return result;
    }

private:
    ExpressionGraph& graph_;
    std::unordered_map<std::uint64_t, std::uint32_t> derivatives_;
    std::unordered_map<std::uint64_t, std::uint32_t> directional_;
};

class BytecodeCompiler {
public:
    BytecodeCompiler(const ExpressionGraph& graph, const std::uint32_t maximumStack)
        : graph_(graph), maximumAllowed_(maximumStack) {}

    [[nodiscard]] ScalarBytecode compile(
        const std::uint32_t root,
        std::vector<Diagnostic>& diagnostics
    ) {
        ScalarBytecode output;
        stack_ = 0u;
        maximum_ = 0u;
        emit(root, output.instructions, diagnostics);
        output.maximumStack = maximum_;
        if (stack_ != 1u) {
            diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "constitutive bytecode did not finish with one scalar on the stack",
            });
        }
        if (maximum_ > maximumAllowed_) {
            diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "constitutive expression requires " + std::to_string(maximum_) +
                " stack values; compiled capacity is " + std::to_string(maximumAllowed_),
            });
        }
        return output;
    }

private:
    void push() {
        ++stack_;
        maximum_ = std::max(maximum_, stack_);
    }

    void binaryReduction() {
        if (stack_ >= 2u) {
            --stack_;
        }
    }

    void emit(
        const std::uint32_t node,
        std::vector<NMExpressionInstructionGPU>& output,
        std::vector<Diagnostic>& diagnostics
    ) {
        if (node == NM_INVALID_INDEX || node >= graph_.nodes.size()) {
            diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "invalid expression root during bytecode emission",
            });
            return;
        }
        const Expr& expression = graph_.nodes[node];
        NMExpressionInstructionGPU instruction{};
        instruction.index = expression.index;
        instruction.integer = expression.integer;
        instruction.immediate = float4(static_cast<float>(expression.constant));
        const auto child = [&](const std::size_t index) {
            emit(expression.arguments[index], output, diagnostics);
        };
        switch (expression.kind) {
        case ExprKind::constant:
            instruction.opcode = NM_EXPR_CONSTANT;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::parameter:
            instruction.opcode = NM_EXPR_PARAMETER;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::deformation:
            instruction.opcode = NM_EXPR_F;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::deformationDirection:
            instruction.opcode = NM_EXPR_DF;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::deformationRate:
            instruction.opcode = NM_EXPR_RATE;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::timeStep:
            instruction.opcode = NM_EXPR_DT;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::temperature:
            instruction.opcode = NM_EXPR_TEMPERATURE;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::internalState:
            instruction.opcode = NM_EXPR_STATE;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::candidateState:
            instruction.opcode = NM_EXPR_NEXT_STATE;
            output.push_back(instruction);
            push();
            break;
        case ExprKind::add:
        case ExprKind::subtract:
        case ExprKind::multiply:
        case ExprKind::divide:
        case ExprKind::minimum:
        case ExprKind::maximum:
            child(0u);
            child(1u);
            instruction.opcode = expression.kind == ExprKind::add ? NM_EXPR_ADD :
                expression.kind == ExprKind::subtract ? NM_EXPR_SUBTRACT :
                expression.kind == ExprKind::multiply ? NM_EXPR_MULTIPLY :
                expression.kind == ExprKind::divide ? NM_EXPR_DIVIDE :
                expression.kind == ExprKind::minimum ? NM_EXPR_MIN : NM_EXPR_MAX;
            output.push_back(instruction);
            binaryReduction();
            break;
        case ExprKind::negate:
        case ExprKind::logarithm:
        case ExprKind::exponential:
        case ExprKind::squareRoot:
        case ExprKind::absolute:
        case ExprKind::integerPower:
            child(0u);
            instruction.opcode = expression.kind == ExprKind::negate ? NM_EXPR_NEGATE :
                expression.kind == ExprKind::logarithm ? NM_EXPR_LOG :
                expression.kind == ExprKind::exponential ? NM_EXPR_EXP :
                expression.kind == ExprKind::squareRoot ? NM_EXPR_SQRT :
                expression.kind == ExprKind::absolute ? NM_EXPR_ABS : NM_EXPR_POW_INTEGER;
            output.push_back(instruction);
            break;
        case ExprKind::clamp:
            child(0u);
            child(1u);
            child(2u);
            instruction.opcode = NM_EXPR_CLAMP;
            output.push_back(instruction);
            if (stack_ >= 3u) stack_ -= 2u;
            break;
        }
    }

    const ExpressionGraph& graph_;
    std::uint32_t maximumAllowed_ = 0u;
    std::uint32_t stack_ = 0u;
    std::uint32_t maximum_ = 0u;
};

[[nodiscard]] std::optional<std::uint32_t> parameterIndex(
    const MaterialProgram& material,
    const std::string_view name
) {
    for (std::uint32_t index = 0u; index < material.parameters.size(); ++index) {
        if (material.parameters[index].name == name) return index;
    }
    return std::nullopt;
}

[[nodiscard]] std::uint32_t constitutiveKind(const ConstitutiveHint hint) {
    switch (hint) {
    case ConstitutiveHint::neoHookean: return NM_CONSTITUTIVE_NEO_HOOKEAN;
    case ConstitutiveHint::corotated: return NM_CONSTITUTIVE_COROTATED;
    case ConstitutiveHint::druckerPrager: return NM_CONSTITUTIVE_DRUCKER_PRAGER;
    case ConstitutiveHint::vonMises: return NM_CONSTITUTIVE_VON_MISES;
    case ConstitutiveHint::viscoHyperelastic: return NM_CONSTITUTIVE_VISCO_HYPERELASTIC;
    case ConstitutiveHint::polyconvexICNN: return NM_CONSTITUTIVE_POLYCONVEX_ICNN;
    case ConstitutiveHint::generic: return NM_CONSTITUTIVE_BYTECODE;
    }
    return NM_CONSTITUTIVE_BYTECODE;
}

} // namespace

bool ConstitutiveCompileResult::succeeded() const noexcept {
    return std::none_of(
        diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& diagnostic) {
            return diagnostic.severity == Diagnostic::Severity::error;
        }
    );
}

std::uint64_t hashBytes(
    const void* data,
    const std::size_t size,
    std::uint64_t seed
) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        seed ^= bytes[index];
        seed *= 1099511628211ull;
    }
    return seed == 0u ? 1u : seed;
}

std::uint64_t hashString(
    const std::string_view value,
    const std::uint64_t seed
) noexcept {
    return hashBytes(value.data(), value.size(), seed);
}

ConstitutiveCompileResult compileConstitutive(
    const MaterialProgram& material,
    const std::uint32_t maximumStack
) {
    ConstitutiveCompileResult result;
    result.program.material = material;
    if (material.energyRoot == NM_INVALID_INDEX ||
        material.energyRoot >= material.expressions.nodes.size()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "material has no valid stored-energy root",
        });
        return result;
    }
    if (material.internalState.size() > NM_MAX_MATERIAL_STATE) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "material state count exceeds the fixed GPU state capacity",
        });
        return result;
    }
    if (material.stateUpdateRoots.size() > material.internalState.size()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "material contains more state updates than declared states",
        });
        return result;
    }
    if (material.stateImplicitRoots.size() > material.internalState.size()) {
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "material contains more implicit residuals than declared states",
        });
        return result;
    }

    Symbolic symbolic(result.program.material.expressions);
    std::array<std::uint32_t, 9> stressRoots{};
    std::array<std::uint32_t, 9> tangentRoots{};
    std::array<std::uint32_t, 9> viscousStressRoots{};
    std::array<std::uint32_t, 9> viscousTangentRoots{};
    for (std::uint32_t component = 0u; component < 9u; ++component) {
        stressRoots[component] = symbolic.derivative(
            result.program.material.energyRoot,
            component,
            Symbolic::VariableDomain::deformation,
            result.diagnostics
        );
        tangentRoots[component] = symbolic.directional(
            stressRoots[component],
            Symbolic::VariableDomain::deformation,
            result.diagnostics
        );
        if (result.program.material.dissipationRoot != NM_INVALID_INDEX) {
            viscousStressRoots[component] = symbolic.derivative(
                result.program.material.dissipationRoot,
                component,
                Symbolic::VariableDomain::rate,
                result.diagnostics
            );
            viscousTangentRoots[component] = symbolic.directional(
                viscousStressRoots[component],
                Symbolic::VariableDomain::rate,
                result.diagnostics
            );
        } else {
            viscousStressRoots[component] = NM_INVALID_INDEX;
            viscousTangentRoots[component] = NM_INVALID_INDEX;
        }
    }

    const std::uint32_t stateCount =
        static_cast<std::uint32_t>(material.internalState.size());
    const bool hasImplicit = std::ranges::any_of(
        material.stateImplicitRoots,
        [](const std::uint32_t root) { return root != NM_INVALID_INDEX; }
    );
    std::vector<std::uint32_t> updateRoots(stateCount, NM_INVALID_INDEX);
    std::vector<std::uint32_t> residualRoots;
    std::vector<std::uint32_t> jacobianRoots;
    std::vector<std::uint32_t> residualDirectionRoots;
    std::vector<std::uint32_t> stressStateRoots;
    residualRoots.reserve(stateCount);
    jacobianRoots.reserve(static_cast<std::size_t>(stateCount) * stateCount);
    residualDirectionRoots.reserve(stateCount);
    stressStateRoots.reserve(static_cast<std::size_t>(9u) * stateCount);
    for (std::uint32_t state = 0u; state < stateCount; ++state) {
        std::uint32_t update = state < material.stateUpdateRoots.size()
            ? material.stateUpdateRoots[state] : NM_INVALID_INDEX;
        if (update == NM_INVALID_INDEX) {
            Expr identity;
            identity.kind = ExprKind::internalState;
            identity.dimension = material.internalState[state].dimension;
            identity.index = state;
            update = result.program.material.expressions.append(identity);
        }
        updateRoots[state] = update;

        if (!hasImplicit) continue;
        std::uint32_t residual = state < material.stateImplicitRoots.size()
            ? material.stateImplicitRoots[state] : NM_INVALID_INDEX;
        if (residual == NM_INVALID_INDEX) {
            Expr candidate;
            candidate.kind = ExprKind::candidateState;
            candidate.dimension = material.internalState[state].dimension;
            candidate.index = state;
            const std::uint32_t candidateRoot =
                result.program.material.expressions.append(candidate);
            residual = symbolic.binary(
                ExprKind::subtract,
                candidateRoot,
                update,
                material.internalState[state].dimension
            );
        }
        residualRoots.push_back(residual);
        residualDirectionRoots.push_back(symbolic.directional(
            residual, Symbolic::VariableDomain::deformation,
            result.diagnostics
        ));
        for (std::uint32_t candidate = 0u; candidate < stateCount; ++candidate) {
            jacobianRoots.push_back(symbolic.derivative(
                residual, candidate, Symbolic::VariableDomain::candidateState,
                result.diagnostics
            ));
        }
    }
    if (hasImplicit) {
        for (std::uint32_t component = 0u; component < 9u; ++component) {
            for (std::uint32_t state = 0u; state < stateCount; ++state) {
                stressStateRoots.push_back(symbolic.derivative(
                    stressRoots[component], state,
                    Symbolic::VariableDomain::candidateState,
                    result.diagnostics
                ));
            }
        }
    }

    BytecodeCompiler compiler(
        result.program.material.expressions,
        maximumStack
    );
    for (std::uint32_t component = 0u; component < 9u; ++component) {
        result.program.stress[component] = compiler.compile(
            stressRoots[component],
            result.diagnostics
        );
        result.program.tangentVector[component] = compiler.compile(
            tangentRoots[component],
            result.diagnostics
        );
        if (viscousStressRoots[component] != NM_INVALID_INDEX) {
            result.program.viscousStress[component] = compiler.compile(
                viscousStressRoots[component],
                result.diagnostics
            );
            result.program.viscousTangentVector[component] = compiler.compile(
                viscousTangentRoots[component],
                result.diagnostics
            );
        }
    }

    result.program.stateUpdates.reserve(material.internalState.size());
    for (std::uint32_t state = 0u;
         state < material.internalState.size();
         ++state) {
        result.program.stateUpdates.push_back(
            compiler.compile(updateRoots[state], result.diagnostics)
        );
    }
    if (hasImplicit) {
        for (const std::uint32_t root : residualRoots) {
            result.program.implicitResiduals.push_back(
                compiler.compile(root, result.diagnostics));
        }
        for (const std::uint32_t root : jacobianRoots) {
            result.program.implicitJacobians.push_back(
                compiler.compile(root, result.diagnostics));
        }
        for (const std::uint32_t root : residualDirectionRoots) {
            result.program.implicitDeformationDirections.push_back(
                compiler.compile(root, result.diagnostics));
        }
        for (const std::uint32_t root : stressStateRoots) {
            result.program.stressStateDerivatives.push_back(
                compiler.compile(root, result.diagnostics));
        }
    }
    if (result.program.material.dissipationRoot != NM_INVALID_INDEX) {
        result.program.dissipation = compiler.compile(
            result.program.material.dissipationRoot,
            result.diagnostics
        );
    }
    if (result.program.material.validityRoot != NM_INVALID_INDEX) {
        result.program.validity = compiler.compile(
            result.program.material.validityRoot,
            result.diagnostics
        );
    }

    NMMaterialGPU gpu{};
    gpu.constitutiveKind = constitutiveKind(material.hint);
    gpu.parameterCount = static_cast<nm_u32>(material.parameters.size());
    gpu.stateCount = static_cast<nm_u32>(material.internalState.size());
    gpu.flags =
        (!material.internalState.empty() ? NM_MATERIAL_HAS_STATE : 0u) |
        (material.dissipationRoot != NM_INVALID_INDEX
             ? NM_MATERIAL_HAS_DISSIPATION
             : 0u) |
        (hasImplicit ? NM_MATERIAL_HAS_IMPLICIT_STATE : 0u);
    gpu.stateInitialOffset = NM_INVALID_INDEX;
    gpu.stateUpdateProgramOffset = NM_INVALID_INDEX;
    gpu.dissipationProgram = NM_INVALID_INDEX;
    gpu.projectionKind = material.hint == ConstitutiveHint::vonMises
        ? NM_MATERIAL_PROJECTION_VON_MISES
        : material.hint == ConstitutiveHint::druckerPrager
            ? NM_MATERIAL_PROJECTION_DRUCKER_PRAGER
            : NM_MATERIAL_PROJECTION_GENERIC;
    gpu.viscousStressProgramOffset = NM_INVALID_INDEX;
    gpu.viscousTangentProgramOffset = NM_INVALID_INDEX;
    gpu.implicitResidualProgramOffset = NM_INVALID_INDEX;
    gpu.implicitJacobianProgramOffset = NM_INVALID_INDEX;
    gpu.implicitDeformationProgramOffset = NM_INVALID_INDEX;
    gpu.stressStateDerivativeProgramOffset = NM_INVALID_INDEX;
    gpu.localNewtonIterations = 8u;
    gpu.stateTransferMask = 0u;
    for (std::uint32_t state = 0u; state < material.internalState.size(); ++state) {
        const auto transfer = material.internalState[state].transfer;
        const std::uint32_t encoded = transfer == InternalState::Transfer::maximum
            ? NM_MATERIAL_TRANSFER_MAXIMUM
            : transfer == InternalState::Transfer::sum
                ? NM_MATERIAL_TRANSFER_SUM
                : NM_MATERIAL_TRANSFER_AVERAGE;
        gpu.stateTransferMask |= encoded << (2u * state);
    }
    const auto density = parameterIndex(material, "density");
    gpu.bulk = float4(
        density.has_value() ? static_cast<float>(material.parameters[*density].defaultValue) : 0.0f,
        293.15f,
        0.0f,
        0.0f
    );
    gpu.interfaceResponse = float4(
        static_cast<float>(material.staticFriction),
        static_cast<float>(material.dynamicFriction),
        static_cast<float>(material.restitution),
        static_cast<float>(material.adhesion)
    );
    gpu.validity = float4(
        static_cast<float>(material.minimumDeterminant),
        static_cast<float>(material.maximumDeterminant),
        static_cast<float>(material.maximumStress),
        static_cast<float>(material.maximumEnergyDensity)
    );
    result.program.gpu = gpu;

    result.program.parameters.reserve(material.parameters.size());
    for (const Parameter& parameter : material.parameters) {
        NMParameterRangeGPU range{};
        range.valueAndBounds = float4(
            static_cast<float>(parameter.defaultValue),
            static_cast<float>(parameter.lower),
            static_cast<float>(parameter.upper),
            parameter.logarithmic ? 1.0f : 0.0f
        );
        result.program.parameters.push_back(range);
    }

    std::uint64_t fingerprint = hashString(material.name, material.fingerprint);
    for (const auto& bytecode : result.program.stress) {
        fingerprint = hashBytes(bytecode.instructions.data(),
            bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU), fingerprint);
    }
    for (const auto& bytecode : result.program.tangentVector) {
        fingerprint = hashBytes(bytecode.instructions.data(),
            bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU), fingerprint);
    }
    for (const auto& bytecode : result.program.viscousStress) {
        fingerprint = hashBytes(
            bytecode.instructions.data(),
            bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU),
            fingerprint
        );
    }
    for (const auto& bytecode : result.program.viscousTangentVector) {
        fingerprint = hashBytes(
            bytecode.instructions.data(),
            bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU),
            fingerprint
        );
    }
    for (const auto& bytecode : result.program.stateUpdates) {
        fingerprint = hashBytes(
            bytecode.instructions.data(),
            bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU),
            fingerprint
        );
    }
    const auto hashPrograms = [&](const std::vector<ScalarBytecode>& programs) {
        for (const auto& bytecode : programs) {
            fingerprint = hashBytes(
                bytecode.instructions.data(),
                bytecode.instructions.size() * sizeof(NMExpressionInstructionGPU),
                fingerprint
            );
        }
    };
    hashPrograms(result.program.implicitResiduals);
    hashPrograms(result.program.implicitJacobians);
    hashPrograms(result.program.implicitDeformationDirections);
    hashPrograms(result.program.stressStateDerivatives);
    fingerprint = hashBytes(
        &result.program.gpu.stateTransferMask,
        sizeof(result.program.gpu.stateTransferMask),
        fingerprint
    );
    if (result.program.dissipation.has_value()) {
        fingerprint = hashBytes(
            result.program.dissipation->instructions.data(),
            result.program.dissipation->instructions.size() *
                sizeof(NMExpressionInstructionGPU),
            fingerprint
        );
    }
    if (result.program.validity.has_value()) {
        fingerprint = hashBytes(
            result.program.validity->instructions.data(),
            result.program.validity->instructions.size() *
                sizeof(NMExpressionInstructionGPU),
            fingerprint
        );
    }
    result.program.fingerprint = fingerprint;
    return result;
}

} // namespace numi::matter::detail

namespace numi::matter {

std::string emitSpecializedMetal(
    const std::span<const ConstitutiveProgram> programs
) {
    std::ostringstream source;
    source << "// Generated by Numi Matter. Fingerprinted scene specialization.\n"
           << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    for (std::size_t index = 0u; index < programs.size(); ++index) {
        const ConstitutiveProgram& program = programs[index];
        source << "// material " << program.material.name
               << " fingerprint " << program.fingerprint << "\n";
        const auto find = [&](const std::string_view name) -> std::optional<std::uint32_t> {
            for (std::uint32_t parameter = 0u;
                 parameter < program.material.parameters.size(); ++parameter) {
                if (program.material.parameters[parameter].name == name) return parameter;
            }
            return std::nullopt;
        };
        if (program.material.hint == ConstitutiveHint::neoHookean &&
            find("mu").has_value() && find("lambda").has_value()) {
            source << "inline float3x3 nm_specialized_stress_" << index
                   << "(float3x3 F, device const float* parameters) {\n"
                   << "  const float mu = parameters[" << *find("mu") << "];\n"
                   << "  const float lambda = parameters[" << *find("lambda") << "];\n"
                   << "  const float J = determinant(F);\n"
                   << "  const float3x3 FinvT = transpose(inverse(F));\n"
                   << "  return mu * (F - FinvT) + lambda * log(J) * FinvT;\n"
                   << "}\n\n";
        } else {
            source << "// Generic bytecode evaluator is authoritative for this material.\n\n";
        }
    }
    return source.str();
}

} // namespace numi::matter
