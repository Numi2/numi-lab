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
#include <unordered_set>
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

    // Recognize the exact smoothed complementarity expression, rather than
    // changing the authored value or a solver tolerance. The ordinary chain
    // rule subtracts nearly equal O(1) terms on its inactive branch. In FP32
    // that roundoff can become a false plastic tangent after multiplication
    // by anisotropic yield coefficients.
    [[nodiscard]] bool sameExpression(std::uint32_t a, std::uint32_t b) const {
        if (a == b) return true;
        if (a >= graph_.nodes.size() || b >= graph_.nodes.size()) return false;
        const Expr& x = graph_.nodes[a];
        const Expr& y = graph_.nodes[b];
        if (x.kind != y.kind || x.dimension != y.dimension ||
            x.constant != y.constant || x.index != y.index || x.integer != y.integer)
            return false;
        for (unsigned i = 0; i < x.arguments.size(); ++i) {
            if (x.arguments[i] == NM_INVALID_INDEX || y.arguments[i] == NM_INVALID_INDEX) {
                if (x.arguments[i] != y.arguments[i]) return false;
            } else if (!sameExpression(x.arguments[i], y.arguments[i])) return false;
        }
        return true;
    }

    struct Complementarity { std::uint32_t a, b, epsilon, radius; };

    [[nodiscard]] std::optional<Complementarity> complementarity(std::uint32_t node) const {
        const Expr& outer = graph_.nodes[node];
        if (outer.kind != ExprKind::subtract) return std::nullopt;
        const Expr& inner = graph_.nodes[outer.arguments[0]];
        if (inner.kind != ExprKind::subtract) return std::nullopt;
        const auto radius = inner.arguments[0];
        if (graph_.nodes[radius].kind != ExprKind::squareRoot) return std::nullopt;
        const auto a = inner.arguments[1], b = outer.arguments[1];
        std::vector<std::uint32_t> terms;
        const auto flatten = [&](auto&& self, std::uint32_t id) -> void {
            const Expr& expression = graph_.nodes[id];
            if (expression.kind == ExprKind::add) {
                self(self, expression.arguments[0]); self(self, expression.arguments[1]);
            } else terms.push_back(id);
        };
        flatten(flatten, graph_.nodes[radius].arguments[0]);
        if (terms.size() != 3) return std::nullopt;
        bool foundA = false, foundB = false;
        std::uint32_t epsilon = NM_INVALID_INDEX;
        for (auto term : terms) {
            const Expr& square = graph_.nodes[term];
            if (square.kind != ExprKind::integerPower || square.integer != 2)
                return std::nullopt;
            const auto base = square.arguments[0];
            if (!foundA && sameExpression(base, a)) foundA = true;
            else if (!foundB && sameExpression(base, b)) foundB = true;
            else if (epsilon == NM_INVALID_INDEX &&
                (graph_.nodes[base].kind == ExprKind::constant ||
                 graph_.nodes[base].kind == ExprKind::parameter)) epsilon = base;
            else return std::nullopt;
        }
        if (!foundA || !foundB || epsilon == NM_INVALID_INDEX ||
            graph_.nodes[a].dimension != graph_.nodes[b].dimension ||
            graph_.nodes[a].dimension != graph_.nodes[epsilon].dimension) return std::nullopt;
        return Complementarity{a, b, epsilon, radius};
    }

    [[nodiscard]] std::uint32_t rationalizedComplementarityValue(
        const std::uint32_t original,
        const Complementarity& c
    ) {
        const Dimension valueDimension = graph_.nodes[c.a].dimension;
        const Dimension squaredDimension = valueDimension * 2;
        const std::uint32_t sum = binary(
            ExprKind::add, c.a, c.b, valueDimension);
        const std::uint32_t denominator = binary(
            ExprKind::add,
            c.radius,
            sum,
            valueDimension);
        // Accurate bilinear-15 layout admission requires epsilon > 0. On the
        // selected sum>=0 branch, denominator >= epsilon; on the unselected
        // sum<0 branch this floor keeps eager SELECT evaluation finite.
        const std::uint32_t safeDenominator = binary(
            ExprKind::maximum, denominator, c.epsilon, valueDimension);
        const std::uint32_t epsilonSquared = power(
            c.epsilon, 2, squaredDimension);
        const std::uint32_t twiceProduct = binary(
            ExprKind::multiply,
            graph_.constant(2.0),
            binary(ExprKind::multiply, c.a, c.b, squaredDimension),
            squaredDimension);
        const std::uint32_t numerator = binary(
            ExprKind::subtract, epsilonSquared, twiceProduct,
            squaredDimension);
        const std::uint32_t stableValue = binary(
            ExprKind::divide, numerator, safeDenominator, valueDimension);
        Expr branch;
        branch.kind = ExprKind::selectNonnegative;
        branch.dimension = graph_.nodes[original].dimension;
        branch.arguments[0] = sum;
        branch.arguments[1] = stableValue;
        branch.arguments[2] = original;
        return graph_.append(branch);
    }

    [[nodiscard]] std::uint32_t rationalizeComplementarityValues(
        const std::uint32_t node,
        std::unordered_map<std::uint32_t, std::uint32_t>& cache
    ) {
        if (node == NM_INVALID_INDEX || node >= graph_.nodes.size()) return node;
        if (const auto found = cache.find(node); found != cache.end())
            return found->second;
        if (const auto c = complementarity(node)) {
            const std::uint32_t result = rationalizedComplementarityValue(node, *c);
            cache.emplace(node, result);
            return result;
        }
        const Expr original = graph_.nodes[node];
        Expr rewritten = original;
        bool changed = false;
        for (std::uint32_t slot = 0u; slot < original.arguments.size(); ++slot) {
            const std::uint32_t argument = original.arguments[slot];
            if (argument == NM_INVALID_INDEX) continue;
            const std::uint32_t replacement =
                rationalizeComplementarityValues(argument, cache);
            if (replacement != argument) {
                rewritten.arguments[slot] = replacement;
                changed = true;
            }
        }
        const std::uint32_t result = changed
            ? graph_.append(rewritten) : node;
        cache.emplace(node, result);
        return result;
    }

    [[nodiscard]] std::uint32_t complementarityCoefficient(
        std::uint32_t x, std::uint32_t y, const Complementarity& c
    ) {
        const auto ratio = [&](std::uint32_t v) {
            return binary(ExprKind::divide, v, c.radius, kDimensionless);
        };
        const auto xr = ratio(x), yr = ratio(y), er = ratio(c.epsilon);
        const auto numerator = binary(ExprKind::add,
            power(yr, 2, kDimensionless), power(er, 2, kDimensionless), kDimensionless);
        // Both select arms are evaluated. Guard the unused negative-x arm's
        // stable denominator without altering the positive-x formula.
        const auto denominator = binary(ExprKind::maximum,
            binary(ExprKind::add, one(), xr, kDimensionless), one(), kDimensionless);
        const auto stable = unary(ExprKind::negate,
            binary(ExprKind::divide, numerator, denominator, kDimensionless), kDimensionless);
        const auto ordinary = binary(ExprKind::subtract, xr, one(), kDimensionless);
        Expr branch;
        branch.kind = ExprKind::selectNonnegative;
        branch.dimension = kDimensionless;
        branch.arguments[0] = x;
        branch.arguments[1] = stable;
        branch.arguments[2] = ordinary;
        return graph_.append(branch);
    }

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
            case ExprKind::expm1MinusArgument:
                return graph_.constant(std::expm1(value) - value, dimension);
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
        case ExprKind::subtract:
            if (const auto c = complementarity(node)) {
                const auto pair = binary(ExprKind::add,
                    binary(ExprKind::multiply, complementarityCoefficient(c->a, c->b, *c), d(c->a), resultDimension),
                    binary(ExprKind::multiply, complementarityCoefficient(c->b, c->a, *c), d(c->b), resultDimension),
                    resultDimension);
                result = binary(ExprKind::add, pair,
                    binary(ExprKind::multiply,
                        binary(ExprKind::divide, c->epsilon, c->radius, kDimensionless),
                        d(c->epsilon), resultDimension), resultDimension);
                break;
            }
            [[fallthrough]];
        case ExprKind::add:
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
        case ExprKind::fiberExpLinear: {
            bool shapeConstant=true;
            for(unsigned i=1;i<5;++i) {
                double derivative=0.0;
                if(!constantValue(graph_,d(source.arguments[i]),derivative) || derivative!=0.0)
                    shapeConstant=false;
            }
            if(!shapeConstant || source.integer>=2) {
                diagnostics.push_back({Diagnostic::Severity::error,0u,0u,
                    "fiber_exp_linear supports two stretch derivatives with constant material shape inputs"});
                result=zero(resultDimension);
                break;
            }
            Expr next=source;
            ++next.integer;
            const auto factor=graph_.append(next);
            result=binary(ExprKind::multiply,factor,d(source.arguments[0]),resultDimension);
            break;
        }
        case ExprKind::expm1MinusArgument: {
            const std::uint32_t exponential = unary(
                ExprKind::exponential,
                source.arguments[0],
                kDimensionless
            );
            const std::uint32_t factor = binary(
                ExprKind::subtract,
                exponential,
                graph_.constant(1.0),
                kDimensionless
            );
            result = binary(
                ExprKind::multiply,
                factor,
                d(source.arguments[0]),
                resultDimension
            );
            break;
        }
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
        case ExprKind::selectNonnegative: {
            Expr branch;
            branch.kind = ExprKind::selectNonnegative;
            branch.dimension = resultDimension;
            branch.arguments[0] = source.arguments[0];
            branch.arguments[1] = d(source.arguments[1]);
            branch.arguments[2] = d(source.arguments[2]);
            result = graph_.append(branch);
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
        case ExprKind::subtract:
            if (const auto c = complementarity(node)) {
                const auto pair = binary(ExprKind::add,
                    binary(ExprKind::multiply, complementarityCoefficient(c->a, c->b, *c), d(c->a), source.dimension),
                    binary(ExprKind::multiply, complementarityCoefficient(c->b, c->a, *c), d(c->b), source.dimension),
                    source.dimension);
                result = binary(ExprKind::add, pair,
                    binary(ExprKind::multiply,
                        binary(ExprKind::divide, c->epsilon, c->radius, kDimensionless),
                        d(c->epsilon), source.dimension), source.dimension);
                break;
            }
            [[fallthrough]];
        case ExprKind::add:
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
        case ExprKind::fiberExpLinear: {
            bool shapeConstant=true;
            for(unsigned i=1;i<5;++i) {
                double derivative=0.0;
                if(!constantValue(graph_,d(source.arguments[i]),derivative) || derivative!=0.0)
                    shapeConstant=false;
            }
            if(!shapeConstant || source.integer>=2) {
                diagnostics.push_back({Diagnostic::Severity::error,0u,0u,
                    "fiber_exp_linear supports two stretch derivatives with constant material shape inputs"});
                result=zero(source.dimension);
                break;
            }
            Expr next=source;
            ++next.integer;
            const auto factor=graph_.append(next);
            result=binary(ExprKind::multiply,factor,d(source.arguments[0]),source.dimension);
            break;
        }
        case ExprKind::expm1MinusArgument: {
            const std::uint32_t exponential = unary(
                ExprKind::exponential,
                source.arguments[0],
                kDimensionless
            );
            const std::uint32_t factor = binary(
                ExprKind::subtract,
                exponential,
                graph_.constant(1.0),
                kDimensionless
            );
            result = binary(
                ExprKind::multiply,
                factor,
                d(source.arguments[0]),
                source.dimension
            );
            break;
        }
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
        case ExprKind::selectNonnegative: {
            Expr branch;
            branch.kind = ExprKind::selectNonnegative;
            branch.dimension = source.dimension;
            branch.arguments[0] = source.arguments[0];
            branch.arguments[1] = d(source.arguments[1]);
            branch.arguments[2] = d(source.arguments[2]);
            result = graph_.append(branch);
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
    BytecodeCompiler(const ExpressionGraph& graph, const std::uint32_t maximumStack,
                     const bool cacheExpressions)
        : graph_(graph), maximumAllowed_(maximumStack), cacheExpressions_(cacheExpressions) {}

    [[nodiscard]] ScalarBytecode compile(
        const std::uint32_t root,
        std::vector<Diagnostic>& diagnostics
    ) {
        ScalarBytecode output;
        stack_ = 0u;
        maximum_ = 0u;
        cacheSlots_.clear(); cacheNodes_.clear(); remainingUses_.clear(); freeSlots_.clear();
        emissionLimitReported_ = false;
        if (cacheExpressions_) {
            std::unordered_set<std::uint32_t> visited;
            const auto visit = [&](auto&& self, std::uint32_t node) -> void {
                if (node == NM_INVALID_INDEX || node >= graph_.nodes.size()) return;
                ++remainingUses_[node];
                if (!visited.insert(node).second) return;
                for (auto child : graph_.nodes[node].arguments)
                    if (child != NM_INVALID_INDEX) self(self, child);
            };
            visit(visit, root);
            for (const auto& [node, count] : remainingUses_)
                if (count > 1u && graph_.nodes[node].arguments[0] != NM_INVALID_INDEX)
                    cacheNodes_.insert(node);
            for (std::uint32_t i = NM_EXPRESSION_CACHE_CAPACITY; i > 0u; --i)
                freeSlots_.push_back(i - 1u);
        }
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
        if (cacheExpressions_ && output.size() >= 65536u) {
            if (!emissionLimitReported_) {
                diagnostics.push_back({Diagnostic::Severity::error, 0u, 0u,
                    "cached constitutive scalar exceeds the bounded instruction limit"});
                emissionLimitReported_ = true;
            }
            return;
        }
        if (node == NM_INVALID_INDEX || node >= graph_.nodes.size()) {
            diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "invalid expression root during bytecode emission",
            });
            return;
        }
        const Expr& expression = graph_.nodes[node];
        const auto cache = cacheSlots_.find(node);
        if (cache != cacheSlots_.end()) {
            NMExpressionInstructionGPU load{};
            load.opcode = NM_EXPR_CACHE_LOAD;
            load.index = cache->second;
            output.push_back(load);
            push();
            consume(node);
            return;
        }
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
        case ExprKind::expm1MinusArgument:
        case ExprKind::squareRoot:
        case ExprKind::absolute:
        case ExprKind::integerPower:
            child(0u);
            instruction.opcode = expression.kind == ExprKind::negate ? NM_EXPR_NEGATE :
                expression.kind == ExprKind::logarithm ? NM_EXPR_LOG :
                expression.kind == ExprKind::exponential ? NM_EXPR_EXP :
                expression.kind == ExprKind::expm1MinusArgument ? NM_EXPR_EXPM1_MINUS_X :
                expression.kind == ExprKind::squareRoot ? NM_EXPR_SQRT :
                expression.kind == ExprKind::absolute ? NM_EXPR_ABS : NM_EXPR_POW_INTEGER;
            output.push_back(instruction);
            break;
        case ExprKind::fiberExpLinear:
            for(unsigned i=0;i<5;++i) child(i);
            instruction.opcode=NM_EXPR_FIBER_EXP_LINEAR;
            instruction.integer=expression.integer;
            output.push_back(instruction);
            if(stack_>=5u) stack_-=4u;
            break;
        case ExprKind::selectNonnegative:
            child(0u); child(1u); child(2u);
            instruction.opcode = NM_EXPR_SELECT_NONNEGATIVE;
            output.push_back(instruction);
            if (stack_ >= 3u) stack_ -= 2u;
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
        if (cacheExpressions_ && cacheNodes_.contains(node)) {
            if (freeSlots_.empty()) {
                if (!emissionLimitReported_) diagnostics.push_back({
                    Diagnostic::Severity::error, 0u, 0u,
                    "constitutive expression exceeds the bounded live cache capacity"});
                emissionLimitReported_ = true;
                return;
            }
            const auto slot = freeSlots_.back(); freeSlots_.pop_back();
            cacheSlots_[node] = slot;
            NMExpressionInstructionGPU save{};
            save.opcode = NM_EXPR_CACHE_STORE; save.index = slot;
            output.push_back(save);
        }
        consume(node);
    }

    void consume(std::uint32_t node) {
        if (!cacheExpressions_) return;
        auto found = remainingUses_.find(node);
        if (found == remainingUses_.end() || found->second == 0u) return;
        if (--found->second != 0u) return;
        const auto slot = cacheSlots_.find(node);
        if (slot != cacheSlots_.end()) {
            freeSlots_.push_back(slot->second);
            cacheSlots_.erase(slot);
        }
    }

    const ExpressionGraph& graph_;
    std::uint32_t maximumAllowed_ = 0u;
    std::uint32_t stack_ = 0u;
    std::uint32_t maximum_ = 0u;
    bool cacheExpressions_ = false;
    bool emissionLimitReported_ = false;
    std::unordered_map<std::uint32_t, std::uint32_t> cacheSlots_;
    std::unordered_set<std::uint32_t> cacheNodes_;
    std::unordered_map<std::uint32_t, std::uint32_t> remainingUses_;
    std::vector<std::uint32_t> freeSlots_;
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
    bool bilinear15AccurateLayout = false;
    if (material.paperHillScalarInitializer || material.paperCompensatedArithmetic ||
        material.paperHillAccurateProjection ||
        material.genericCompensatedImplicitProjection ||
        material.genericBilinear15AccurateProjection ||
        material.genericBilinear15StrictStateCertification) {
        const std::array<std::string_view,20> parameters{"density","c11","c22","c33",
            "c12","c13","c23","g12","g13","g23","sigma_ref","hill_F","hill_G",
            "hill_H","hill_L","hill_M","hill_N","q_floor","fb_epsilon","residual_scale"};
        const std::array<std::string_view,13> states{"ep11","ep22","ep33","ep23","ep13","ep12",
            "sbar11","sbar22","sbar33","sbar23","sbar13","sbar12","accumulated_multiplier"};
        bool valid=material.hint==ConstitutiveHint::generic &&
            material.dissipationRoot==NM_INVALID_INDEX && material.parameters.size()==parameters.size() &&
            material.internalState.size()==states.size() && material.stateImplicitRoots.size()==states.size();
        for(unsigned i=0;valid && i<parameters.size();++i)
            valid=material.parameters[i].name==parameters[i] &&
                std::isfinite(material.parameters[i].defaultValue);
        for(unsigned i=0;valid && i<states.size();++i)
            valid=material.internalState[i].name==states[i] && material.stateImplicitRoots[i]!=NM_INVALID_INDEX;
        const bool finiteElasticStarch = !material.paperHillScalarInitializer &&
            !material.paperHillAccurateProjection &&
            !material.genericCompensatedImplicitProjection &&
            material.paperCompensatedArithmetic && material.hint==ConstitutiveHint::neoHookean &&
            material.dissipationRoot==NM_INVALID_INDEX && material.parameters.size()==3u &&
            material.internalState.empty() && material.stateImplicitRoots.empty() &&
            material.parameters[0].name=="density" && material.parameters[1].name=="mu" &&
            material.parameters[2].name=="lambda";
        const bool genericProjectionLayout =
            material.genericCompensatedImplicitProjection &&
            !material.paperHillScalarInitializer &&
            !material.paperHillAccurateProjection &&
            material.hint==ConstitutiveHint::generic &&
            !material.internalState.empty() &&
            material.internalState.size()<=NM_MAX_MATERIAL_STATE &&
            material.stateImplicitRoots.size()==material.internalState.size() &&
            std::ranges::all_of(material.stateImplicitRoots,
                [](const std::uint32_t root) { return root!=NM_INVALID_INDEX; });
        const std::array<std::string_view,28> bilinear15Parameters{
            "density","c11","c22","c33","c12","c13","c23","g12","g13","g23",
            "sigma_ref","hill_F","hill_G","hill_H","hill_L","hill_M","hill_N",
            "hardening_md","hardening_cd","fit_transfer_scale_md","fit_transfer_scale_cd",
            "hardening_md_flow_scale","hardening_cd_flow_scale","hardening_smoothing_width",
            "hardening_weight_scale","q_floor","fb_epsilon","residual_scale"};
        const std::array<std::string_view,15> bilinear15States{
            "ep11","ep22","ep33","ep23","ep13","ep12","sbar11","sbar22",
            "sbar33","sbar23","sbar13","sbar12","accumulated_multiplier",
            "hardening_md_accumulation","hardening_cd_accumulation"};
        bilinear15AccurateLayout = material.genericBilinear15AccurateProjection &&
            material.genericCompensatedImplicitProjection &&
            !material.paperHillScalarInitializer &&
            !material.paperHillAccurateProjection && material.hint==ConstitutiveHint::generic &&
            material.dissipationRoot==NM_INVALID_INDEX && !material.learned.has_value() &&
            material.parameters.size()==bilinear15Parameters.size() &&
            material.internalState.size()==bilinear15States.size() &&
            material.stateImplicitRoots.size()==bilinear15States.size() &&
            std::ranges::all_of(material.stateImplicitRoots,
                [](const std::uint32_t root) { return root!=NM_INVALID_INDEX; });
        for (unsigned i=0;bilinear15AccurateLayout && i<bilinear15Parameters.size();++i)
            bilinear15AccurateLayout=material.parameters[i].name==bilinear15Parameters[i] &&
                std::isfinite(material.parameters[i].defaultValue);
        for (unsigned i=0;bilinear15AccurateLayout && i<bilinear15States.size();++i)
            bilinear15AccurateLayout=material.internalState[i].name==bilinear15States[i];
        if (bilinear15AccurateLayout)
            bilinear15AccurateLayout=material.parameters[26].defaultValue>0.0;
        const auto rateOrHistoryDependency = [&](const std::uint32_t root) {
            if (root == NM_INVALID_INDEX || root >= material.expressions.nodes.size())
                return root != NM_INVALID_INDEX;
            std::vector<std::uint32_t> pending{root};
            std::vector<std::uint8_t> visited(material.expressions.nodes.size(), 0u);
            while (!pending.empty()) {
                const std::uint32_t index = pending.back();
                pending.pop_back();
                if (index >= material.expressions.nodes.size()) return true;
                if (visited[index] != 0u) continue;
                visited[index] = 1u;
                const Expr& expression = material.expressions.nodes[index];
                switch (expression.kind) {
                case ExprKind::deformationDirection:
                case ExprKind::deformationRate:
                case ExprKind::timeStep:
                case ExprKind::temperature:
                case ExprKind::internalState:
                case ExprKind::candidateState:
                    return true;
                default:
                    break;
                }
                for (const std::uint32_t argument : expression.arguments) {
                    if (argument != NM_INVALID_INDEX) pending.push_back(argument);
                }
            }
            return false;
        };
        const bool statelessGenericStoredEnergy =
            material.paperCompensatedArithmetic &&
            !material.paperHillScalarInitializer &&
            !material.paperHillAccurateProjection &&
            !material.genericCompensatedImplicitProjection &&
            material.hint == ConstitutiveHint::generic &&
            material.internalState.empty() &&
            material.stateUpdateRoots.empty() &&
            material.stateImplicitRoots.empty() &&
            material.dissipationRoot == NM_INVALID_INDEX &&
            !material.learned.has_value() &&
            !rateOrHistoryDependency(material.energyRoot) &&
            !rateOrHistoryDependency(material.validityRoot);
        if((material.genericCompensatedImplicitProjection &&
                !genericProjectionLayout) ||
           (material.genericBilinear15AccurateProjection && !bilinear15AccurateLayout) ||
           (material.genericBilinear15StrictStateCertification &&
            (!material.genericBilinear15AccurateProjection || !bilinear15AccurateLayout)) ||
           ((material.paperHillScalarInitializer ||
             material.paperCompensatedArithmetic ||
             material.paperHillAccurateProjection) &&
                !valid && !finiteElasticStarch && !genericProjectionLayout &&
                !statelessGenericStoredEnergy) ||
           (material.paperHillAccurateProjection && !material.paperCompensatedArithmetic)) {
            result.diagnostics.push_back({Diagnostic::Severity::error,0u,0u,
                genericProjectionLayout || finiteElasticStarch
                    ? "accurate Hill projection requires compensated arithmetic"
                    : "compensated arithmetic requires source-ordered 13-state Hill, stateless finite elastic starch, stateless rate-independent generic stored energy, or a fully implicit generic bytecode layout; generic compensated projection cannot combine with Hill-only policies"});
            return result;
        }
    }
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
    const bool specializedPlasticity =
        material.hint == ConstitutiveHint::vonMises ||
        material.hint == ConstitutiveHint::druckerPrager;
    if (specializedPlasticity) {
        constexpr std::array<std::string_view, 10u> requiredState{{
            "plastic_f00", "plastic_f01", "plastic_f02",
            "plastic_f10", "plastic_f11", "plastic_f12",
            "plastic_f20", "plastic_f21", "plastic_f22",
            "equivalent_plastic_strain",
        }};
        bool validLayout = material.internalState.size() ==
            requiredState.size();
        for (std::size_t state = 0u;
             validLayout && state < requiredState.size(); ++state) {
            const double expected = state < 9u && state % 4u == 0u
                ? 1.0 : 0.0;
            validLayout = material.internalState[state].name ==
                    requiredState[state] &&
                material.internalState[state].initialValue == expected &&
                (state != 9u || material.internalState[state].transfer ==
                    InternalState::Transfer::maximum);
        }
        const bool validParameters = parameterIndex(material, "mu").has_value() &&
            parameterIndex(material, "lambda").has_value() &&
            parameterIndex(material, "hardening").has_value() &&
            ((material.hint == ConstitutiveHint::vonMises &&
              parameterIndex(material, "yield_stress").has_value()) ||
             (material.hint == ConstitutiveHint::druckerPrager &&
              parameterIndex(material, "friction_angle").has_value() &&
              parameterIndex(material, "cohesion").has_value()));
        if (!validLayout || !validParameters) {
            result.diagnostics.push_back({
                Diagnostic::Severity::error, 0u, 0u,
                "specialized plasticity requires row-major plastic_f00..plastic_f22 identity state, equivalent_plastic_strain transfer max, and mu/lambda/hardening parameters; von Mises requires yield_stress and Drucker-Prager requires friction_angle and cohesion",
            });
            return result;
        }
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
        const std::uint32_t authoredResidual = residual;
        if (bilinear15AccurateLayout) {
            std::unordered_map<std::uint32_t, std::uint32_t> valueRewriteCache;
            residual = symbolic.rationalizeComplementarityValues(
                authoredResidual, valueRewriteCache);
        }
        residualRoots.push_back(residual);
        // The exact authored residual remains the source for both derivative
        // programs. Its recognized FB derivative already uses the stable
        // analytic coefficient; only the value program is rationalized.
        residualDirectionRoots.push_back(symbolic.directional(
            authoredResidual, Symbolic::VariableDomain::deformation,
            result.diagnostics
        ));
        for (std::uint32_t candidate = 0u; candidate < stateCount; ++candidate) {
            jacobianRoots.push_back(symbolic.derivative(
                authoredResidual, candidate, Symbolic::VariableDomain::candidateState,
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
        maximumStack,
        material.cacheExpressions
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
        (hasImplicit ? NM_MATERIAL_HAS_IMPLICIT_STATE : 0u) |
        (material.paperHillScalarInitializer ? NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER : 0u) |
        (material.paperCompensatedArithmetic ? NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC : 0u) |
        (material.paperHillAccurateProjection ? NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION : 0u) |
        (material.genericCompensatedImplicitProjection
            ? NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION : 0u) |
        (material.genericBilinear15AccurateProjection
            ? NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION : 0u) |
        (material.genericBilinear15StrictStateCertification
            ? NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION : 0u);
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
    gpu.localNewtonIterations = material.genericBilinear15AccurateProjection ? 16u : 8u;
    gpu.stateTransferMask = 0u;
    for (std::uint32_t state = 0u; state < material.internalState.size(); ++state) {
        const auto transfer = material.internalState[state].transfer;
        const std::uint32_t encoded = transfer == InternalState::Transfer::maximum
            ? NM_MATERIAL_TRANSFER_MAXIMUM
            : transfer == InternalState::Transfer::sum
                ? NM_MATERIAL_TRANSFER_SUM
                : NM_MATERIAL_TRANSFER_AVERAGE;
        gpu.stateTransferMask |= static_cast<nm_u64>(encoded) << (2u * state);
    }
    if ((gpu.projectionKind == NM_MATERIAL_PROJECTION_VON_MISES ||
         gpu.projectionKind == NM_MATERIAL_PROJECTION_DRUCKER_PRAGER) &&
        gpu.stateCount == 10u) {
        // The specialized layout is Fp[0..8], accumulated plastic strain[9].
        // Fp is projected by the authored/default average policy, while the
        // irreversible scalar must never decrease during coarsening.
        constexpr std::uint32_t accumulatedShift = 2u * 9u;
        gpu.stateTransferMask &= ~(3u << accumulatedShift);
        gpu.stateTransferMask |=
            NM_MATERIAL_TRANSFER_MAXIMUM << accumulatedShift;
    }
    const auto density = parameterIndex(material, "density");
    const auto mu = parameterIndex(material, "mu");
    const auto lambda = parameterIndex(material, "lambda");
    const auto yieldStress = parameterIndex(material, "yield_stress");
    const auto hardening = parameterIndex(material, "hardening");
    const auto frictionAngle = parameterIndex(material, "friction_angle");
    const auto cohesion = parameterIndex(material, "cohesion");
    const auto parameterDefault = [&](
        const std::optional<std::uint32_t> index,
        const double fallback = 0.0
    ) {
        return index.has_value()
            ? material.parameters[*index].defaultValue : fallback;
    };
    gpu.bulk = float4(
        density.has_value() ? static_cast<float>(material.parameters[*density].defaultValue) : 0.0f,
        293.15f,
        static_cast<float>(parameterDefault(mu)),
        static_cast<float>(parameterDefault(lambda))
    );
    const double angle = parameterDefault(frictionAngle);
    const double sinAngle = std::sin(angle);
    const double denominator = std::max(3.0 - sinAngle, 1.0e-12);
    const double druckerAlpha = 2.0 * sinAngle / denominator;
    const double druckerCohesion =
        6.0 * parameterDefault(cohesion) * std::cos(angle) / denominator;
    gpu.inelastic = float4(
        static_cast<float>(parameterDefault(yieldStress)),
        static_cast<float>(parameterDefault(hardening)),
        static_cast<float>(druckerAlpha),
        static_cast<float>(druckerCohesion)
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
    if(material.paperHillScalarInitializer) {
        const std::uint32_t policy=NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
    if(material.paperCompensatedArithmetic) {
        const std::uint32_t policy=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
    if(material.paperHillAccurateProjection) {
        const std::uint32_t policy=NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
    if(material.genericCompensatedImplicitProjection) {
        const std::uint32_t policy=NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
    if(material.genericBilinear15AccurateProjection) {
        const std::uint32_t policy=NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
    if(material.genericBilinear15StrictStateCertification) {
        const std::uint32_t policy=NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION;
        fingerprint=hashBytes(&policy,sizeof(policy),fingerprint);
    }
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
