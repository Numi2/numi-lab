#include "numi/matter/matter.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace numi::matter {

std::uint32_t ExpressionGraph::append(Expr value) {
    if (nodes.size() >= static_cast<std::size_t>(NM_INVALID_INDEX)) {
        return NM_INVALID_INDEX;
    }
    nodes.push_back(std::move(value));
    return static_cast<std::uint32_t>(nodes.size() - 1u);
}

std::uint32_t ExpressionGraph::constant(
    const double value,
    const Dimension dimension
) {
    Expr expression;
    expression.kind = ExprKind::constant;
    expression.dimension = dimension;
    expression.constant = value;
    return append(expression);
}

bool ParseResult::succeeded() const noexcept {
    return std::none_of(
        diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& diagnostic) {
            return diagnostic.severity == Diagnostic::Severity::error;
        }
    );
}

bool CompileResult::succeeded() const noexcept {
    return std::none_of(
        diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& diagnostic) {
            return diagnostic.severity == Diagnostic::Severity::error;
        }
    );
}

std::string dimensionName(const Dimension dimension) {
    if (dimension == kDimensionless) {
        return "1";
    }
    std::ostringstream result;
    const auto append = [&result](
        const char* symbol,
        const int exponent
    ) {
        if (exponent == 0) {
            return;
        }
        if (result.tellp() > 0) {
            result << ' ';
        }
        result << symbol;
        if (exponent != 1) {
            result << '^' << exponent;
        }
    };
    append("m", dimension.length);
    append("kg", dimension.mass);
    append("s", dimension.time);
    append("K", dimension.temperature);
    return result.str();
}

} // namespace numi::matter
