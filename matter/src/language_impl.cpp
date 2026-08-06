#include "numi/matter/matter.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace numi::matter {
namespace {

enum class TokenKind : std::uint8_t {
    end,
    identifier,
    number,
    leftBrace,
    rightBrace,
    leftParen,
    rightParen,
    leftBracket,
    rightBracket,
    colon,
    semicolon,
    comma,
    equal,
    plus,
    minus,
    star,
    slash,
    caret,
};

struct Token {
    TokenKind kind = TokenKind::end;
    std::string text;
    double number = 0.0;
    std::size_t line = 1u;
    std::size_t column = 1u;
};

class Lexer {
public:
    explicit Lexer(std::string_view source) : source_(source) {}

    [[nodiscard]] std::vector<Token> run(std::vector<Diagnostic>& diagnostics) {
        std::vector<Token> result;
        while (true) {
            skipSpaceAndComments();
            if (offset_ >= source_.size()) {
                result.push_back({TokenKind::end, {}, 0.0, line_, column_});
                break;
            }
            const std::size_t line = line_;
            const std::size_t column = column_;
            const char c = source_[offset_];
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                const std::size_t begin = offset_;
                advance();
                while (offset_ < source_.size()) {
                    const char value = source_[offset_];
                    if (!std::isalnum(static_cast<unsigned char>(value)) &&
                        value != '_') {
                        break;
                    }
                    advance();
                }
                result.push_back({
                    TokenKind::identifier,
                    std::string(source_.substr(begin, offset_ - begin)),
                    0.0,
                    line,
                    column,
                });
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(c)) ||
                (c == '.' && offset_ + 1u < source_.size() &&
                 std::isdigit(static_cast<unsigned char>(source_[offset_ + 1u])))) {
                const std::size_t begin = offset_;
                bool exponent = false;
                advance();
                while (offset_ < source_.size()) {
                    const char value = source_[offset_];
                    if (std::isdigit(static_cast<unsigned char>(value)) || value == '.') {
                        advance();
                        continue;
                    }
                    if ((value == 'e' || value == 'E') && !exponent) {
                        exponent = true;
                        advance();
                        if (offset_ < source_.size() &&
                            (source_[offset_] == '+' || source_[offset_] == '-')) {
                            advance();
                        }
                        continue;
                    }
                    break;
                }
                const std::string text(source_.substr(begin, offset_ - begin));
                double number = 0.0;
                const auto [pointer, error] = std::from_chars(
                    text.data(), text.data() + text.size(), number
                );
                if (error != std::errc{} || pointer != text.data() + text.size() ||
                    !std::isfinite(number)) {
                    diagnostics.push_back({
                        Diagnostic::Severity::error,
                        line,
                        column,
                        "invalid finite number '" + text + "'",
                    });
                }
                result.push_back({TokenKind::number, text, number, line, column});
                continue;
            }
            const auto single = [&](const TokenKind kind) {
                result.push_back({kind, std::string(1u, c), 0.0, line, column});
                advance();
            };
            switch (c) {
            case '{': single(TokenKind::leftBrace); break;
            case '}': single(TokenKind::rightBrace); break;
            case '(': single(TokenKind::leftParen); break;
            case ')': single(TokenKind::rightParen); break;
            case '[': single(TokenKind::leftBracket); break;
            case ']': single(TokenKind::rightBracket); break;
            case ':': single(TokenKind::colon); break;
            case ';': single(TokenKind::semicolon); break;
            case ',': single(TokenKind::comma); break;
            case '=': single(TokenKind::equal); break;
            case '+': single(TokenKind::plus); break;
            case '-': single(TokenKind::minus); break;
            case '*': single(TokenKind::star); break;
            case '/': single(TokenKind::slash); break;
            case '^': single(TokenKind::caret); break;
            default:
                diagnostics.push_back({
                    Diagnostic::Severity::error,
                    line,
                    column,
                    std::string("unexpected character '") + c + "'",
                });
                advance();
                break;
            }
        }
        return result;
    }

private:
    void advance() {
        if (source_[offset_] == '\n') {
            ++line_;
            column_ = 1u;
        } else {
            ++column_;
        }
        ++offset_;
    }

    void skipSpaceAndComments() {
        while (offset_ < source_.size()) {
            if (std::isspace(static_cast<unsigned char>(source_[offset_]))) {
                advance();
                continue;
            }
            if (source_[offset_] == '/' && offset_ + 1u < source_.size() &&
                source_[offset_ + 1u] == '/') {
                while (offset_ < source_.size() && source_[offset_] != '\n') {
                    advance();
                }
                continue;
            }
            if (source_[offset_] == '/' && offset_ + 1u < source_.size() &&
                source_[offset_ + 1u] == '*') {
                advance();
                advance();
                while (offset_ + 1u < source_.size() &&
                       !(source_[offset_] == '*' && source_[offset_ + 1u] == '/')) {
                    advance();
                }
                if (offset_ + 1u < source_.size()) {
                    advance();
                    advance();
                }
                continue;
            }
            break;
        }
    }

    std::string_view source_;
    std::size_t offset_ = 0u;
    std::size_t line_ = 1u;
    std::size_t column_ = 1u;
};

struct Unit {
    Dimension dimension{};
    double scale = 1.0;
};

[[nodiscard]] std::optional<Unit> baseUnit(const std::string_view name) {
    static const std::unordered_map<std::string_view, Unit> units{
        {"one", {kDimensionless, 1.0}},
        {"m", {kLength, 1.0}},
        {"mm", {kLength, 1.0e-3}},
        {"cm", {kLength, 1.0e-2}},
        {"kg", {kMass, 1.0}},
        {"g", {kMass, 1.0e-3}},
        {"s", {kTime, 1.0}},
        {"ms", {kTime, 1.0e-3}},
        {"K", {kTemperature, 1.0}},
        {"Pa", {kPressure, 1.0}},
        {"kPa", {kPressure, 1.0e3}},
        {"MPa", {kPressure, 1.0e6}},
        {"GPa", {kPressure, 1.0e9}},
        {"N", {{1, 1, -2, 0}, 1.0}},
    };
    const auto iterator = units.find(name);
    return iterator == units.end() ? std::nullopt : std::optional<Unit>{iterator->second};
}

[[nodiscard]] std::uint64_t fnv1a(const std::string_view value) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash == 0u ? 1u : hash;
}

class Parser {
public:
    Parser(
        std::string_view source,
        std::vector<Token> tokens,
        std::vector<Diagnostic> diagnostics
    ) : source_(source), tokens_(std::move(tokens)), diagnostics_(std::move(diagnostics)) {}

    [[nodiscard]] ParseResult run() {
        MaterialProgram material;
        expectIdentifier("material");
        material.name = identifier("material name");
        expect(TokenKind::leftBrace, "'{' after material name");
        while (!at(TokenKind::rightBrace) && !at(TokenKind::end)) {
            if (matchIdentifier("parameter")) {
                parseParameter(material);
            } else if (matchIdentifier("state")) {
                parseState(material);
            } else if (matchIdentifier("model")) {
                parseModel(material);
            } else if (matchIdentifier("energy")) {
                parseRoot(material, material.energyRoot, kPressure, "energy");
            } else if (matchIdentifier("dissipation")) {
                parseRoot(material, material.dissipationRoot, kPressure, "dissipation");
            } else if (matchIdentifier("valid")) {
                parseRoot(material, material.validityRoot, kDimensionless, "validity");
            } else if (matchIdentifier("supports")) {
                parseSupports(material);
            } else if (matchIdentifier("interface")) {
                parseInterface(material);
            } else if (matchIdentifier("limits")) {
                parseLimits(material);
            } else {
                error(peek(), "expected parameter, state, model, energy, dissipation, valid, supports, interface or limits");
                synchronize();
            }
        }
        expect(TokenKind::rightBrace, "'}' after material body");
        if (material.energyRoot == NM_INVALID_INDEX) {
            error(peek(), "material requires an energy expression");
        }
        if (material.supportedRepresentations.empty()) {
            material.supportedRepresentations = {Representation::mpm, Representation::fem};
        }
        const auto density = parameterIndex(material, "density");
        if (!density.has_value()) {
            diagnostics_.push_back({
                Diagnostic::Severity::error, 1u, 1u,
                "material requires a density parameter",
            });
        } else if (material.parameters[*density].dimension != kDensity) {
            diagnostics_.push_back({
                Diagnostic::Severity::error, 1u, 1u,
                "density parameter must have kg/m^3 dimensions",
            });
        }
        material.fingerprint = fnv1a(source_);
        return {std::move(material), std::move(diagnostics_)};
    }

private:
    [[nodiscard]] const Token& peek(const std::size_t lookahead = 0u) const {
        return tokens_[std::min(offset_ + lookahead, tokens_.size() - 1u)];
    }

    [[nodiscard]] bool at(const TokenKind kind) const { return peek().kind == kind; }

    [[nodiscard]] bool match(const TokenKind kind) {
        if (!at(kind)) {
            return false;
        }
        ++offset_;
        return true;
    }

    [[nodiscard]] bool matchIdentifier(const std::string_view value) {
        if (peek().kind != TokenKind::identifier || peek().text != value) {
            return false;
        }
        ++offset_;
        return true;
    }

    void expect(const TokenKind kind, const std::string_view message) {
        if (!match(kind)) {
            error(peek(), std::string("expected ") + std::string(message));
        }
    }

    void expectIdentifier(const std::string_view value) {
        if (!matchIdentifier(value)) {
            error(peek(), "expected '" + std::string(value) + "'");
        }
    }

    [[nodiscard]] std::string identifier(const std::string_view role) {
        if (peek().kind != TokenKind::identifier) {
            error(peek(), "expected " + std::string(role));
            return {};
        }
        return tokens_[offset_++].text;
    }

    void error(const Token& token, std::string message) {
        diagnostics_.push_back({
            Diagnostic::Severity::error,
            token.line,
            token.column,
            std::move(message),
        });
    }

    void synchronize() {
        while (!at(TokenKind::end) && !at(TokenKind::rightBrace)) {
            if (match(TokenKind::semicolon)) {
                return;
            }
            ++offset_;
        }
    }

    [[nodiscard]] std::optional<std::uint32_t> parameterIndex(
        const MaterialProgram& material,
        const std::string_view name
    ) const {
        for (std::uint32_t index = 0u; index < material.parameters.size(); ++index) {
            if (material.parameters[index].name == name) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::uint32_t> stateIndex(
        const MaterialProgram& material,
        const std::string_view name
    ) const {
        for (std::uint32_t index = 0u; index < material.internalState.size(); ++index) {
            if (material.internalState[index].name == name) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] Unit parseUnit() {
        if (match(TokenKind::number)) {
            return {kDimensionless, 1.0};
        }
        if (peek().kind != TokenKind::identifier) {
            error(peek(), "expected a physical unit");
            return {};
        }
        Unit result{};
        bool divide = false;
        bool first = true;
        while (peek().kind == TokenKind::identifier) {
            const Token token = tokens_[offset_++];
            const auto base = baseUnit(token.text);
            if (!base.has_value()) {
                error(token, "unknown unit '" + token.text + "'");
                return {};
            }
            int exponent = 1;
            if (match(TokenKind::caret)) {
                bool negative = match(TokenKind::minus);
                if (peek().kind != TokenKind::number || std::floor(peek().number) != peek().number) {
                    error(peek(), "unit exponent must be an integer");
                } else {
                    exponent = static_cast<int>(tokens_[offset_++].number);
                    if (negative) {
                        exponent = -exponent;
                    }
                }
            }
            if (divide) {
                exponent = -exponent;
            }
            result.dimension = result.dimension + base->dimension * exponent;
            result.scale *= std::pow(base->scale, exponent);
            first = false;
            if (match(TokenKind::star)) {
                divide = false;
                continue;
            }
            if (match(TokenKind::slash)) {
                divide = true;
                continue;
            }
            break;
        }
        if (first) {
            error(peek(), "empty unit");
        }
        return result;
    }

    [[nodiscard]] double signedNumber(const std::string_view role) {
        const bool negative = match(TokenKind::minus);
        const bool positive = !negative && match(TokenKind::plus);
        (void)positive;
        if (peek().kind != TokenKind::number) {
            error(peek(), "expected numeric " + std::string(role));
            return 0.0;
        }
        const double result = tokens_[offset_++].number;
        return negative ? -result : result;
    }

    void parseParameter(MaterialProgram& material) {
        Parameter parameter;
        parameter.name = identifier("parameter name");
        if (parameter.name.empty()) {
            synchronize();
            return;
        }
        if (parameterIndex(material, parameter.name).has_value()) {
            error(peek(), "duplicate parameter '" + parameter.name + "'");
        }
        expect(TokenKind::colon, "':' after parameter name");
        const Unit unit = parseUnit();
        parameter.dimension = unit.dimension;
        expect(TokenKind::equal, "'=' after parameter unit");
        parameter.defaultValue = signedNumber("parameter value") * unit.scale;
        parameter.lower = parameter.defaultValue;
        parameter.upper = parameter.defaultValue;
        if (matchIdentifier("in")) {
            expect(TokenKind::leftBracket, "'[' before parameter range");
            parameter.lower = signedNumber("lower bound") * unit.scale;
            expect(TokenKind::comma, "',' in parameter range");
            parameter.upper = signedNumber("upper bound") * unit.scale;
            expect(TokenKind::rightBracket, "']' after parameter range");
        }
        while (peek().kind == TokenKind::identifier) {
            if (matchIdentifier("log")) {
                parameter.logarithmic = true;
            } else if (matchIdentifier("identifiable")) {
                parameter.identifiable = true;
            } else {
                break;
            }
        }
        expect(TokenKind::semicolon, "';' after parameter declaration");
        if (!(std::isfinite(parameter.defaultValue) &&
              std::isfinite(parameter.lower) &&
              std::isfinite(parameter.upper) &&
              parameter.lower <= parameter.defaultValue &&
              parameter.defaultValue <= parameter.upper)) {
            error(peek(), "parameter value must lie within finite ordered bounds");
        }
        if (parameter.logarithmic && !(parameter.lower > 0.0)) {
            error(peek(), "log-scaled parameter requires a positive lower bound");
        }
        material.parameters.push_back(std::move(parameter));
    }

    void parseState(MaterialProgram& material) {
        InternalState state;
        state.name = identifier("state name");
        expect(TokenKind::colon, "':' after state name");
        const Unit unit = parseUnit();
        state.dimension = unit.dimension;
        expect(TokenKind::equal, "'=' after state unit");
        state.initialValue = signedNumber("state value") * unit.scale;
        expect(TokenKind::semicolon, "';' after state declaration");
        if (stateIndex(material, state.name).has_value()) {
            error(peek(), "duplicate state '" + state.name + "'");
        }
        material.internalState.push_back(std::move(state));
    }

    void parseModel(MaterialProgram& material) {
        const std::string model = identifier("constitutive model");
        if (model == "generic") material.hint = ConstitutiveHint::generic;
        else if (model == "neo_hookean") material.hint = ConstitutiveHint::neoHookean;
        else if (model == "corotated") material.hint = ConstitutiveHint::corotated;
        else if (model == "drucker_prager") material.hint = ConstitutiveHint::druckerPrager;
        else if (model == "von_mises") material.hint = ConstitutiveHint::vonMises;
        else if (model == "visco_hyperelastic") material.hint = ConstitutiveHint::viscoHyperelastic;
        else error(peek(), "unknown constitutive model '" + model + "'");
        expect(TokenKind::semicolon, "';' after model");
    }

    [[nodiscard]] std::uint32_t binary(
        MaterialProgram& material,
        const ExprKind kind,
        const std::uint32_t left,
        const std::uint32_t right,
        const Token& token
    ) {
        if (left == NM_INVALID_INDEX || right == NM_INVALID_INDEX) {
            return NM_INVALID_INDEX;
        }
        const Dimension leftDimension = material.expressions.nodes[left].dimension;
        const Dimension rightDimension = material.expressions.nodes[right].dimension;
        Dimension dimension{};
        if (kind == ExprKind::add || kind == ExprKind::subtract ||
            kind == ExprKind::minimum || kind == ExprKind::maximum) {
            if (leftDimension != rightDimension) {
                error(token, "dimension mismatch: " + dimensionName(leftDimension) +
                    " versus " + dimensionName(rightDimension));
            }
            dimension = leftDimension;
        } else if (kind == ExprKind::multiply) {
            dimension = leftDimension + rightDimension;
        } else if (kind == ExprKind::divide) {
            dimension = leftDimension - rightDimension;
        }
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = left;
        expression.arguments[1] = right;
        return material.expressions.append(expression);
    }

    [[nodiscard]] std::uint32_t fComponent(MaterialProgram& material, const std::uint32_t index) {
        Expr expression;
        expression.kind = ExprKind::deformation;
        expression.dimension = kDimensionless;
        expression.index = index;
        return material.expressions.append(expression);
    }

    [[nodiscard]] std::uint32_t invariantI1(MaterialProgram& material) {
        std::uint32_t result = material.expressions.constant(0.0);
        const Token synthetic{};
        for (std::uint32_t index = 0u; index < 9u; ++index) {
            const std::uint32_t value = fComponent(material, index);
            result = binary(material, ExprKind::add, result,
                binary(material, ExprKind::multiply, value, value, synthetic), synthetic);
        }
        return result;
    }

    [[nodiscard]] std::uint32_t determinant(MaterialProgram& material) {
        const Token synthetic{};
        const auto f = [&](const std::uint32_t row, const std::uint32_t column) {
            return fComponent(material, 3u * row + column);
        };
        const auto mul = [&](const std::uint32_t a, const std::uint32_t b) {
            return binary(material, ExprKind::multiply, a, b, synthetic);
        };
        const auto sub = [&](const std::uint32_t a, const std::uint32_t b) {
            return binary(material, ExprKind::subtract, a, b, synthetic);
        };
        const auto add = [&](const std::uint32_t a, const std::uint32_t b) {
            return binary(material, ExprKind::add, a, b, synthetic);
        };
        const std::uint32_t a = mul(f(0, 0), sub(mul(f(1, 1), f(2, 2)), mul(f(1, 2), f(2, 1))));
        const std::uint32_t b = mul(f(0, 1), sub(mul(f(1, 0), f(2, 2)), mul(f(1, 2), f(2, 0))));
        const std::uint32_t c = mul(f(0, 2), sub(mul(f(1, 0), f(2, 1)), mul(f(1, 1), f(2, 0))));
        return add(sub(a, b), c);
    }

    [[nodiscard]] std::uint32_t unary(
        MaterialProgram& material,
        const ExprKind kind,
        const std::uint32_t argument,
        const Token& token
    ) {
        if (argument == NM_INVALID_INDEX) {
            return argument;
        }
        Dimension dimension = material.expressions.nodes[argument].dimension;
        if (kind == ExprKind::logarithm || kind == ExprKind::exponential) {
            if (dimension != kDimensionless) {
                error(token, "log/exp argument must be dimensionless");
            }
            dimension = kDimensionless;
        } else if (kind == ExprKind::squareRoot) {
            if ((dimension.length & 1) || (dimension.mass & 1) ||
                (dimension.time & 1) || (dimension.temperature & 1)) {
                error(token, "sqrt argument has odd physical exponents");
            }
            dimension = {
                static_cast<std::int8_t>(dimension.length / 2),
                static_cast<std::int8_t>(dimension.mass / 2),
                static_cast<std::int8_t>(dimension.time / 2),
                static_cast<std::int8_t>(dimension.temperature / 2),
            };
        }
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = argument;
        return material.expressions.append(expression);
    }

    [[nodiscard]] std::uint32_t primary(MaterialProgram& material) {
        if (match(TokenKind::leftParen)) {
            const std::uint32_t result = expression(material);
            expect(TokenKind::rightParen, "')'");
            return result;
        }
        if (peek().kind == TokenKind::number) {
            const double value = tokens_[offset_++].number;
            return material.expressions.constant(value);
        }
        if (peek().kind != TokenKind::identifier) {
            error(peek(), "expected expression");
            return material.expressions.constant(0.0);
        }
        const Token token = tokens_[offset_++];
        if (!at(TokenKind::leftParen)) {
            if (const auto parameter = parameterIndex(material, token.text); parameter.has_value()) {
                Expr result;
                result.kind = ExprKind::parameter;
                result.dimension = material.parameters[*parameter].dimension;
                result.index = *parameter;
                return material.expressions.append(result);
            }
            if (const auto state = stateIndex(material, token.text); state.has_value()) {
                Expr result;
                result.kind = ExprKind::internalState;
                result.dimension = material.internalState[*state].dimension;
                result.index = *state;
                return material.expressions.append(result);
            }
            error(token, "unknown symbol '" + token.text + "'");
            return material.expressions.constant(0.0);
        }
        expect(TokenKind::leftParen, "'('");
        if (token.text == "I1" || token.text == "traceC") {
            expect(TokenKind::rightParen, "')'");
            return invariantI1(material);
        }
        if (token.text == "J") {
            expect(TokenKind::rightParen, "')'");
            return determinant(material);
        }
        if (token.text == "F") {
            const int row = static_cast<int>(signedNumber("F row"));
            expect(TokenKind::comma, "',' in F(row,column)");
            const int column = static_cast<int>(signedNumber("F column"));
            expect(TokenKind::rightParen, "')'");
            if (row < 0 || row > 2 || column < 0 || column > 2) {
                error(token, "F indices must lie in [0, 2]");
                return material.expressions.constant(0.0);
            }
            return fComponent(material, static_cast<std::uint32_t>(3 * row + column));
        }
        if (token.text == "neo_hookean") {
            const std::uint32_t mu = expression(material);
            expect(TokenKind::comma, "',' after shear modulus");
            const std::uint32_t lambda = expression(material);
            expect(TokenKind::rightParen, "')'");
            const Token synthetic{};
            const std::uint32_t i1 = invariantI1(material);
            const std::uint32_t three = material.expressions.constant(3.0);
            const std::uint32_t half = material.expressions.constant(0.5);
            const std::uint32_t j = determinant(material);
            const std::uint32_t logJ = unary(material, ExprKind::logarithm, j, token);
            const std::uint32_t distort = binary(material, ExprKind::subtract, i1, three, synthetic);
            const std::uint32_t first = binary(material, ExprKind::multiply, half,
                binary(material, ExprKind::multiply, mu, distort, synthetic), synthetic);
            const std::uint32_t second = binary(material, ExprKind::multiply, mu, logJ, synthetic);
            const std::uint32_t logSquared = binary(material, ExprKind::multiply, logJ, logJ, synthetic);
            const std::uint32_t third = binary(material, ExprKind::multiply, half,
                binary(material, ExprKind::multiply, lambda, logSquared, synthetic), synthetic);
            return binary(material, ExprKind::add,
                binary(material, ExprKind::subtract, first, second, synthetic), third, synthetic);
        }
        const std::uint32_t first = expression(material);
        if (token.text == "log" || token.text == "exp" || token.text == "sqrt" || token.text == "abs") {
            expect(TokenKind::rightParen, "')'");
            const ExprKind kind = token.text == "log" ? ExprKind::logarithm :
                token.text == "exp" ? ExprKind::exponential :
                token.text == "sqrt" ? ExprKind::squareRoot : ExprKind::absolute;
            return unary(material, kind, first, token);
        }
        if (token.text == "pow") {
            expect(TokenKind::comma, "',' in pow");
            const int exponent = static_cast<int>(signedNumber("integer exponent"));
            expect(TokenKind::rightParen, "')'");
            Expr result;
            result.kind = ExprKind::integerPower;
            result.dimension = material.expressions.nodes[first].dimension * exponent;
            result.integer = exponent;
            result.arguments[0] = first;
            return material.expressions.append(result);
        }
        const std::uint32_t second = [&]() {
            expect(TokenKind::comma, "',' in function");
            return expression(material);
        }();
        if (token.text == "min" || token.text == "max") {
            expect(TokenKind::rightParen, "')'");
            return binary(material,
                token.text == "min" ? ExprKind::minimum : ExprKind::maximum,
                first, second, token);
        }
        if (token.text == "clamp") {
            expect(TokenKind::comma, "',' before clamp upper bound");
            const std::uint32_t third = expression(material);
            expect(TokenKind::rightParen, "')'");
            const Dimension dimension = material.expressions.nodes[first].dimension;
            if (material.expressions.nodes[second].dimension != dimension ||
                material.expressions.nodes[third].dimension != dimension) {
                error(token, "clamp operands require identical dimensions");
            }
            Expr result;
            result.kind = ExprKind::clamp;
            result.dimension = dimension;
            result.arguments = {first, second, third};
            return material.expressions.append(result);
        }
        error(token, "unknown function '" + token.text + "'");
        while (!at(TokenKind::rightParen) && !at(TokenKind::end)) {
            ++offset_;
        }
        (void)match(TokenKind::rightParen);
        return first;
    }

    [[nodiscard]] std::uint32_t unaryExpression(MaterialProgram& material) {
        if (match(TokenKind::minus)) {
            return unary(material, ExprKind::negate, unaryExpression(material), peek());
        }
        if (match(TokenKind::plus)) {
            return unaryExpression(material);
        }
        return primary(material);
    }

    [[nodiscard]] std::uint32_t product(MaterialProgram& material) {
        std::uint32_t left = unaryExpression(material);
        while (at(TokenKind::star) || at(TokenKind::slash)) {
            const Token operation = tokens_[offset_++];
            const std::uint32_t right = unaryExpression(material);
            left = binary(material,
                operation.kind == TokenKind::star ? ExprKind::multiply : ExprKind::divide,
                left, right, operation);
        }
        return left;
    }

    [[nodiscard]] std::uint32_t expression(MaterialProgram& material) {
        std::uint32_t left = product(material);
        while (at(TokenKind::plus) || at(TokenKind::minus)) {
            const Token operation = tokens_[offset_++];
            const std::uint32_t right = product(material);
            left = binary(material,
                operation.kind == TokenKind::plus ? ExprKind::add : ExprKind::subtract,
                left, right, operation);
        }
        return left;
    }

    void parseRoot(
        MaterialProgram& material,
        std::uint32_t& root,
        const Dimension expectedDimension,
        const std::string_view label
    ) {
        expect(TokenKind::equal, "'=' after " + std::string(label));
        root = expression(material);
        expect(TokenKind::semicolon, "';' after " + std::string(label));
        if (root != NM_INVALID_INDEX &&
            material.expressions.nodes[root].dimension != expectedDimension) {
            error(peek(), std::string(label) + " expression has " +
                dimensionName(material.expressions.nodes[root].dimension) +
                " dimensions; expected " + dimensionName(expectedDimension));
        }
    }

    void parseSupports(MaterialProgram& material) {
        do {
            const std::string value = identifier("representation");
            if (value == "rigid") material.supportedRepresentations.push_back(Representation::rigid);
            else if (value == "mpm") material.supportedRepresentations.push_back(Representation::mpm);
            else if (value == "fem") material.supportedRepresentations.push_back(Representation::fem);
            else error(peek(), "unknown representation '" + value + "'");
        } while (match(TokenKind::comma));
        expect(TokenKind::semicolon, "';' after supports");
        std::ranges::sort(material.supportedRepresentations);
        material.supportedRepresentations.erase(
            std::unique(material.supportedRepresentations.begin(), material.supportedRepresentations.end()),
            material.supportedRepresentations.end()
        );
    }

    void parseInterface(MaterialProgram& material) {
        expect(TokenKind::leftBrace, "'{' after interface");
        while (!at(TokenKind::rightBrace) && !at(TokenKind::end)) {
            const Token key = peek();
            const std::string name = identifier("interface property");
            expect(TokenKind::equal, "'=' after interface property");
            const double value = signedNumber("interface value");
            expect(TokenKind::semicolon, "';' after interface property");
            if (name == "static_friction") material.staticFriction = value;
            else if (name == "dynamic_friction") material.dynamicFriction = value;
            else if (name == "restitution") material.restitution = value;
            else if (name == "adhesion") material.adhesion = value;
            else error(key, "unknown interface property '" + name + "'");
        }
        expect(TokenKind::rightBrace, "'}' after interface");
        if (material.staticFriction < 0.0 || material.dynamicFriction < 0.0 ||
            material.restitution < 0.0 || material.restitution > 1.0) {
            error(peek(), "interface friction must be nonnegative and restitution in [0,1]");
        }
    }

    void parseLimits(MaterialProgram& material) {
        expect(TokenKind::leftBrace, "'{' after limits");
        while (!at(TokenKind::rightBrace) && !at(TokenKind::end)) {
            const Token key = peek();
            const std::string name = identifier("limit property");
            expect(TokenKind::equal, "'=' after limit property");
            const double value = signedNumber("limit value");
            expect(TokenKind::semicolon, "';' after limit property");
            if (name == "minimum_J") material.minimumDeterminant = value;
            else if (name == "maximum_J") material.maximumDeterminant = value;
            else if (name == "maximum_stress") material.maximumStress = value;
            else if (name == "maximum_energy_density") material.maximumEnergyDensity = value;
            else error(key, "unknown material limit '" + name + "'");
        }
        expect(TokenKind::rightBrace, "'}' after limits");
    }

    std::string_view source_;
    std::vector<Token> tokens_;
    std::vector<Diagnostic> diagnostics_;
    std::size_t offset_ = 0u;
};

} // namespace

ParseResult parseMatter(const std::string_view source) {
    std::vector<Diagnostic> diagnostics;
    Lexer lexer(source);
    auto tokens = lexer.run(diagnostics);
    Parser parser(source, std::move(tokens), std::move(diagnostics));
    return parser.run();
}

ParseResult parseMatterFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        ParseResult result;
        result.diagnostics.push_back({
            Diagnostic::Severity::error, 0u, 0u,
            "cannot open Matter Language file: " + path.string(),
        });
        return result;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    return parseMatter(contents.str());
}

} // namespace numi::matter
