#include "numi/matter/language.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace numi::matter {
namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

void hashBytes(std::uint64_t& hash, const void* bytes, const std::size_t count) {
    const auto* data = static_cast<const unsigned char*>(bytes);
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= data[index];
        hash *= kFNVPrime;
    }
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(value));
}

void hashString(std::uint64_t& hash, const std::string_view value) {
    hashBytes(hash, value.data(), value.size());
    const unsigned char terminator = 0xffu;
    hashBytes(hash, &terminator, 1u);
}

enum class TokenKind : std::uint8_t {
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
    end,
    invalid,
};

struct Token {
    TokenKind kind = TokenKind::invalid;
    std::string_view text;
    std::size_t line = 1u;
    std::size_t column = 1u;
};

class Lexer {
public:
    explicit Lexer(const std::string_view source) : source_(source) {}

    [[nodiscard]] std::vector<Token> scan() {
        std::vector<Token> tokens;
        while (!atEnd()) {
            skipWhitespaceAndComments();
            if (atEnd()) break;
            const std::size_t begin = cursor_;
            const std::size_t line = line_;
            const std::size_t column = column_;
            const char current = advance();
            if (isIdentifierStart(current)) {
                while (!atEnd() && isIdentifierPart(peek())) advance();
                tokens.push_back({
                    TokenKind::identifier,
                    source_.substr(begin, cursor_ - begin),
                    line,
                    column,
                });
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(current)) ||
                (current == '.' && !atEnd() &&
                 std::isdigit(static_cast<unsigned char>(peek())))) {
                scanNumber();
                tokens.push_back({
                    TokenKind::number,
                    source_.substr(begin, cursor_ - begin),
                    line,
                    column,
                });
                continue;
            }
            TokenKind kind = TokenKind::invalid;
            switch (current) {
            case '{': kind = TokenKind::leftBrace; break;
            case '}': kind = TokenKind::rightBrace; break;
            case '(': kind = TokenKind::leftParen; break;
            case ')': kind = TokenKind::rightParen; break;
            case '[': kind = TokenKind::leftBracket; break;
            case ']': kind = TokenKind::rightBracket; break;
            case ':': kind = TokenKind::colon; break;
            case ';': kind = TokenKind::semicolon; break;
            case ',': kind = TokenKind::comma; break;
            case '=': kind = TokenKind::equal; break;
            case '+': kind = TokenKind::plus; break;
            case '-': kind = TokenKind::minus; break;
            case '*': kind = TokenKind::star; break;
            case '/': kind = TokenKind::slash; break;
            case '^': kind = TokenKind::caret; break;
            default: kind = TokenKind::invalid; break;
            }
            tokens.push_back({
                kind,
                source_.substr(begin, 1u),
                line,
                column,
            });
        }
        tokens.push_back({TokenKind::end, {}, line_, column_});
        return tokens;
    }

private:
    [[nodiscard]] bool atEnd() const noexcept { return cursor_ >= source_.size(); }
    [[nodiscard]] char peek() const noexcept { return atEnd() ? '\0' : source_[cursor_]; }
    [[nodiscard]] char peekNext() const noexcept {
        return cursor_ + 1u >= source_.size() ? '\0' : source_[cursor_ + 1u];
    }

    char advance() noexcept {
        const char value = source_[cursor_++];
        if (value == '\n') {
            ++line_;
            column_ = 1u;
        } else {
            ++column_;
        }
        return value;
    }

    static bool isIdentifierStart(const char value) noexcept {
        return std::isalpha(static_cast<unsigned char>(value)) || value == '_';
    }
    static bool isIdentifierPart(const char value) noexcept {
        return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
    }

    void skipWhitespaceAndComments() {
        for (;;) {
            while (!atEnd() && std::isspace(static_cast<unsigned char>(peek()))) {
                advance();
            }
            if (peek() == '/' && peekNext() == '/') {
                while (!atEnd() && peek() != '\n') advance();
                continue;
            }
            if (peek() == '#') {
                while (!atEnd() && peek() != '\n') advance();
                continue;
            }
            break;
        }
    }

    void scanNumber() {
        while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) advance();
        if (peek() == '.' &&
            std::isdigit(static_cast<unsigned char>(peekNext()))) {
            advance();
            while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) advance();
        }
        if (peek() == 'e' || peek() == 'E') {
            const std::size_t savedCursor = cursor_;
            const std::size_t savedColumn = column_;
            advance();
            if (peek() == '+' || peek() == '-') advance();
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                cursor_ = savedCursor;
                column_ = savedColumn;
                return;
            }
            while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) advance();
        }
    }

    std::string_view source_;
    std::size_t cursor_ = 0u;
    std::size_t line_ = 1u;
    std::size_t column_ = 1u;
};

struct Unit {
    double scale = 1.0;
    Dimension dimension{};
};

std::optional<Unit> namedUnit(const std::string_view name) {
    if (name == "1") return Unit{};
    if (name == "m") return Unit{1.0, lengthDimension};
    if (name == "mm") return Unit{1.0e-3, lengthDimension};
    if (name == "cm") return Unit{1.0e-2, lengthDimension};
    if (name == "kg") return Unit{1.0, massDimension};
    if (name == "g") return Unit{1.0e-3, massDimension};
    if (name == "s") return Unit{1.0, timeDimension};
    if (name == "ms") return Unit{1.0e-3, timeDimension};
    if (name == "K") return Unit{1.0, temperatureDimension};
    if (name == "Pa") return Unit{1.0, pressureDimension};
    if (name == "kPa") return Unit{1.0e3, pressureDimension};
    if (name == "MPa") return Unit{1.0e6, pressureDimension};
    if (name == "GPa") return Unit{1.0e9, pressureDimension};
    if (name == "N") return Unit{1.0, {1, 1, -2, 0}};
    return std::nullopt;
}

class Parser {
public:
    explicit Parser(const std::string_view source)
        : tokens_(Lexer{source}.scan()) {}

    [[nodiscard]] ParseResult parse() {
        ParseResult result;
        output_ = &result;
        if (!matchIdentifier("material")) {
            error(current(), "expected 'material'");
            return result;
        }
        if (current().kind != TokenKind::identifier) {
            error(current(), "expected material name");
            return result;
        }
        result.material.name = std::string{advance().text};
        expect(TokenKind::leftBrace, "expected '{' after material name");
        while (!atEnd() && current().kind != TokenKind::rightBrace) {
            parseStatement(result.material);
        }
        expect(TokenKind::rightBrace, "expected '}' after material body");
        if (!atEnd()) {
            error(current(), "unexpected tokens after material body");
        }
        validateMaterial(result.material);
        result.material.fingerprint = materialFingerprint(result.material);
        return result;
    }

private:
    [[nodiscard]] const Token& current() const noexcept { return tokens_[cursor_]; }
    [[nodiscard]] const Token& previous() const noexcept { return tokens_[cursor_ - 1u]; }
    [[nodiscard]] bool atEnd() const noexcept { return current().kind == TokenKind::end; }
    const Token& advance() noexcept {
        if (!atEnd()) ++cursor_;
        return previous();
    }
    [[nodiscard]] bool check(const TokenKind kind) const noexcept {
        return current().kind == kind;
    }
    bool match(const TokenKind kind) noexcept {
        if (!check(kind)) return false;
        advance();
        return true;
    }
    bool matchIdentifier(const std::string_view value) noexcept {
        if (current().kind != TokenKind::identifier || current().text != value) return false;
        advance();
        return true;
    }

    void error(const Token& token, std::string message) {
        output_->diagnostics.push_back({
            Diagnostic::Severity::error,
            token.line,
            token.column,
            std::move(message),
        });
    }

    bool expect(const TokenKind kind, const std::string_view message) {
        if (match(kind)) return true;
        error(current(), std::string{message});
        recover();
        return false;
    }

    void recover() {
        while (!atEnd() && current().kind != TokenKind::semicolon &&
               current().kind != TokenKind::rightBrace) {
            advance();
        }
        if (current().kind == TokenKind::semicolon) advance();
    }

    void parseStatement(MaterialProgram& material) {
        if (matchIdentifier("parameter")) {
            parseParameter(material);
        } else if (matchIdentifier("state")) {
            parseState(material);
        } else if (matchIdentifier("model")) {
            parseModel(material);
        } else if (matchIdentifier("energy")) {
            parseExpressionAssignment(material.energyRoot, material, pressureDimension);
        } else if (matchIdentifier("dissipation")) {
            parseExpressionAssignment(
                material.dissipationRoot,
                material,
                pressureDimension - timeDimension
            );
        } else if (matchIdentifier("valid")) {
            parseExpressionAssignment(material.validityRoot, material, dimensionless);
        } else if (matchIdentifier("supports")) {
            parseSupports(material);
        } else if (matchIdentifier("interface")) {
            parseInterface(material);
        } else if (matchIdentifier("limits")) {
            parseLimits(material);
        } else {
            error(current(), "unknown material statement");
            recover();
        }
    }

    void parseParameter(MaterialProgram& material) {
        if (current().kind != TokenKind::identifier) {
            error(current(), "expected parameter name");
            recover();
            return;
        }
        Parameter parameter;
        parameter.name = std::string{advance().text};
        if (std::ranges::any_of(material.parameters, [&](const Parameter& value) {
                return value.name == parameter.name;
            })) {
            error(previous(), "duplicate parameter '" + parameter.name + "'");
        }
        if (!expect(TokenKind::colon, "expected ':' after parameter name")) return;
        const std::optional<Unit> unit = parseUnit();
        if (!unit) {
            recover();
            return;
        }
        parameter.dimension = unit->dimension;
        if (!expect(TokenKind::equal, "expected '=' after parameter unit")) return;
        const auto defaultValue = parseSignedNumber();
        if (!defaultValue) { recover(); return; }
        parameter.defaultValue = *defaultValue * unit->scale;
        if (!matchIdentifier("in")) {
            error(current(), "expected 'in [lower, upper]' parameter domain");
            recover();
            return;
        }
        if (!expect(TokenKind::leftBracket, "expected '[' before parameter bounds")) return;
        const auto lower = parseSignedNumber();
        if (!lower || !expect(TokenKind::comma, "expected ',' in parameter bounds")) return;
        const auto upper = parseSignedNumber();
        if (!upper || !expect(TokenKind::rightBracket, "expected ']' after parameter bounds")) return;
        parameter.lower = *lower * unit->scale;
        parameter.upper = *upper * unit->scale;
        while (!check(TokenKind::semicolon) && !atEnd()) {
            if (matchIdentifier("log")) {
                parameter.logarithmic = true;
            } else if (matchIdentifier("identifiable")) {
                parameter.identifiable = true;
            } else if (matchIdentifier("sigma")) {
                const auto sigma = parseSignedNumber();
                if (!sigma) break;
                parameter.proposalSigma = *sigma;
            } else {
                error(current(), "unknown parameter modifier");
                advance();
            }
        }
        expect(TokenKind::semicolon, "expected ';' after parameter");
        if (!(parameter.lower <= parameter.defaultValue &&
              parameter.defaultValue <= parameter.upper) ||
            !(parameter.lower < parameter.upper) ||
            !std::isfinite(parameter.defaultValue) ||
            !std::isfinite(parameter.lower) ||
            !std::isfinite(parameter.upper) ||
            !(parameter.proposalSigma > 0.0)) {
            error(previous(), "invalid parameter range or default value");
        }
        if (parameter.logarithmic && parameter.lower <= 0.0) {
            error(previous(), "log-space parameter lower bound must be positive");
        }
        material.parameters.push_back(std::move(parameter));
    }

    void parseState(MaterialProgram& material) {
        if (current().kind != TokenKind::identifier) {
            error(current(), "expected state name");
            recover();
            return;
        }
        InternalState state;
        state.name = std::string{advance().text};
        if (!expect(TokenKind::colon, "expected ':' after state name")) return;
        const std::optional<Unit> unit = parseUnit();
        if (!unit || !expect(TokenKind::equal, "expected '=' after state unit")) return;
        const auto initial = parseSignedNumber();
        if (!initial) { recover(); return; }
        state.dimension = unit->dimension;
        state.initialValue = *initial * unit->scale;
        expect(TokenKind::semicolon, "expected ';' after state");
        material.internalState.push_back(std::move(state));
    }

    void parseModel(MaterialProgram& material) {
        if (current().kind != TokenKind::identifier) {
            error(current(), "expected constitutive model hint");
            recover();
            return;
        }
        const std::string_view value = advance().text;
        if (value == "generic") material.hint = ConstitutiveHint::generic;
        else if (value == "neo_hookean") material.hint = ConstitutiveHint::neoHookean;
        else if (value == "corotated") material.hint = ConstitutiveHint::corotated;
        else if (value == "hencky") material.hint = ConstitutiveHint::hencky;
        else if (value == "drucker_prager") material.hint = ConstitutiveHint::druckerPrager;
        else if (value == "von_mises") material.hint = ConstitutiveHint::vonMises;
        else if (value == "newtonian") material.hint = ConstitutiveHint::newtonian;
        else if (value == "visco_hyperelastic") material.hint = ConstitutiveHint::viscoHyperelastic;
        else error(previous(), "unknown constitutive model hint");
        expect(TokenKind::semicolon, "expected ';' after model hint");
    }

    void parseExpressionAssignment(
        std::uint32_t& target,
        MaterialProgram& material,
        const Dimension expected
    ) {
        if (!expect(TokenKind::equal, "expected '=' before expression")) return;
        const std::uint32_t root = parseExpression(material, 0);
        expect(TokenKind::semicolon, "expected ';' after expression");
        if (root != NM_INVALID_INDEX) {
            if (!(material.expressions.nodes[root].dimension == expected)) {
                error(previous(), "expression has dimension " +
                    dimensionName(material.expressions.nodes[root].dimension) +
                    ", expected " + dimensionName(expected));
            } else {
                target = root;
            }
        }
    }

    void parseSupports(MaterialProgram& material) {
        for (;;) {
            if (current().kind != TokenKind::identifier) {
                error(current(), "expected representation name");
                recover();
                return;
            }
            const std::string_view value = advance().text;
            std::optional<Representation> representation;
            if (value == "rigid") representation = Representation::rigid;
            else if (value == "mpm") representation = Representation::mpm;
            else if (value == "fem") representation = Representation::fem;
            else if (value == "rod") representation = Representation::rod;
            else if (value == "surface") representation = Representation::surface;
            else error(previous(), "unknown representation '" + std::string{value} + "'");
            if (representation &&
                std::ranges::find(material.supportedRepresentations, *representation) ==
                    material.supportedRepresentations.end()) {
                material.supportedRepresentations.push_back(*representation);
            }
            if (!match(TokenKind::comma)) break;
        }
        expect(TokenKind::semicolon, "expected ';' after supports list");
    }

    void parseInterface(MaterialProgram& material) {
        if (!expect(TokenKind::leftBrace, "expected '{' after interface")) return;
        while (!atEnd() && !check(TokenKind::rightBrace)) {
            if (current().kind != TokenKind::identifier) {
                error(current(), "expected interface field");
                recover();
                continue;
            }
            const std::string field{advance().text};
            if (!expect(TokenKind::equal, "expected '=' after interface field")) continue;
            const auto value = parseSignedNumber();
            expect(TokenKind::semicolon, "expected ';' after interface field");
            if (!value) continue;
            if (field == "static_friction") material.staticFriction = *value;
            else if (field == "dynamic_friction") material.dynamicFriction = *value;
            else if (field == "restitution") material.restitution = *value;
            else if (field == "adhesion") material.adhesion = *value;
            else error(previous(), "unknown interface field '" + field + "'");
        }
        expect(TokenKind::rightBrace, "expected '}' after interface block");
    }

    void parseLimits(MaterialProgram& material) {
        if (!expect(TokenKind::leftBrace, "expected '{' after limits")) return;
        while (!atEnd() && !check(TokenKind::rightBrace)) {
            if (current().kind != TokenKind::identifier) {
                error(current(), "expected limit field");
                recover();
                continue;
            }
            const std::string field{advance().text};
            if (!expect(TokenKind::equal, "expected '=' after limit field")) continue;
            const auto value = parseSignedNumber();
            expect(TokenKind::semicolon, "expected ';' after limit field");
            if (!value) continue;
            if (field == "minimum_J") material.minimumDeterminant = *value;
            else if (field == "maximum_J") material.maximumDeterminant = *value;
            else if (field == "maximum_stress") material.maximumStress = *value;
            else if (field == "maximum_energy") material.maximumEnergyDensity = *value;
            else error(previous(), "unknown limit field '" + field + "'");
        }
        expect(TokenKind::rightBrace, "expected '}' after limits block");
    }

    std::optional<Unit> parseUnit() {
        Unit result{};
        bool divide = false;
        bool first = true;
        while (!atEnd() && !check(TokenKind::equal)) {
            if (match(TokenKind::star)) { divide = false; continue; }
            if (match(TokenKind::slash)) { divide = true; continue; }
            if (current().kind != TokenKind::identifier &&
                current().kind != TokenKind::number) {
                error(current(), "invalid unit expression");
                return std::nullopt;
            }
            const std::string_view name = advance().text;
            const auto unit = namedUnit(name);
            if (!unit) {
                error(previous(), "unknown unit '" + std::string{name} + "'");
                return std::nullopt;
            }
            int exponent = 1;
            if (match(TokenKind::caret)) {
                bool negative = match(TokenKind::minus);
                if (current().kind != TokenKind::number) {
                    error(current(), "unit exponent must be an integer");
                    return std::nullopt;
                }
                const auto parsed = integer(previousOrAdvanceNumber());
                if (!parsed) {
                    error(previous(), "invalid unit exponent");
                    return std::nullopt;
                }
                exponent = negative ? -*parsed : *parsed;
            }
            if (divide) exponent = -exponent;
            result.scale *= std::pow(unit->scale, exponent);
            result.dimension = result.dimension + unit->dimension * exponent;
            first = false;
            divide = false;
        }
        if (first) {
            error(current(), "unit expression is empty");
            return std::nullopt;
        }
        return result;
    }

    const Token& previousOrAdvanceNumber() {
        return advance();
    }

    static std::optional<int> integer(const Token& token) {
        int value = 0;
        const auto [end, code] = std::from_chars(
            token.text.data(), token.text.data() + token.text.size(), value
        );
        if (code != std::errc{} || end != token.text.data() + token.text.size()) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<double> parseSignedNumber() {
        double sign = 1.0;
        if (match(TokenKind::minus)) sign = -1.0;
        else match(TokenKind::plus);
        if (current().kind != TokenKind::number) {
            error(current(), "expected numeric literal");
            return std::nullopt;
        }
        const Token token = advance();
        double value = 0.0;
        const auto [end, code] = std::from_chars(
            token.text.data(), token.text.data() + token.text.size(), value
        );
        if (code != std::errc{} || end != token.text.data() + token.text.size() ||
            !std::isfinite(value)) {
            error(token, "invalid numeric literal");
            return std::nullopt;
        }
        return sign * value;
    }

    [[nodiscard]] std::uint32_t parseExpression(
        MaterialProgram& material,
        const int minimumPrecedence
    ) {
        std::uint32_t left = parsePrefix(material);
        if (left == NM_INVALID_INDEX) return left;
        while (true) {
            const int precedence = binaryPrecedence(current().kind);
            if (precedence < minimumPrecedence) break;
            const TokenKind operation = advance().kind;
            const int nextMinimum = operation == TokenKind::caret
                ? precedence
                : precedence + 1;
            const std::uint32_t right = parseExpression(material, nextMinimum);
            if (right == NM_INVALID_INDEX) return NM_INVALID_INDEX;
            left = makeBinary(material, operation, left, right);
            if (left == NM_INVALID_INDEX) return left;
        }
        return left;
    }

    static int binaryPrecedence(const TokenKind kind) noexcept {
        switch (kind) {
        case TokenKind::plus:
        case TokenKind::minus: return 10;
        case TokenKind::star:
        case TokenKind::slash: return 20;
        case TokenKind::caret: return 30;
        default: return -1;
        }
    }

    std::uint32_t parsePrefix(MaterialProgram& material) {
        if (match(TokenKind::minus)) {
            const std::uint32_t argument = parseExpression(material, 40);
            if (argument == NM_INVALID_INDEX) return argument;
            return material.expressions.append({
                .kind = ExprKind::negate,
                .dimension = material.expressions.nodes[argument].dimension,
                .arguments = {argument, NM_INVALID_INDEX, NM_INVALID_INDEX},
            });
        }
        if (match(TokenKind::plus)) return parseExpression(material, 40);
        if (match(TokenKind::leftParen)) {
            const std::uint32_t value = parseExpression(material, 0);
            expect(TokenKind::rightParen, "expected ')' after expression");
            return value;
        }
        if (current().kind == TokenKind::number) {
            const auto value = parseSignedNumber();
            return value ? material.expressions.constant(*value) : NM_INVALID_INDEX;
        }
        if (current().kind != TokenKind::identifier) {
            error(current(), "expected expression");
            return NM_INVALID_INDEX;
        }
        const Token identifier = advance();
        if (match(TokenKind::leftParen)) {
            return parseFunction(material, identifier);
        }
        return parseVariable(material, identifier);
    }

    std::uint32_t parseVariable(MaterialProgram& material, const Token& token) {
        const auto parameter = std::ranges::find_if(
            material.parameters,
            [&](const Parameter& value) { return value.name == token.text; }
        );
        if (parameter != material.parameters.end()) {
            return material.expressions.append({
                .kind = ExprKind::parameter,
                .dimension = parameter->dimension,
                .index = static_cast<std::uint32_t>(
                    std::distance(material.parameters.begin(), parameter)
                ),
            });
        }
        const auto state = std::ranges::find_if(
            material.internalState,
            [&](const InternalState& value) { return value.name == token.text; }
        );
        if (state != material.internalState.end()) {
            return material.expressions.append({
                .kind = ExprKind::state,
                .dimension = state->dimension,
                .index = static_cast<std::uint32_t>(
                    std::distance(material.internalState.begin(), state)
                ),
            });
        }
        const auto matrixIndex = [](const std::string_view name, const std::string_view prefix)
            -> std::optional<std::uint32_t> {
            if (!name.starts_with(prefix) || name.size() != prefix.size() + 2u) return std::nullopt;
            const char row = name[prefix.size()];
            const char column = name[prefix.size() + 1u];
            if (row < '0' || row > '2' || column < '0' || column > '2') return std::nullopt;
            return static_cast<std::uint32_t>((row - '0') * 3 + (column - '0'));
        };
        if (const auto index = matrixIndex(token.text, "F")) {
            return material.expressions.append({
                .kind = ExprKind::deformation,
                .dimension = dimensionless,
                .index = *index,
            });
        }
        if (const auto index = matrixIndex(token.text, "dF")) {
            return material.expressions.append({
                .kind = ExprKind::deformationDirection,
                .dimension = dimensionless,
                .index = *index,
            });
        }
        if (const auto index = matrixIndex(token.text, "D")) {
            return material.expressions.append({
                .kind = ExprKind::rate,
                .dimension = dimensionless - timeDimension,
                .index = *index,
            });
        }
        if (const auto index = matrixIndex(token.text, "dD")) {
            return material.expressions.append({
                .kind = ExprKind::rateDirection,
                .dimension = dimensionless - timeDimension,
                .index = *index,
            });
        }
        error(token, "unknown expression symbol '" + std::string{token.text} + "'");
        return NM_INVALID_INDEX;
    }

    std::uint32_t parseFunction(MaterialProgram& material, const Token& function) {
        std::vector<std::uint32_t> arguments;
        if (!check(TokenKind::rightParen)) {
            do {
                arguments.push_back(parseExpression(material, 0));
            } while (match(TokenKind::comma));
        }
        expect(TokenKind::rightParen, "expected ')' after function arguments");
        if (std::ranges::find(arguments, NM_INVALID_INDEX) != arguments.end()) {
            return NM_INVALID_INDEX;
        }
        const std::string_view name = function.text;
        if (name == "J") {
            if (!arguments.empty()) return arityError(function, 0u);
            return determinant(material);
        }
        if (name == "I1") {
            if (!arguments.empty()) return arityError(function, 0u);
            return firstInvariant(material);
        }
        if (name == "neo_hookean") {
            if (arguments.size() != 2u) return arityError(function, 2u);
            if (!(node(material, arguments[0]).dimension == pressureDimension) ||
                !(node(material, arguments[1]).dimension == pressureDimension)) {
                error(function, "neo_hookean arguments must have pressure units");
                return NM_INVALID_INDEX;
            }
            return neoHookean(material, arguments[0], arguments[1]);
        }
        if (name == "log" || name == "exp" || name == "sqrt" || name == "abs") {
            if (arguments.size() != 1u) return arityError(function, 1u);
            return makeUnary(material, function, arguments[0]);
        }
        if (name == "min" || name == "max") {
            if (arguments.size() != 2u) return arityError(function, 2u);
            return makeMinMax(material, function, arguments[0], arguments[1]);
        }
        if (name == "clamp") {
            if (arguments.size() != 3u) return arityError(function, 3u);
            const Dimension dimension = node(material, arguments[0]).dimension;
            if (!(node(material, arguments[1]).dimension == dimension) ||
                !(node(material, arguments[2]).dimension == dimension)) {
                error(function, "clamp arguments must have matching dimensions");
                return NM_INVALID_INDEX;
            }
            return material.expressions.append({
                .kind = ExprKind::clamp,
                .dimension = dimension,
                .arguments = {arguments[0], arguments[1], arguments[2]},
            });
        }
        error(function, "unknown function '" + std::string{name} + "'");
        return NM_INVALID_INDEX;
    }

    std::uint32_t arityError(const Token& function, const std::size_t expected) {
        error(function, "function '" + std::string{function.text} +
            "' expects " + std::to_string(expected) + " arguments");
        return NM_INVALID_INDEX;
    }

    static const Expr& node(const MaterialProgram& material, const std::uint32_t index) {
        return material.expressions.nodes[index];
    }

    std::uint32_t makeUnary(
        MaterialProgram& material,
        const Token& operation,
        const std::uint32_t argument
    ) {
        const Dimension dimension = node(material, argument).dimension;
        ExprKind kind = ExprKind::absolute;
        Dimension output = dimension;
        if (operation.text == "log") {
            kind = ExprKind::logarithm;
            if (!(dimension == dimensionless)) {
                error(operation, "log argument must be dimensionless");
                return NM_INVALID_INDEX;
            }
            output = dimensionless;
        } else if (operation.text == "exp") {
            kind = ExprKind::exponential;
            if (!(dimension == dimensionless)) {
                error(operation, "exp argument must be dimensionless");
                return NM_INVALID_INDEX;
            }
            output = dimensionless;
        } else if (operation.text == "sqrt") {
            kind = ExprKind::squareRoot;
            if ((dimension.length & 1) || (dimension.mass & 1) ||
                (dimension.time & 1) || (dimension.temperature & 1)) {
                error(operation, "sqrt requires even dimension exponents");
                return NM_INVALID_INDEX;
            }
            output = {
                static_cast<std::int8_t>(dimension.length / 2),
                static_cast<std::int8_t>(dimension.mass / 2),
                static_cast<std::int8_t>(dimension.time / 2),
                static_cast<std::int8_t>(dimension.temperature / 2),
            };
        }
        return material.expressions.append({
            .kind = kind,
            .dimension = output,
            .arguments = {argument, NM_INVALID_INDEX, NM_INVALID_INDEX},
        });
    }

    std::uint32_t makeMinMax(
        MaterialProgram& material,
        const Token& operation,
        const std::uint32_t left,
        const std::uint32_t right
    ) {
        const Dimension dimension = node(material, left).dimension;
        if (!(node(material, right).dimension == dimension)) {
            error(operation, "min/max arguments must have matching dimensions");
            return NM_INVALID_INDEX;
        }
        return material.expressions.append({
            .kind = operation.text == "min" ? ExprKind::minimum : ExprKind::maximum,
            .dimension = dimension,
            .arguments = {left, right, NM_INVALID_INDEX},
        });
    }

    std::uint32_t makeBinary(
        MaterialProgram& material,
        const TokenKind operation,
        const std::uint32_t left,
        const std::uint32_t right
    ) {
        const Dimension leftDimension = node(material, left).dimension;
        const Dimension rightDimension = node(material, right).dimension;
        Expr expression;
        expression.arguments = {left, right, NM_INVALID_INDEX};
        switch (operation) {
        case TokenKind::plus:
        case TokenKind::minus:
            if (!(leftDimension == rightDimension)) {
                error(previous(), "addition/subtraction requires matching dimensions");
                return NM_INVALID_INDEX;
            }
            expression.kind = operation == TokenKind::plus
                ? ExprKind::add
                : ExprKind::subtract;
            expression.dimension = leftDimension;
            break;
        case TokenKind::star:
            expression.kind = ExprKind::multiply;
            expression.dimension = leftDimension + rightDimension;
            break;
        case TokenKind::slash:
            expression.kind = ExprKind::divide;
            expression.dimension = leftDimension - rightDimension;
            break;
        case TokenKind::caret: {
            const Expr& exponent = node(material, right);
            if (exponent.kind != ExprKind::constant ||
                !(exponent.dimension == dimensionless) ||
                std::floor(exponent.constant) != exponent.constant ||
                exponent.constant < -16.0 || exponent.constant > 16.0) {
                error(previous(), "power exponent must be an integer literal in [-16, 16]");
                return NM_INVALID_INDEX;
            }
            expression.kind = ExprKind::integerPower;
            expression.integer = static_cast<int>(exponent.constant);
            expression.dimension = leftDimension * expression.integer;
            expression.arguments[1] = NM_INVALID_INDEX;
            break;
        }
        default:
            error(previous(), "invalid binary operator");
            return NM_INVALID_INDEX;
        }
        return material.expressions.append(expression);
    }

    std::uint32_t determinant(MaterialProgram& material) {
        std::array<std::uint32_t, 9> F{};
        for (std::uint32_t index = 0u; index < 9u; ++index) {
            F[index] = material.expressions.append({
                .kind = ExprKind::deformation,
                .dimension = dimensionless,
                .index = index,
            });
        }
        const auto mul = [&](const std::uint32_t a, const std::uint32_t b) {
            return material.expressions.append({
                .kind = ExprKind::multiply,
                .dimension = dimensionless,
                .arguments = {a, b, NM_INVALID_INDEX},
            });
        };
        const auto add = [&](const std::uint32_t a, const std::uint32_t b) {
            return material.expressions.append({
                .kind = ExprKind::add,
                .dimension = dimensionless,
                .arguments = {a, b, NM_INVALID_INDEX},
            });
        };
        const auto sub = [&](const std::uint32_t a, const std::uint32_t b) {
            return material.expressions.append({
                .kind = ExprKind::subtract,
                .dimension = dimensionless,
                .arguments = {a, b, NM_INVALID_INDEX},
            });
        };
        const std::uint32_t c0 = sub(mul(F[4], F[8]), mul(F[5], F[7]));
        const std::uint32_t c1 = sub(mul(F[3], F[8]), mul(F[5], F[6]));
        const std::uint32_t c2 = sub(mul(F[3], F[7]), mul(F[4], F[6]));
        return add(sub(mul(F[0], c0), mul(F[1], c1)), mul(F[2], c2));
    }

    std::uint32_t firstInvariant(MaterialProgram& material) {
        std::uint32_t result = material.expressions.constant(0.0);
        for (std::uint32_t index = 0u; index < 9u; ++index) {
            const std::uint32_t value = material.expressions.append({
                .kind = ExprKind::deformation,
                .dimension = dimensionless,
                .index = index,
            });
            const std::uint32_t square = material.expressions.append({
                .kind = ExprKind::multiply,
                .dimension = dimensionless,
                .arguments = {value, value, NM_INVALID_INDEX},
            });
            result = material.expressions.append({
                .kind = ExprKind::add,
                .dimension = dimensionless,
                .arguments = {result, square, NM_INVALID_INDEX},
            });
        }
        return result;
    }

    std::uint32_t neoHookean(
        MaterialProgram& material,
        const std::uint32_t mu,
        const std::uint32_t lambda
    ) {
        const std::uint32_t half = material.expressions.constant(0.5);
        const std::uint32_t three = material.expressions.constant(3.0);
        const std::uint32_t I1 = firstInvariant(material);
        const std::uint32_t J = determinant(material);
        const std::uint32_t logJ = material.expressions.append({
            .kind = ExprKind::logarithm,
            .dimension = dimensionless,
            .arguments = {J, NM_INVALID_INDEX, NM_INVALID_INDEX},
        });
        const auto multiply = [&](const std::uint32_t a, const std::uint32_t b,
                                  const Dimension dimension) {
            return material.expressions.append({
                .kind = ExprKind::multiply,
                .dimension = dimension,
                .arguments = {a, b, NM_INVALID_INDEX},
            });
        };
        const std::uint32_t shifted = material.expressions.append({
            .kind = ExprKind::subtract,
            .dimension = dimensionless,
            .arguments = {I1, three, NM_INVALID_INDEX},
        });
        const std::uint32_t first = multiply(
            half,
            multiply(mu, shifted, pressureDimension),
            pressureDimension
        );
        const std::uint32_t second = multiply(mu, logJ, pressureDimension);
        const std::uint32_t logSquared = multiply(logJ, logJ, dimensionless);
        const std::uint32_t third = multiply(
            half,
            multiply(lambda, logSquared, pressureDimension),
            pressureDimension
        );
        const std::uint32_t difference = material.expressions.append({
            .kind = ExprKind::subtract,
            .dimension = pressureDimension,
            .arguments = {first, second, NM_INVALID_INDEX},
        });
        return material.expressions.append({
            .kind = ExprKind::add,
            .dimension = pressureDimension,
            .arguments = {difference, third, NM_INVALID_INDEX},
        });
    }

    void validateMaterial(MaterialProgram& material) {
        if (material.energyRoot == NM_INVALID_INDEX) {
            error(current(), "material must define an energy expression");
        }
        if (material.supportedRepresentations.empty()) {
            error(current(), "material must declare at least one supported representation");
        }
        if (!(material.staticFriction >= 0.0) ||
            !(material.dynamicFriction >= 0.0) ||
            !(material.restitution >= 0.0 && material.restitution <= 1.0) ||
            !(material.adhesion >= 0.0)) {
            error(current(), "invalid interface response values");
        }
        if (!(material.minimumDeterminant > 0.0) ||
            !(material.maximumDeterminant > material.minimumDeterminant) ||
            !(material.maximumStress > 0.0) ||
            !(material.maximumEnergyDensity > 0.0)) {
            error(current(), "invalid material validity limits");
        }
        if (material.parameters.size() > NM_MAX_MATERIAL_PARAMETERS) {
            error(current(), "material exceeds GPU parameter capacity");
        }
    }

    std::vector<Token> tokens_;
    std::size_t cursor_ = 0u;
    ParseResult* output_ = nullptr;
};

} // namespace

std::uint32_t ExpressionGraph::append(Expr expression) {
    if (nodes.size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Matter expression graph exceeds uint32 capacity");
    }
    nodes.push_back(std::move(expression));
    return static_cast<std::uint32_t>(nodes.size() - 1u);
}

std::uint32_t ExpressionGraph::constant(
    const double value,
    const Dimension dimension
) {
    return append({
        .kind = ExprKind::constant,
        .dimension = dimension,
        .constant = value,
    });
}

std::uint32_t ExpressionGraph::cloneFrom(
    const ExpressionGraph& source,
    const std::uint32_t root,
    std::unordered_map<std::uint32_t, std::uint32_t>& memo
) {
    if (const auto found = memo.find(root); found != memo.end()) return found->second;
    if (root >= source.nodes.size()) {
        throw std::out_of_range("invalid source expression root");
    }
    Expr expression = source.nodes[root];
    for (std::uint32_t& argument : expression.arguments) {
        if (argument != NM_INVALID_INDEX) argument = cloneFrom(source, argument, memo);
    }
    const std::uint32_t result = append(std::move(expression));
    memo.emplace(root, result);
    return result;
}

bool ParseResult::succeeded() const noexcept {
    return std::ranges::none_of(diagnostics, [](const Diagnostic& diagnostic) {
        return diagnostic.severity == Diagnostic::Severity::error;
    });
}

ParseResult parseMatterLanguage(const std::string_view source) {
    return Parser{source}.parse();
}

ParseResult parseMatterLanguageFile(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        ParseResult result;
        result.diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "could not open Matter source '" + path.string() + "'",
        });
        return result;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        ParseResult result;
        result.diagnostics.push_back({
            Diagnostic::Severity::error,
            0u,
            0u,
            "could not read Matter source '" + path.string() + "'",
        });
        return result;
    }
    return parseMatterLanguage(contents.str());
}

std::string dimensionName(const Dimension dimension) {
    if (dimension == dimensionless) return "1";
    std::ostringstream output;
    bool first = true;
    auto append = [&](const char* name, const int exponent) {
        if (exponent == 0) return;
        if (!first) output << '*';
        output << name;
        if (exponent != 1) output << '^' << exponent;
        first = false;
    };
    append("m", dimension.length);
    append("kg", dimension.mass);
    append("s", dimension.time);
    append("K", dimension.temperature);
    return output.str();
}

std::uint64_t materialFingerprint(const MaterialProgram& material) noexcept {
    std::uint64_t hash = kFNVOffset;
    hashString(hash, material.name);
    hashValue(hash, material.hint);
    for (const Parameter& parameter : material.parameters) {
        hashString(hash, parameter.name);
        hashValue(hash, parameter.dimension);
        hashValue(hash, parameter.defaultValue);
        hashValue(hash, parameter.lower);
        hashValue(hash, parameter.upper);
        hashValue(hash, parameter.proposalSigma);
        hashValue(hash, parameter.logarithmic);
        hashValue(hash, parameter.identifiable);
    }
    for (const InternalState& state : material.internalState) {
        hashString(hash, state.name);
        hashValue(hash, state.dimension);
        hashValue(hash, state.initialValue);
    }
    for (const Expr& expression : material.expressions.nodes) {
        hashValue(hash, expression.kind);
        hashValue(hash, expression.dimension);
        hashValue(hash, expression.constant);
        hashValue(hash, expression.index);
        hashValue(hash, expression.integer);
        for (const std::uint32_t argument : expression.arguments) hashValue(hash, argument);
    }
    hashValue(hash, material.energyRoot);
    hashValue(hash, material.dissipationRoot);
    hashValue(hash, material.validityRoot);
    for (const Representation representation : material.supportedRepresentations) {
        hashValue(hash, representation);
    }
    hashValue(hash, material.staticFriction);
    hashValue(hash, material.dynamicFriction);
    hashValue(hash, material.restitution);
    hashValue(hash, material.adhesion);
    hashValue(hash, material.minimumDeterminant);
    hashValue(hash, material.maximumDeterminant);
    hashValue(hash, material.maximumStress);
    hashValue(hash, material.maximumEnergyDensity);
    return hash == 0u ? 1u : hash;
}

} // namespace numi::matter
