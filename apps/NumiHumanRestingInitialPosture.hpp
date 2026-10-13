#pragma once

#import <Foundation/Foundation.h>
#include <CoreFoundation/CoreFoundation.h>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>
#include <vector>

namespace numi::human::resting_initial_posture {

// This value is an initializer for a fresh posture only. It deliberately
// exposes no velocity, activation, controller, or solver history and is not a
// restart/checkpoint representation.
struct InitialPosture {
    std::vector<float> q;
    std::size_t stepCount = 0u;
    double timestepSeconds = 0.0;
};

namespace detail {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline bool hasPrefix(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
        value.substr(0u, prefix.size()) == prefix;
}

inline std::vector<std::string> whitespaceTokens(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) tokens.push_back(std::move(token));
    return tokens;
}

inline bool validSHA256(const std::string& value) {
    if (value.size() != 64u) return false;
    for (const unsigned char byte : value) {
        const bool digit = byte >= '0' && byte <= '9';
        const bool lower = byte >= 'a' && byte <= 'f';
        const bool upper = byte >= 'A' && byte <= 'F';
        if (!digit && !lower && !upper) return false;
    }
    return true;
}

inline bool isJSONBoolean(id value) {
    return [value isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

inline bool isJSONNumber(id value) {
    return [value isKindOfClass:NSNumber.class] && !isJSONBoolean(value);
}

inline double requiredFiniteNumber(NSDictionary* document,
                                   NSString* key,
                                   const char* error) {
    id value = document[key];
    require(isJSONNumber(value), error);
    const double number = [value doubleValue];
    require(std::isfinite(number), error);
    return number;
}

inline void skipJSONWhitespace(std::string_view json, std::size_t& offset) {
    while (offset < json.size()) {
        const char value = json[offset];
        if (value != ' ' && value != '\t' &&
            value != '\n' && value != '\r') {
            break;
        }
        ++offset;
    }
}

inline std::string_view skipJSONString(
    std::string_view json,
    std::size_t& offset
) {
    require(offset < json.size() && json[offset] == '"',
            "terminal JSON string token is malformed");
    const std::size_t start = offset++;
    while (offset < json.size()) {
        const char value = json[offset++];
        if (value == '"') return json.substr(start, offset - start);
        if (value != '\\') continue;
        require(offset < json.size(),
                "terminal JSON string escape is truncated");
        const char escape = json[offset++];
        if (escape == 'u') {
            require(json.size() - offset >= 4u,
                    "terminal JSON unicode escape is truncated");
            offset += 4u;
        }
    }
    require(false, "terminal JSON string is unterminated");
    return {};
}

inline void skipJSONValue(std::string_view json, std::size_t& offset) {
    skipJSONWhitespace(json, offset);
    require(offset < json.size(), "terminal JSON value is missing");
    if (json[offset] == '"') {
        (void)skipJSONString(json, offset);
        return;
    }
    if (json[offset] == '[' || json[offset] == '{') {
        std::vector<char> closers;
        const char first = json[offset++];
        closers.push_back(first == '[' ? ']' : '}');
        while (!closers.empty()) {
            require(offset < json.size(),
                    "terminal JSON compound value is truncated");
            const char value = json[offset];
            if (value == '"') {
                (void)skipJSONString(json, offset);
                continue;
            }
            ++offset;
            if (value == '[') {
                closers.push_back(']');
            } else if (value == '{') {
                closers.push_back('}');
            } else if (value == ']' || value == '}') {
                require(closers.back() == value,
                        "terminal JSON compound value is unbalanced");
                closers.pop_back();
            }
        }
        return;
    }
    const std::size_t start = offset;
    while (offset < json.size()) {
        const char value = json[offset];
        if (value == ',' || value == ']' || value == '}' ||
            value == ' ' || value == '\t' ||
            value == '\n' || value == '\r') {
            break;
        }
        ++offset;
    }
    require(offset > start, "terminal JSON primitive is empty");
}

// NSNumber.stringValue may shorten a native 17-digit Float32-as-double
// decimal. Read the q tokens from the original JSON text before enforcing the
// exact binary32 round-trip.
inline std::vector<std::string_view> topLevelJSONArrayTokens(
    std::string_view json,
    std::string_view key
) {
    std::size_t offset = 0u;
    skipJSONWhitespace(json, offset);
    require(offset < json.size() && json[offset] == '{',
            "terminal raw JSON is not an object");
    ++offset;
    bool found = false;
    std::unordered_set<std::string> memberNames;
    std::vector<std::string_view> result;
    while (true) {
        skipJSONWhitespace(json, offset);
        require(offset < json.size(),
                "terminal raw JSON object is truncated");
        if (json[offset] == '}') {
            ++offset;
            break;
        }
        const std::string_view member = skipJSONString(json, offset);
        skipJSONWhitespace(json, offset);
        require(offset < json.size() && json[offset] == ':',
                "terminal raw JSON member lacks a colon");
        ++offset;
        skipJSONWhitespace(json, offset);
        NSData* memberData = [NSData dataWithBytes:member.data() length:member.size()];
        id decodedMember = [NSJSONSerialization JSONObjectWithData:memberData
            options:NSJSONReadingFragmentsAllowed error:nil];
        require([decodedMember isKindOfClass:NSString.class],
                "terminal raw JSON member name is invalid");
        const char* memberUTF8 = [(NSString*)decodedMember UTF8String];
        require(memberUTF8 != nullptr && memberNames.insert(memberUTF8).second,
                "terminal raw JSON has duplicate member names");
        const bool matchesKey = std::string_view(memberUTF8) == key;
        if (matchesKey) {
            require(!found, "terminal raw JSON has duplicate q arrays");
            found = true;
            require(offset < json.size() && json[offset] == '[',
                    "terminal raw q value is not an array");
            ++offset;
            while (true) {
                skipJSONWhitespace(json, offset);
                require(offset < json.size(),
                        "terminal raw q array is truncated");
                if (json[offset] == ']') {
                    ++offset;
                    break;
                }
                const std::size_t start = offset;
                skipJSONValue(json, offset);
                std::size_t end = offset;
                while (end > start &&
                       (json[end - 1u] == ' ' || json[end - 1u] == '\t' ||
                        json[end - 1u] == '\n' || json[end - 1u] == '\r')) {
                    --end;
                }
                require(end > start, "terminal raw q number is empty");
                result.push_back(json.substr(start, end - start));
                skipJSONWhitespace(json, offset);
                require(offset < json.size(),
                        "terminal raw q array is truncated");
                if (json[offset] == ',') {
                    ++offset;
                } else {
                    require(json[offset] == ']',
                            "terminal raw q array separator is invalid");
                }
            }
        } else {
            skipJSONValue(json, offset);
        }
        skipJSONWhitespace(json, offset);
        require(offset < json.size(),
                "terminal raw JSON object is truncated");
        if (json[offset] == ',') {
            ++offset;
        } else {
            require(json[offset] == '}',
                    "terminal raw JSON object separator is invalid");
            ++offset;
            break;
        }
    }
    skipJSONWhitespace(json, offset);
    require(offset == json.size() && found,
            "terminal raw JSON q array is missing or has trailing bytes");
    return result;
}

inline double exactJSONNumberValue(std::string_view decimal,
                                   const char* error) {
    require(!decimal.empty(), error);
    const char* begin = decimal.data();
    const char* end = begin + decimal.size();
    double number = 0.0;
    const auto parsed = std::from_chars(
        begin, end, number, std::chars_format::general);
    require(parsed.ec == std::errc{} && parsed.ptr == end &&
                std::isfinite(number), error);
    return number;
}

} // namespace detail

// Read the terminal q from the legacy stand log as a starting posture for a
// newly initialized simulation. The caller supplies the expected rigid-payload
// identity and generalized-coordinate dimension. File SHA ownership stays
// with the caller.
inline InitialPosture readRestingInitialPosture(
    const std::string& path,
    const std::string& expectedRigidSHA,
    std::size_t nq
) {
    using namespace detail;
    require(!path.empty(), "initial-posture log path is empty");
    require(validSHA256(expectedRigidSHA),
            "expected rigid payload SHA-256 is malformed");
    require(nq >= 7u, "initial-posture q dimension must include a root quaternion");

    InitialPosture result;
    @autoreleasepool {
        NSString* filePath = [NSString stringWithUTF8String:path.c_str()];
        require(filePath != nil, "initial-posture log path is not valid UTF-8");
        NSError* attributesError = nil;
        NSDictionary* attributes = [[NSFileManager defaultManager]
            attributesOfItemAtPath:filePath error:&attributesError];
        require(attributes != nil &&
                    [attributes[NSFileType] isEqualToString:NSFileTypeRegular],
                "initial-posture log must be a regular file");
        id fileSizeValue = attributes[NSFileSize];
        require(isJSONNumber(fileSizeValue),
                "initial-posture log size is unavailable");
        constexpr unsigned long long kMaximumLogBytes =
            256ull * 1024ull * 1024ull;
        const unsigned long long fileSize =
            [fileSizeValue unsignedLongLongValue];
        require(fileSize <= kMaximumLogBytes,
                "initial-posture log exceeds the 256 MiB admission bound");
        NSData* data = [NSData dataWithContentsOfFile:filePath];
        require(data != nil, "initial-posture log could not be read");
        require(data.length <= kMaximumLogBytes,
                "initial-posture log grew beyond the 256 MiB admission bound");
        NSString* text = [[NSString alloc] initWithData:data
                                               encoding:NSUTF8StringEncoding];
        require(text != nil, "initial-posture log is not UTF-8");

        constexpr std::string_view kSourcePrefix =
            "rigid_source_rest_frames=";
        constexpr std::string_view kTerminalPrefix =
            "stand_terminal_state=";
        constexpr std::string_view kFooterPrefix =
            "resting_integrated_body=";

        std::vector<std::string> sourceLines;
        std::vector<std::string> terminalLines;
        std::vector<std::string> footerLines;
        std::size_t lineIndex = 0u, sourceIndex = 0u, terminalIndex = 0u, footerIndex = 0u;
        NSArray<NSString*>* lines = [text componentsSeparatedByCharactersInSet:
            [NSCharacterSet newlineCharacterSet]];
        for (NSString* line in lines) {
            const char* utf8 = line.UTF8String;
            require(utf8 != nullptr, "initial-posture log contains invalid text");
            const std::string value(utf8);
            if (hasPrefix(value, kSourcePrefix)) { sourceLines.push_back(value); sourceIndex = lineIndex; }
            if (hasPrefix(value, kTerminalPrefix)) { terminalLines.push_back(value); terminalIndex = lineIndex; }
            if (hasPrefix(value, kFooterPrefix)) { footerLines.push_back(value); footerIndex = lineIndex; }
            ++lineIndex;
        }

        require(sourceLines.size() == 1u,
                "initial-posture log must have exactly one rigid source record");
        const auto sourceTokens = whitespaceTokens(sourceLines.front());
        require(!sourceTokens.empty() &&
                    sourceTokens.front() == "rigid_source_rest_frames=passed",
                "rigid source record did not pass");
        std::size_t rigidSHAOccurrences = 0u;
        bool rigidSHAMatches = false;
        for (const auto& token : sourceTokens) {
            constexpr std::string_view kRigidSHA = "rigid_sha256=";
            if (!hasPrefix(token, kRigidSHA)) continue;
            ++rigidSHAOccurrences;
            rigidSHAMatches = token.substr(kRigidSHA.size()) == expectedRigidSHA;
        }
        require(rigidSHAOccurrences == 1u && rigidSHAMatches,
                "rigid source record does not contain the exact expected SHA-256 token");

        require(terminalLines.size() == 1u,
                "initial-posture log must have exactly one terminal state record");
        require(footerLines.size() == 1u,
                "initial-posture log must have exactly one resting completion footer");
        require(sourceIndex < terminalIndex && terminalIndex < footerIndex,
                "initial-posture source, terminal and completion records are out of order");
        const auto footerTokens = whitespaceTokens(footerLines.front());
        require(!footerTokens.empty() &&
                    footerTokens.front() == "resting_integrated_body=completed",
                "resting integration did not complete successfully");
        bool footerDisablesRootAssistance = false;
        std::size_t footerRootAssistanceOccurrences = 0u;
        for (const auto& token : footerTokens) {
            constexpr std::string_view kRootAssistance = "root_assistance=";
            if (!hasPrefix(token, kRootAssistance)) continue;
            ++footerRootAssistanceOccurrences;
            footerDisablesRootAssistance =
                token.substr(kRootAssistance.size()) == "false";
        }
        require(footerRootAssistanceOccurrences == 1u &&
                    footerDisablesRootAssistance,
                "resting completion footer does not confirm root assistance is disabled");

        const std::string& terminalLine = terminalLines.front();
        const std::string jsonText = terminalLine.substr(kTerminalPrefix.size());
        NSData* jsonData = [NSData dataWithBytes:jsonText.data()
                                          length:jsonText.size()];
        NSError* error = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:jsonData
                                                     options:0
                                                       error:&error];
        require([parsed isKindOfClass:NSDictionary.class],
                "terminal state is not a JSON object");
        NSDictionary* state = (NSDictionary*)parsed;
        id schema = state[@"schema"];
        require([schema isKindOfClass:NSString.class] &&
                    [schema isEqualToString:
                        @"numi.human.legacy-stand-terminal.v1"],
                "terminal state has an unsupported schema");

        id rootAssistance = state[@"root_assistance"];
        require(isJSONBoolean(rootAssistance) &&
                    ![rootAssistance boolValue],
                "terminal state must declare root assistance false");

        const double stepCount =
            requiredFiniteNumber(state, @"step_count",
                                 "terminal step_count is invalid");
        require(stepCount > 0.0 && std::floor(stepCount) == stepCount &&
                    stepCount < std::ldexp(
                        1.0, std::numeric_limits<std::size_t>::digits),
                "terminal step_count must be a positive integer in range");

        const double timestepSeconds =
            requiredFiniteNumber(state, @"timestep_seconds",
                                 "terminal timestep_seconds is invalid");
        require(timestepSeconds > 0.0,
                "terminal timestep_seconds must be positive");

        id coordinates = state[@"q"];
        require([coordinates isKindOfClass:NSArray.class] &&
                    [(NSArray*)coordinates count] == nq,
                "terminal q dimension differs from the requested dimension");
        NSArray* qArray = (NSArray*)coordinates;
        const auto qTokens = topLevelJSONArrayTokens(jsonText, "q");
        require(qTokens.size() == nq,
                "terminal raw q dimension differs from the requested dimension");
        result.q.reserve(nq);
        for (std::size_t index = 0u; index < nq; ++index) {
            id coordinate = qArray[index];
            require(isJSONNumber(coordinate),
                    "terminal q entry must be a JSON number, not a boolean");
            const double value = exactJSONNumberValue(
                qTokens[index],
                "terminal q entry must be a finite canonical JSON number");
            const float narrowed = static_cast<float>(value);
            require(std::isfinite(narrowed),
                    "terminal q entry is outside finite Float32 range");
            const double roundTrip = static_cast<double>(narrowed);
            require(roundTrip == value &&
                        (value != 0.0 ||
                         std::signbit(roundTrip) == std::signbit(value)),
                    "terminal q entry does not round-trip exactly through Float32");
            result.q.push_back(narrowed);
        }

        double quaternionNormSquared = 0.0;
        for (std::size_t index = 3u; index < 7u; ++index) {
            const double component = result.q[index];
            quaternionNormSquared += component * component;
        }
        const double quaternionNorm = std::sqrt(quaternionNormSquared);
        require(std::isfinite(quaternionNorm) &&
                    std::abs(quaternionNorm - 1.0) <= 1.0e-5,
                "terminal root quaternion is not unit length");

        // Do not read or return initial_q, v, initial_v, activations, or solver
        // state. The caller must initialize those afresh.
        result.stepCount = static_cast<std::size_t>(stepCount);
        result.timestepSeconds = timestepSeconds;
    }
    return result;
}

} // namespace numi::human::resting_initial_posture
