#pragma once

// Strict metadata and source-prefix contracts for diagnostic coupon resume.
// The native RuntimeStateSnapshot remains the sole source of restored state.

#include <algorithm>
#include <charconv>
#include <array>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace numi::cardboard::coupon_resume {

// Registered only by source-bound diagnostic runner 161-002. Selector/root
// controls intentionally remain in the strict runtime digest.
inline constexpr std::array<std::string_view, 7u> kRegisteredOutputPathKeys{
    "NM_FEM_BASE_CGLS_CAPTURE_PATH",
    "NM_FEM_COUPLED_CAPTURE_PATH",
    "NM_FEM_COUPLED_TRIAL_CAPTURE_PATH",
    "NM_FEM_COUPLED_TRIAL_SUMMARY_PATH",
    "NM_FEM_OPERATOR_FD_CAPTURE_PATH",
    "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH",
    "NM_FEM_STATUS5_CAPTURE_PATH",
};

// These selectors only request bounded, read-only diagnostics.  They remain
// strict in ordinary resume mode; the explicit diagnostic-replay mode may
// change only these five selectors in addition to the seven path bindings.
inline constexpr std::array<std::string_view, 5u>
    kRegisteredDiagnosticReplaySelectorKeys{
        "NM_FEM_BASE_CGLS_CAPTURE_NEWTON",
        "NM_FEM_BASE_CGLS_STATE_CAPTURE",
        "NM_FEM_OPERATOR_FD_SELECTOR",
        "NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR",
        "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
    };

struct RuntimePathBinding {
    std::string key;
    bool present = false;
    std::string value;
    bool operator==(const RuntimePathBinding&) const = default;
};
using RuntimePathMap = std::vector<RuntimePathBinding>;

inline bool isRegisteredOutputPathKey(const std::string_view name) {
    return std::find(kRegisteredOutputPathKeys.begin(),
                     kRegisteredOutputPathKeys.end(), name) !=
        kRegisteredOutputPathKeys.end();
}

inline bool isRegisteredDiagnosticReplaySelectorKey(
    const std::string_view name) {
    return std::find(kRegisteredDiagnosticReplaySelectorKeys.begin(),
                     kRegisteredDiagnosticReplaySelectorKeys.end(), name) !=
        kRegisteredDiagnosticReplaySelectorKeys.end();
}

inline bool isLowerSha256(const std::string_view value) {
    if (value.size() != 64u) return false;
    for (const char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

inline std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1u);
}

inline std::map<std::string, std::string> parseProperties(
    std::istream& input, const std::set<std::string>& allowed,
    const std::string_view sourceName) {
    std::map<std::string, std::string> result;
    std::string line;
    std::uint32_t lineNumber = 0u;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(std::move(line));
        if (line.empty() || line.front() == '#') continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            throw std::invalid_argument(std::string(sourceName) + ":" +
                std::to_string(lineNumber) + " must use key=value syntax");
        const std::string key = trim(line.substr(0u, separator));
        const std::string value = trim(line.substr(separator + 1u));
        if (key.empty() || value.empty() || !allowed.contains(key))
            throw std::invalid_argument(std::string(sourceName) + ":" +
                std::to_string(lineNumber) + " has an empty or unsupported property");
        if (!result.emplace(key, value).second)
            throw std::invalid_argument(std::string(sourceName) + ":" +
                std::to_string(lineNumber) + " repeats property '" + key + "'");
    }
    if (!input.eof())
        throw std::invalid_argument(std::string(sourceName) + " read failed");
    return result;
}

inline const std::string& property(
    const std::map<std::string, std::string>& values, const char* name) {
    const auto found = values.find(name);
    if (found == values.end())
        throw std::invalid_argument(std::string("coupon resume is missing '") + name + "'");
    return found->second;
}

template<class UInt>
inline UInt parseUnsigned(const std::string& value, const char* name) {
    static_assert(std::is_unsigned_v<UInt>);
    if (value.empty()) throw std::invalid_argument(std::string("invalid ") + name);
    for (const char c : value)
        if (c < '0' || c > '9')
            throw std::invalid_argument(std::string("invalid ") + name);
    UInt parsed{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

struct Metadata {
    std::string checkpointSha256;
    std::uint64_t checkpointBytes = 0u;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::string protocolManifestSha256;
    std::string runtimeEnvironmentSha256;
    std::string runtimeOutputPathMapHex;
    std::string runtimeOutputPathMapSha256;
    std::string sourceRunnerCaseDirectory = "none";
    std::string sourceRunnerEnvironmentRecordSha256 = "none";
    std::string sourceRunnerNonOutputSettingsSha256 = "none";
    std::string sourceRunnerPredictionSha256 = "none";
    std::string sourceRunnerContractSha256 = "none";
    std::string prefixManifestSha256;
    std::string prefixManifestFile;
    std::string sourceRunDirectory;
    std::string sourceRunId;
    std::string phase;
    std::string nextPhase;
    std::uint32_t environmentCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t materialStateStride = 0u;
    std::uint32_t acceptedControlIndex = 0u;
    std::uint32_t nextControlIndex = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t sourcePrefixControlCount = 0u;
    std::uint32_t timestepFloatBits = 0u;
    std::uint64_t scheduledElapsedSecondsBits = 0u;
    std::uint64_t nextTargetAngleBits = 0u;
};

inline std::string canonicalRuntimeEnvironment(
    std::vector<std::pair<std::string, std::string>> variables) {
    variables.erase(std::remove_if(variables.begin(), variables.end(),
        [](const auto& variable) {
            return !variable.first.starts_with("NM_") &&
                !variable.first.starts_with("NUMI_MATTER_");
        }), variables.end());
    std::sort(variables.begin(), variables.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    std::ostringstream output;
    output << "numi.cardboard.coupon-runtime-environment.v1\n";
    std::string previous;
    for (const auto& [name, value] : variables) {
        if (name.empty() || name == previous ||
            name.find('\n') != std::string::npos ||
            name.find('=') != std::string::npos ||
            value.find('\n') != std::string::npos ||
            value.find('\r') != std::string::npos)
            throw std::invalid_argument(
                "coupon runtime environment contains an invalid or duplicate binding");
        previous = name;
        output << name.size() << ':' << name << value.size() << ':' << value << '\n';
    }
    return output.str();
}

inline std::string canonicalCouponRuntimeSettings(
    std::vector<std::pair<std::string, std::string>> variables) {
    variables.erase(std::remove_if(variables.begin(), variables.end(),
        [](const auto& variable) {
            const bool native = variable.first.starts_with("NM_") ||
                variable.first.starts_with("NUMI_MATTER_");
            return !native || isRegisteredOutputPathKey(variable.first);
        }), variables.end());
    std::sort(variables.begin(), variables.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    std::ostringstream output;
    output << "numi.cardboard.coupon-runtime-settings.v2\n";
    std::string previous;
    for (const auto& [name, value] : variables) {
        if (name.empty() || name == previous ||
            name.find('\n') != std::string::npos ||
            name.find('=') != std::string::npos ||
            value.find('\n') != std::string::npos ||
            value.find('\r') != std::string::npos ||
            value.find('\0') != std::string::npos)
            throw std::invalid_argument(
                "coupon runtime settings contain an invalid or duplicate binding");
        previous = name;
        output << name.size() << ':' << name << value.size() << ':' << value << '\n';
    }
    return output.str();
}

inline std::string canonicalCouponDiagnosticReplaySettings(
    std::vector<std::pair<std::string, std::string>> variables) {
    variables.erase(std::remove_if(variables.begin(), variables.end(),
        [](const auto& variable) {
            return isRegisteredDiagnosticReplaySelectorKey(variable.first) ||
                isRegisteredOutputPathKey(variable.first);
        }), variables.end());
    std::sort(variables.begin(), variables.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    std::ostringstream output;
    output << "numi.cardboard.diagnostic-replay-settings.v1\n";
    std::string previous;
    for (const auto& [name, value] : variables) {
        if (name.empty() || name == previous ||
            name.find('\n') != std::string::npos ||
            name.find('=') != std::string::npos ||
            value.find('\n') != std::string::npos ||
            value.find('\r') != std::string::npos ||
            value.find('\0') != std::string::npos)
            throw std::invalid_argument(
                "diagnostic replay settings contain an invalid or duplicate binding");
        previous = name;
        output << name.size() << ':' << name << value.size() << ':' << value << '\n';
    }
    return output.str();
}

inline bool couponDiagnosticReplaySettingsMatch(
    const std::vector<std::pair<std::string, std::string>>& source,
    const std::vector<std::pair<std::string, std::string>>& current) {
    return canonicalCouponDiagnosticReplaySettings(source) ==
        canonicalCouponDiagnosticReplaySettings(current);
}

inline bool couponSourceBoundEnvironmentMatches(
    std::vector<std::pair<std::string, std::string>> contract,
    std::vector<std::pair<std::string, std::string>> actual) {
    const auto orderedUnique = [](auto& values) {
        std::sort(values.begin(), values.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
        return std::adjacent_find(values.begin(), values.end(),
            [](const auto& left, const auto& right) {
                return left.first == right.first;
            }) == values.end();
    };
    return orderedUnique(contract) && orderedUnique(actual) &&
        contract == actual;
}

inline std::optional<std::uint32_t> parseDecimalUint32(
    const std::string_view text) {
    if (text.empty()) return std::nullopt;
    for (const char c : text)
        if (c < '0' || c > '9') return std::nullopt;
    std::uint32_t value = 0u;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    if (error != std::errc{} || end != text.data() + text.size())
        return std::nullopt;
    return value;
}

inline std::optional<std::uint32_t> parseCanonicalUint32(
    const std::string_view text) {
    const auto value = parseDecimalUint32(text);
    if (!value || std::to_string(*value) != text) return std::nullopt;
    return value;
}

inline void validateDiagnosticReplaySelectors(
    const std::vector<std::pair<std::string, std::string>>& settings,
    const std::uint32_t acceptedControl,
    const std::uint32_t environmentCount) {
    std::map<std::string, std::string> values;
    for (const auto& [key, value] : settings) {
        if (!values.emplace(key, value).second)
            throw std::invalid_argument(
                "diagnostic replay selector settings contain duplicate keys");
    }
    const auto find = [&](const std::string_view key) -> const std::string* {
        const auto it = values.find(std::string(key));
        return it == values.end() ? nullptr : &it->second;
    };
    const auto requirePair = [&](const std::string_view pathKey,
                                 const std::string_view selectorKey) {
        if ((find(pathKey) == nullptr) != (find(selectorKey) == nullptr))
            throw std::invalid_argument(
                "diagnostic capture path and selector must be present together");
    };
    requirePair("NM_FEM_BASE_CGLS_CAPTURE_PATH",
                "NM_FEM_BASE_CGLS_CAPTURE_NEWTON");
    requirePair("NM_FEM_OPERATOR_FD_CAPTURE_PATH",
                "NM_FEM_OPERATOR_FD_SELECTOR");
    requirePair("NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH",
                "NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR");

    if (const std::string* state = find("NM_FEM_BASE_CGLS_STATE_CAPTURE")) {
        if (*state != "0" && *state != "1")
            throw std::invalid_argument(
                "CGLS state-capture selector must be exactly 0 or 1");
        if (*state == "1" && find("NM_FEM_BASE_CGLS_CAPTURE_PATH") == nullptr)
            throw std::invalid_argument(
                "CGLS state capture requires its paired path and Newton selector");
    }
    if (const std::string* newton = find("NM_FEM_BASE_CGLS_CAPTURE_NEWTON")) {
        const auto parsed = parseCanonicalUint32(*newton);
        if (!parsed || *parsed > 256u)
            throw std::invalid_argument(
                "CGLS capture Newton selector must be canonical uint32 in [0,256]");
    }
    const auto requireFutureRoot = [&](const std::string_view key,
                                       const std::string_view raw) {
        const auto root = parseDecimalUint32(raw);
        if (!root || *root <= acceptedControl)
            throw std::invalid_argument(
                std::string(key) +
                " must select a canonical control strictly after the accepted checkpoint");
    };
    if (const std::string* selector = find("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT"))
        requireFutureRoot("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", *selector);
    if (const std::string* selector = find("NM_FEM_OPERATOR_FD_SELECTOR")) {
        const std::size_t separator = selector->find(':');
        if (separator == std::string::npos ||
            selector->find(':', separator + 1u) != std::string::npos)
            throw std::invalid_argument(
                "operator-FD selector must have exactly root:iteration fields");
        requireFutureRoot("NM_FEM_OPERATOR_FD_SELECTOR",
                          std::string_view(*selector).substr(0u, separator));
        if (!parseDecimalUint32(
                std::string_view(*selector).substr(separator + 1u)))
            throw std::invalid_argument(
                "operator-FD iteration selector must be canonical uint32");
    }
    if (const std::string* selector =
            find("NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR")) {
        std::array<std::uint32_t, 6u> fields{};
        std::size_t begin = 0u;
        for (std::size_t index = 0u; index < fields.size(); ++index) {
            const std::size_t separator = selector->find(':', begin);
            const std::size_t end = separator == std::string::npos
                ? selector->size() : separator;
            if ((index + 1u < fields.size()) ==
                    (separator == std::string::npos))
                throw std::invalid_argument(
                    "regional consumer selector must contain exactly six fields");
            const auto field = parseDecimalUint32(
                std::string_view(*selector).substr(begin, end - begin));
            if (!field)
                throw std::invalid_argument(
                    "regional consumer selector fields must be canonical uint32");
            fields[index] = *field;
            begin = end + 1u;
        }
        if (fields[0] <= acceptedControl || fields[3] == 0u ||
            environmentCount == 0u || fields[5] >= environmentCount)
            throw std::invalid_argument(
                "regional consumer selector must target a future control, positive restart, and valid environment");
    }
}

inline RuntimePathMap couponRuntimePathMap(
    const std::vector<std::pair<std::string, std::string>>& variables) {
    std::map<std::string, std::string> present;
    for (const auto& [name, value] : variables) {
        if (!isRegisteredOutputPathKey(name)) continue;
        if (!present.emplace(name, value).second)
            throw std::invalid_argument("duplicate registered output-path binding");
    }
    RuntimePathMap result;
    result.reserve(kRegisteredOutputPathKeys.size());
    for (const std::string_view key : kRegisteredOutputPathKeys) {
        const auto found = present.find(std::string(key));
        if (found == present.end())
            result.push_back({std::string(key), false, {}});
        else
            result.push_back({std::string(key), true, found->second});
    }
    return result;
}

inline std::string couponHexEncode(const std::string_view input) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(input.size() * 2u);
    for (const unsigned char byte : input) {
        output.push_back(digits[byte >> 4u]);
        output.push_back(digits[byte & 0x0fu]);
    }
    return output;
}

inline std::string couponHexDecode(const std::string_view input) {
    if (input.size() % 2u != 0u)
        throw std::invalid_argument("coupon path-map hex has odd length");
    auto nibble = [](const char value) -> unsigned char {
        if (value >= '0' && value <= '9')
            return static_cast<unsigned char>(value - '0');
        if (value >= 'a' && value <= 'f')
            return static_cast<unsigned char>(value - 'a' + 10);
        throw std::invalid_argument("coupon path-map hex is not lowercase hexadecimal");
    };
    std::string output;
    output.reserve(input.size() / 2u);
    for (std::size_t i = 0u; i < input.size(); i += 2u)
        output.push_back(static_cast<char>((nibble(input[i]) << 4u) |
                                           nibble(input[i + 1u])));
    return output;
}

inline std::string canonicalCouponOutputPathMap(const RuntimePathMap& map) {
    if (map.size() != kRegisteredOutputPathKeys.size())
        throw std::invalid_argument("coupon output-path map has the wrong key count");
    std::ostringstream output;
    output << "numi.cardboard.registered-diagnostic-output-path-map.v1\n";
    for (std::size_t i = 0u; i < kRegisteredOutputPathKeys.size(); ++i) {
        const RuntimePathBinding& binding = map[i];
        if (binding.key != kRegisteredOutputPathKeys[i] ||
            (!binding.present && !binding.value.empty()) ||
            binding.value.size() > 4096u ||
            binding.value.find('\0') != std::string::npos ||
            binding.value.find('\n') != std::string::npos ||
            binding.value.find('\r') != std::string::npos)
            throw std::invalid_argument("coupon output-path map is malformed");
        output << binding.key << '\t';
        if (!binding.present) output << "absent\n";
        else output << "set\t" << couponHexEncode(binding.value) << '\n';
    }
    const std::string result = output.str();
    if (result.size() > 60u * 1024u)
        throw std::invalid_argument("coupon output-path map exceeds its metadata bound");
    return result;
}

inline RuntimePathMap parseCouponOutputPathMap(const std::string_view input) {
    constexpr std::string_view header =
        "numi.cardboard.registered-diagnostic-output-path-map.v1\n";
    if (!input.starts_with(header) || input.size() > 60u * 1024u)
        throw std::invalid_argument("coupon output-path map schema or size is invalid");
    RuntimePathMap result;
    std::size_t offset = header.size();
    while (offset < input.size()) {
        const std::size_t end = input.find('\n', offset);
        if (end == std::string_view::npos)
            throw std::invalid_argument("coupon output-path map has a partial record");
        const std::string_view line = input.substr(offset, end - offset);
        const std::size_t firstTab = line.find('\t');
        if (firstTab == std::string_view::npos)
            throw std::invalid_argument("coupon output-path map record is malformed");
        const std::string key(line.substr(0u, firstTab));
        const std::string_view remainder = line.substr(firstTab + 1u);
        if (remainder == "absent") {
            result.push_back({key, false, {}});
        } else if (remainder.starts_with("set\t")) {
            result.push_back({key, true, couponHexDecode(remainder.substr(4u))});
        } else {
            throw std::invalid_argument("coupon output-path map state is invalid");
        }
        offset = end + 1u;
    }
    if (canonicalCouponOutputPathMap(result) != input)
        throw std::invalid_argument("coupon output-path map is noncanonical or duplicated");
    return result;
}

inline bool couponOutputPathMapHasAnySetValue(const RuntimePathMap& map) {
    (void)canonicalCouponOutputPathMap(map);
    return std::any_of(map.begin(), map.end(),
        [](const RuntimePathBinding& binding) { return binding.present; });
}

inline bool validateCouponOutputPathMapTransition(
    const RuntimePathMap& source, const RuntimePathMap& current,
    const bool sourceRunnerValidated, const bool currentRunnerValidated,
    const bool allowDiagnosticPathPresenceChange = false) {
    const std::string sourceCanonical = canonicalCouponOutputPathMap(source);
    const std::string currentCanonical = canonicalCouponOutputPathMap(current);
    if (sourceCanonical == currentCanonical) return false;
    bool changed = false;
    for (std::size_t i = 0u; i < source.size(); ++i) {
        if (source[i] == current[i]) continue;
        changed = true;
        const auto isAbsoluteNonempty = [](const RuntimePathBinding& binding) {
            return binding.present && !binding.value.empty() &&
                binding.value.front() == '/';
        };
        const bool bothPresent = source[i].present && current[i].present;
        const bool permittedPresenceChange = allowDiagnosticPathPresenceChange &&
            source[i].present != current[i].present;
        if ((!bothPresent && !permittedPresenceChange) ||
            (source[i].present && !isAbsoluteNonempty(source[i])) ||
            (current[i].present && !isAbsoluteNonempty(current[i])))
            throw std::invalid_argument(
                "registered output paths must be nonempty absolute paths; presence changes require explicit diagnostic replay");
    }
    if (changed && (!sourceRunnerValidated || !currentRunnerValidated))
        throw std::invalid_argument(
            "registered output-path relocation requires source-bound runner validation on both segments");
    return changed;
}

struct DiagnosticRunnerProof {
    std::string caseDirectory;
    std::string environmentRecordSha256;
    std::string nonOutputSettingsSha256;
    std::string predictionSha256;
    std::string contractSha256;
    RuntimePathMap outputPaths;
    std::vector<std::pair<std::string, std::string>> effectiveNativeSettings;
};

inline void validateDiagnosticRunnerProof(
    const DiagnosticRunnerProof& proof,
    const std::string_view expectedCaseDirectory,
    const RuntimePathMap& expectedPaths) {
    if (proof.caseDirectory != expectedCaseDirectory ||
        !isLowerSha256(proof.environmentRecordSha256) ||
        !isLowerSha256(proof.nonOutputSettingsSha256) ||
        !isLowerSha256(proof.predictionSha256) ||
        !isLowerSha256(proof.contractSha256) ||
        canonicalCouponOutputPathMap(proof.outputPaths) !=
            canonicalCouponOutputPathMap(expectedPaths))
        throw std::invalid_argument(
            "source-bound diagnostic runner proof does not match the exact case/path map");
}

inline void validateRuntimeFingerprints(const Metadata& metadata,
                                        const std::uint64_t sourcePhysics,
                                        const std::uint64_t deviceProgram) {
    if (metadata.sourcePhysicsFingerprint != sourcePhysics ||
        metadata.deviceProgramFingerprint != deviceProgram)
        throw std::invalid_argument(
            "coupon checkpoint runtime source/program fingerprints differ");
}

struct Request {
    bool saveControlRequested = false;
    std::uint32_t saveControlIndex = 0u;
    bool checkpointProvided = false;
    bool metadataProvided = false;
    bool diagnosticReplayRequested = false;
    bool shipping = false;
    bool pin = false;
    bool blank = false;
    bool assembly = false;
    bool fixtureJoint = false;
    bool crease = false;
    bool compileOnly = false;
    std::uint32_t environmentCount = 0u;
    std::uint64_t totalControls = 0u;
};

inline void validateRequest(const Request& request) {
    if (request.checkpointProvided != request.metadataProvided)
        throw std::invalid_argument(
            "coupon resume requires both checkpoint and metadata paths");
    const bool requested = request.saveControlRequested ||
        request.checkpointProvided;
    if (request.diagnosticReplayRequested &&
        (!request.checkpointProvided || request.saveControlRequested))
        throw std::invalid_argument(
            "diagnostic replay mode requires an accepted checkpoint resume");
    if (requested && (request.shipping || request.pin || request.blank ||
        request.assembly || request.fixtureJoint || request.crease ||
        request.compileOnly || request.environmentCount != 2u))
        throw std::invalid_argument(
            "accepted coupon resume is limited to the two-environment corrugated-strip diagnostic protocol");
    if (request.saveControlRequested && request.checkpointProvided)
        throw std::invalid_argument(
            "coupon checkpoint creation and resume cannot be combined in one invocation");
    if (request.saveControlRequested &&
        std::uint64_t(request.saveControlIndex) + 1u >= request.totalControls)
        throw std::invalid_argument(
            "coupon checkpoint must leave at least one scheduled control for continuation");
}

inline bool stopAfterAcceptedCheckpoint(const bool checkpointRequested,
                                        const std::uint32_t requestedControl,
                                        const std::uint32_t controlIndex) {
    return checkpointRequested && requestedControl == controlIndex;
}

inline bool safeSingleFilename(const std::string& value) {
    return !value.empty() && value != "." && value != ".." &&
        value.find('/') == std::string::npos &&
        value.find('\\') == std::string::npos &&
        value.find('\n') == std::string::npos &&
        value.find('\r') == std::string::npos;
}

inline void validate(const Metadata& value) {
    if (!isLowerSha256(value.checkpointSha256) || value.checkpointBytes == 0u ||
        !isLowerSha256(value.protocolManifestSha256) ||
        !isLowerSha256(value.runtimeEnvironmentSha256) ||
        !isLowerSha256(value.runtimeOutputPathMapSha256) ||
        (value.sourceRunnerEnvironmentRecordSha256 != "none" &&
            !isLowerSha256(value.sourceRunnerEnvironmentRecordSha256)) ||
        (value.sourceRunnerNonOutputSettingsSha256 != "none" &&
            !isLowerSha256(value.sourceRunnerNonOutputSettingsSha256)) ||
        (value.sourceRunnerPredictionSha256 != "none" &&
            !isLowerSha256(value.sourceRunnerPredictionSha256)) ||
        (value.sourceRunnerContractSha256 != "none" &&
            !isLowerSha256(value.sourceRunnerContractSha256)) ||
        ((value.sourceRunnerEnvironmentRecordSha256 == "none") !=
            (value.sourceRunnerPredictionSha256 == "none")) ||
        ((value.sourceRunnerNonOutputSettingsSha256 == "none") !=
            (value.sourceRunnerPredictionSha256 == "none")) ||
        ((value.sourceRunnerPredictionSha256 == "none") !=
            (value.sourceRunnerContractSha256 == "none")) ||
        value.runtimeOutputPathMapHex.empty() ||
        value.runtimeOutputPathMapHex.size() > 120u * 1024u ||
        !isLowerSha256(value.prefixManifestSha256) ||
        !safeSingleFilename(value.prefixManifestFile) ||
        value.sourceRunDirectory.empty() || value.sourceRunId.empty() ||
        value.sourceRunDirectory.find('\n') != std::string::npos ||
        value.sourceRunDirectory.find('\r') != std::string::npos ||
        value.sourceRunnerCaseDirectory.find('\n') != std::string::npos ||
        value.sourceRunnerCaseDirectory.find('\r') != std::string::npos ||
        ((value.sourceRunnerPredictionSha256 == "none") !=
            (value.sourceRunnerCaseDirectory == "none")) ||
        value.environmentCount != 2u || value.nodeCount == 0u ||
        value.tetrahedronCount == 0u || value.materialStateStride == 0u ||
        value.acceptedControlIndex == std::numeric_limits<std::uint32_t>::max() ||
        value.nextControlIndex == 0u ||
        value.nextControlIndex != value.acceptedControlIndex + 1u ||
        value.sourcePrefixControlCount != value.nextControlIndex ||
        value.timestepFloatBits == 0u ||
        value.scheduledElapsedSecondsBits == 0u || value.phase.empty() ||
        value.nextPhase.empty())
        throw std::invalid_argument("coupon resume metadata is inconsistent");
    const RuntimePathMap pathMap = parseCouponOutputPathMap(
        couponHexDecode(value.runtimeOutputPathMapHex));
    for (const RuntimePathBinding& binding : pathMap) {
        if (binding.present &&
            (binding.value.empty() || binding.value.front() != '/'))
            throw std::invalid_argument(
                "coupon checkpoint output paths must be nonempty absolute paths");
    }
    if (value.sourceRunnerPredictionSha256 != "none" &&
        (value.sourceRunnerCaseDirectory.empty() ||
         value.sourceRunnerCaseDirectory.front() != '/'))
        throw std::invalid_argument(
            "coupon source-bound runner case directory must be absolute");
    if (couponOutputPathMapHasAnySetValue(pathMap) &&
        value.sourceRunnerPredictionSha256 == "none")
        throw std::invalid_argument(
            "coupon output paths require a bound source runner contract");
}

inline Metadata parseMetadata(std::istream& input,
                              const std::string_view sourceName) {
    const std::set<std::string> keys{
        "schema", "checkpoint_sha256", "checkpoint_bytes",
        "source_physics_fingerprint", "device_program_fingerprint",
        "protocol_manifest_sha256", "prefix_manifest_sha256",
        "runtime_environment_sha256", "runtime_output_path_map_hex",
        "runtime_output_path_map_sha256", "source_runner_prediction_sha256",
        "source_runner_contract_sha256", "source_runner_case_directory",
        "source_runner_environment_record_sha256",
        "source_runner_non_output_settings_sha256",
        "prefix_manifest_file", "source_run_directory", "source_run_id",
        "environment_count", "node_count", "tetrahedron_count",
        "material_state_stride", "accepted_control_index",
        "next_control_index", "physics_substep",
        "source_prefix_control_count", "timestep_f32_bits",
        "scheduled_elapsed_seconds_f64_bits", "phase", "next_phase",
        "next_target_angle_f64_bits",
        "both_environments_accepted"};
    const auto values = parseProperties(input, keys, sourceName);
    if (property(values, "schema") !=
            "numi.cardboard.accepted-coupon-resume.v2" ||
        property(values, "both_environments_accepted") != "true")
        throw std::invalid_argument("coupon resume schema or accepted-state gate is invalid");
    Metadata result;
    result.checkpointSha256 = property(values, "checkpoint_sha256");
    result.checkpointBytes = parseUnsigned<std::uint64_t>(
        property(values, "checkpoint_bytes"), "checkpoint byte count");
    result.sourcePhysicsFingerprint = parseUnsigned<std::uint64_t>(
        property(values, "source_physics_fingerprint"), "source physics fingerprint");
    result.deviceProgramFingerprint = parseUnsigned<std::uint64_t>(
        property(values, "device_program_fingerprint"), "device program fingerprint");
    result.protocolManifestSha256 = property(values, "protocol_manifest_sha256");
    result.runtimeEnvironmentSha256 = property(values, "runtime_environment_sha256");
    result.runtimeOutputPathMapHex = property(values, "runtime_output_path_map_hex");
    result.runtimeOutputPathMapSha256 = property(values, "runtime_output_path_map_sha256");
    result.sourceRunnerCaseDirectory = property(values, "source_runner_case_directory");
    result.sourceRunnerEnvironmentRecordSha256 = property(values, "source_runner_environment_record_sha256");
    result.sourceRunnerNonOutputSettingsSha256 = property(values, "source_runner_non_output_settings_sha256");
    result.sourceRunnerPredictionSha256 = property(values, "source_runner_prediction_sha256");
    result.sourceRunnerContractSha256 = property(values, "source_runner_contract_sha256");
    result.prefixManifestSha256 = property(values, "prefix_manifest_sha256");
    result.prefixManifestFile = property(values, "prefix_manifest_file");
    result.sourceRunDirectory = property(values, "source_run_directory");
    result.sourceRunId = property(values, "source_run_id");
    result.environmentCount = parseUnsigned<std::uint32_t>(
        property(values, "environment_count"), "environment count");
    result.nodeCount = parseUnsigned<std::uint32_t>(
        property(values, "node_count"), "node count");
    result.tetrahedronCount = parseUnsigned<std::uint32_t>(
        property(values, "tetrahedron_count"), "tetrahedron count");
    result.materialStateStride = parseUnsigned<std::uint32_t>(
        property(values, "material_state_stride"), "material state stride");
    result.acceptedControlIndex = parseUnsigned<std::uint32_t>(
        property(values, "accepted_control_index"), "accepted control index");
    result.nextControlIndex = parseUnsigned<std::uint32_t>(
        property(values, "next_control_index"), "next control index");
    result.physicsSubstep = parseUnsigned<std::uint32_t>(
        property(values, "physics_substep"), "physics substep");
    result.sourcePrefixControlCount = parseUnsigned<std::uint32_t>(
        property(values, "source_prefix_control_count"), "source prefix control count");
    result.timestepFloatBits = parseUnsigned<std::uint32_t>(
        property(values, "timestep_f32_bits"), "timestep bits");
    result.scheduledElapsedSecondsBits = parseUnsigned<std::uint64_t>(
        property(values, "scheduled_elapsed_seconds_f64_bits"),
        "scheduled elapsed time bits");
    result.phase = property(values, "phase");
    result.nextPhase = property(values, "next_phase");
    result.nextTargetAngleBits = parseUnsigned<std::uint64_t>(
        property(values, "next_target_angle_f64_bits"), "next target angle bits");
    validate(result);
    return result;
}

inline std::string serializeMetadata(const Metadata& value) {
    validate(value);
    std::ostringstream output;
    output << "schema=numi.cardboard.accepted-coupon-resume.v2\n"
        << "checkpoint_sha256=" << value.checkpointSha256 << '\n'
        << "checkpoint_bytes=" << value.checkpointBytes << '\n'
        << "source_physics_fingerprint=" << value.sourcePhysicsFingerprint << '\n'
        << "device_program_fingerprint=" << value.deviceProgramFingerprint << '\n'
        << "protocol_manifest_sha256=" << value.protocolManifestSha256 << '\n'
        << "runtime_environment_sha256=" << value.runtimeEnvironmentSha256 << '\n'
        << "runtime_output_path_map_hex=" << value.runtimeOutputPathMapHex << '\n'
        << "runtime_output_path_map_sha256=" << value.runtimeOutputPathMapSha256 << '\n'
        << "source_runner_case_directory=" << value.sourceRunnerCaseDirectory << '\n'
        << "source_runner_environment_record_sha256=" << value.sourceRunnerEnvironmentRecordSha256 << '\n'
        << "source_runner_non_output_settings_sha256=" << value.sourceRunnerNonOutputSettingsSha256 << '\n'
        << "source_runner_prediction_sha256=" << value.sourceRunnerPredictionSha256 << '\n'
        << "source_runner_contract_sha256=" << value.sourceRunnerContractSha256 << '\n'
        << "prefix_manifest_sha256=" << value.prefixManifestSha256 << '\n'
        << "prefix_manifest_file=" << value.prefixManifestFile << '\n'
        << "source_run_directory=" << value.sourceRunDirectory << '\n'
        << "source_run_id=" << value.sourceRunId << '\n'
        << "environment_count=" << value.environmentCount << '\n'
        << "node_count=" << value.nodeCount << '\n'
        << "tetrahedron_count=" << value.tetrahedronCount << '\n'
        << "material_state_stride=" << value.materialStateStride << '\n'
        << "accepted_control_index=" << value.acceptedControlIndex << '\n'
        << "next_control_index=" << value.nextControlIndex << '\n'
        << "physics_substep=" << value.physicsSubstep << '\n'
        << "source_prefix_control_count=" << value.sourcePrefixControlCount << '\n'
        << "timestep_f32_bits=" << value.timestepFloatBits << '\n'
        << "scheduled_elapsed_seconds_f64_bits="
        << value.scheduledElapsedSecondsBits << '\n'
        << "phase=" << value.phase << '\n'
        << "next_phase=" << value.nextPhase << '\n'
        << "next_target_angle_f64_bits=" << value.nextTargetAngleBits << '\n'
        << "both_environments_accepted=true\n";
    return output.str();
}

struct PrefixEntry {
    std::string relativePath;
    std::uint64_t bytes = 0u;
    std::string sha256;
    bool observationsPrefix = false;
};

struct PrefixManifest {
    std::string sourceRunDirectory;
    std::uint32_t acceptedControlIndex = 0u;
    std::uint64_t observationsPrefixBytes = 0u;
    std::string observationsPrefixSha256;
    std::vector<PrefixEntry> files;
};

inline bool safeRelativePath(const std::string& path) {
    if (path.empty() || path.front() == '/' ||
        path.find('\\') != std::string::npos || path.find('|') != std::string::npos ||
        path.find('\n') != std::string::npos || path.find('\r') != std::string::npos)
        return false;
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/'))
        if (part.empty() || part == "." || part == "..") return false;
    return true;
}

inline std::string serializePrefixManifest(const PrefixManifest& value) {
    if (value.sourceRunDirectory.empty() || value.files.empty() ||
        value.files.size() > 8192u ||
        !isLowerSha256(value.observationsPrefixSha256) ||
        value.sourceRunDirectory.find('\n') != std::string::npos ||
        value.sourceRunDirectory.find('\r') != std::string::npos)
        throw std::invalid_argument("coupon prefix manifest is incomplete");
    std::ostringstream output;
    output << "schema=numi.cardboard.accepted-coupon-prefix.v1\n"
        << "source_run_directory=" << value.sourceRunDirectory << '\n'
        << "accepted_control_index=" << value.acceptedControlIndex << '\n'
        << "observations_prefix_bytes=" << value.observationsPrefixBytes << '\n'
        << "observations_prefix_sha256=" << value.observationsPrefixSha256 << '\n';
    const PrefixEntry* observationsEntry = nullptr;
    std::set<std::string> unique;
    std::uint64_t totalBytes = 0u;
    for (const auto& entry : value.files) {
        if (!safeRelativePath(entry.relativePath) ||
            !isLowerSha256(entry.sha256) || !unique.insert(entry.relativePath).second)
            throw std::invalid_argument("coupon prefix manifest has an invalid entry");
        if (entry.bytes > 2ull * 1024ull * 1024ull * 1024ull - totalBytes)
            throw std::invalid_argument("coupon prefix manifest exceeds its byte bound");
        totalBytes += entry.bytes;
        if (entry.relativePath == "observations.csv") {
            if (!entry.observationsPrefix)
                throw std::invalid_argument("observations.csv must be a prefix entry");
            observationsEntry = &entry;
        } else if (entry.observationsPrefix) {
            throw std::invalid_argument("only observations.csv may be a prefix entry");
        }
        output << "file=" << entry.relativePath << '|' << entry.bytes << '|'
            << entry.sha256 << '|' << (entry.observationsPrefix ? 1 : 0) << '\n';
    }
    if (observationsEntry == nullptr ||
        observationsEntry->bytes != value.observationsPrefixBytes ||
        observationsEntry->sha256 != value.observationsPrefixSha256)
        throw std::invalid_argument("coupon prefix manifest omits observations.csv");
    return output.str();
}

inline PrefixManifest parsePrefixManifest(std::istream& input,
                                          const std::string_view sourceName) {
    PrefixManifest result;
    std::set<std::string> unique;
    bool schemaSeen = false, dirSeen = false, controlSeen = false;
    bool bytesSeen = false, hashSeen = false, observationsSeen = false;
    std::uint64_t totalBytes = 0u;
    std::string line;
    std::uint32_t lineNumber = 0u;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(std::move(line));
        if (line.empty()) continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            throw std::invalid_argument(std::string(sourceName) + ":" +
                std::to_string(lineNumber) + " has malformed line");
        const std::string key = line.substr(0u, separator);
        const std::string value = line.substr(separator + 1u);
        if (key == "schema") {
            if (schemaSeen || value != "numi.cardboard.accepted-coupon-prefix.v1")
                throw std::invalid_argument("coupon prefix schema is invalid");
            schemaSeen = true;
        } else if (key == "source_run_directory") {
            if (dirSeen || value.empty() ||
                value.find('\n') != std::string::npos ||
                value.find('\r') != std::string::npos)
                throw std::invalid_argument("coupon prefix source directory is invalid");
            result.sourceRunDirectory = value; dirSeen = true;
        } else if (key == "accepted_control_index") {
            if (controlSeen) throw std::invalid_argument("coupon prefix control is duplicated");
            result.acceptedControlIndex = parseUnsigned<std::uint32_t>(value, "prefix control");
            controlSeen = true;
        } else if (key == "observations_prefix_bytes") {
            if (bytesSeen) throw std::invalid_argument("coupon prefix observation size is duplicated");
            result.observationsPrefixBytes = parseUnsigned<std::uint64_t>(value, "prefix observation size");
            bytesSeen = true;
        } else if (key == "observations_prefix_sha256") {
            if (hashSeen || !isLowerSha256(value)) throw std::invalid_argument("coupon prefix observation hash is invalid");
            result.observationsPrefixSha256 = value; hashSeen = true;
        } else if (key == "file") {
            if (result.files.size() >= 8192u)
                throw std::invalid_argument("coupon prefix manifest exceeds its file-count bound");
            std::array<std::string, 4> fields{};
            std::size_t cursor = 0u;
            for (std::size_t field = 0u; field < fields.size(); ++field) {
                const auto end = value.find('|', cursor);
                if (field + 1u == fields.size()) {
                    if (end != std::string::npos) throw std::invalid_argument("coupon prefix file has extra fields");
                    fields[field] = value.substr(cursor);
                } else {
                    if (end == std::string::npos) throw std::invalid_argument("coupon prefix file is truncated");
                    fields[field] = value.substr(cursor, end - cursor);
                    cursor = end + 1u;
                }
            }
            PrefixEntry entry;
            entry.relativePath = fields[0];
            entry.bytes = parseUnsigned<std::uint64_t>(fields[1], "prefix file byte count");
            entry.sha256 = fields[2];
            const auto isPrefix = parseUnsigned<std::uint32_t>(fields[3], "prefix file kind");
            if (isPrefix > 1u || !safeRelativePath(entry.relativePath) ||
                !isLowerSha256(entry.sha256) || !unique.insert(entry.relativePath).second)
                throw std::invalid_argument("coupon prefix file entry is invalid or repeated");
            entry.observationsPrefix = isPrefix == 1u;
            if (entry.relativePath == "observations.csv") {
                if (!entry.observationsPrefix || observationsSeen ||
                    entry.bytes != result.observationsPrefixBytes ||
                    entry.sha256 != result.observationsPrefixSha256)
                    throw std::invalid_argument("coupon prefix observations entry does not match its prefix digest");
                observationsSeen = true;
            } else if (entry.observationsPrefix) {
                throw std::invalid_argument("only observations.csv may be a prefix entry");
            }
            if (entry.bytes > 2ull * 1024ull * 1024ull * 1024ull - totalBytes)
                throw std::invalid_argument("coupon prefix manifest exceeds its byte bound");
            totalBytes += entry.bytes;
            result.files.push_back(std::move(entry));
        } else {
            throw std::invalid_argument(std::string(sourceName) + " has unsupported key '" + key + "'");
        }
    }
    if (!input.eof() || !schemaSeen || !dirSeen || !controlSeen || !bytesSeen ||
        !hashSeen || !observationsSeen || result.files.empty())
        throw std::invalid_argument("coupon prefix manifest is incomplete");
    return result;
}

} // namespace numi::cardboard::coupon_resume
