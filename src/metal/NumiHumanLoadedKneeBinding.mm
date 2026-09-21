#import <Foundation/Foundation.h>

#include "metalrobo/NumiHumanLoadedKneeBinding.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::string_view kManifestName =
    "HumanPack.loaded-anatomy-knee.v1.json";
constexpr std::string_view kBindingName =
    "HumanPack.loaded-anatomy-knee.binding.v1.json";
constexpr std::string_view kSourceComplianceName =
    "HumanPack.loaded-anatomy-knee.source-compliance.v1.json";
constexpr std::string_view kCanonicalization =
    "utf8-json-sorted-keys-compact-ensure_ascii=false-allow_nan=false";
constexpr std::string_view kOwnershipManifestSHA256 =
    "7083f238815a4baf40542801b2fb632efdf81f930e56de0c016ec76c1278276e";
constexpr std::string_view kOwnershipFileSHA256 =
    "db8ccacde8ecc49b29de50f07b21340d4abfc040e606117a8d8ff011ea8d5824";
constexpr std::string_view kRawF32NodeMassAlgorithm =
    "matter-referenced-f32-volume-f32-density-fp64-source-order-accumulate-"
    "final-f32.1";
constexpr std::string_view kSourceToReferenceMappingAlgorithm =
    "adaptive-dyadic-first-success-slerp-geodesic-inverse-distance-moving-"
    "enthesis.1";
constexpr std::string_view kMappingCodeIdentityEncoding =
    "sha256(domain-utf8||header-file-sha256||core-file-sha256||adapter-file-"
    "sha256)";
constexpr std::string_view kSourceComplianceSchema =
    "HumanPack.loaded-anatomy-knee.source-compliance.v1";
constexpr std::string_view kSourceComplianceCompiler =
    "numilab-human.loaded-anatomy-knee-source-compliance.1";
constexpr std::string_view kSourceArchiveSHA256 =
    "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975";
constexpr std::string_view kSourceRigidPayloadSHA256 =
    "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44";
constexpr std::string_view kSourceComplianceBoundary =
    "Candidate-only additive source-compliance binding for "
    "HumanPack.loaded-anatomy-knee.v1. Human authors the immutable NHEQ2 and "
    "NHLIM1 source-law bytes; Matter alone owns runtime constraint force and "
    "accepted constraint state. This companion binds no prepared-state identity "
    "and does not establish runtime execution, production physical ownership, "
    "standing, walking, clinical validity, or integrated Human qualification.";
constexpr std::size_t kMaximumJSONNestingDepth = 128u;

class AdmissionError final : public std::runtime_error {
public:
  explicit AdmissionError(const std::string &message)
      : std::runtime_error("loaded-knee binding: " + message) {}
};

[[noreturn]] void reject(const std::string &message) {
  throw AdmissionError(message);
}

void require(const bool condition, const std::string &message) {
  if (!condition)
    reject(message);
}

NumiHumanLoadedKneeDigest sha256(const std::span<const std::uint8_t> bytes) {
  NumiHumanLoadedKneeDigest result{};
  CC_SHA256_CTX context{};
  require(CC_SHA256_Init(&context) == 1, "SHA-256 initialization failed");
  std::size_t offset = 0u;
  while (offset < bytes.size()) {
    const std::size_t count = std::min<std::size_t>(
        bytes.size() - offset,
        static_cast<std::size_t>(std::numeric_limits<CC_LONG>::max()));
    require(CC_SHA256_Update(&context, bytes.data() + offset,
                             static_cast<CC_LONG>(count)) == 1,
            "SHA-256 update failed");
    offset += count;
  }
  require(CC_SHA256_Final(result.data(), &context) == 1,
          "SHA-256 finalization failed");
  return result;
}

std::vector<std::uint8_t> readRegularFile(const std::filesystem::path &path,
                                          const std::uint64_t maximumBytes,
                                          const std::string_view label) {
  const std::string native = path.string();
  require(!native.empty(), std::string(label) + " path is empty");
  const int descriptor =
      ::open(native.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0)
    reject(std::string(label) + " cannot be opened");
  struct DescriptorGuard {
    int value;
    ~DescriptorGuard() {
      if (value >= 0)
        ::close(value);
    }
  } guard{descriptor};
  struct stat status{};
  require(::fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode),
          std::string(label) + " is not a regular file");
  require(status.st_size > 0 &&
              static_cast<std::uint64_t>(status.st_size) <= maximumBytes,
          std::string(label) + " byte count is outside its admission bound");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(status.st_size));
  std::size_t offset = 0u;
  while (offset < result.size()) {
    const ssize_t count =
        ::read(descriptor, result.data() + offset, result.size() - offset);
    require(count > 0, std::string(label) + " read was incomplete");
    offset += static_cast<std::size_t>(count);
  }
  std::uint8_t trailing = 0u;
  require(::read(descriptor, &trailing, 1u) == 0,
          std::string(label) + " changed while it was read");
  return result;
}

struct JSONMemberSpan {
  std::string key;
  std::size_t pairBegin = 0u;
  std::size_t valueBegin = 0u;
  std::size_t valueEnd = 0u;
  std::size_t pairEnd = 0u;
};

// json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,
// allow_nan=False) uses Python's shortest-roundtrip float representation and
// switches to exponent notation outside [-4, 16).  std::to_chars supplies the
// same shortest significant digits; this normalizes only the presentation.
std::string pythonFloat(const double value) {
  require(std::isfinite(value), "canonical JSON contains a non-finite number");
  if (value == 0.0)
    return std::signbit(value) ? "-0.0" : "0.0";
  std::array<char, 128u> storage{};
  const auto converted =
      std::to_chars(storage.data(), storage.data() + storage.size(), value,
                    std::chars_format::general);
  require(converted.ec == std::errc{},
          "canonical JSON number formatting failed");
  std::string rendered(storage.data(), converted.ptr);
  std::string sign;
  if (!rendered.empty() && rendered.front() == '-') {
    sign = "-";
    rendered.erase(rendered.begin());
  }
  int explicitExponent = 0;
  const auto exponentPosition = rendered.find_first_of("eE");
  if (exponentPosition != std::string::npos) {
    std::string exponent = rendered.substr(exponentPosition + 1u);
    bool negativeExponent = false;
    if (!exponent.empty() &&
        (exponent.front() == '+' || exponent.front() == '-')) {
      negativeExponent = exponent.front() == '-';
      exponent.erase(exponent.begin());
    }
    const auto parsed = std::from_chars(
        exponent.data(), exponent.data() + exponent.size(), explicitExponent);
    require(parsed.ec == std::errc{} &&
                parsed.ptr == exponent.data() + exponent.size(),
            "canonical JSON exponent formatting failed");
    if (negativeExponent)
      explicitExponent = -explicitExponent;
    rendered.resize(exponentPosition);
  }
  const auto decimalPosition = rendered.find('.');
  const std::size_t integerDigits =
      decimalPosition == std::string::npos ? rendered.size() : decimalPosition;
  std::string digits = rendered;
  if (decimalPosition != std::string::npos)
    digits.erase(decimalPosition, 1u);
  const auto firstNonzero = digits.find_first_not_of('0');
  require(firstNonzero != std::string::npos,
          "canonical JSON number normalization failed");
  if (firstNonzero != 0u)
    digits.erase(0u, firstNonzero);
  int scientificExponent = explicitExponent + static_cast<int>(integerDigits) -
                           1 - static_cast<int>(firstNonzero);

  std::string result = sign;
  if (scientificExponent < -4 || scientificExponent >= 16) {
    result.push_back(digits.front());
    if (digits.size() > 1u) {
      result.push_back('.');
      result.append(digits.begin() + 1, digits.end());
    }
    result.push_back('e');
    result.push_back(scientificExponent < 0 ? '-' : '+');
    const unsigned magnitude = static_cast<unsigned>(
        scientificExponent < 0 ? -scientificExponent : scientificExponent);
    if (magnitude < 10u)
      result.push_back('0');
    result += std::to_string(magnitude);
    return result;
  }

  const int decimal = scientificExponent + 1;
  if (decimal <= 0) {
    result += "0.";
    result.append(static_cast<std::size_t>(-decimal), '0');
    result += digits;
  } else if (static_cast<std::size_t>(decimal) >= digits.size()) {
    result += digits;
    result.append(static_cast<std::size_t>(decimal) - digits.size(), '0');
    result += ".0";
  } else {
    result.append(digits.data(), static_cast<std::size_t>(decimal));
    result.push_back('.');
    result.append(digits.data() + decimal,
                  digits.size() - static_cast<std::size_t>(decimal));
  }
  return result;
}

class CanonicalJSONScanner final {
public:
  explicit CanonicalJSONScanner(const std::span<const std::uint8_t> bytes)
      : bytes_(bytes) {}

  std::vector<JSONMemberSpan> validateDocument() {
    require(bytes_.size() >= 3u && bytes_.back() == '\n' &&
                bytes_[bytes_.size() - 2u] != '\n',
            "JSON artifact must end in exactly one LF");
    end_ = bytes_.size() - 1u;
    position_ = 0u;
    rootMembers_.clear();
    parseObject(true, 1u);
    require(position_ == end_,
            "JSON artifact has bytes outside its root object");
    return rootMembers_;
  }

private:
  void expect(const char value) {
    require(position_ < end_ &&
                bytes_[position_] == static_cast<std::uint8_t>(value),
            "JSON artifact is not compact canonical JSON");
    ++position_;
  }

  static bool isLowerHex(const std::uint8_t value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  }

  std::string parseString(const bool key) {
    expect('"');
    std::string keyValue;
    while (position_ < end_) {
      const std::uint8_t value = bytes_[position_++];
      if (value == '"')
        return keyValue;
      require(value >= 0x20u, "JSON string contains an unescaped control byte");
      if (value == '\\') {
        require(position_ < end_, "JSON string escape is truncated");
        const std::uint8_t escape = bytes_[position_++];
        require(!key, "canonical JSON object keys must be unescaped ASCII");
        if (escape == '"' || escape == '\\' || escape == 'b' || escape == 'f' ||
            escape == 'n' || escape == 'r' || escape == 't')
          continue;
        require(escape == 'u' && position_ + 4u <= end_,
                "JSON string has a non-canonical escape");
        for (std::size_t index = 0u; index < 4u; ++index) {
          require(isLowerHex(bytes_[position_ + index]),
                  "JSON Unicode escape is not lowercase canonical hex");
        }
        require(bytes_[position_] == '0' && bytes_[position_ + 1u] == '0',
                "ensure_ascii=false string uses an unnecessary Unicode escape");
        const auto nibble = [](const std::uint8_t byte) {
          return byte <= '9' ? static_cast<unsigned>(byte - '0')
                             : static_cast<unsigned>(byte - 'a' + 10u);
        };
        const unsigned control = nibble(bytes_[position_ + 2u]) * 16u +
                                 nibble(bytes_[position_ + 3u]);
        require(control != 0u, "JSON string contains an embedded NUL");
        require(control < 0x20u && control != 0x08u && control != 0x09u &&
                    control != 0x0au && control != 0x0cu && control != 0x0du,
                "JSON string uses a non-minimal Unicode escape");
        position_ += 4u;
        continue;
      }
      if (key) {
        require(value < 0x80u, "canonical JSON object keys must be ASCII");
        keyValue.push_back(static_cast<char>(value));
      }
    }
    reject("JSON string is unterminated");
  }

  void parseNumber() {
    const std::size_t begin = position_;
    if (bytes_[position_] == '-')
      ++position_;
    require(position_ < end_, "JSON number is truncated");
    if (bytes_[position_] == '0') {
      ++position_;
      require(position_ == end_ || bytes_[position_] < '0' ||
                  bytes_[position_] > '9',
              "JSON number has a leading zero");
    } else {
      require(bytes_[position_] >= '1' && bytes_[position_] <= '9',
              "JSON number integer part is invalid");
      while (position_ < end_ && bytes_[position_] >= '0' &&
             bytes_[position_] <= '9')
        ++position_;
    }
    bool floating = false;
    if (position_ < end_ && bytes_[position_] == '.') {
      floating = true;
      ++position_;
      const std::size_t fraction = position_;
      while (position_ < end_ && bytes_[position_] >= '0' &&
             bytes_[position_] <= '9')
        ++position_;
      require(position_ != fraction, "JSON number fraction is empty");
    }
    if (position_ < end_ &&
        (bytes_[position_] == 'e' || bytes_[position_] == 'E')) {
      floating = true;
      ++position_;
      if (position_ < end_ &&
          (bytes_[position_] == '+' || bytes_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent = position_;
      while (position_ < end_ && bytes_[position_] >= '0' &&
             bytes_[position_] <= '9')
        ++position_;
      require(position_ != exponent, "JSON number exponent is empty");
    }
    const std::string token(
        reinterpret_cast<const char *>(bytes_.data() + begin),
        position_ - begin);
    if (!floating) {
      // Python emits arbitrary precision base-10 integers in this form.
      return;
    }
    double value = 0.0;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value,
                        std::chars_format::general);
    require(parsed.ec == std::errc{} &&
                parsed.ptr == token.data() + token.size() &&
                pythonFloat(value) == token,
            "JSON float is not Python canonical shortest-roundtrip form");
  }

  void literal(const std::string_view value) {
    require(position_ + value.size() <= end_ &&
                std::memcmp(bytes_.data() + position_, value.data(),
                            value.size()) == 0,
            "JSON literal is invalid");
    position_ += value.size();
  }

  void parseArray(const std::size_t depth) {
    require(depth <= kMaximumJSONNestingDepth,
            "JSON nesting exceeds its admission bound");
    expect('[');
    if (position_ < end_ && bytes_[position_] == ']') {
      ++position_;
      return;
    }
    for (;;) {
      parseValue(depth + 1u);
      require(position_ < end_, "JSON array is unterminated");
      if (bytes_[position_] == ']') {
        ++position_;
        return;
      }
      expect(',');
    }
  }

  void parseObject(const bool root, const std::size_t depth) {
    require(depth <= kMaximumJSONNestingDepth,
            "JSON nesting exceeds its admission bound");
    expect('{');
    if (position_ < end_ && bytes_[position_] == '}') {
      ++position_;
      return;
    }
    std::string previous;
    bool first = true;
    for (;;) {
      const std::size_t pairBegin = position_;
      const std::string key = parseString(true);
      require(first || previous < key,
              "JSON object keys are duplicated or not sorted");
      first = false;
      previous = key;
      expect(':');
      const std::size_t valueBegin = position_;
      parseValue(depth + 1u);
      const std::size_t valueEnd = position_;
      if (root)
        rootMembers_.push_back(
            {key, pairBegin, valueBegin, valueEnd, position_});
      require(position_ < end_, "JSON object is unterminated");
      if (bytes_[position_] == '}') {
        ++position_;
        return;
      }
      expect(',');
    }
  }

  void parseValue(const std::size_t depth) {
    require(position_ < end_, "JSON value is missing");
    switch (bytes_[position_]) {
    case '{':
      parseObject(false, depth);
      return;
    case '[':
      parseArray(depth);
      return;
    case '"':
      (void)parseString(false);
      return;
    case 't':
      literal("true");
      return;
    case 'f':
      literal("false");
      return;
    case 'n':
      literal("null");
      return;
    default:
      require(bytes_[position_] == '-' ||
                  (bytes_[position_] >= '0' && bytes_[position_] <= '9'),
              "JSON value token is invalid");
      parseNumber();
    }
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t position_ = 0u;
  std::size_t end_ = 0u;
  std::vector<JSONMemberSpan> rootMembers_;
};

const JSONMemberSpan &memberSpan(const std::vector<JSONMemberSpan> &members,
                                 const std::string_view key) {
  const auto found = std::find_if(
      members.begin(), members.end(),
      [key](const JSONMemberSpan &value) { return value.key == key; });
  require(found != members.end(),
          "canonical JSON omits required top-level field " + std::string(key));
  return *found;
}

NumiHumanLoadedKneeDigest
hashExcludingTopLevelMember(const std::span<const std::uint8_t> bytes,
                            const std::vector<JSONMemberSpan> &members,
                            const std::string_view excluded) {
  std::vector<std::uint8_t> reduced;
  reduced.reserve(bytes.size());
  reduced.push_back('{');
  bool first = true;
  bool found = false;
  for (const auto &member : members) {
    if (member.key == excluded) {
      found = true;
      continue;
    }
    if (!first)
      reduced.push_back(',');
    first = false;
    reduced.insert(reduced.end(), bytes.begin() + member.pairBegin,
                   bytes.begin() + member.pairEnd);
  }
  reduced.push_back('}');
  require(found, "hash-excluded JSON field is absent");
  return sha256(reduced);
}

NSDictionary *dictionary(id value, const std::string_view label) {
  require(value != nil && [value isKindOfClass:[NSDictionary class]],
          std::string(label) + " must be an object");
  return static_cast<NSDictionary *>(value);
}

void requireKeys(NSDictionary *object,
                 const std::initializer_list<std::string_view> expected,
                 const std::string_view label) {
  require(object.count == expected.size(),
          std::string(label) + " field set differs");
  std::set<std::string> names;
  for (id key in object) {
    require([key isKindOfClass:[NSString class]],
            std::string(label) + " has a non-string key");
    const char *bytes = [static_cast<NSString *>(key) UTF8String];
    require(bytes != nullptr, std::string(label) + " has a non-UTF-8 key");
    names.emplace(bytes);
  }
  for (const auto name : expected) {
    require(names.contains(std::string(name)),
            std::string(label) + " omits field " + std::string(name));
  }
}

NSArray *array(id value, const std::string_view label) {
  require(value != nil && [value isKindOfClass:[NSArray class]],
          std::string(label) + " must be an array");
  return static_cast<NSArray *>(value);
}

std::string stringValue(id value, const std::string_view label) {
  require(value != nil && [value isKindOfClass:[NSString class]],
          std::string(label) + " must be a string");
  NSString *string = static_cast<NSString *>(value);
  for (NSUInteger index = 0u; index < string.length; ++index) {
    require([string characterAtIndex:index] != 0u,
            std::string(label) + " contains an embedded NUL");
  }
  NSData *encoded = [string dataUsingEncoding:NSUTF8StringEncoding
                         allowLossyConversion:NO];
  require(encoded != nil, std::string(label) + " is not valid UTF-8");
  if (encoded.length == 0u)
    return {};
  return std::string(static_cast<const char *>(encoded.bytes), encoded.length);
}

bool isBoolean(id value) {
  return value != nil &&
         CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

bool booleanValue(id value, const std::string_view label) {
  require(isBoolean(value), std::string(label) + " must be a Boolean");
  return [static_cast<NSNumber *>(value) boolValue];
}

double doubleValue(id value, const std::string_view label) {
  require(value != nil && [value isKindOfClass:[NSNumber class]] &&
              !isBoolean(value),
          std::string(label) + " must be numeric");
  const double result = [static_cast<NSNumber *>(value) doubleValue];
  require(std::isfinite(result), std::string(label) + " must be finite");
  return result;
}

std::uint64_t unsignedValue(id value, const std::string_view label) {
  const double asDouble = doubleValue(value, label);
  require(asDouble >= 0.0 && std::floor(asDouble) == asDouble &&
              asDouble <= static_cast<double>(
                              std::numeric_limits<std::uint64_t>::max()),
          std::string(label) + " must be an unsigned integer");
  const unsigned long long result =
      [static_cast<NSNumber *>(value) unsignedLongLongValue];
  require(static_cast<double>(result) == asDouble,
          std::string(label) + " is not exactly representable");
  return static_cast<std::uint64_t>(result);
}

id field(NSDictionary *object, NSString *key, const std::string_view label) {
  id value = object[key];
  require(value != nil, std::string(label) + " is missing");
  return value;
}

std::string stringField(NSDictionary *object, NSString *key,
                        const std::string_view label) {
  return stringValue(field(object, key, label), label);
}

bool booleanField(NSDictionary *object, NSString *key,
                  const std::string_view label) {
  return booleanValue(field(object, key, label), label);
}

double doubleField(NSDictionary *object, NSString *key,
                   const std::string_view label) {
  return doubleValue(field(object, key, label), label);
}

std::uint64_t unsignedField(NSDictionary *object, NSString *key,
                            const std::string_view label) {
  return unsignedValue(field(object, key, label), label);
}

std::uint32_t unsigned32Field(NSDictionary *object, NSString *key,
                              const std::string_view label) {
  const std::uint64_t value = unsignedField(object, key, label);
  require(value <= std::numeric_limits<std::uint32_t>::max(),
          std::string(label) + " exceeds uint32");
  return static_cast<std::uint32_t>(value);
}

NumiHumanLoadedKneeDigest parseDigest(const std::string &text,
                                      const std::string_view label) {
  require(text.size() == 64u,
          std::string(label) + " must be a lowercase SHA-256");
  NumiHumanLoadedKneeDigest result{};
  const auto nibble = [label](const char value) -> std::uint8_t {
    if (value >= '0' && value <= '9')
      return static_cast<std::uint8_t>(value - '0');
    if (value >= 'a' && value <= 'f')
      return static_cast<std::uint8_t>(value - 'a' + 10);
    reject(std::string(label) + " must be a lowercase SHA-256");
  };
  for (std::size_t index = 0u; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>((nibble(text[index * 2u]) << 4u) |
                                              nibble(text[index * 2u + 1u]));
  }
  return result;
}

NumiHumanLoadedKneeDigest digestField(NSDictionary *object, NSString *key,
                                      const std::string_view label) {
  return parseDigest(stringField(object, key, label), label);
}

NSDictionary *parseJSONObject(const std::vector<std::uint8_t> &bytes,
                              const std::string_view label) {
  NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
  NSError *error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data
                                             options:0
                                               error:&error];
  require(error == nil && value != nil,
          std::string(label) + " is not valid UTF-8 JSON");
  return dictionary(value, label);
}

NumiHumanLoadedKneeImmutableIdentityV1
immutableIdentity(NSDictionary *value, const std::string_view label) {
  return {
      .schema = stringField(value, @"schema", std::string(label) + " schema"),
      .fileSHA256 = digestField(value, @"file_sha256",
                                std::string(label) + " file SHA-256"),
      .identitySHA256 = digestField(value, @"identity_sha256",
                                    std::string(label) + " identity SHA-256"),
  };
}

std::array<double, 3u> vector3(id value, const std::string_view label) {
  NSArray *values = array(value, label);
  require(values.count == 3u, std::string(label) + " must have three values");
  return {{
      doubleValue(values[0], std::string(label) + "[0]"),
      doubleValue(values[1], std::string(label) + "[1]"),
      doubleValue(values[2], std::string(label) + "[2]"),
  }};
}

std::vector<std::uint8_t>
documentFromValueSpan(const std::span<const std::uint8_t> parent,
                      const JSONMemberSpan &span) {
  std::vector<std::uint8_t> result(parent.begin() + span.valueBegin,
                                   parent.begin() + span.valueEnd);
  result.push_back('\n');
  return result;
}

std::vector<NumiHumanLoadedKneeDigest>
digestArray(id value, const std::string_view label) {
  NSArray *values = array(value, label);
  std::vector<NumiHumanLoadedKneeDigest> result;
  result.reserve(values.count);
  for (NSUInteger index = 0u; index < values.count; ++index) {
    result.push_back(parseDigest(
        stringValue(values[index], std::string(label) + " item"), label));
  }
  return result;
}

std::vector<std::string> stringArray(id value, const std::string_view label) {
  NSArray *values = array(value, label);
  std::vector<std::string> result;
  result.reserve(values.count);
  for (NSUInteger index = 0u; index < values.count; ++index) {
    result.push_back(stringValue(values[index], std::string(label) + " item"));
  }
  return result;
}

std::vector<std::string> sortedUniqueStrings(id value,
                                             const std::string_view label) {
  auto result = stringArray(value, label);
  require(std::is_sorted(result.begin(), result.end()) &&
              std::adjacent_find(result.begin(), result.end()) == result.end(),
          std::string(label) + " must be sorted and unique");
  return result;
}

void validateIdentityObject(NSDictionary *value,
                            const std::string_view expectedSchema,
                            const std::string_view label) {
  requireKeys(value, {"file_sha256", "identity_sha256", "schema"}, label);
  require(stringField(value, @"schema", std::string(label) + " schema") ==
              expectedSchema,
          std::string(label) + " schema differs");
  (void)digestField(value, @"file_sha256",
                    std::string(label) + " file SHA-256");
  (void)digestField(value, @"identity_sha256",
                    std::string(label) + " identity SHA-256");
}

NSDictionary *ownershipRole(NSDictionary *record, NSString *role,
                            const std::string_view label) {
  NSDictionary *owners = dictionary(field(record, @"owners", label), label);
  return dictionary(field(owners, role, label), label);
}

void requireOwnershipRole(NSDictionary *record, NSString *role,
                          const std::string_view expectedStatus,
                          const std::string_view expectedOwner,
                          const std::string_view label) {
  NSDictionary *owner = ownershipRole(record, role, label);
  requireKeys(owner, {"owner_id", "status"}, label);
  require(stringField(owner, @"status", std::string(label) + " status") ==
              expectedStatus,
          std::string(label) + " status differs");
  id ownerID = field(owner, @"owner_id", std::string(label) + " owner ID");
  if (expectedStatus == "unresolved") {
    require(ownerID == [NSNull null],
            std::string(label) + " unresolved owner has an ID");
  } else {
    require(stringValue(ownerID, std::string(label) + " owner ID") ==
                expectedOwner,
            std::string(label) + " owner ID differs");
  }
}

using OwnershipRecordMap = std::map<std::string, NSDictionary *>;

OwnershipRecordMap validateOwnershipContract(NSDictionary *ownership) {
  constexpr std::string_view kOwnershipBoundary =
      "Stable Human authoring identities and candidate moment bookkeeping "
      "only. "
      "Source topology, a candidate volume or mass, a semantic action, or a "
      "declared replacement rule does not assign a production physical owner, "
      "material, motor compartment, active-force path, state authority, "
      "mechanics, calibration, or integrated Human qualification.";
  require(stringField(ownership, @"boundary", "ownership boundary") ==
                  kOwnershipBoundary &&
              !stringField(ownership, @"subject", "ownership subject").empty(),
          "ownership boundary or subject differs");

  NSDictionary *inputs = dictionary(
      field(ownership, @"inputs", "ownership inputs"), "ownership inputs");
  requireKeys(inputs,
              {"authoring", "body_composition", "muscle_route_identities",
               "muscle_routes", "target_coverage"},
              "ownership inputs");
  validateIdentityObject(
      dictionary(field(inputs, @"target_coverage", "target-coverage input"),
                 "target-coverage input"),
      "HumanPack.target-coverage.v1", "target-coverage input");
  validateIdentityObject(
      dictionary(field(inputs, @"body_composition", "body-composition input"),
                 "body-composition input"),
      "HumanPack.body-composition-release-join.v1", "body-composition input");
  validateIdentityObject(
      dictionary(field(inputs, @"muscle_routes", "muscle-route input"),
                 "muscle-route input"),
      "HumanPack.muscle-route-mass-partition-candidate.v1",
      "muscle-route input");
  validateIdentityObject(dictionary(field(inputs, @"muscle_route_identities",
                                          "muscle-route identity input"),
                                    "muscle-route identity input"),
                         "HumanPack.muscle-route-volume-join-candidate.v1",
                         "muscle-route identity input");
  validateIdentityObject(
      dictionary(field(inputs, @"authoring", "ownership authoring input"),
                 "ownership authoring input"),
      "numi.human.ownership-authoring.v1", "ownership authoring input");

  NSDictionary *coverage = dictionary(
      field(ownership, @"source_coverage", "source-coverage summary"),
      "source-coverage summary");
  requireKeys(coverage,
              {"leaf_count", "manifest_sha256", "scope_status",
               "unresolved_current_registers"},
              "source-coverage summary");
  const auto coverageIdentity = digestField(
      coverage, @"manifest_sha256", "source-coverage manifest identity");
  NSDictionary *targetCoverageInput =
      dictionary(field(inputs, @"target_coverage", "target-coverage input"),
                 "target-coverage input");
  require(coverageIdentity == digestField(targetCoverageInput,
                                          @"identity_sha256",
                                          "target-coverage input identity"),
          "source-coverage identity differs from its input");
  const std::uint64_t unresolvedRegisters =
      unsignedField(coverage, @"unresolved_current_registers",
                    "unresolved current registers");
  const std::string expectedScopeStatus =
      unresolvedRegisters == 0u ? "source_union_materialized" : "blocked";
  require(stringField(coverage, @"scope_status", "source-coverage status") ==
              expectedScopeStatus,
          "source-coverage status hides unresolved current registers");
  const std::uint64_t expectedLeafCount =
      unsignedField(coverage, @"leaf_count", "source-coverage leaf count");

  NSDictionary *body = dictionary(
      field(ownership, @"body_composition", "body-composition summary"),
      "body-composition summary");
  requireKeys(body,
              {"candidate_scopes_disjoint", "domain_counts", "ownership",
               "production_physical_owner_count", "schema", "status"},
              "body-composition summary");
  require(stringField(body, @"schema", "body-composition schema") ==
                  "HumanPack.body-composition-release-join.v1" &&
              stringField(body, @"status", "body-composition status") ==
                  "partial" &&
              !booleanField(body, @"candidate_scopes_disjoint",
                            "candidate-scope disjointness") &&
              unsignedField(body, @"production_physical_owner_count",
                            "production physical-owner count") == 0u,
          "body-composition summary promotes candidate ownership");
  NSDictionary *domainCounts = dictionary(
      field(body, @"domain_counts", "body-composition domain counts"),
      "body-composition domain counts");
  requireKeys(domainCounts,
              {"fat_mass_candidates", "fat_surfaces", "fat_volume_candidates",
               "muscle_closed_geometry_candidates",
               "muscle_routes_without_surface_binding", "muscle_source_routes",
               "native_active_support_contacts", "native_recruited_muscles",
               "organ_blood_source_members", "organ_blood_source_owners",
               "organ_blood_tissue_beds"},
              "body-composition domain counts");
  for (id key in domainCounts) {
    require([key isKindOfClass:[NSString class]],
            "body-composition domain-count key is not a string");
    (void)unsignedValue(domainCounts[key], "body-composition domain count");
  }
  NSDictionary *bodyOwnership =
      dictionary(field(body, @"ownership", "body-composition ownership"),
                 "body-composition ownership");
  requireKeys(bodyOwnership,
              {"fat_mechanical_mass_owner", "mechanical_blood_mass_owner",
               "mechanical_tissue_mass_owner", "organ_physical_volume_owner",
               "physical_owner_count", "skeletal_muscle_tissue_mass_owner",
               "whole_body_dynamic_mass_matrix_owner"},
              "body-composition ownership");
  require(!booleanField(bodyOwnership, @"fat_mechanical_mass_owner",
                        "fat mechanical-mass ownership") &&
              !booleanField(bodyOwnership, @"mechanical_blood_mass_owner",
                            "blood mechanical-mass ownership") &&
              !booleanField(bodyOwnership, @"mechanical_tissue_mass_owner",
                            "tissue mechanical-mass ownership") &&
              !booleanField(bodyOwnership, @"organ_physical_volume_owner",
                            "organ physical-volume ownership") &&
              !booleanField(bodyOwnership, @"skeletal_muscle_tissue_mass_owner",
                            "skeletal-muscle tissue-mass ownership") &&
              !booleanField(bodyOwnership,
                            @"whole_body_dynamic_mass_matrix_owner",
                            "whole-body dynamic-mass ownership") &&
              unsignedField(bodyOwnership, @"physical_owner_count",
                            "body-composition physical-owner count") == 0u,
          "body-composition summary promotes production ownership");

  NSArray *records = array(field(ownership, @"records", "ownership records"),
                           "ownership records");
  require(records.count != 0u, "ownership records are empty");
  OwnershipRecordMap result;
  std::set<std::string> allLeaves;
  std::set<std::uint64_t> actionIndices;
  std::set<std::string> actionIDs;
  std::set<std::string> declaredActiveOwners;
  std::set<std::string> reservedSourceForceOwners;
  constexpr std::array<std::string_view, 5u> ownerRoles{
      {"physical_volume", "mechanical_mass", "material", "active_force",
       "state"}};
  std::map<std::string, std::array<std::uint64_t, 2u>> ownerCounts;
  std::uint64_t semanticActions = 0u;
  std::uint64_t unmappedRouteRecords = 0u;
  std::uint64_t compositionConflicts = 0u;
  std::uint64_t candidateMomentRecords = 0u;
  std::string previousSemantic;

  const auto validateBinding = [](NSDictionary *binding,
                                  const std::string_view label) {
    requireKeys(binding, {"ids", "status"}, label);
    const std::string status =
        stringField(binding, @"status", std::string(label) + " status");
    require(status == "unresolved" || status == "source" ||
                status == "candidate",
            std::string(label) + " status is invalid");
    const auto ids =
        sortedUniqueStrings(field(binding, @"ids", std::string(label) + " IDs"),
                            std::string(label) + " IDs");
    require((status == "unresolved") == ids.empty(),
            std::string(label) + " status and IDs disagree");
  };

  for (NSUInteger index = 0u; index < records.count; ++index) {
    NSDictionary *record = dictionary(records[index], "ownership record");
    requireKeys(record,
                {"action", "composition", "coverage_leaf_sha256s",
                 "entity_kind", "fields", "force_semantics", "moments",
                 "motor_compartments", "owners", "qualification_status",
                 "semantic_id", "topology"},
                "ownership record");
    const std::string semantic =
        stringField(record, @"semantic_id", "ownership record semantic ID");
    require(!semantic.empty() && (index == 0u || previousSemantic < semantic) &&
                result.emplace(semantic, record).second,
            "ownership records are empty, repeated, or not sorted");
    previousSemantic = semantic;

    const auto leaves = sortedUniqueStrings(
        field(record, @"coverage_leaf_sha256s", "ownership coverage leaves"),
        "ownership coverage leaves");
    for (const auto &leaf : leaves) {
      (void)parseDigest(leaf, "ownership coverage leaf");
      require(allLeaves.insert(leaf).second,
              "ownership coverage leaf belongs to multiple records");
    }
    const std::string entityKind =
        stringField(record, @"entity_kind", "ownership entity kind");
    require(!entityKind.empty(), "ownership entity kind is empty");
    if (entityKind == "muscle_route" && leaves.empty())
      ++unmappedRouteRecords;

    validateBinding(dictionary(field(record, @"topology", "topology binding"),
                               "topology binding"),
                    "topology binding");
    validateBinding(
        dictionary(field(record, @"fields", "field binding"), "field binding"),
        "field binding");
    validateBinding(dictionary(field(record, @"motor_compartments",
                                     "motor-compartment binding"),
                               "motor-compartment binding"),
                    "motor-compartment binding");

    NSDictionary *composition =
        dictionary(field(record, @"composition", "ownership composition"),
                   "ownership composition");
    requireKeys(composition,
                {"component_semantic_ids", "conflict_ids", "status"},
                "ownership composition");
    const std::string compositionStatus =
        stringField(composition, @"status", "ownership composition status");
    require(compositionStatus == "unresolved" ||
                compositionStatus == "single_source" ||
                compositionStatus == "composed" ||
                compositionStatus == "conflict",
            "ownership composition status is invalid");
    (void)sortedUniqueStrings(field(composition, @"component_semantic_ids",
                                    "composition semantic IDs"),
                              "composition semantic IDs");
    const auto conflictIDs = sortedUniqueStrings(
        field(composition, @"conflict_ids", "composition conflict IDs"),
        "composition conflict IDs");
    require((compositionStatus == "conflict") == !conflictIDs.empty(),
            "ownership composition conflict status and IDs disagree");
    if (compositionStatus == "conflict")
      ++compositionConflicts;

    NSDictionary *owners = dictionary(
        field(record, @"owners", "ownership roles"), "ownership roles");
    requireKeys(owners,
                {"active_force", "material", "mechanical_mass",
                 "physical_volume", "state"},
                "ownership roles");
    for (const auto role : ownerRoles) {
      NSString *key = [NSString stringWithUTF8String:role.data()];
      NSDictionary *owner =
          dictionary(field(owners, key, "ownership role"), "ownership role");
      requireKeys(owner, {"owner_id", "status"}, "ownership role");
      const std::string status =
          stringField(owner, @"status", "ownership role status");
      require(status == "unresolved" || status == "candidate",
              "ownership role status is invalid");
      id ownerID = field(owner, @"owner_id", "ownership role owner ID");
      if (status == "unresolved") {
        require(ownerID == [NSNull null],
                "unresolved ownership role carries an owner ID");
        ++ownerCounts[std::string(role)][0u];
      } else {
        const std::string identifier =
            stringValue(ownerID, "ownership role owner ID");
        require(!identifier.empty(), "candidate ownership role ID is empty");
        ++ownerCounts[std::string(role)][1u];
        if (role == "active_force")
          declaredActiveOwners.insert(identifier);
      }
    }

    NSDictionary *moments = dictionary(
        field(record, @"moments", "ownership moments"), "ownership moments");
    requireKeys(moments,
                {"first_mass_moment_kg_m", "frame_id",
                 "second_mass_moment_kg_m2", "status", "volume_m3",
                 "zeroth_mass_kg"},
                "ownership moments");
    const std::string momentStatus =
        stringField(moments, @"status", "ownership moment status");
    require(momentStatus == "unresolved" || momentStatus == "candidate",
            "ownership moment status is invalid");
    if (momentStatus == "candidate")
      ++candidateMomentRecords;
    if (momentStatus == "unresolved") {
      require(
          field(moments, @"frame_id", "moment frame") == [NSNull null] &&
              field(moments, @"volume_m3", "moment volume") == [NSNull null] &&
              field(moments, @"zeroth_mass_kg", "moment mass") ==
                  [NSNull null] &&
              field(moments, @"first_mass_moment_kg_m", "first mass moment") ==
                  [NSNull null] &&
              field(moments, @"second_mass_moment_kg_m2",
                    "second mass moment") == [NSNull null],
          "unresolved ownership moments carry values");
    } else {
      require(field(moments, @"frame_id", "moment frame") == [NSNull null] &&
                  field(moments, @"first_mass_moment_kg_m",
                        "first mass moment") == [NSNull null] &&
                  field(moments, @"second_mass_moment_kg_m2",
                        "second mass moment") == [NSNull null] &&
                  doubleField(moments, @"volume_m3", "moment volume") >= 0.0 &&
                  doubleField(moments, @"zeroth_mass_kg", "moment mass") >= 0.0,
              "candidate ownership moments are invalid");
    }

    NSDictionary *force =
        dictionary(field(record, @"force_semantics", "force semantics"),
                   "force semantics");
    requireKeys(force, {"mode", "replaces_owner_ids", "status"},
                "force semantics");
    const std::string forceStatus =
        stringField(force, @"status", "force-semantics status");
    require(forceStatus == "unresolved" || forceStatus == "candidate",
            "force-semantics status is invalid");
    const auto replaced = sortedUniqueStrings(
        field(force, @"replaces_owner_ids", "replaced owner IDs"),
        "replaced owner IDs");
    id mode = field(force, @"mode", "force-semantics mode");
    if (forceStatus == "unresolved") {
      require(mode == [NSNull null] && replaced.empty(),
              "unresolved force semantics carry a rule");
    } else {
      const std::string value = stringValue(mode, "force-semantics mode");
      require((value == "replacement" && !replaced.empty()) ||
                  (value == "additive" && replaced.empty()),
              "candidate force semantics are invalid");
    }

    const std::string qualification = stringField(
        record, @"qualification_status", "record qualification status");
    require(qualification == "unresolved" || qualification == "source_only" ||
                qualification == "candidate",
            "record qualification status is invalid");

    id actionValue = field(record, @"action", "semantic action");
    if (actionValue != [NSNull null]) {
      NSDictionary *action = dictionary(actionValue, "semantic action");
      requireKeys(action, {"semantic_action_id", "source_index", "status"},
                  "semantic action");
      const std::string actionStatus =
          stringField(action, @"status", "semantic action status");
      const std::string actionID =
          stringField(action, @"semantic_action_id", "semantic action ID");
      const std::uint64_t actionIndex =
          unsignedField(action, @"source_index", "semantic action index");
      require((actionStatus == "source" || actionStatus == "candidate") &&
                  !actionID.empty() &&
                  actionIndices.insert(actionIndex).second &&
                  actionIDs.insert(actionID).second,
              "semantic action is invalid or repeated");
      ++semanticActions;
      if (entityKind == "muscle_route")
        reservedSourceForceOwners.insert(semantic + "/owner/source-jt");
    }
  }

  require(allLeaves.size() == expectedLeafCount,
          "ownership coverage leaf count differs from its summary");
  for (std::uint64_t index = 0u; index < semanticActions; ++index) {
    require(actionIndices.contains(index),
            "ownership semantic action indices are not contiguous");
  }
  require(unsignedField(domainCounts, @"muscle_source_routes",
                        "muscle source-route count") == semanticActions &&
              unsignedField(domainCounts, @"native_recruited_muscles",
                            "native recruited-muscle count") == semanticActions,
          "body-composition action counts differ from ownership records");

  for (const auto &entry : result) {
    NSDictionary *record = entry.second;
    NSDictionary *composition =
        dictionary(field(record, @"composition", "ownership composition"),
                   "ownership composition");
    for (const auto &component :
         sortedUniqueStrings(field(composition, @"component_semantic_ids",
                                   "composition semantic IDs"),
                             "composition semantic IDs")) {
      require(result.contains(component),
              "ownership composition names an unknown semantic ID");
    }
    NSDictionary *force =
        dictionary(field(record, @"force_semantics", "force semantics"),
                   "force semantics");
    for (const auto &replaced : sortedUniqueStrings(
             field(force, @"replaces_owner_ids", "replaced owner IDs"),
             "replaced owner IDs")) {
      require(declaredActiveOwners.contains(replaced) ||
                  reservedSourceForceOwners.contains(replaced),
              "force semantics replace an unknown owner");
    }
  }

  NSArray *closures =
      array(field(ownership, @"moment_closures", "ownership moment closures"),
            "ownership moment closures");
  std::string previousClosure;
  for (id value in closures) {
    NSDictionary *closure = dictionary(value, "ownership moment closure");
    requireKeys(closure,
                {"absolute_tolerance", "actual", "expected", "frame_id", "id",
                 "member_semantic_ids", "residual", "status"},
                "ownership moment closure");
    const std::string identifier =
        stringField(closure, @"id", "moment-closure ID");
    require(!identifier.empty() &&
                (previousClosure.empty() || previousClosure < identifier) &&
                stringField(closure, @"status", "moment-closure status") ==
                    "candidate" &&
                doubleField(closure, @"absolute_tolerance",
                            "moment-closure tolerance") >= 0.0,
            "ownership moment closure is invalid or unsorted");
    previousClosure = identifier;
    NSDictionary *actual =
        dictionary(field(closure, @"actual", "actual closure moments"),
                   "actual closure moments");
    NSDictionary *expected =
        dictionary(field(closure, @"expected", "expected closure moments"),
                   "expected closure moments");
    NSDictionary *residual =
        dictionary(field(closure, @"residual", "residual closure moments"),
                   "residual closure moments");
    for (NSDictionary *moments in @[ actual, expected, residual ]) {
      requireKeys(moments,
                  {"first_mass_moment_kg_m", "second_mass_moment_kg_m2",
                   "volume_m3", "zeroth_mass_kg"},
                  "closure moments");
      require(field(moments, @"first_mass_moment_kg_m",
                    "closure first mass moment") == [NSNull null] &&
                  field(moments, @"second_mass_moment_kg_m2",
                        "closure second mass moment") == [NSNull null],
              "ownership closure claims unavailable spatial moments");
    }
    require(field(closure, @"frame_id", "moment-closure frame") ==
                [NSNull null],
            "ownership closure claims an unavailable spatial frame");
    const double tolerance =
        doubleField(closure, @"absolute_tolerance", "moment-closure tolerance");
    const auto members = sortedUniqueStrings(
        field(closure, @"member_semantic_ids", "moment-closure members"),
        "moment-closure members");
    require(!members.empty(), "ownership moment closure has no members");
    double memberVolume = 0.0;
    double memberMass = 0.0;
    for (const auto &member : members) {
      const auto found = result.find(member);
      require(found != result.end(),
              "ownership moment closure names an unknown record");
      NSDictionary *moments =
          dictionary(field(found->second, @"moments", "closure-member moments"),
                     "closure-member moments");
      require(stringField(moments, @"status", "closure-member status") ==
                  "candidate",
              "ownership moment closure includes unresolved moments");
      memberVolume +=
          doubleField(moments, @"volume_m3", "closure-member volume");
      memberMass +=
          doubleField(moments, @"zeroth_mass_kg", "closure-member mass");
    }
    for (NSString *key in @[ @"volume_m3", @"zeroth_mass_kg" ]) {
      const double actualValue =
          doubleField(actual, key, "actual closure value");
      const double expectedValue =
          doubleField(expected, key, "expected closure value");
      const double residualValue =
          doubleField(residual, key, "residual closure value");
      const double memberValue =
          [key isEqualToString:@"volume_m3"] ? memberVolume : memberMass;
      require(std::abs(memberValue - actualValue) <= tolerance &&
                  std::abs(actualValue - expectedValue) <= tolerance &&
                  std::abs(residualValue - (actualValue - expectedValue)) <=
                      tolerance &&
                  std::abs(residualValue) <= tolerance,
              "ownership moment closure does not close within tolerance");
    }
  }

  NSDictionary *counts = dictionary(
      field(ownership, @"counts", "ownership counts"), "ownership counts");
  requireKeys(counts,
              {"candidate_moment_records", "composition_conflicts",
               "moment_closures", "owners_by_role_and_status",
               "production_moment_records", "records", "semantic_actions",
               "unmapped_route_records"},
              "ownership counts");
  require(
      unsignedField(counts, @"records", "ownership record count") ==
              records.count &&
          unsignedField(counts, @"semantic_actions",
                        "ownership semantic-action count") == semanticActions &&
          unsignedField(counts, @"unmapped_route_records",
                        "unmapped route-record count") ==
              unmappedRouteRecords &&
          unsignedField(counts, @"composition_conflicts",
                        "composition-conflict count") == compositionConflicts &&
          unsignedField(counts, @"candidate_moment_records",
                        "candidate moment-record count") ==
              candidateMomentRecords &&
          unsignedField(counts, @"production_moment_records",
                        "production moment-record count") == 0u &&
          unsignedField(counts, @"moment_closures", "moment-closure count") ==
              closures.count,
      "ownership counts differ from records");
  NSDictionary *ownerCountTable =
      dictionary(field(counts, @"owners_by_role_and_status", "owner counts"),
                 "owner counts");
  requireKeys(ownerCountTable,
              {"active_force", "material", "mechanical_mass", "physical_volume",
               "state"},
              "owner counts");
  for (const auto role : ownerRoles) {
    NSString *key = [NSString stringWithUTF8String:role.data()];
    NSDictionary *roleCounts = dictionary(
        field(ownerCountTable, key, "owner role counts"), "owner role counts");
    requireKeys(roleCounts, {"candidate", "unresolved"}, "owner role counts");
    require(
        unsignedField(roleCounts, @"unresolved", "unresolved owner count") ==
                ownerCounts[std::string(role)][0u] &&
            unsignedField(roleCounts, @"candidate", "candidate owner count") ==
                ownerCounts[std::string(role)][1u],
        "owner role counts differ from records");
  }

  const std::string derivedStatus =
      compositionConflicts != 0u || unmappedRouteRecords != 0u ||
              expectedScopeStatus == "blocked" || unresolvedRegisters != 0u
          ? "blocked"
          : "partial";
  require(stringField(ownership, @"status", "ownership status") ==
              derivedStatus,
          "ownership status hides unresolved source facts");

  NSDictionary *qualification =
      dictionary(field(ownership, @"qualification", "ownership qualification"),
                 "ownership qualification");
  requireKeys(
      qualification,
      {"all_source_routes_retained", "body_composition_join_bound",
       "candidate_moment_closure_validated", "integrated_human_qualification",
       "production_physical_ownership", "route_semantic_ids_coverage_bound",
       "target_coverage_validated"},
      "ownership qualification");
  require(booleanField(qualification, @"target_coverage_validated",
                       "target-coverage validation") &&
              booleanField(qualification, @"body_composition_join_bound",
                           "body-composition binding") &&
              booleanField(qualification, @"all_source_routes_retained",
                           "source-route retention") &&
              booleanField(qualification, @"route_semantic_ids_coverage_bound",
                           "route semantic coverage") ==
                  (unmappedRouteRecords == 0u) &&
              booleanField(qualification, @"candidate_moment_closure_validated",
                           "candidate moment-closure validation") ==
                  (closures.count != 0u) &&
              !booleanField(qualification, @"production_physical_ownership",
                            "production physical ownership") &&
              !booleanField(qualification, @"integrated_human_qualification",
                            "integrated Human qualification"),
          "ownership qualification boundary differs");
  return result;
}

void validateManifestShape(NSDictionary *manifest) {
  NSDictionary *inputs =
      dictionary(field(manifest, @"inputs", "Human inputs"), "Human inputs");
  requireKeys(inputs,
              {"authoring_profile", "lab_export", "open_knee_payload",
               "ownership", "tendon_payload", "x_ref"},
              "Human inputs");
  for (id key in inputs) {
    requireKeys(dictionary(inputs[key], "Human input"),
                {"file_sha256", "identity_sha256", "schema"}, "Human input");
  }
  requireKeys(dictionary(field(manifest, @"semantic_scope", "semantic scope"),
                         "semantic scope"),
              {"coverage_leaf_sha256s", "ownership_manifest_sha256"},
              "semantic scope");
  NSDictionary *qualification =
      dictionary(field(manifest, @"qualification", "Human qualification"),
                 "Human qualification");
  requireKeys(qualification,
              {"candidate_only", "clinical_validity_qualified",
               "donor_mass_and_raw_moments_subtracted",
               "integrated_human_qualification", "mesh_convergence_qualified",
               "ownership_identity_bound", "prestrain_reference_reset_executed",
               "production_active_force", "production_physical_ownership",
               "projected_reference_identity_bound",
               "runtime_x_current_accepted", "source_identity_bound",
               "specimen_load_validation_qualified",
               "subject_material_calibrated", "topology_identity_bound",
               "unloaded_reference_qualified"},
              "Human qualification");
  require(
      booleanField(qualification, @"candidate_only", "candidate-only status") &&
          booleanField(qualification, @"source_identity_bound",
                       "source identity status") &&
          booleanField(qualification, @"ownership_identity_bound",
                       "ownership identity status") &&
          booleanField(qualification, @"topology_identity_bound",
                       "topology identity status") &&
          booleanField(qualification, @"projected_reference_identity_bound",
                       "projected-reference identity status") &&
          booleanField(qualification, @"donor_mass_and_raw_moments_subtracted",
                       "donor subtraction status") &&
          !booleanField(qualification, @"unloaded_reference_qualified",
                        "unloaded reference qualification") &&
          !booleanField(qualification, @"prestrain_reference_reset_executed",
                        "prestrain reset qualification") &&
          !booleanField(qualification, @"subject_material_calibrated",
                        "subject material calibration") &&
          !booleanField(qualification, @"mesh_convergence_qualified",
                        "mesh convergence qualification") &&
          !booleanField(qualification, @"specimen_load_validation_qualified",
                        "specimen-load qualification") &&
          !booleanField(qualification, @"clinical_validity_qualified",
                        "clinical validity qualification") &&
          !booleanField(qualification, @"production_physical_ownership",
                        "production ownership status") &&
          !booleanField(qualification, @"production_active_force",
                        "production active-force status") &&
          !booleanField(qualification, @"runtime_x_current_accepted",
                        "runtime x_current acceptance status") &&
          !booleanField(qualification, @"integrated_human_qualification",
                        "integrated Human status"),
      "Human qualification overclaims or omits its candidate-only boundary");

  NSDictionary *source =
      dictionary(field(manifest, @"source", "Human source"), "Human source");
  requireKeys(source,
              {"dataset_id", "doi", "files", "license",
               "material_and_density_status", "nhknee", "tendon_payload"},
              "Human source");
  requireKeys(
      dictionary(field(source, @"files", "source files"), "source files"),
      {"FeBio_custom.feb", "Geometry.feb", "ModelProperties.xml",
       "license.txt"},
      "source files");
  requireKeys(dictionary(field(source, @"nhknee", "NHKNEE1 identity"),
                         "NHKNEE1 identity"),
              {"abi", "bytes", "magic", "sha256"}, "NHKNEE1 identity");
  requireKeys(
      dictionary(field(source, @"tendon_payload", "tendon payload identity"),
                 "tendon payload identity"),
      {"abi", "body_count", "bodyparts3d_bone_payload_sha256", "bone_count",
       "bytes", "endpoint_count", "envelope_count", "magic", "muscle_count",
       "myosim_archive_sha256", "myosim_muscle_payload_sha256",
       "point_fallback_count", "registration_fingerprint32", "schema", "sha256",
       "site_count", "wire_magic_hex"},
      "tendon payload identity");

  NSDictionary *topology = dictionary(
      field(manifest, @"topology", "Human topology"), "Human topology");
  requireKeys(
      topology,
      {"anchor_ownership_encoding", "anchor_ownership_sha256",
       "executable_fem_topology_encoding", "executable_fem_topology_sha256",
       "identity_sha256", "loaded_node_count", "loaded_tetrahedron_count",
       "node_count", "node_set_count", "node_set_membership_count", "ordering",
       "region_count", "region_spans", "source_global_node_index_encoding",
       "source_global_node_index_sha256", "surface_count", "surface_face_count",
       "surface_pair_count", "tetrahedron_count"},
      "Human topology");
  NSArray *spans =
      array(field(topology, @"region_spans", "topology region spans"),
            "topology region spans");
  for (id value in spans) {
    requireKeys(dictionary(value, "topology region span"),
                {"first_node", "first_tetrahedron", "name", "node_count",
                 "runtime_first_node", "runtime_first_tetrahedron",
                 "runtime_node_count", "runtime_tetrahedron_count",
                 "tetrahedron_count"},
                "topology region span");
  }

  NSDictionary *coordinates =
      dictionary(field(manifest, @"coordinates", "Human coordinates"),
                 "Human coordinates");
  requireKeys(coordinates, {"x_current", "x_ref", "x_source"},
              "Human coordinates");
  requireKeys(dictionary(field(coordinates, @"x_source", "x_source identity"),
                         "x_source identity"),
              {"body_pose_id", "encoding", "frame_id", "node_count",
               "registration_id", "scope", "sha256"},
              "x_source identity");
  NSDictionary *xReference = dictionary(
      field(coordinates, @"x_ref", "x_ref identity"), "x_ref identity");
  requireKeys(xReference,
              {"construction_id", "encoding", "frame_id",
               "mapping_identity_sha256", "node_count",
               "prestrain_reset_method", "reference_body_pose_id",
               "reference_state_class", "scope", "sha256",
               "source_body_pose_id", "source_frame_id",
               "source_registration_id", "source_to_reference_mapping_id",
               "unloaded_reference_qualified", "volumetric_prestress_status"},
              "x_ref identity");
  requireKeys(dictionary(field(xReference, @"prestrain_reset_method",
                               "prestrain reset method"),
                         "prestrain reset method"),
              {"method_id", "status"}, "prestrain reset method");
  requireKeys(
      dictionary(field(coordinates, @"x_current", "x_current authority"),
                 "x_current authority"),
      {"acceptance_receipt_sha256", "accepted_sha256", "authority", "encoding",
       "hash_algorithm", "scope"},
      "x_current authority");
  requireKeys(
      dictionary(field(manifest, @"density_conversion", "density conversion"),
                 "density conversion"),
      {"calibration_status", "conversion_factor_to_kg_per_m3",
       "runtime_value_kg_per_m3", "source_unit", "source_value"},
      "density conversion");

  const auto validateMomentPayload = [](NSDictionary *value,
                                        const std::string_view label,
                                        const bool regional) {
    if (regional) {
      requireKeys(value,
                  {"first_mass_moment_kg_m", "maximum_jacobian_determinant",
                   "minimum_jacobian_determinant", "name",
                   "raw_f32_node_mass_sha256", "raw_second_mass_moment_kg_m2",
                   "volume_m3", "zeroth_mass_kg"},
                  label);
    } else {
      requireKeys(value,
                  {"first_mass_moment_kg_m", "raw_second_mass_moment_kg_m2",
                   "zeroth_mass_kg"},
                  label);
    }
  };

  NSArray *regions =
      array(field(manifest, @"regions", "Human regions"), "Human regions");
  for (id value in regions) {
    NSDictionary *region = dictionary(value, "Human region");
    requireKeys(region,
                {"donor_body_semantic_id", "material", "material_semantic_ids",
                 "name", "owners", "payload", "projected_reference_moments",
                 "topology_semantic_id"},
                "Human region");
    requireKeys(dictionary(field(region, @"payload", "region payload"),
                           "region payload"),
                {"first_node", "first_surface", "first_tetrahedron", "index",
                 "kind", "material_flags", "name", "node_count",
                 "runtime_first_node", "runtime_first_tetrahedron",
                 "runtime_node_count", "runtime_tetrahedron_count",
                 "surface_count", "tetrahedra_sha256", "tetrahedron_count",
                 "visual_body_index"},
                "region payload");
    requireKeys(dictionary(field(region, @"material", "region material"),
                           "region material"),
                {"bulk_modulus_mpa", "c1_mpa", "c2_mpa", "c3_mpa", "c4",
                 "c5_mpa", "calibration_status", "fiber_world", "lambda_max",
                 "source_initial_stretch", "type"},
                "region material");
    NSDictionary *owners =
        dictionary(field(region, @"owners", "region owners"), "region owners");
    requireKeys(owners,
                {"active_force", "material", "mechanical_mass",
                 "physical_volume", "state"},
                "region owners");
    for (id key in owners)
      requireKeys(dictionary(owners[key], "region owner"),
                  {"owner_id", "status"}, "region owner");
    validateMomentPayload(
        dictionary(field(region, @"projected_reference_moments",
                         "projected reference moments"),
                   "projected reference moments"),
        "projected reference moments", true);
  }

  NSDictionary *mass = dictionary(
      field(manifest, @"mass_partition", "mass partition"), "mass partition");
  requireKeys(mass,
              {"density_kg_per_m3", "donors", "excluded_bodies", "frame_id",
               "identity_sha256", "policy_id", "provenance",
               "qualification_status", "raw_f32_node_mass", "regions",
               "source_kind"},
              "mass partition");
  NSDictionary *rawMass = dictionary(
      field(mass, @"raw_f32_node_mass", "raw node mass"), "raw node mass");
  requireKeys(rawMass,
              {"algorithm", "encoding", "node_count", "sha256"},
              "raw node mass");
  require(stringField(rawMass, @"algorithm", "raw node mass algorithm") ==
                  kRawF32NodeMassAlgorithm &&
              stringField(rawMass, @"encoding", "raw node mass encoding") ==
                  "float32-le-lumped-tet-mass-quarter-source-element-order-"
                  "profile-region-node-order",
          "raw node mass arithmetic identity differs");
  NSDictionary *provenance = dictionary(
      field(mass, @"provenance", "mass provenance"), "mass provenance");
  requireKeys(provenance,
              {"equality_payload_sha256", "lab_export_manifest_sha256",
               "mapping", "nhknee_sha256", "poses",
               "source_model_fingerprint_sha256", "source_rigid_payload_sha256",
               "x_ref_sha256", "x_source_sha256"},
              "mass provenance");
  NSDictionary *poses = dictionary(
      field(provenance, @"poses", "pose provenance"), "pose provenance");
  requireKeys(poses, {"projected_reference", "source_default"},
              "pose provenance");
  for (id key in poses)
    requireKeys(dictionary(poses[key], "pose identity"),
                {"id", "identity_sha256"}, "pose identity");
  NSDictionary *mapping =
      dictionary(field(provenance, @"mapping", "mapping provenance"),
                 "mapping provenance");
  requireKeys(mapping,
              {"algorithm", "code_identity_encoding", "code_identity_sha256",
               "diagnostics", "id"},
              "mapping provenance");
  require(stringField(mapping, @"algorithm", "mapping algorithm") ==
                  kSourceToReferenceMappingAlgorithm &&
              stringField(mapping, @"code_identity_encoding",
                          "mapping code identity encoding") ==
                  kMappingCodeIdentityEncoding,
          "source-to-reference mapping identity differs");
  NSDictionary *diagnostics =
      dictionary(field(mapping, @"diagnostics", "mapping diagnostics"),
                 "mapping diagnostics");
  requireKeys(diagnostics,
              {"equality_residual_maximum", "finite", "jacobian",
               "maximum_displacement_m", "output_node_count",
               "raw_f32_node_mass_algorithm", "raw_f32_node_mass_sha256",
               "regions", "source_node_count"},
              "mapping diagnostics");
  require(stringField(diagnostics, @"raw_f32_node_mass_algorithm",
                      "mapping raw node mass algorithm") ==
                  kRawF32NodeMassAlgorithm &&
              stringField(diagnostics, @"raw_f32_node_mass_algorithm",
                          "mapping raw node mass algorithm") ==
                  stringField(rawMass, @"algorithm",
                              "raw node mass algorithm") &&
              digestField(diagnostics, @"raw_f32_node_mass_sha256",
                          "mapping raw node mass SHA-256") ==
                  digestField(rawMass, @"sha256",
                              "raw node mass SHA-256") &&
              unsignedField(diagnostics, @"output_node_count",
                            "mapping output node count") ==
                  unsignedField(rawMass, @"node_count",
                                "raw node mass node count"),
          "mapping and mass arithmetic provenance differ");
  requireKeys(dictionary(field(diagnostics, @"jacobian", "mapping Jacobian"),
                         "mapping Jacobian"),
              {"finite", "maximum_determinant", "minimum_determinant",
               "orientation_preserving"},
              "mapping Jacobian");
  NSArray *mappingRegions =
      array(field(diagnostics, @"regions", "mapping regions"),
            "mapping regions");
  for (id value in mappingRegions)
    requireKeys(dictionary(value, "mapping region"),
                {"direct_failure_jacobian", "direct_failure_tetrahedron",
                 "direct_map_status", "maximum_anchor_residual_m",
                 "maximum_jacobian",
                 "maximum_persisted_f32_anchor_residual_m",
                 "maximum_source_anchor_reconstruction_residual_m",
                 "minimum_jacobian", "name", "substep_count"},
                "mapping region");
  constexpr std::array<std::string_view, 6u> kMappingRegionNames{{
      "ACL", "LCL", "MCL", "PCL", "PTL", "QAT",
  }};
  require(mappingRegions.count == kMappingRegionNames.size(),
          "mapping diagnostic region count differs");
  double derivedMinimumJacobian = std::numeric_limits<double>::infinity();
  double derivedMaximumJacobian = -std::numeric_limits<double>::infinity();
  for (NSUInteger index = 0u; index < mappingRegions.count; ++index) {
    NSDictionary *region = dictionary(mappingRegions[index], "mapping region");
    const bool directRejected = index == 4u;
    require(stringField(region, @"name", "mapping region name") ==
                    kMappingRegionNames[index] &&
                stringField(region, @"direct_map_status",
                            "mapping direct-map status") ==
                    (directRejected ? "rejected_inversion" : "accepted") &&
                unsignedField(region, @"substep_count",
                              "mapping substep count") ==
                    (directRejected ? 2u : 1u),
            "mapping diagnostic region identity differs");
    id directFailureJacobian = field(region, @"direct_failure_jacobian",
                                     "mapping direct-failure Jacobian");
    id directFailureTetrahedron = field(
        region, @"direct_failure_tetrahedron",
        "mapping direct-failure tetrahedron");
    if (directRejected) {
      require(doubleValue(directFailureJacobian,
                          "mapping direct-failure Jacobian") < 0.0 &&
                  unsignedField(region, @"direct_failure_tetrahedron",
                                "mapping direct-failure tetrahedron") == 419u,
              "PTL direct-map rejection diagnostic differs");
    } else {
      require(directFailureJacobian == [NSNull null] &&
                  directFailureTetrahedron == [NSNull null],
              "accepted direct map carries a failure diagnostic");
    }
    const double minimum =
        doubleField(region, @"minimum_jacobian", "mapping minimum Jacobian");
    const double maximum =
        doubleField(region, @"maximum_jacobian", "mapping maximum Jacobian");
    require(minimum > 0.0 && maximum >= minimum &&
                doubleField(region, @"maximum_anchor_residual_m",
                            "mapping anchor residual") >= 0.0 &&
                doubleField(region, @"maximum_anchor_residual_m",
                            "mapping anchor residual") <= 5.0e-5 &&
                doubleField(region,
                            @"maximum_persisted_f32_anchor_residual_m",
                            "mapping persisted anchor residual") >= 0.0 &&
                doubleField(region,
                            @"maximum_persisted_f32_anchor_residual_m",
                            "mapping persisted anchor residual") <= 5.0e-5 &&
                doubleField(
                    region,
                    @"maximum_source_anchor_reconstruction_residual_m",
                    "mapping source-anchor residual") >= 0.0 &&
                doubleField(
                    region,
                    @"maximum_source_anchor_reconstruction_residual_m",
                    "mapping source-anchor residual") <= 5.0e-5,
            "mapping diagnostic is inverted or exceeds the anchor bound");
    derivedMinimumJacobian = std::min(derivedMinimumJacobian, minimum);
    derivedMaximumJacobian = std::max(derivedMaximumJacobian, maximum);
  }
  NSDictionary *mappingJacobian = dictionary(
      field(diagnostics, @"jacobian", "mapping Jacobian"),
      "mapping Jacobian");
  require(booleanField(diagnostics, @"finite", "mapping finite status") &&
              booleanField(mappingJacobian, @"finite",
                           "mapping Jacobian finite status") &&
              booleanField(mappingJacobian, @"orientation_preserving",
                           "mapping orientation status") &&
              doubleField(mappingJacobian, @"minimum_determinant",
                          "mapping aggregate minimum Jacobian") ==
                  derivedMinimumJacobian &&
              doubleField(mappingJacobian, @"maximum_determinant",
                          "mapping aggregate maximum Jacobian") ==
                  derivedMaximumJacobian,
          "mapping aggregate diagnostic differs from its regions");
  for (id value in array(field(mass, @"regions", "mass regions"),
                         "mass regions"))
    validateMomentPayload(dictionary(value, "mass region"), "mass region",
                          true);
  for (id value in array(field(mass, @"donors", "mass donors"),
                         "mass donors")) {
    NSDictionary *donor = dictionary(value, "mass donor");
    requireKeys(donor,
                {"closure_residual", "core_body_index", "region_names",
                 "remaining", "semantic_id", "source", "source_body_id",
                 "subtracted"},
                "mass donor");
    for (NSString *key in
         @[ @"closure_residual", @"remaining", @"source", @"subtracted" ])
      validateMomentPayload(
          dictionary(field(donor, key, "donor moments"), "donor moments"),
          "donor moments", false);
  }
  for (id value in array(field(mass, @"excluded_bodies", "excluded bodies"),
                         "excluded bodies"))
    requireKeys(dictionary(value, "excluded body"),
                {"core_body_index", "reason", "semantic_id", "source_body_id",
                 "source_mass_kg"},
                "excluded body");

  for (id value in array(
           field(manifest, @"articular_contact_pairs", "Human contact pairs"),
           "Human contact pairs")) {
    NSDictionary *pair = dictionary(value, "Human contact pair");
    requireKeys(pair,
                {"master_surface", "name", "semantic_id", "slave_surface",
                 "state_owner"},
                "Human contact pair");
    requireKeys(dictionary(field(pair, @"state_owner", "contact owner"),
                           "contact owner"),
                {"owner_id", "status"}, "contact owner");
  }
  for (id value in array(
           field(manifest, @"active_force_replacements", "force replacements"),
           "force replacements")) {
    NSDictionary *replacement = dictionary(value, "force replacement");
    requireKeys(replacement,
                {"endpoints", "name", "replacement_owner_id",
                 "replacement_scale", "replaces_owner_id", "route_semantic_id",
                 "semantic_action_id", "source_actuator_index", "status"},
                "force replacement");
    for (id endpoint in array(
             field(replacement, @"endpoints", "replacement endpoints"),
             "replacement endpoints"))
      requireKeys(dictionary(endpoint, "replacement endpoint"),
                  {"attachment_mode", "body_index", "bone_stable_id",
                   "endpoint_index", "endpoint_ordinal", "envelope_index",
                   "role", "route_node_index", "source_site_index"},
                  "replacement endpoint");
  }
  for (id value in array(field(manifest, @"passive_ligament_owners",
                               "passive ligament owners"),
                         "passive ligament owners")) {
    NSDictionary *passive = dictionary(value, "passive ligament owner");
    requireKeys(passive,
                {"active_force", "name", "passive_force_owner", "semantic_id"},
                "passive ligament owner");
    requireKeys(
        dictionary(field(passive, @"active_force", "passive active force"),
                   "passive active force"),
        {"owner_id", "status"}, "passive active force");
    requireKeys(dictionary(field(passive, @"passive_force_owner",
                                 "passive-force owner"),
                           "passive-force owner"),
                {"owner_id", "status"}, "passive-force owner");
  }
  requireKeys(dictionary(field(manifest, @"full_state_authority",
                               "full-state authority"),
                         "full-state authority"),
              {"identity_sha256", "owner_id", "required_snapshot_components",
               "schema", "semantic_id"},
              "full-state authority");

  NSDictionary *lab = dictionary(
      field(manifest, @"lab_authoring_export", "embedded Lab export"),
      "embedded Lab export");
  requireKeys(lab,
              {"boundary", "donor_moments", "manifest_canonicalization",
               "manifest_hash_exclusion", "manifest_sha256", "mapping", "poses",
               "schema", "side", "source", "status", "subject_id", "x_ref"},
              "embedded Lab export");
  require(stringField(lab, @"schema", "embedded Lab export schema") ==
                  "numi.lab.loaded-knee-authoring-export.v1" &&
              stringField(lab, @"status", "embedded Lab export status") ==
                  "candidate" &&
              stringField(lab, @"boundary", "embedded Lab export boundary") ==
                  kNumiHumanLoadedKneeLabAuthoringBoundaryV1,
          "embedded Lab export schema, status, or qualification boundary differs");
  requireKeys(dictionary(field(lab, @"source", "Lab source"), "Lab source"),
              {"equality_payload_sha256", "nhknee_sha256",
               "source_model_fingerprint_sha256", "source_rigid_payload_sha256",
               "x_source_sha256"},
              "Lab source");
  NSDictionary *labPoses =
      dictionary(field(lab, @"poses", "Lab poses"), "Lab poses");
  requireKeys(labPoses, {"projected_reference", "source_default"}, "Lab poses");
  for (id key in labPoses)
    requireKeys(dictionary(labPoses[key], "Lab pose"),
                {"id", "identity_sha256"}, "Lab pose");
  NSDictionary *labMapping =
      dictionary(field(lab, @"mapping", "Lab mapping"), "Lab mapping");
  requireKeys(labMapping,
              {"algorithm", "code_identity_encoding", "code_identity_sha256",
               "diagnostics", "id"},
              "Lab mapping");
  require([mapping isEqualToDictionary:labMapping],
          "mass mapping provenance differs from embedded Lab export");
  NSDictionary *labDiagnostics =
      dictionary(field(labMapping, @"diagnostics", "Lab mapping diagnostics"),
                 "Lab mapping diagnostics");
  requireKeys(labDiagnostics,
              {"equality_residual_maximum", "finite", "jacobian",
               "maximum_displacement_m", "output_node_count",
               "raw_f32_node_mass_algorithm", "raw_f32_node_mass_sha256",
               "regions", "source_node_count"},
              "Lab mapping diagnostics");
  require(stringField(labDiagnostics, @"raw_f32_node_mass_algorithm",
                      "Lab raw node mass algorithm") ==
                  kRawF32NodeMassAlgorithm &&
              digestField(labDiagnostics, @"raw_f32_node_mass_sha256",
                          "Lab raw node mass SHA-256") ==
                  digestField(rawMass, @"sha256",
                              "raw node mass SHA-256"),
          "embedded Lab and mass arithmetic provenance differ");
  requireKeys(dictionary(field(labDiagnostics, @"jacobian", "Lab Jacobian"),
                         "Lab Jacobian"),
              {"finite", "maximum_determinant", "minimum_determinant",
               "orientation_preserving"},
              "Lab Jacobian");
  for (id value in array(field(labDiagnostics, @"regions", "Lab mapping regions"),
                         "Lab mapping regions"))
    requireKeys(dictionary(value, "Lab mapping region"),
                {"direct_failure_jacobian", "direct_failure_tetrahedron",
                 "direct_map_status", "maximum_anchor_residual_m",
                 "maximum_jacobian",
                 "maximum_persisted_f32_anchor_residual_m",
                 "maximum_source_anchor_reconstruction_residual_m",
                 "minimum_jacobian", "name", "substep_count"},
                "Lab mapping region");
  NSDictionary *labDonors = dictionary(
      field(lab, @"donor_moments", "Lab donor moments"), "Lab donor moments");
  requireKeys(labDonors, {"donors", "frame_id", "policy_id", "source_kind"},
              "Lab donor moments");
  for (id value in array(field(labDonors, @"donors", "Lab donors"),
                         "Lab donors")) {
    NSDictionary *donor = dictionary(value, "Lab donor");
    requireKeys(donor,
                {"core_body_index", "moments", "semantic_id", "source_body_id"},
                "Lab donor");
    validateMomentPayload(
        dictionary(field(donor, @"moments", "Lab donor moments"),
                   "Lab donor moments"),
        "Lab donor moments", false);
  }
  NSDictionary *labXReference =
      dictionary(field(lab, @"x_ref", "Lab x_ref"), "Lab x_ref");
  requireKeys(labXReference,
              {"anchor_ownership_encoding", "anchor_ownership_sha256", "bytes",
               "encoding", "executable_fem_topology_encoding",
               "executable_fem_topology_sha256", "file_sha256", "local_order",
               "node_count", "region_order", "region_spans",
               "source_global_node_index_encoding",
               "source_global_node_index_sha256"},
              "Lab x_ref");
  for (id value in array(
           field(labXReference, @"region_spans", "Lab x_ref region spans"),
           "Lab x_ref region spans"))
    requireKeys(dictionary(value, "Lab x_ref region span"),
                {"first_node", "first_tetrahedron", "name", "node_count",
                 "runtime_first_node", "runtime_first_tetrahedron",
                 "runtime_node_count", "runtime_tetrahedron_count",
                 "tetrahedron_count"},
                "Lab x_ref region span");
}

std::vector<NumiHumanLoadedKneeDigest>
deriveOwnershipScope(const OwnershipRecordMap &records,
                     NSDictionary *manifest) {
  std::set<std::string> uniqueLeaves;
  std::vector<NumiHumanLoadedKneeDigest> result;
  const auto recordFor = [&records](const std::string &semantic,
                                    const std::string_view label) {
    const auto found = records.find(semantic);
    require(found != records.end(),
            "ownership manifest omits " + std::string(label) + " " + semantic);
    return found->second;
  };
  const auto appendLeaves = [&uniqueLeaves, &result](NSDictionary *record) {
    for (const auto &leaf : sortedUniqueStrings(
             field(record, @"coverage_leaf_sha256s", "ownership record leaves"),
             "ownership record leaves")) {
      require(uniqueLeaves.insert(leaf).second,
              "Human semantic ownership scope repeats a coverage leaf");
      result.push_back(parseDigest(leaf, "ownership coverage leaf"));
    }
  };
  const auto manifestOwner = [](NSDictionary *owners, NSString *role,
                                const std::string_view label) {
    NSDictionary *owner =
        dictionary(field(owners, role, label), std::string(label));
    requireKeys(owner, {"owner_id", "status"}, label);
    require(stringField(owner, @"status", std::string(label) + " status") ==
                "candidate",
            std::string(label) + " is not candidate-scoped");
    return stringField(owner, @"owner_id", std::string(label) + " owner ID");
  };

  NSArray *regions =
      array(field(manifest, @"regions", "Human regions"), "Human regions");
  std::set<std::string> donors;
  for (NSUInteger index = 0u; index < regions.count; ++index) {
    NSDictionary *region = dictionary(regions[index], "Human region");
    const std::string regionName =
        stringField(region, @"name", "Human region name");
    const std::string topologySemantic = stringField(
        region, @"topology_semantic_id", "region topology semantic ID");
    NSDictionary *topologyRecord =
        recordFor(topologySemantic, "Human topology semantic ID");
    appendLeaves(topologyRecord);
    NSDictionary *owners = dictionary(
        field(region, @"owners", "Human region owners"), "Human region owners");
    requireKeys(owners,
                {"active_force", "material", "mechanical_mass",
                 "physical_volume", "state"},
                "Human region owners");
    requireOwnershipRole(
        topologyRecord, @"physical_volume", "candidate",
        manifestOwner(owners, @"physical_volume", "physical-volume owner"),
        "topology physical-volume owner");
    requireOwnershipRole(
        topologyRecord, @"mechanical_mass", "candidate",
        manifestOwner(owners, @"mechanical_mass", "mechanical-mass owner"),
        "topology mechanical-mass owner");
    requireOwnershipRole(topologyRecord, @"state", "candidate",
                         manifestOwner(owners, @"state", "state owner"),
                         "topology state owner");
    NSDictionary *activeOwner =
        dictionary(field(owners, @"active_force", "active-force owner"),
                   "active-force owner");
    requireKeys(activeOwner, {"owner_id", "status"}, "active-force owner");
    if (regionName == "QAT") {
      requireOwnershipRole(
          topologyRecord, @"active_force", "candidate",
          manifestOwner(owners, @"active_force", "active-force owner"),
          "topology active-force owner");
    } else {
      require(stringField(activeOwner, @"status",
                          "active-force owner status") == "none" &&
                  field(activeOwner, @"owner_id", "active-force owner ID") ==
                      [NSNull null],
              "passive topology gained a manifest active-force owner");
      requireOwnershipRole(topologyRecord, @"active_force", "unresolved", {},
                           "topology active-force owner");
    }
    NSArray *materialIDs = array(
        field(region, @"material_semantic_ids", "region material semantic IDs"),
        "region material semantic IDs");
    for (id materialID in materialIDs) {
      const std::string semantic =
          stringValue(materialID, "region material semantic ID");
      NSDictionary *materialRecord =
          recordFor(semantic, "Human material semantic ID");
      appendLeaves(materialRecord);
      requireOwnershipRole(materialRecord, @"material", "candidate",
                           manifestOwner(owners, @"material", "material owner"),
                           "material owner");
    }
    donors.insert(stringField(region, @"donor_body_semantic_id",
                              "region donor semantic ID"));
  }
  NSArray *pairs =
      array(field(manifest, @"articular_contact_pairs", "Human contact pairs"),
            "Human contact pairs");
  for (id value in pairs) {
    NSDictionary *pair = dictionary(value, "Human contact pair");
    const std::string semantic =
        stringField(pair, @"semantic_id", "contact semantic ID");
    NSDictionary *record = recordFor(semantic, "Human contact semantic ID");
    appendLeaves(record);
    NSDictionary *stateOwner =
        dictionary(field(pair, @"state_owner", "contact state owner"),
                   "contact state owner");
    requireKeys(stateOwner, {"owner_id", "status"}, "contact state owner");
    require(stringField(stateOwner, @"status", "contact owner status") ==
                "candidate",
            "contact state owner is not candidate-scoped");
    requireOwnershipRole(
        record, @"state", "candidate",
        stringField(stateOwner, @"owner_id", "contact owner ID"),
        "contact state owner");
  }
  NSArray *replacements = array(
      field(manifest, @"active_force_replacements", "Human force replacements"),
      "Human force replacements");
  for (id value in replacements) {
    NSDictionary *replacement = dictionary(value, "Human force replacement");
    const std::string semantic =
        stringField(replacement, @"route_semantic_id", "route semantic ID");
    NSDictionary *record = recordFor(semantic, "Human route semantic ID");
    appendLeaves(record);
    NSDictionary *action =
        dictionary(field(record, @"action", "route action"), "route action");
    requireKeys(action, {"semantic_action_id", "source_index", "status"},
                "route action");
    require(stringField(action, @"status", "route action status") == "source" &&
                stringField(action, @"semantic_action_id", "route action ID") ==
                    stringField(replacement, @"semantic_action_id",
                                "replacement action ID") &&
                unsignedField(action, @"source_index", "route action index") ==
                    unsignedField(replacement, @"source_actuator_index",
                                  "replacement source actuator index"),
            "route action differs from the Human replacement");
    NSDictionary *force =
        dictionary(field(record, @"force_semantics", "route force semantics"),
                   "route force semantics");
    require(
        stringField(force, @"status", "route force status") == "candidate" &&
            stringField(force, @"mode", "route force mode") == "replacement",
        "route force semantics are not a candidate replacement");
    const auto replaced = sortedUniqueStrings(
        field(force, @"replaces_owner_ids", "route replaced owners"),
        "route replaced owners");
    require(replaced.size() == 1u &&
                replaced.front() == stringField(replacement,
                                                @"replaces_owner_id",
                                                "replaced owner ID"),
            "route replacement target differs from Human authoring");
    requireOwnershipRole(record, @"active_force", "candidate",
                         stringField(replacement, @"replacement_owner_id",
                                     "replacement owner ID"),
                         "route active-force owner");
    requireOwnershipRole(record, @"state", "candidate",
                         "humanpack:loaded-anatomy-knee:left/full-state",
                         "route state owner");
  }

  NSDictionary *mass = dictionary(
      field(manifest, @"mass_partition", "mass partition"), "mass partition");
  NSArray *massDonors =
      array(field(mass, @"donors", "mass donors"), "mass donors");
  std::set<std::string> admittedDonors;
  for (id value in massDonors) {
    NSDictionary *donor = dictionary(value, "mass donor");
    const std::string semantic =
        stringField(donor, @"semantic_id", "mass donor semantic ID");
    require(donors.contains(semantic) && admittedDonors.insert(semantic).second,
            "mass donor differs from the region donor scope");
    NSDictionary *record = recordFor(semantic, "Human donor semantic ID");
    appendLeaves(record);
    const std::uint64_t sourceBody =
        unsignedField(donor, @"source_body_id", "mass donor source body ID");
    const std::string expectedOwner =
        sourceBody == 98u   ? "humanpack:source-rigid-body-mass/femur_l"
        : sourceBody == 99u ? "humanpack:source-rigid-body-mass/tibia_l"
                            : "";
    require(!expectedOwner.empty(), "mass donor source body is unsupported");
    requireOwnershipRole(record, @"mechanical_mass", "candidate", expectedOwner,
                         "donor mechanical-mass owner");
  }
  require(admittedDonors == donors,
          "mass partition does not exactly cover the region donors");
  std::sort(result.begin(), result.end());
  require(!result.empty(), "derived Human semantic ownership scope is empty");
  return result;
}

NSDictionary *endpointWithRole(NSArray *endpoints,
                               const std::string_view role) {
  NSDictionary *found = nil;
  for (id value in endpoints) {
    NSDictionary *endpoint = dictionary(value, "replacement endpoint");
    if (stringField(endpoint, @"role", "replacement endpoint role") == role) {
      require(found == nil,
              "force replacement repeats endpoint role " + std::string(role));
      found = endpoint;
    }
  }
  require(found != nil,
          "force replacement omits endpoint role " + std::string(role));
  return found;
}

void populateAuthoring(
    NSDictionary *manifest, const NumiHumanLoadedKneeDigest &manifestIdentity,
    const std::vector<NumiHumanLoadedKneeDigest> &derivedCoverage,
    NumiHumanLoadedKneeAuthoringV1 &output) {
  NumiHumanLoadedKneeAuthoringV1 candidate{};
  candidate.schema = stringField(manifest, @"schema", "Human schema");
  candidate.compiler = stringField(manifest, @"compiler", "Human compiler");
  candidate.status = stringField(manifest, @"status", "Human status");
  candidate.sourceOwnershipStatus = stringField(
      manifest, @"source_ownership_status", "source ownership status");
  candidate.side = stringField(manifest, @"side", "Human side");
  candidate.manifestSHA256 = manifestIdentity;
  candidate.ownershipManifestSHA256 =
      digestField(manifest, @"ownership_manifest_sha256", "ownership identity");
  candidate.manifestCanonicalization = stringField(
      manifest, @"manifest_canonicalization", "Human canonicalization");
  candidate.manifestHashExclusion =
      stringField(manifest, @"manifest_hash_exclusion", "Human hash exclusion");
  candidate.subjectID = stringField(manifest, @"subject_id", "Human subject");
  candidate.coverageLeafSHA256s = derivedCoverage;
  candidate.boundary = stringField(manifest, @"boundary", "Human boundary");

  NSDictionary *inputs =
      dictionary(field(manifest, @"inputs", "Human inputs"), "Human inputs");
  requireKeys(inputs,
              {"authoring_profile", "lab_export", "open_knee_payload",
               "ownership", "tendon_payload", "x_ref"},
              "Human inputs");
  candidate.ownershipInput = immutableIdentity(
      dictionary(field(inputs, @"ownership", "ownership input"),
                 "ownership input"),
      "ownership input");
  candidate.authoringProfileInput = immutableIdentity(
      dictionary(field(inputs, @"authoring_profile", "authoring profile input"),
                 "authoring profile input"),
      "authoring profile input");
  const auto openKneeInput = immutableIdentity(
      dictionary(field(inputs, @"open_knee_payload", "Open Knee input"),
                 "Open Knee input"),
      "Open Knee input");
  candidate.tendonPayloadInput = immutableIdentity(
      dictionary(field(inputs, @"tendon_payload", "tendon input"),
                 "tendon input"),
      "tendon input");
  candidate.xReferenceInput = immutableIdentity(
      dictionary(field(inputs, @"x_ref", "x_ref input"), "x_ref input"),
      "x_ref input");
  candidate.labExportInput = immutableIdentity(
      dictionary(field(inputs, @"lab_export", "Lab export input"),
                 "Lab export input"),
      "Lab export input");

  NSDictionary *source =
      dictionary(field(manifest, @"source", "Human source"), "Human source");
  candidate.datasetID = stringField(source, @"dataset_id", "source dataset ID");
  NSDictionary *sourceFiles =
      dictionary(field(source, @"files", "source files"), "source files");
  candidate.licenseFileSHA256 =
      digestField(sourceFiles, @"license.txt", "Open Knee license SHA-256");
  NSDictionary *knee = dictionary(field(source, @"nhknee", "NHKNEE1 identity"),
                                  "NHKNEE1 identity");
  candidate.kneePayload = {
      .schema = openKneeInput.schema,
      .magic = stringField(knee, @"magic", "NHKNEE1 magic"),
      .abi = unsigned32Field(knee, @"abi", "NHKNEE1 ABI"),
      .byteCount = unsignedField(knee, @"bytes", "NHKNEE1 bytes"),
      .fileSHA256 = digestField(knee, @"sha256", "NHKNEE1 SHA-256"),
  };
  require(candidate.kneePayload.fileSHA256 == openKneeInput.fileSHA256 &&
              openKneeInput.fileSHA256 == openKneeInput.identitySHA256,
          "Open Knee input and source identities differ");
  NSDictionary *tendon =
      dictionary(field(source, @"tendon_payload", "tendon payload identity"),
                 "tendon payload identity");
  candidate.tendonPayload = {
      .schema = stringField(tendon, @"schema", "tendon payload schema"),
      .magic = stringField(tendon, @"magic", "tendon payload magic"),
      .abi = unsigned32Field(tendon, @"abi", "tendon payload ABI"),
      .byteCount = unsignedField(tendon, @"bytes", "tendon payload bytes"),
      .fileSHA256 = digestField(tendon, @"sha256", "tendon payload SHA-256"),
  };
  require(candidate.tendonPayload.fileSHA256 ==
                  candidate.tendonPayloadInput.fileSHA256 &&
              candidate.tendonPayloadInput.fileSHA256 ==
                  candidate.tendonPayloadInput.identitySHA256,
          "tendon input and source identities differ");

  NSDictionary *mass = dictionary(
      field(manifest, @"mass_partition", "mass partition"), "mass partition");
  candidate.authoredMassPartitionSHA256 =
      digestField(mass, @"identity_sha256", "mass partition identity");
  NSDictionary *rawMass =
      dictionary(field(mass, @"raw_f32_node_mass", "raw node mass identity"),
                 "raw node mass identity");
  candidate.authoredRawF32NodeMassSHA256 =
      digestField(rawMass, @"sha256", "raw node mass SHA-256");
  NSDictionary *provenance = dictionary(
      field(mass, @"provenance", "mass provenance"), "mass provenance");
  candidate.sourceModelFingerprintSHA256 =
      digestField(provenance, @"source_model_fingerprint_sha256",
                  "source model fingerprint");
  candidate.sourceRigidPayloadSHA256 =
      digestField(provenance, @"source_rigid_payload_sha256",
                  "source rigid payload SHA-256");
  candidate.equalityPayloadSHA256 = digestField(
      provenance, @"equality_payload_sha256", "equality payload SHA-256");
  NSDictionary *poses = dictionary(
      field(provenance, @"poses", "pose provenance"), "pose provenance");
  NSDictionary *sourcePose =
      dictionary(field(poses, @"source_default", "source default pose"),
                 "source default pose");
  candidate.sourceDefaultPoseID =
      stringField(sourcePose, @"id", "source default pose ID");
  candidate.sourceDefaultPoseSHA256 = digestField(
      sourcePose, @"identity_sha256", "source default pose identity");
  NSDictionary *referencePose = dictionary(
      field(poses, @"projected_reference", "projected reference pose"),
      "projected reference pose");
  candidate.projectedReferencePoseID =
      stringField(referencePose, @"id", "projected reference pose ID");
  candidate.projectedReferencePoseSHA256 = digestField(
      referencePose, @"identity_sha256", "projected reference pose identity");
  NSDictionary *mapping =
      dictionary(field(provenance, @"mapping", "source-to-reference mapping"),
                 "source-to-reference mapping");
  candidate.sourceToReferenceMappingID =
      stringField(mapping, @"id", "source-to-reference mapping ID");
  candidate.sourceToReferenceMappingAlgorithm = stringField(
      mapping, @"algorithm", "source-to-reference mapping algorithm");
  candidate.sourceToReferenceMappingCodeSHA256 =
      digestField(mapping, @"code_identity_sha256", "mapping code identity");

  NSDictionary *topology = dictionary(
      field(manifest, @"topology", "Human topology"), "Human topology");
  candidate.topology.identitySHA256 =
      digestField(topology, @"identity_sha256", "topology identity");
  candidate.topology.executableFEMTopologySHA256 =
      digestField(topology, @"executable_fem_topology_sha256",
                  "executable FEM topology identity");
  candidate.topology.sourceGlobalNodeIndexSHA256 =
      digestField(topology, @"source_global_node_index_sha256",
                  "source global-node identity");
  candidate.topology.anchorOwnershipSHA256 = digestField(
      topology, @"anchor_ownership_sha256", "source anchor identity");
  candidate.topology.nodeCount =
      unsigned32Field(topology, @"node_count", "topology node count");
  candidate.topology.tetrahedronCount = unsigned32Field(
      topology, @"tetrahedron_count", "topology tetrahedron count");
  candidate.topology.surfaceCount =
      unsigned32Field(topology, @"surface_count", "topology surface count");
  candidate.topology.surfaceFaceCount = unsigned32Field(
      topology, @"surface_face_count", "topology surface-face count");
  candidate.topology.nodeSetCount =
      unsigned32Field(topology, @"node_set_count", "topology node-set count");
  candidate.topology.nodeSetMembershipCount =
      unsigned32Field(topology, @"node_set_membership_count",
                      "topology node-set membership count");
  candidate.topology.sourceSurfacePairCount = unsigned32Field(
      topology, @"surface_pair_count", "source surface-pair count");
  candidate.topology.loadedNodeCount =
      unsigned32Field(topology, @"loaded_node_count", "loaded node count");
  candidate.topology.loadedTetrahedronCount = unsigned32Field(
      topology, @"loaded_tetrahedron_count", "loaded tetrahedron count");
  NSArray *spans =
      array(field(topology, @"region_spans", "topology region spans"),
            "topology region spans");
  require(spans.count == candidate.topology.loadedRegionNames.size(),
          "topology region-span count differs");
  for (NSUInteger index = 0u; index < spans.count; ++index) {
    candidate.topology.loadedRegionNames[index] =
        stringField(dictionary(spans[index], "topology region span"), @"name",
                    "topology region-span name");
  }

  NSDictionary *coordinates =
      dictionary(field(manifest, @"coordinates", "Human coordinates"),
                 "Human coordinates");
  NSDictionary *xSource =
      dictionary(field(coordinates, @"x_source", "x_source identity"),
                 "x_source identity");
  NSDictionary *xReference = dictionary(
      field(coordinates, @"x_ref", "x_ref identity"), "x_ref identity");
  NSDictionary *xCurrent =
      dictionary(field(coordinates, @"x_current", "x_current authority"),
                 "x_current authority");
  candidate.coordinates.sourceStateSHA256 =
      digestField(xSource, @"sha256", "x_source SHA-256");
  candidate.coordinates.referenceStateSHA256 =
      digestField(xReference, @"sha256", "x_ref SHA-256");
  candidate.coordinates.sourceStateNodeCount =
      unsigned32Field(xSource, @"node_count", "x_source node count");
  candidate.coordinates.referenceStateNodeCount =
      unsigned32Field(xReference, @"node_count", "x_ref node count");
  candidate.coordinates.sourceStateEncoding =
      stringField(xSource, @"encoding", "x_source encoding");
  candidate.coordinates.referenceStateEncoding =
      stringField(xReference, @"encoding", "x_ref encoding");
  candidate.coordinates.sourceFrameID =
      stringField(xSource, @"frame_id", "x_source frame");
  candidate.coordinates.referenceFrameID =
      stringField(xReference, @"frame_id", "x_ref frame");
  candidate.coordinates.referenceStateClass =
      stringField(xReference, @"reference_state_class", "x_ref state class");
  candidate.coordinates.constructionID =
      stringField(xReference, @"construction_id", "x_ref construction ID");
  candidate.coordinates.unloadedReferenceQualified = booleanField(
      xReference, @"unloaded_reference_qualified", "unloaded reference status");
  NSDictionary *prestrain = dictionary(
      field(xReference, @"prestrain_reset_method", "prestrain reset method"),
      "prestrain reset method");
  candidate.coordinates.prestrainResetStatus =
      stringField(prestrain, @"status", "prestrain reset status");
  id prestrainMethod =
      field(prestrain, @"method_id", "prestrain reset method ID");
  require(prestrainMethod == [NSNull null] ||
              [prestrainMethod isKindOfClass:[NSString class]],
          "prestrain reset method ID must be null or a string");
  if (prestrainMethod != [NSNull null]) {
    candidate.coordinates.prestrainResetMethodID =
        stringValue(prestrainMethod, "prestrain reset method ID");
  }
  candidate.coordinates.volumetricPrestressStatus =
      stringField(xReference, @"volumetric_prestress_status",
                  "volumetric prestress status");
  candidate.coordinates.currentStateOwner =
      stringField(xCurrent, @"authority", "x_current authority");
  candidate.coordinates.currentStateHashAlgorithm =
      stringField(xCurrent, @"hash_algorithm", "x_current hash algorithm");
  candidate.coordinates.currentStateHashScope =
      stringField(xCurrent, @"scope", "x_current hash scope");
  require(
      field(xCurrent, @"accepted_sha256", "accepted x_current") ==
              [NSNull null] &&
          field(xCurrent, @"acceptance_receipt_sha256",
                "x_current acceptance receipt") == [NSNull null],
      "Human authoring receipt contains accepted runtime x_current evidence");
  candidate.coordinates.currentStateRequired = true;
  require(candidate.coordinates.referenceStateSHA256 ==
                  candidate.xReferenceInput.fileSHA256 &&
              candidate.xReferenceInput.fileSHA256 ==
                  candidate.xReferenceInput.identitySHA256,
          "x_ref coordinate and input identities differ");

  NSDictionary *density =
      dictionary(field(manifest, @"density_conversion", "density conversion"),
                 "density conversion");
  candidate.densitySourceValue =
      doubleField(density, @"source_value", "density source value");
  candidate.densitySourceUnit =
      stringField(density, @"source_unit", "density source unit");
  candidate.densityConversionFactorToKgPerM3 = doubleField(
      density, @"conversion_factor_to_kg_per_m3", "density conversion factor");
  candidate.densityRuntimeKgPerM3 =
      doubleField(density, @"runtime_value_kg_per_m3", "runtime density");
  candidate.densityCalibrationStatus =
      stringField(density, @"calibration_status", "density calibration status");

  std::map<std::string, std::uint32_t> donorByRegion;
  NSArray *donors = array(field(mass, @"donors", "mass donors"), "mass donors");
  for (id donorValue in donors) {
    NSDictionary *donor = dictionary(donorValue, "mass donor");
    const auto body =
        unsigned32Field(donor, @"core_body_index", "mass donor body index");
    for (const auto &region :
         stringArray(field(donor, @"region_names", "mass donor region names"),
                     "mass donor region names")) {
      require(donorByRegion.emplace(region, body).second,
              "mass donor region is assigned more than once");
    }
  }
  NSArray *regions =
      array(field(manifest, @"regions", "Human regions"), "Human regions");
  require(regions.count == candidate.regions.size(),
          "Human region count differs");
  for (NSUInteger index = 0u; index < regions.count; ++index) {
    NSDictionary *region = dictionary(regions[index], "Human region");
    auto &target = candidate.regions[index];
    target.sourceRegionName = stringField(region, @"name", "region name");
    target.semanticID =
        stringField(region, @"topology_semantic_id", "region semantic ID");
    const auto donor = donorByRegion.find(target.sourceRegionName);
    require(donor != donorByRegion.end(), "region has no exact mass donor");
    target.donorBodyIndex = donor->second;
    NSDictionary *payload =
        dictionary(field(region, @"payload", "region payload identity"),
                   "region payload identity");
    target.topologyIdentitySHA256 =
        digestField(payload, @"tetrahedra_sha256", "region topology identity");
    NSDictionary *material = dictionary(
        field(region, @"material", "region material"), "region material");
    target.material.sourceType =
        stringField(material, @"type", "material source type");
    target.material.c1Pascals =
        1.0e6 * doubleField(material, @"c1_mpa", "material c1");
    target.material.c2Pascals =
        1.0e6 * doubleField(material, @"c2_mpa", "material c2");
    target.material.c3Pascals =
        1.0e6 * doubleField(material, @"c3_mpa", "material c3");
    target.material.c4 = doubleField(material, @"c4", "material c4");
    target.material.c5Pascals =
        1.0e6 * doubleField(material, @"c5_mpa", "material c5");
    target.material.lambdaMaximum =
        doubleField(material, @"lambda_max", "material lambda maximum");
    target.material.bulkModulusPascals =
        1.0e6 *
        doubleField(material, @"bulk_modulus_mpa", "material bulk modulus");
    target.material.initialStretch = doubleField(
        material, @"source_initial_stretch", "material initial stretch");
    target.material.homogeneousFiberWorld = vector3(
        field(material, @"fiber_world", "material fiber"), "material fiber");
    target.material.calibrationStatus = stringField(
        material, @"calibration_status", "material calibration status");
    NSDictionary *owners =
        dictionary(field(region, @"owners", "region owners"), "region owners");
    const auto owner = [owners](NSString *name, const std::string_view label) {
      return dictionary(field(owners, name, label), label);
    };
    target.physicalVolumeOwnerID =
        stringField(owner(@"physical_volume", "physical-volume owner"),
                    @"owner_id", "physical-volume owner ID");
    target.mechanicalMassOwnerID =
        stringField(owner(@"mechanical_mass", "mechanical-mass owner"),
                    @"owner_id", "mechanical-mass owner ID");
    target.materialOwnerID = stringField(owner(@"material", "material owner"),
                                         @"owner_id", "material owner ID");
    NSDictionary *active = owner(@"active_force", "active-force owner");
    target.activeForceOwnerStatus =
        stringField(active, @"status", "active-force owner status");
    id activeOwner = field(active, @"owner_id", "active-force owner ID");
    require(activeOwner == [NSNull null] ||
                [activeOwner isKindOfClass:[NSString class]],
            "active-force owner ID must be null or a string");
    if (activeOwner != [NSNull null])
      target.activeForceOwnerID =
          stringValue(activeOwner, "active-force owner ID");
    target.stateOwnerID = stringField(owner(@"state", "state owner"),
                                      @"owner_id", "state owner ID");
  }

  NSArray *pairs =
      array(field(manifest, @"articular_contact_pairs", "Human contact pairs"),
            "Human contact pairs");
  require(pairs.count == candidate.contactPairs.size(),
          "Human contact-pair count differs");
  for (NSUInteger index = 0u; index < pairs.count; ++index) {
    NSDictionary *pair = dictionary(pairs[index], "Human contact pair");
    auto &target = candidate.contactPairs[index];
    target.semanticID =
        stringField(pair, @"semantic_id", "contact semantic ID");
    target.sourcePairName = stringField(pair, @"name", "contact pair name");
    target.masterSurface =
        stringField(pair, @"master_surface", "contact master surface");
    target.slaveSurface =
        stringField(pair, @"slave_surface", "contact slave surface");
    NSDictionary *state =
        dictionary(field(pair, @"state_owner", "contact state owner"),
                   "contact state owner");
    require(stringField(state, @"status", "contact owner status") ==
                "candidate",
            "contact state owner is not candidate-scoped");
    target.ownerID = stringField(state, @"owner_id", "contact owner ID");
  }

  NSArray *replacements =
      array(field(manifest, @"active_force_replacements", "force replacements"),
            "force replacements");
  require(replacements.count == candidate.activeReplacements.size(),
          "force-replacement count differs");
  for (NSUInteger index = 0u; index < replacements.count; ++index) {
    NSDictionary *replacement =
        dictionary(replacements[index], "force replacement");
    auto &target = candidate.activeReplacements[index];
    target.semanticID =
        stringField(replacement, @"route_semantic_id", "route semantic ID");
    target.sourceActuatorIndex = unsigned32Field(
        replacement, @"source_actuator_index", "source actuator index");
    target.sourceRouteName =
        stringField(replacement, @"name", "source route name");
    target.sourceOwnerID =
        stringField(replacement, @"replaces_owner_id", "replaced owner ID");
    target.candidateOwnerID = stringField(replacement, @"replacement_owner_id",
                                          "replacement owner ID");
    target.replacementFraction =
        doubleField(replacement, @"replacement_scale", "replacement scale");
    require(stringField(replacement, @"status", "replacement status") ==
                "candidate",
            "force replacement is not candidate-scoped");
    target.mode = "replacement";
    NSArray *endpoints =
        array(field(replacement, @"endpoints", "replacement endpoints"),
              "replacement endpoints");
    require(endpoints.count == 2u, "force replacement endpoint count differs");
    NSDictionary *load = endpointWithRole(endpoints, "load");
    NSDictionary *anchor = endpointWithRole(endpoints, "anchor");
    target.loadEndpointIndex =
        unsigned32Field(load, @"endpoint_index", "load endpoint index");
    target.loadRouteNodeIndex =
        unsigned32Field(load, @"route_node_index", "load route-node index");
    target.loadSourceSiteIndex =
        unsigned32Field(load, @"source_site_index", "load source-site index");
    target.loadBodyIndex =
        unsigned32Field(load, @"body_index", "load body index");
    target.anchorEndpointIndex =
        unsigned32Field(anchor, @"endpoint_index", "anchor endpoint index");
    target.anchorRouteNodeIndex =
        unsigned32Field(anchor, @"route_node_index", "anchor route-node index");
    target.anchorSourceSiteIndex = unsigned32Field(anchor, @"source_site_index",
                                                   "anchor source-site index");
    target.anchorBodyIndex =
        unsigned32Field(anchor, @"body_index", "anchor body index");
  }

  NSArray *passive = array(
      field(manifest, @"passive_ligament_owners", "passive ligament owners"),
      "passive ligament owners");
  require(passive.count == candidate.passiveOwners.size(),
          "passive-owner count differs");
  for (NSUInteger index = 0u; index < passive.count; ++index) {
    NSDictionary *value = dictionary(passive[index], "passive ligament owner");
    auto &target = candidate.passiveOwners[index];
    target.sourceRegionName =
        stringField(value, @"name", "passive region name");
    target.semanticID =
        stringField(value, @"semantic_id", "passive semantic ID");
    NSDictionary *owner =
        dictionary(field(value, @"passive_force_owner", "passive-force owner"),
                   "passive-force owner");
    require(stringField(owner, @"status", "passive owner status") ==
                "candidate",
            "passive-force owner is not candidate-scoped");
    target.ownerID = stringField(owner, @"owner_id", "passive owner ID");
  }

  NSDictionary *fullState = dictionary(
      field(manifest, @"full_state_authority", "full-state authority"),
      "full-state authority");
  candidate.fullStateSemanticID =
      stringField(fullState, @"semantic_id", "full-state semantic ID");
  candidate.fullStateOwnerID =
      stringField(fullState, @"owner_id", "full-state owner ID");
  candidate.fullStateSchema =
      stringField(fullState, @"schema", "full-state schema");
  candidate.fullStateIdentitySHA256 =
      digestField(fullState, @"identity_sha256", "full-state identity");
  candidate.requiredSnapshotComponents = stringArray(
      field(fullState, @"required_snapshot_components", "snapshot components"),
      "snapshot components");

  NSDictionary *qualification =
      dictionary(field(manifest, @"qualification", "Human qualification"),
                 "Human qualification");
  requireKeys(qualification,
              {"candidate_only", "clinical_validity_qualified",
               "donor_mass_and_raw_moments_subtracted",
               "integrated_human_qualification", "mesh_convergence_qualified",
               "ownership_identity_bound", "prestrain_reference_reset_executed",
               "production_active_force", "production_physical_ownership",
               "projected_reference_identity_bound",
               "runtime_x_current_accepted", "source_identity_bound",
               "specimen_load_validation_qualified",
               "subject_material_calibrated", "topology_identity_bound",
               "unloaded_reference_qualified"},
              "Human qualification");
  require(
      booleanField(qualification, @"candidate_only", "candidate-only status") &&
          booleanField(qualification, @"source_identity_bound",
                       "source identity status") &&
          booleanField(qualification, @"ownership_identity_bound",
                       "ownership identity status") &&
          booleanField(qualification, @"topology_identity_bound",
                       "topology identity status") &&
          booleanField(qualification, @"projected_reference_identity_bound",
                       "projected-reference identity status") &&
          booleanField(qualification, @"donor_mass_and_raw_moments_subtracted",
                       "donor subtraction status") &&
          !booleanField(qualification, @"unloaded_reference_qualified",
                        "unloaded reference qualification") &&
          !booleanField(qualification, @"prestrain_reference_reset_executed",
                        "prestrain reset qualification") &&
          !booleanField(qualification, @"subject_material_calibrated",
                        "subject material calibration") &&
          !booleanField(qualification, @"mesh_convergence_qualified",
                        "mesh convergence qualification") &&
          !booleanField(qualification, @"specimen_load_validation_qualified",
                        "specimen-load qualification") &&
          !booleanField(qualification, @"clinical_validity_qualified",
                        "clinical validity qualification") &&
          !booleanField(qualification, @"production_physical_ownership",
                        "production ownership status") &&
          !booleanField(qualification, @"production_active_force",
                        "production active-force status") &&
          !booleanField(qualification, @"runtime_x_current_accepted",
                        "runtime x_current acceptance status") &&
          !booleanField(qualification, @"integrated_human_qualification",
                        "integrated Human status"),
      "Human qualification overclaims or omits its candidate-only boundary");
  candidate.candidateOnly =
      booleanField(qualification, @"candidate_only", "candidate-only status");
  const bool productionOwnership =
      booleanField(qualification, @"production_physical_ownership",
                   "production ownership status");
  const bool productionActive =
      booleanField(qualification, @"production_active_force",
                   "production active-force status");
  const bool integrated =
      booleanField(qualification, @"integrated_human_qualification",
                   "integrated Human status");
  candidate.productionQualified =
      productionOwnership || productionActive || integrated;
  candidate.productionPromotion = candidate.productionQualified;
  candidate.unloadedReferenceQualified =
      booleanField(qualification, @"unloaded_reference_qualified",
                   "unloaded reference qualification");
  candidate.subjectCalibratedMaterials =
      booleanField(qualification, @"subject_material_calibrated",
                   "subject material calibration");
  candidate.donorMassSubtractionRequired =
      booleanField(qualification, @"donor_mass_and_raw_moments_subtracted",
                   "donor subtraction status");
  candidate.massMomentClosureRequired = candidate.donorMassSubtractionRequired;

  output = std::move(candidate);
}

struct SourceComplianceProgramSpec final {
  std::string_view key;
  std::string_view schema;
  std::string_view filename;
  std::string_view magicLabel;
  std::array<std::uint8_t, 8u> magic;
  std::uint32_t abi = 0u;
  std::uint32_t rowCount = 0u;
  std::uint32_t recordBytes = 0u;
  std::uint64_t byteCount = 0u;
  std::string_view fileSHA256;
};

constexpr SourceComplianceProgramSpec kSourceEqualityProgram{
    .key = "joint_equalities",
    .schema = "numi.human.joint-equality-source-compliance-payload.v1",
    .filename = "myosim-fullbody-joint-equalities-source-compliance.nheq",
    .magicLabel = "NHEQ2",
    .magic = {{'N', 'H', 'E', 'Q', '2', 0u, 0u, 0u}},
    .abi = 2u,
    .rowCount = 51u,
    .recordBytes = 112u,
    .byteCount = 5'792u,
    .fileSHA256 =
        "12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477",
};

constexpr SourceComplianceProgramSpec kSourceLimitProgram{
    .key = "joint_limits",
    .schema = "numi.human.joint-limit-source-compliance-payload.v1",
    .filename = "myosim-fullbody-joint-limits.nhlim",
    .magicLabel = "NHLIM1",
    .magic = {{'N', 'H', 'L', 'I', 'M', '1', 0u, 0u}},
    .abi = 1u,
    .rowCount = 122u,
    .recordBytes = 80u,
    .byteCount = 9'840u,
    .fileSHA256 =
        "c583611fcedc326a32c6f69504a65e675c8e0adc987c6db622ca0d95a02438d3",
};

std::uint32_t sourceProgramU32(
    const std::span<const std::uint8_t> bytes, const std::size_t offset,
    const std::string_view label) {
  require(offset <= bytes.size() && bytes.size() - offset >= 4u,
          std::string(label) + " header is truncated");
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u) |
         (static_cast<std::uint32_t>(bytes[offset + 2u]) << 16u) |
         (static_cast<std::uint32_t>(bytes[offset + 3u]) << 24u);
}

NumiHumanLoadedKneeDigest validateSourceComplianceProgramBytes(
    const std::span<const std::uint8_t> bytes,
    const SourceComplianceProgramSpec &specification) {
  constexpr std::size_t kHeaderBytes = 80u;
  constexpr std::uint32_t kNQ = 129u;
  constexpr std::uint32_t kNV = 128u;
  constexpr std::uint32_t kPolicy = 1u;
  constexpr std::uint32_t kFlags = 1u;
  const std::string label(specification.key);
  require(bytes.size() == specification.byteCount,
          label + " byte count differs");
  require(bytes.size() >= kHeaderBytes &&
              std::equal(specification.magic.begin(),
                         specification.magic.end(), bytes.begin()),
          label + " magic differs");
  require(sourceProgramU32(bytes, 8u, label) == specification.abi,
          label + " ABI differs");
  require(sourceProgramU32(bytes, 12u, label) == kNQ &&
              sourceProgramU32(bytes, 16u, label) == kNV,
          label + " nq/nv differ");
  require(sourceProgramU32(bytes, 20u, label) == specification.rowCount,
          label + " row count differs");
  require(sourceProgramU32(bytes, 24u, label) == specification.recordBytes,
          label + " record ABI differs");
  require(sourceProgramU32(bytes, 28u, label) == specification.rowCount,
          label + " source row count differs");
  require(sourceProgramU32(bytes, 32u, label) == kPolicy,
          label + " source policy differs");
  require(sourceProgramU32(bytes, 36u, label) == kFlags,
          label + " source flags differ");
  require(sourceProgramU32(bytes, 40u, label) == 0u &&
              sourceProgramU32(bytes, 44u, label) == 0u,
          label + " reserved fields differ");
  const auto sourceArchive = parseDigest(std::string(kSourceArchiveSHA256),
                                         "source archive identity");
  require(std::equal(sourceArchive.begin(), sourceArchive.end(),
                     bytes.begin() + 48u),
          label + " source archive identity differs");
  require(kHeaderBytes +
                  static_cast<std::uint64_t>(specification.rowCount) *
                      specification.recordBytes ==
              specification.byteCount,
          label + " internal byte-count contract differs");
  const auto fileIdentity = sha256(bytes);
  require(fileIdentity ==
              parseDigest(std::string(specification.fileSHA256),
                          label + " pinned file identity"),
          label + " exact payload SHA-256 differs");
  return fileIdentity;
}

void validateSourceComplianceProgramDescriptor(
    NSDictionary *program, const SourceComplianceProgramSpec &specification,
    const NumiHumanLoadedKneeDigest &actualFileSHA256) {
  const std::string label(specification.key);
  requireKeys(program,
              {"abi", "bytes", "file_sha256", "filename", "flags", "magic",
               "nq", "nv", "policy_id", "record_bytes", "refsafe",
               "row_count", "schema", "source_archive_sha256",
               "source_row_count"},
              label);
  require(stringField(program, @"schema", label + " schema") ==
                  specification.schema &&
              stringField(program, @"filename", label + " filename") ==
                  specification.filename &&
              stringField(program, @"magic", label + " magic") ==
                  specification.magicLabel,
          label + " identity differs");
  require(unsigned32Field(program, @"abi", label + " ABI") ==
                  specification.abi &&
              unsignedField(program, @"bytes", label + " bytes") ==
                  specification.byteCount &&
              unsigned32Field(program, @"nq", label + " nq") == 129u &&
              unsigned32Field(program, @"nv", label + " nv") == 128u &&
              unsigned32Field(program, @"row_count", label + " row count") ==
                  specification.rowCount &&
              unsigned32Field(program, @"record_bytes",
                              label + " record bytes") ==
                  specification.recordBytes &&
              unsigned32Field(program, @"source_row_count",
                              label + " source row count") ==
                  specification.rowCount &&
              unsigned32Field(program, @"policy_id", label + " policy") ==
                  1u &&
              unsigned32Field(program, @"flags", label + " flags") == 1u &&
              booleanField(program, @"refsafe", label + " refsafe"),
          label + " source-compliance metadata differs");
  require(digestField(program, @"file_sha256", label + " file identity") ==
                  actualFileSHA256 &&
              actualFileSHA256 ==
                  parseDigest(std::string(specification.fileSHA256),
                              label + " pinned file identity") &&
              digestField(program, @"source_archive_sha256",
                          label + " source archive identity") ==
                  parseDigest(std::string(kSourceArchiveSHA256),
                              "source archive identity"),
          label + " bound byte identity differs");
}

void validateSourceComplianceOwnership(
    NSDictionary *ownership,
    const NumiHumanLoadedKneeBindingAdmissionV1 &authenticatedBase) {
  requireKeys(ownership, {"human_source_laws", "matter_runtime_constraints"},
              "source-compliance ownership");
  NSDictionary *human = dictionary(
      field(ownership, @"human_source_laws", "Human source-law ownership"),
      "Human source-law ownership");
  requireKeys(human,
              {"authority", "owner_id", "owner_system",
               "owns_runtime_constraint_force", "owns_runtime_constraint_state",
               "programs"},
              "Human source-law ownership");
  require(stringField(human, @"owner_id", "Human source-law owner") ==
                  "numilab-human:loaded-anatomy-knee/source-constraint-laws" &&
              stringField(human, @"owner_system", "Human owner system") ==
                  "Human" &&
              stringField(human, @"authority", "Human source-law authority") ==
                  "immutable-source-law-program-bytes" &&
              !booleanField(human, @"owns_runtime_constraint_force",
                            "Human runtime-force authority") &&
              !booleanField(human, @"owns_runtime_constraint_state",
                            "Human runtime-state authority"),
          "Human source-law ownership boundary differs");
  const auto programs = stringArray(
      field(human, @"programs", "Human source-law programs"),
      "Human source-law programs");
  require(programs ==
              std::vector<std::string>{"joint_equalities", "joint_limits"},
          "Human source-law program ownership differs");

  NSDictionary *matter = dictionary(
      field(ownership, @"matter_runtime_constraints",
            "Matter runtime-constraint ownership"),
      "Matter runtime-constraint ownership");
  requireKeys(matter,
              {"accepted_state_authority_identity_sha256",
               "accepted_state_schema", "accepted_state_semantic_id",
               "authors_source_laws", "authority", "owner_id", "owner_system",
               "owns_runtime_constraint_force", "owns_runtime_constraint_state"},
              "Matter runtime-constraint ownership");
  require(stringField(matter, @"owner_id", "Matter runtime owner") ==
                  authenticatedBase.authoring.fullStateOwnerID &&
              stringField(matter, @"owner_id", "Matter runtime owner") ==
                  "numi-lab:human-matter/accepted-step-transaction" &&
              stringField(matter, @"owner_system", "Matter owner system") ==
                  "Matter" &&
              stringField(matter, @"authority", "Matter runtime authority") ==
                  "runtime-constraint-force-and-accepted-state" &&
              stringField(matter, @"accepted_state_schema",
                          "Matter accepted-state schema") ==
                  authenticatedBase.authoring.fullStateSchema &&
              stringField(matter, @"accepted_state_semantic_id",
                          "Matter accepted-state semantic ID") ==
                  authenticatedBase.authoring.fullStateSemanticID &&
              digestField(matter, @"accepted_state_authority_identity_sha256",
                          "Matter accepted-state authority identity") ==
                  authenticatedBase.authoring.fullStateIdentitySHA256 &&
              !booleanField(matter, @"authors_source_laws",
                            "Matter source-law authority") &&
              booleanField(matter, @"owns_runtime_constraint_force",
                           "Matter runtime-force authority") &&
              booleanField(matter, @"owns_runtime_constraint_state",
                           "Matter runtime-state authority"),
          "Matter runtime-constraint ownership boundary differs");
}

void validateSourceComplianceQualification(NSDictionary *qualification) {
  requireKeys(qualification,
              {"base_manifest_identity_bound", "candidate_only",
               "clinical_validity_qualified",
               "cross_program_consistency_validated",
               "exact_source_program_bytes_bound",
               "integrated_human_qualification",
               "prepared_state_identity_bound", "production_physical_ownership",
               "program_headers_validated",
               "runtime_constraint_force_or_state_executed",
               "source_rigid_identity_bound"},
              "source-compliance qualification");
  require(
      booleanField(qualification, @"candidate_only", "candidate-only status") &&
          booleanField(qualification, @"base_manifest_identity_bound",
                       "base manifest binding") &&
          booleanField(qualification, @"exact_source_program_bytes_bound",
                       "exact source-program binding") &&
          booleanField(qualification, @"source_rigid_identity_bound",
                       "source rigid binding") &&
          booleanField(qualification, @"program_headers_validated",
                       "program header validation") &&
          booleanField(qualification, @"cross_program_consistency_validated",
                       "cross-program validation") &&
          !booleanField(qualification, @"prepared_state_identity_bound",
                        "prepared-state binding") &&
          !booleanField(qualification,
                        @"runtime_constraint_force_or_state_executed",
                        "runtime constraint execution") &&
          !booleanField(qualification, @"production_physical_ownership",
                        "production ownership") &&
          !booleanField(qualification, @"clinical_validity_qualified",
                        "clinical qualification") &&
          !booleanField(qualification, @"integrated_human_qualification",
                        "integrated Human qualification"),
      "source-compliance qualification overclaims its authoring boundary");
}

} // namespace

bool loadNumiHumanLoadedKneeBindingV1(
    const std::filesystem::path &manifestPath,
    const std::filesystem::path &bindingPath,
    const std::filesystem::path &ownershipManifestPath,
    NumiHumanLoadedKneeBindingAdmissionV1 &output, std::string &error) {
  try {
    require(manifestPath.filename() == kManifestName &&
                bindingPath.filename() == kBindingName &&
                manifestPath.parent_path() == bindingPath.parent_path(),
            "Human manifest and binding must be the standard same-directory "
            "bundle");
    const auto manifestBytes =
        readRegularFile(manifestPath, 16u * 1024u * 1024u, "Human manifest");
    const auto bindingBytes =
        readRegularFile(bindingPath, 24u * 1024u * 1024u, "Human binding");
    const auto ownershipBytes = readRegularFile(
        ownershipManifestPath, 128u * 1024u * 1024u, "ownership manifest");

    CanonicalJSONScanner manifestScanner(manifestBytes);
    const auto manifestMembers = manifestScanner.validateDocument();
    CanonicalJSONScanner bindingScanner(bindingBytes);
    const auto bindingMembers = bindingScanner.validateDocument();
    CanonicalJSONScanner ownershipScanner(ownershipBytes);
    const auto ownershipMembers = ownershipScanner.validateDocument();

    NSDictionary *manifest = parseJSONObject(manifestBytes, "Human manifest");
    NSDictionary *binding = parseJSONObject(bindingBytes, "Human binding");
    NSDictionary *ownership =
        parseJSONObject(ownershipBytes, "ownership manifest");
    requireKeys(manifest,
                {
                    "active_force_replacements",
                    "articular_contact_pairs",
                    "boundary",
                    "compiler",
                    "coordinates",
                    "density_conversion",
                    "full_state_authority",
                    "inputs",
                    "lab_authoring_export",
                    "manifest_canonicalization",
                    "manifest_hash_exclusion",
                    "manifest_sha256",
                    "mass_partition",
                    "ownership_manifest_sha256",
                    "passive_ligament_owners",
                    "qualification",
                    "regions",
                    "schema",
                    "semantic_scope",
                    "side",
                    "source",
                    "source_ownership_status",
                    "status",
                    "subject_id",
                    "topology",
                },
                "Human manifest");
    requireKeys(binding,
                {
                    "binding_hash_exclusion",
                    "binding_sha256",
                    "manifest",
                    "manifest_canonicalization",
                    "manifest_file_sha256",
                    "schema",
                },
                "Human binding");
    requireKeys(ownership,
                {
                    "body_composition",
                    "boundary",
                    "compiler",
                    "counts",
                    "inputs",
                    "manifest_sha256",
                    "moment_closures",
                    "qualification",
                    "records",
                    "schema",
                    "source_coverage",
                    "status",
                    "subject",
                },
                "ownership manifest");
    validateManifestShape(manifest);

    require(stringField(manifest, @"manifest_canonicalization",
                        "Human canonicalization") == kCanonicalization &&
                stringField(manifest, @"manifest_hash_exclusion",
                            "Human hash exclusion") ==
                    "top-level manifest_sha256",
            "Human manifest canonicalization contract differs");
    require(stringField(binding, @"schema", "binding schema") ==
                    "HumanPack.loaded-anatomy-knee.binding.v1" &&
                stringField(binding, @"manifest_canonicalization",
                            "binding canonicalization") == kCanonicalization &&
                stringField(binding, @"binding_hash_exclusion",
                            "binding hash exclusion") ==
                    "top-level binding_sha256",
            "Human binding envelope differs");
    require(stringField(ownership, @"schema", "ownership schema") ==
                    "HumanPack.ownership.v1" &&
                stringField(ownership, @"compiler", "ownership compiler") ==
                    "numilab-human.ownership.1",
            "ownership manifest schema or compiler differs");

    const auto manifestIdentity = hashExcludingTopLevelMember(
        manifestBytes, manifestMembers, "manifest_sha256");
    require(manifestIdentity == digestField(manifest, @"manifest_sha256",
                                            "Human manifest identity"),
            "Human manifest identity hash mismatch");
    const auto bindingIdentity = hashExcludingTopLevelMember(
        bindingBytes, bindingMembers, "binding_sha256");
    require(bindingIdentity ==
                digestField(binding, @"binding_sha256", "binding identity"),
            "Human binding identity hash mismatch");
    const auto ownershipIdentity = hashExcludingTopLevelMember(
        ownershipBytes, ownershipMembers, "manifest_sha256");
    require(ownershipIdentity == digestField(ownership, @"manifest_sha256",
                                             "ownership identity"),
            "ownership manifest identity hash mismatch");

    const auto manifestFileIdentity = sha256(manifestBytes);
    require(manifestFileIdentity == digestField(binding,
                                                @"manifest_file_sha256",
                                                "bound Human file identity"),
            "binding names a different Human manifest file");
    const auto &embeddedSpan = memberSpan(bindingMembers, "manifest");
    require(embeddedSpan.valueEnd - embeddedSpan.valueBegin ==
                    manifestBytes.size() - 1u &&
                std::equal(bindingBytes.begin() + embeddedSpan.valueBegin,
                           bindingBytes.begin() + embeddedSpan.valueEnd,
                           manifestBytes.begin()),
            "binding does not embed the exact companion Human manifest");

    NSDictionary *inputs =
        dictionary(field(manifest, @"inputs", "Human inputs"), "Human inputs");
    NSDictionary *ownershipInput = dictionary(
        field(inputs, @"ownership", "ownership input"), "ownership input");
    const auto ownershipFileIdentity = sha256(ownershipBytes);
    require(
        ownershipFileIdentity == digestField(ownershipInput, @"file_sha256",
                                             "ownership input file identity") &&
            ownershipIdentity == digestField(ownershipInput, @"identity_sha256",
                                             "ownership input identity") &&
            ownershipIdentity == digestField(manifest,
                                             @"ownership_manifest_sha256",
                                             "Human ownership identity"),
        "Human manifest names a different ownership companion");
    const auto ownershipRecords = validateOwnershipContract(ownership);
    const std::string sourceStatus = stringField(
        manifest, @"source_ownership_status", "source ownership status");
    require(
        (sourceStatus == "blocked" || sourceStatus == "partial") &&
            sourceStatus ==
                stringField(ownership, @"status", "ownership status"),
        "Human source ownership status is not authenticated by its companion");

    const auto derivedCoverage =
        deriveOwnershipScope(ownershipRecords, manifest);
    NSDictionary *scope =
        dictionary(field(manifest, @"semantic_scope", "Human semantic scope"),
                   "Human semantic scope");
    requireKeys(scope, {"coverage_leaf_sha256s", "ownership_manifest_sha256"},
                "Human semantic scope");
    require(digestField(scope, @"ownership_manifest_sha256",
                        "semantic-scope ownership identity") ==
                ownershipIdentity,
            "semantic scope names a different ownership manifest");
    auto declaredCoverage = digestArray(
        field(scope, @"coverage_leaf_sha256s", "declared semantic coverage"),
        "declared semantic coverage");
    require(std::is_sorted(declaredCoverage.begin(), declaredCoverage.end()) &&
                std::adjacent_find(declaredCoverage.begin(),
                                   declaredCoverage.end()) ==
                    declaredCoverage.end() &&
                declaredCoverage == derivedCoverage,
            "Human semantic coverage is not exactly derived from ownership "
            "records");
    require(ownershipFileIdentity ==
                    parseDigest(std::string(kOwnershipFileSHA256),
                                "pinned ownership file identity") &&
                ownershipIdentity ==
                    parseDigest(std::string(kOwnershipManifestSHA256),
                                "pinned ownership manifest identity"),
            "exact ownership companion identity differs from Human authoring");

    // The embedded Lab export has its own content and file identities; the
    // top-level Human hash alone is not a substitute for either.
    const auto labBytes = documentFromValueSpan(
        manifestBytes, memberSpan(manifestMembers, "lab_authoring_export"));
    CanonicalJSONScanner labScanner(labBytes);
    const auto labMembers = labScanner.validateDocument();
    NSDictionary *labExport = dictionary(
        field(manifest, @"lab_authoring_export", "embedded Lab export"),
        "embedded Lab export");
    const auto labIdentity =
        hashExcludingTopLevelMember(labBytes, labMembers, "manifest_sha256");
    require(
        labIdentity == digestField(labExport, @"manifest_sha256",
                                   "Lab export identity") &&
            labIdentity ==
                digestField(
                    dictionary(field(inputs, @"lab_export", "Lab export input"),
                               "Lab export input"),
                    @"identity_sha256", "Lab export input identity") &&
            sha256(labBytes) ==
                digestField(
                    dictionary(field(inputs, @"lab_export", "Lab export input"),
                               "Lab export input"),
                    @"file_sha256", "Lab export file identity"),
        "embedded Lab export identity differs from its Human input binding");

    NumiHumanLoadedKneeBindingAdmissionV1 candidate{};
    populateAuthoring(manifest, manifestIdentity, derivedCoverage,
                      candidate.authoring);
    std::string validationError;
    require(validateNumiHumanLoadedKneeAuthoringV1(candidate.authoring, nullptr,
                                                   validationError),
            "native Human authoring validation failed: " + validationError);
    candidate.manifestFileSHA256 = manifestFileIdentity;
    candidate.bindingSHA256 = bindingIdentity;
    candidate.bindingFileSHA256 = sha256(bindingBytes);
    candidate.ownershipFileSHA256 = ownershipFileIdentity;
    output = std::move(candidate);
    error.clear();
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  } catch (...) {
    error = "loaded-knee binding admission failed";
    return false;
  }
}

bool loadNumiHumanLoadedKneeSourceComplianceV1(
    const std::filesystem::path &sourceCompliancePath,
    const std::filesystem::path &jointEqualityPath,
    const std::filesystem::path &jointLimitPath,
    const NumiHumanLoadedKneeBindingAdmissionV1 &authenticatedBase,
    NumiHumanLoadedKneeSourceComplianceAdmissionV1 &output,
    std::string &error) {
  try {
    require(sourceCompliancePath.filename() == kSourceComplianceName,
            "source-compliance companion must use its standard filename");
    require(jointEqualityPath.filename() == kSourceEqualityProgram.filename &&
                jointLimitPath.filename() == kSourceLimitProgram.filename,
            "source-compliance programs must use their standard filenames");

    std::string baseError;
    require(validateNumiHumanLoadedKneeAuthoringV1(
                authenticatedBase.authoring, nullptr, baseError),
            "authenticated base admission is invalid: " + baseError);
    require(authenticatedBase.authoring.schema ==
                    "HumanPack.loaded-anatomy-knee.v1" &&
                authenticatedBase.authoring.sourceRigidPayloadSHA256 ==
                    parseDigest(std::string(kSourceRigidPayloadSHA256),
                                "source rigid payload identity") &&
                authenticatedBase.authoring.fullStateOwnerID ==
                    "numi-lab:human-matter/accepted-step-transaction",
            "authenticated base admission differs from the source-compliance "
            "contract");

    const auto companionBytes = readRegularFile(
        sourceCompliancePath, 1024u * 1024u, "source-compliance companion");
    const auto equalityBytes = readRegularFile(
        jointEqualityPath, kSourceEqualityProgram.byteCount,
        "NHEQ2 source program");
    const auto limitBytes = readRegularFile(
        jointLimitPath, kSourceLimitProgram.byteCount, "NHLIM1 source program");
    const auto equalityFileIdentity =
        validateSourceComplianceProgramBytes(equalityBytes,
                                             kSourceEqualityProgram);
    const auto limitFileIdentity =
        validateSourceComplianceProgramBytes(limitBytes, kSourceLimitProgram);

    CanonicalJSONScanner companionScanner(companionBytes);
    const auto companionMembers = companionScanner.validateDocument();
    NSDictionary *companion =
        parseJSONObject(companionBytes, "source-compliance companion");
    requireKeys(companion,
                {"base_manifest", "binding_hash_exclusion", "binding_sha256",
                 "boundary", "compiler", "cross_program",
                 "manifest_canonicalization", "ownership", "prepared_state",
                 "programs", "qualification", "schema", "source_model",
                 "status"},
                "source-compliance companion");
    require(stringField(companion, @"schema", "source-compliance schema") ==
                    kSourceComplianceSchema &&
                stringField(companion, @"compiler",
                            "source-compliance compiler") ==
                    kSourceComplianceCompiler &&
                stringField(companion, @"manifest_canonicalization",
                            "source-compliance canonicalization") ==
                    kCanonicalization &&
                stringField(companion, @"binding_hash_exclusion",
                            "source-compliance hash exclusion") ==
                    "top-level binding_sha256" &&
                stringField(companion, @"status", "source-compliance status") ==
                    "candidate" &&
                stringField(companion, @"boundary",
                            "source-compliance boundary") ==
                    kSourceComplianceBoundary,
            "source-compliance envelope or boundary differs");

    const auto bindingIdentity = hashExcludingTopLevelMember(
        companionBytes, companionMembers, "binding_sha256");
    require(bindingIdentity ==
                digestField(companion, @"binding_sha256",
                            "source-compliance binding identity"),
            "source-compliance binding identity hash mismatch");

    NSDictionary *base = dictionary(
        field(companion, @"base_manifest", "base manifest binding"),
        "base manifest binding");
    requireKeys(base, {"file_sha256", "manifest_sha256", "schema"},
                "base manifest binding");
    require(stringField(base, @"schema", "base manifest schema") ==
                    authenticatedBase.authoring.schema &&
                digestField(base, @"manifest_sha256",
                            "base manifest identity") ==
                    authenticatedBase.authoring.manifestSHA256 &&
                digestField(base, @"file_sha256", "base manifest file identity") ==
                    authenticatedBase.manifestFileSHA256,
            "source-compliance companion names a different base manifest");

    NSDictionary *source = dictionary(
        field(companion, @"source_model", "source model binding"),
        "source model binding");
    requireKeys(source,
                {"nq", "nv", "source_archive_sha256",
                 "source_rigid_payload_sha256"},
                "source model binding");
    require(unsigned32Field(source, @"nq", "source nq") == 129u &&
                unsigned32Field(source, @"nv", "source nv") == 128u &&
                digestField(source, @"source_archive_sha256",
                            "source archive identity") ==
                    parseDigest(std::string(kSourceArchiveSHA256),
                                "source archive identity") &&
                digestField(source, @"source_rigid_payload_sha256",
                            "source rigid payload identity") ==
                    authenticatedBase.authoring.sourceRigidPayloadSHA256,
            "source model binding differs");

    NSDictionary *programs = dictionary(
        field(companion, @"programs", "source programs"), "source programs");
    requireKeys(programs, {"joint_equalities", "joint_limits"},
                "source programs");
    validateSourceComplianceProgramDescriptor(
        dictionary(field(programs, @"joint_equalities", "joint equalities"),
                   "joint equalities"),
        kSourceEqualityProgram, equalityFileIdentity);
    validateSourceComplianceProgramDescriptor(
        dictionary(field(programs, @"joint_limits", "joint limits"),
                   "joint limits"),
        kSourceLimitProgram, limitFileIdentity);

    NSDictionary *cross = dictionary(
        field(companion, @"cross_program", "cross-program binding"),
        "cross-program binding");
    requireKeys(cross,
                {"same_flags", "same_nq_nv", "same_policy_id", "same_refsafe",
                 "same_source_archive_sha256"},
                "cross-program binding");
    require(booleanField(cross, @"same_nq_nv", "cross-program nq/nv") &&
                booleanField(cross, @"same_source_archive_sha256",
                             "cross-program source archive") &&
                booleanField(cross, @"same_policy_id", "cross-program policy") &&
                booleanField(cross, @"same_flags", "cross-program flags") &&
                booleanField(cross, @"same_refsafe", "cross-program refsafe"),
            "cross-program consistency record differs");

    validateSourceComplianceOwnership(
        dictionary(field(companion, @"ownership",
                         "source-compliance ownership"),
                   "source-compliance ownership"),
        authenticatedBase);

    NSDictionary *prepared = dictionary(
        field(companion, @"prepared_state", "prepared-state boundary"),
        "prepared-state boundary");
    requireKeys(prepared,
                {"identity_sha256", "qualification_receipt_sha256", "qualified",
                 "status"},
                "prepared-state boundary");
    require(stringField(prepared, @"status", "prepared-state status") ==
                    "absent" &&
                [field(prepared, @"identity_sha256",
                       "prepared-state identity") isEqual:[NSNull null]] &&
                [field(prepared, @"qualification_receipt_sha256",
                       "prepared-state qualification receipt")
                    isEqual:[NSNull null]] &&
                !booleanField(prepared, @"qualified",
                              "prepared-state qualification"),
            "prepared-state identity must remain explicitly absent and "
            "unqualified");

    validateSourceComplianceQualification(
        dictionary(field(companion, @"qualification",
                         "source-compliance qualification"),
                   "source-compliance qualification"));

    NumiHumanLoadedKneeSourceComplianceAdmissionV1 candidate{};
    candidate.bindingSHA256 = bindingIdentity;
    candidate.fileSHA256 = sha256(companionBytes);
    candidate.baseManifestSHA256 = authenticatedBase.authoring.manifestSHA256;
    candidate.baseManifestFileSHA256 = authenticatedBase.manifestFileSHA256;
    candidate.jointEqualityFileSHA256 = equalityFileIdentity;
    candidate.jointLimitFileSHA256 = limitFileIdentity;
    output = candidate;
    error.clear();
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  } catch (...) {
    error = "loaded-knee source-compliance admission failed";
    return false;
  }
}

} // namespace metalrobo
