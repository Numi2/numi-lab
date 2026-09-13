#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void requireCondition(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

id requiredValue(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    requireCondition(value != nil, std::string("missing mass-moment key: ") + [key UTF8String]);
    return value;
}

std::string stringValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSString class]], std::string(label) + " is not a string");
    return std::string([(NSString*)value UTF8String]);
}

bool booleanValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSNumber class]], std::string(label) + " is not a boolean");
    requireCondition(CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID(),
                     std::string(label) + " is not a boolean");
    return [(NSNumber*)value boolValue];
}

double finiteNumber(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSNumber class]], std::string(label) + " is not a number");
    requireCondition(CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
                     std::string(label) + " is a boolean");
    const double result = [(NSNumber*)value doubleValue];
    requireCondition(std::isfinite(result), std::string(label) + " is not finite");
    return result;
}

NSArray* arrayValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSArray class]], std::string(label) + " is not an array");
    return (NSArray*)value;
}

std::vector<double> finiteVector(id value, const char* label, const NSUInteger expected) {
    NSArray* values = arrayValue(value, label);
    requireCondition(values.count == expected, std::string(label) + " has wrong length");
    std::vector<double> result;
    result.reserve(expected);
    for (NSUInteger i = 0; i < values.count; ++i) {
        result.push_back(finiteNumber(values[i], label));
    }
    return result;
}

std::array<std::array<double, 3>, 3> finiteMatrix3(id value, const char* label) {
    NSArray* rows = arrayValue(value, label);
    requireCondition(rows.count == 3u, std::string(label) + " has wrong row count");
    std::array<std::array<double, 3>, 3> result{};
    for (NSUInteger i = 0; i < 3u; ++i) {
        const auto row = finiteVector(rows[i], label, 3u);
        for (NSUInteger j = 0; j < 3u; ++j) result[i][j] = row[j];
    }
    return result;
}

void requireNull(id value, const char* label) {
    requireCondition(value == [NSNull null], std::string(label) + " was promoted");
}

void requireSha(const std::string& value, const char* label) {
    requireCondition(value.size() == 64u, std::string(label) + " is not SHA-256");
    for (const char ch : value) {
        requireCondition((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'),
                         std::string(label) + " is not lowercase hex");
    }
}

NSData* readData(const char* path, const char* label) {
    NSError* error = nil;
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]
                                          options:0 error:&error];
    requireCondition(data != nil && data.length > 0 && data.length <= 64u * 1024u * 1024u,
                     std::string("cannot read ") + label);
    return data;
}

std::string sha256(NSData* data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), digest);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(CC_SHA256_DIGEST_LENGTH * 2u);
    for (const unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0fu]);
    }
    return result;
}

NSDictionary* readJSON(NSData* data, const char* label) {
    NSError* error = nil;
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    requireCondition(json != nil && [json isKindOfClass:[NSDictionary class]],
                     std::string(label) + " is not an object");
    return (NSDictionary*)json;
}

bool closeEnough(const double actual, const double expected) {
    const double scale = std::max({std::abs(actual), std::abs(expected), 1.0e-30});
    return std::abs(actual - expected) <= 1.0e-12 * scale + 1.0e-30;
}

void compareNumber(id value, const double expected, const char* label) {
    const double actual = finiteNumber(value, label);
    requireCondition(closeEnough(actual, expected), std::string(label) + " disagrees");
}

void compareVector(id value, const std::array<double, 3>& expected, const char* label) {
    const auto actual = finiteVector(value, label, 3u);
    for (std::size_t i = 0; i < 3u; ++i) compareNumber(@(actual[i]), expected[i], label);
}

void compareMatrix(id value, const std::array<std::array<double, 3>, 3>& expected, const char* label) {
    const auto actual = finiteMatrix3(value, label);
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            requireCondition(closeEnough(actual[i][j], expected[i][j]),
                             std::string(label) + " disagrees");
        }
    }
}

std::array<double, 3> vectorFrom(id value, const char* label) {
    const auto values = finiteVector(value, label, 3u);
    return {values[0], values[1], values[2]};
}

std::array<double, 3> add(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

std::array<std::array<double, 3>, 3> add(
    const std::array<std::array<double, 3>, 3>& a,
    const std::array<std::array<double, 3>, 3>& b) {
    std::array<std::array<double, 3>, 3> result{};
    for (std::size_t i = 0; i < 3u; ++i)
        for (std::size_t j = 0; j < 3u; ++j) result[i][j] = a[i][j] + b[i][j];
    return result;
}

std::array<std::array<double, 3>, 3> scale(
    const std::array<std::array<double, 3>, 3>& a, const double value) {
    std::array<std::array<double, 3>, 3> result{};
    for (std::size_t i = 0; i < 3u; ++i)
        for (std::size_t j = 0; j < 3u; ++j) result[i][j] = value * a[i][j];
    return result;
}

std::array<std::array<double, 3>, 3> outer(const std::array<double, 3>& value) {
    std::array<std::array<double, 3>, 3> result{};
    for (std::size_t i = 0; i < 3u; ++i)
        for (std::size_t j = 0; j < 3u; ++j) result[i][j] = value[i] * value[j];
    return result;
}

void requireQualification(NSDictionary* qualification) {
    for (const char* key : {"zeroth_first_second_mass_moments", "momentum_at_authored_initial_velocity",
                            "single_owner_per_registered_surface", "atomic_checkpoint_restore",
                            "hydraulic_volume_storage_unchanged"}) {
        requireCondition(booleanValue(requiredValue(qualification, [NSString stringWithUTF8String:key]), key),
                         std::string("qualification not admitted: ") + key);
    }
    for (const char* key : {"source_surface_is_lumen", "vessel_tube_or_lumen_mechanics",
                            "pressure_gradient_momentum_transfer", "two_way_blood_tissue_transfer",
                            "anatomical_blood_mass_owner", "material_density_calibrated",
                            "subject_calibration", "standing_walking"}) {
        requireCondition(!booleanValue(requiredValue(qualification, [NSString stringWithUTF8String:key]), key),
                         std::string("qualification was promoted: ") + key);
    }
}
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            requireCondition(argc == 4,
                             "usage: numi-matter-vessel-mass-moment-check candidate.json registration.json body-links.json");
            NSData* candidateData = readData(argv[1], "candidate receipt");
            NSData* registrationData = readData(argv[2], "registration receipt");
            NSData* bodyLinksData = readData(argv[3], "body-link receipt");
            NSDictionary* candidate = readJSON(candidateData, "candidate receipt");
            NSDictionary* registration = readJSON(registrationData, "registration receipt");
            NSDictionary* bodyLinks = readJSON(bodyLinksData, "body-link receipt");

            requireCondition(stringValue(requiredValue(candidate, @"schema"), "schema") ==
                                 "HumanPack.vessel-mass-moment-owner-candidate.v1",
                             "unexpected mass-moment schema");
            requireCondition(stringValue(requiredValue(candidate, @"compiler"), "compiler") ==
                                 "numilab-human.vessel-mass-moments.1",
                             "unexpected mass-moment compiler");
            requireCondition(stringValue(requiredValue(candidate, @"status"), "status") == "partial",
                             "mass-moment candidate was promoted");

            NSDictionary* source = (NSDictionary*)requiredValue(candidate, @"source");
            requireCondition([source isKindOfClass:[NSDictionary class]], "candidate source is not an object");
            const std::string candidateRegistrationSha = stringValue(
                requiredValue(source, @"registration_sha256"), "registration_sha256");
            const std::string candidateBodySha = stringValue(
                requiredValue(source, @"body_links_sha256"), "body_links_sha256");
            requireSha(candidateRegistrationSha, "registration_sha256");
            requireSha(candidateBodySha, "body_links_sha256");
            requireCondition(candidateRegistrationSha == sha256(registrationData),
                             "candidate registration hash disagrees with input");
            requireCondition(candidateBodySha == sha256(bodyLinksData),
                             "candidate body-link hash disagrees with input");
            requireCondition(finiteNumber(requiredValue(source, @"vessel_count"), "vessel_count") == 6.0,
                             "candidate vessel count changed");

            NSDictionary* density = (NSDictionary*)requiredValue(candidate, @"density");
            requireCondition([density isKindOfClass:[NSDictionary class]], "density is not an object");
            const double rho = finiteNumber(requiredValue(density, @"value_kg_per_m3"), "density");
            requireCondition(rho > 0.0, "density is not positive");
            requireCondition(stringValue(requiredValue(density, @"status"), "density status") ==
                                 "candidate_not_subject_calibrated",
                             "density was promoted");

            requireCondition(stringValue(requiredValue(registration, @"schema"), "registration schema") ==
                                 "HumanPack.organ-vessel-registration.v1",
                             "unexpected registration schema");
            requireCondition(stringValue(requiredValue(bodyLinks, @"schema"), "body-link schema") ==
                                 "HumanPack.organ-vessel-body-link-registration.v1",
                             "unexpected body-link schema");
            NSArray* registrationRows = arrayValue(requiredValue(registration, @"bindings"), "registration bindings");
            NSArray* bodyLinkRows = arrayValue(requiredValue(bodyLinks, @"bindings"), "body-link bindings");
            requireCondition(registrationRows.count == 6u && bodyLinkRows.count == 6u,
                             "source receipts must contain six vessels");
            std::map<std::string, NSDictionary*> registrationByMember;
            std::map<std::string, NSDictionary*> bodyByMember;
            for (id value in registrationRows) {
                NSDictionary* row = (NSDictionary*)value;
                requireCondition([row isKindOfClass:[NSDictionary class]], "registration row is not an object");
                const std::string member = stringValue(requiredValue(row, @"member_id"), "registration member");
                requireCondition(registrationByMember.emplace(member, row).second, "duplicate registration member");
            }
            for (id value in bodyLinkRows) {
                NSDictionary* row = (NSDictionary*)value;
                requireCondition([row isKindOfClass:[NSDictionary class]], "body-link row is not an object");
                const std::string member = stringValue(requiredValue(row, @"member_id"), "body-link member");
                requireCondition(bodyByMember.emplace(member, row).second, "duplicate body-link member");
            }

            NSArray* owners = arrayValue(requiredValue(candidate, @"owners"), "owners");
            requireCondition(owners.count == 6u, "candidate must contain six owners");
            std::array<double, 3> velocity{0.0, 0.0, 0.0};
            bool velocitySet = false;
            double sourceVolumeSum = 0.0;
            double worldVolumeSum = 0.0;
            double massSum = 0.0;
            std::array<double, 3> firstSum{0.0, 0.0, 0.0};
            std::array<double, 3> momentumSum{0.0, 0.0, 0.0};
            std::array<std::array<double, 3>, 3> rawSum{};
            std::size_t ownerCount = 0;
            for (id value : owners) {
                NSDictionary* row = (NSDictionary*)value;
                requireCondition([row isKindOfClass:[NSDictionary class]], "candidate owner is not an object");
                const std::string member = stringValue(requiredValue(row, @"member_id"), "owner member");
                requireCondition(registrationByMember.contains(member), "owner has no registration member");
                requireCondition(bodyByMember.contains(member), "owner has no body-link member");
                NSDictionary* registrationRow = registrationByMember.at(member);
                NSDictionary* bodyRow = bodyByMember.at(member);
                for (NSString* key in @[@"source_name", @"region_id", @"myosim_body"]) {
                    requireCondition(stringValue(requiredValue(row, key), "owner identity") ==
                                         stringValue(requiredValue(registrationRow, key), "registration identity"),
                                     "owner identity disagrees with registration");
                    requireCondition(stringValue(requiredValue(row, key), "owner identity") ==
                                         stringValue(requiredValue(bodyRow, key), "body-link identity"),
                                     "owner identity disagrees with body link");
                }
                requireSha(stringValue(requiredValue(row, @"source_member_sha256"), "owner source hash"),
                           "owner source hash");
                requireCondition(stringValue(requiredValue(row, @"source_member_sha256"), "owner source hash") ==
                                     stringValue(requiredValue(registrationRow, @"source_member_sha256"), "registration source hash"),
                                 "owner source hash disagrees");
                requireCondition(booleanValue(requiredValue(bodyRow, @"body_link_registration"), "body-link registration"),
                                 "body link is not registered");
                requireNull(requiredValue(bodyRow, @"mechanical_mass_owner"), "body-link mass owner");
                requireNull(requiredValue(registrationRow, @"mechanical_mass_owner"), "registration mass owner");
                requireCondition(!booleanValue(requiredValue(row, @"lumen_or_tube_admitted"), "lumen admission") &&
                                 !booleanValue(requiredValue(row, @"pressure_gradient_momentum_transfer"), "pressure transfer") &&
                                 !booleanValue(requiredValue(row, @"two_way_tissue_exchange"), "tissue exchange") &&
                                 !booleanValue(requiredValue(row, @"subject_calibration"), "subject calibration"),
                                 "owner promoted an unresolved gate");

                const double sourceVolume = finiteNumber(requiredValue(row, @"source_surface_integral_volume_m3"), "source volume");
                const double worldVolume = finiteNumber(requiredValue(row, @"registered_world_surface_integral_volume_m3"), "world volume");
                compareNumber(requiredValue(row, @"source_surface_integral_volume_m3"),
                              finiteNumber(requiredValue(registrationRow, @"source_surface_integral_volume_m3"), "registration source volume"),
                              "source volume");
                compareNumber(requiredValue(row, @"registered_world_surface_integral_volume_m3"),
                              finiteNumber(requiredValue(registrationRow, @"registered_world_surface_integral_volume_m3"), "registration world volume"),
                              "world volume");
                requireCondition(sourceVolume > 0.0 && worldVolume > 0.0, "surface volume is not positive");
                compareNumber(requiredValue(row, @"density_kg_per_m3"), rho, "row density");
                const double mass = finiteNumber(requiredValue(row, @"mass_kg"), "mass");
                compareNumber(requiredValue(row, @"mass_kg"), rho * worldVolume, "mass");
                const auto centroid = vectorFrom(requiredValue(row, @"world_centroid_m"), "centroid");
                const auto registrationCentroid = vectorFrom(requiredValue(registrationRow, @"registered_world_centroid_m"), "registration centroid");
                for (std::size_t i = 0; i < 3u; ++i) requireCondition(closeEnough(centroid[i], registrationCentroid[i]), "centroid disagrees");
                const auto registrationCentral = finiteMatrix3(requiredValue(registrationRow, @"registered_world_central_second_volume_moment_m5"), "registration central moment");
                const auto raw = finiteMatrix3(requiredValue(row, @"raw_second_mass_moment_kg_m2"), "raw mass moment");
                const auto expectedCentral = scale(registrationCentral, rho);
                const auto expectedRaw = add(expectedCentral, scale(outer(centroid), mass));
                compareMatrix(requiredValue(row, @"central_second_mass_moment_kg_m2"), expectedCentral, "central mass moment");
                compareMatrix(requiredValue(row, @"raw_second_mass_moment_kg_m2"), expectedRaw, "raw mass moment");
                const auto first = vectorFrom(requiredValue(row, @"first_mass_moment_kg_m"), "first mass moment");
                const auto expectedFirst = std::array<double, 3>{mass * centroid[0], mass * centroid[1], mass * centroid[2]};
                for (std::size_t i = 0; i < 3u; ++i) requireCondition(closeEnough(first[i], expectedFirst[i]), "first mass moment disagrees");
                const auto rowVelocity = vectorFrom(requiredValue(row, @"initial_velocity_mps"), "initial velocity");
                if (!velocitySet) { velocity = rowVelocity; velocitySet = true; }
                for (std::size_t i = 0; i < 3u; ++i) requireCondition(closeEnough(rowVelocity[i], velocity[i]), "initial velocity disagrees");
                const auto momentum = vectorFrom(requiredValue(row, @"linear_momentum_kg_m_per_s"), "momentum");
                for (std::size_t i = 0; i < 3u; ++i) requireCondition(closeEnough(momentum[i], mass * velocity[i]), "momentum disagrees");
                requireCondition(finiteNumber(requiredValue(row, @"mass_owner_count"), "mass owner count") == 1.0,
                                 "owner count is not one");
                sourceVolumeSum += sourceVolume;
                worldVolumeSum += worldVolume;
                massSum += mass;
                firstSum = add(firstSum, first);
                momentumSum = add(momentumSum, momentum);
                rawSum = add(rawSum, raw);
                ++ownerCount;
            }
            requireCondition(ownerCount == 6u, "candidate owner count changed");
            NSDictionary* totals = (NSDictionary*)requiredValue(candidate, @"totals");
            requireCondition([totals isKindOfClass:[NSDictionary class]], "totals is not an object");
            compareNumber(requiredValue(totals, @"source_surface_integral_volume_m3"), sourceVolumeSum, "total source volume");
            compareNumber(requiredValue(totals, @"registered_world_surface_integral_volume_m3"), worldVolumeSum, "total world volume");
            compareNumber(requiredValue(totals, @"mass_kg"), massSum, "total mass");
            compareVector(requiredValue(totals, @"first_mass_moment_kg_m"), firstSum, "total first mass moment");
            compareMatrix(requiredValue(totals, @"raw_second_mass_moment_kg_m2"), rawSum, "total raw mass moment");
            compareVector(requiredValue(totals, @"linear_momentum_kg_m_per_s"), momentumSum, "total momentum");
            requireCondition(finiteNumber(requiredValue(totals, @"owner_count"), "total owner count") == 6.0 &&
                             finiteNumber(requiredValue(totals, @"unique_member_count"), "unique member count") == 6.0,
                             "total owner identity changed");
            requireQualification((NSDictionary*)requiredValue(candidate, @"qualification"));
            std::printf("numi_matter_vessel_mass_moment=pass vessels=6 source_identity=pass body_link_identity=pass moments=pass ownership=single surface_proxy=pass mechanics=unqualified calibration=unqualified\n");
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "numi_matter_vessel_mass_moment=fail error=\"%s\"\n", error.what());
            return 1;
        }
    }
}
