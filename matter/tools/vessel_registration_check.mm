#include <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void requireCondition(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

id requiredValue(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    requireCondition(value != nil, std::string("missing vessel-registration key: ") + [key UTF8String]);
    return value;
}

std::string stringValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSString class]], std::string(label) + " is not a string");
    return std::string([(NSString*)value UTF8String]);
}

bool booleanValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSNumber class]], std::string(label) + " is not a boolean");
    requireCondition([(NSNumber*)value objCType][0] == 'c', std::string(label) + " is not a boolean");
    return [(NSNumber*)value boolValue];
}

double finiteNumber(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSNumber class]], std::string(label) + " is not a number");
    const double result = [(NSNumber*)value doubleValue];
    requireCondition(std::isfinite(result), std::string(label) + " is not finite");
    return result;
}

void sha256Text(id value, const char* label) {
    const std::string text = stringValue(value, label);
    requireCondition(text.size() == 64u, std::string(label) + " is not SHA-256");
    for (const char c : text) {
        requireCondition((c >= '0' && c <= '9') ||
                         (c >= 'a' && c <= 'f') ||
                         (c >= 'A' && c <= 'F'),
                         std::string(label) + " has non-hex data");
    }
}

void finiteVector(id value, const char* label, const NSUInteger expected) {
    requireCondition([value isKindOfClass:[NSArray class]], std::string(label) + " is not an array");
    NSArray* values = (NSArray*)value;
    requireCondition(values.count == expected, std::string(label) + " has wrong length");
    for (NSUInteger i = 0; i < values.count; ++i) {
        (void)finiteNumber(values[i], label);
    }
}

void finiteMatrix3(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSArray class]], std::string(label) + " is not an array");
    NSArray* rows = (NSArray*)value;
    requireCondition(rows.count == 3u, std::string(label) + " has wrong row count");
    for (NSUInteger i = 0; i < rows.count; ++i) finiteVector(rows[i], label, 3u);
}

void finiteMatrix4(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSArray class]], std::string(label) + " is not an array");
    NSArray* rows = (NSArray*)value;
    requireCondition(rows.count == 4u, std::string(label) + " has wrong row count");
    for (NSUInteger i = 0; i < rows.count; ++i) finiteVector(rows[i], label, 4u);
}

void requireNull(id value, const char* label) {
    requireCondition(value == [NSNull null], std::string(label) + " was mechanically promoted");
}

void checkBodyLinkReceipt(NSDictionary* root) {
    requireCondition(stringValue(requiredValue(root, @"schema"), "schema") ==
                         "HumanPack.organ-vessel-body-link-registration.v1",
                     "unexpected vessel body-link schema");
    requireCondition(stringValue(requiredValue(root, @"compiler"), "compiler") ==
                         "numilab-human.organ-vessel-body-links.1",
                     "unexpected vessel body-link compiler");
    sha256Text(requiredValue(root, @"identity_sha256"), "identity_sha256");
    NSDictionary* source = (NSDictionary*)requiredValue(root, @"source");
    requireCondition([source isKindOfClass:[NSDictionary class]], "body-link source is not an object");
    sha256Text(requiredValue(source, @"human_manifest_sha256"), "human_manifest_sha256");
    sha256Text(requiredValue(source, @"rigid_payload_sha256"), "rigid_payload_sha256");
    sha256Text(requiredValue(source, @"source_archive_sha256"), "source_archive_sha256");
    requireCondition(finiteNumber(requiredValue(source, @"body_count"), "body_count") == 157.0 &&
                     finiteNumber(requiredValue(source, @"source_body_count"), "source_body_count") == 103.0,
                     "body-link topology changed");
    NSDictionary* qualification = (NSDictionary*)requiredValue(root, @"qualification");
    requireCondition([qualification isKindOfClass:[NSDictionary class]], "body-link qualification is not an object");
    requireCondition(booleanValue(requiredValue(qualification, @"source_to_world_frame_registered"), "world frame") &&
                     booleanValue(requiredValue(qualification, @"body_link_registration"), "body link") &&
                     booleanValue(requiredValue(qualification, @"body_link_transform_registered"), "body transform") &&
                     !booleanValue(requiredValue(qualification, @"tubular_vessel_field"), "tubular field") &&
                     !booleanValue(requiredValue(qualification, @"blood_mass_owner"), "blood mass") &&
                     !booleanValue(requiredValue(qualification, @"pressure_gradient_momentum_transfer"), "pressure momentum") &&
                     !booleanValue(requiredValue(qualification, @"two_way_tissue_exchange"), "tissue exchange") &&
                     !booleanValue(requiredValue(qualification, @"subject_calibration"), "subject calibration"),
                     "body-link mechanics boundary changed");

    NSArray* rows = (NSArray*)requiredValue(root, @"bindings");
    requireCondition([rows isKindOfClass:[NSArray class]] && rows.count == 6u,
                     "body-link receipt must contain six bindings");
    struct Expected { const char* member; const char* body; double source; double core; };
    const std::array<Expected, 6u> expected{{
        {"FJ1932", "Abdomen", 4.0, 7.0}, {"FJ3411", "torso", 9.0, 20.0},
        {"FJ3413", "torso", 9.0, 20.0}, {"FJ3427", "torso", 9.0, 20.0},
        {"FJ3441", "Abdomen", 4.0, 7.0}, {"FJ3645", "torso", 9.0, 20.0}}};
    for (NSUInteger i = 0; i < rows.count; ++i) {
        requireCondition([rows[i] isKindOfClass:[NSDictionary class]], "body-link binding is not an object");
        NSDictionary* row = (NSDictionary*)rows[i];
        const Expected& item = expected[i];
        requireCondition(stringValue(requiredValue(row, @"member_id"), "member_id") == item.member &&
                         stringValue(requiredValue(row, @"myosim_body"), "myosim_body") == item.body,
                         "body-link vessel/body identity changed");
        requireCondition(finiteNumber(requiredValue(row, @"source_body_id"), "source_body_id") == item.source &&
                         finiteNumber(requiredValue(row, @"core_body_index"), "core_body_index") == item.core,
                         "body-link source/core index changed");
        requireCondition(!stringValue(requiredValue(row, @"source_name"), "source_name").empty() &&
                         !stringValue(requiredValue(row, @"region_id"), "region_id").empty(),
                         "body-link source identity is empty");
        finiteVector(requiredValue(row, @"default_com_position_world_m"), "body-link COM", 3u);
        finiteVector(requiredValue(row, @"default_inertial_quaternion_world_xyzw"), "body-link quaternion", 4u);
        requireCondition(booleanValue(requiredValue(row, @"body_link_registration"), "binding body link") &&
                         !booleanValue(requiredValue(row, @"tubular_field_registered"), "binding tubular field") &&
                         !booleanValue(requiredValue(row, @"centreline_registered"), "binding centreline") &&
                         !booleanValue(requiredValue(row, @"pressure_gradient_momentum_transfer"), "binding pressure momentum") &&
                         !booleanValue(requiredValue(row, @"subject_calibration"), "binding subject calibration"),
                         "body-link binding mechanics boundary changed");
        requireNull(requiredValue(row, @"cross_section_area_m2"), "cross_section_area_m2");
        requireNull(requiredValue(row, @"material_density_kg_per_m3"), "material_density_kg_per_m3");
        requireNull(requiredValue(row, @"mechanical_mass_owner"), "mechanical_mass_owner");
    }
}
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            requireCondition(argc == 2, "usage: numi-matter-vessel-registration-check registration.json");
            NSString* path = [NSString stringWithUTF8String:argv[1]];
            NSError* readError = nil;
            NSData* data = [NSData dataWithContentsOfFile:path options:0 error:&readError];
            requireCondition(data != nil, "cannot read vessel-registration receipt");
            NSError* jsonError = nil;
            id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
            requireCondition(json != nil && [json isKindOfClass:[NSDictionary class]],
                             jsonError == nil ? "vessel-registration is not an object"
                                               : std::string([[jsonError localizedDescription] UTF8String]));
            NSDictionary* root = (NSDictionary*)json;
            if (stringValue(requiredValue(root, @"schema"), "schema") ==
                    "HumanPack.organ-vessel-body-link-registration.v1") {
                checkBodyLinkReceipt(root);
                std::printf("numi_matter_vessel_body_link=pass vessels=6 source_body_frame=pass mechanics=unqualified calibration=unqualified\n");
                return 0;
            }
            requireCondition(stringValue(requiredValue(root, @"schema"), "schema") ==
                                 "HumanPack.organ-vessel-registration.v1",
                             "unexpected vessel-registration schema");
            requireCondition(stringValue(requiredValue(root, @"compiler"), "compiler") ==
                                 "numilab-human.organ-vessel-registration.1",
                             "unexpected vessel-registration compiler");
            sha256Text(requiredValue(root, @"identity_sha256"), "identity_sha256");
            NSDictionary* source = (NSDictionary*)requiredValue(root, @"source");
            requireCondition([source isKindOfClass:[NSDictionary class]], "source is not an object");
            sha256Text(requiredValue(source, @"registration_sha256"), "registration_sha256");
            sha256Text(requiredValue(source, @"source_archive_sha256"), "source_archive_sha256");
            finiteMatrix4(requiredValue(source, @"global_source_mm_to_myosim_world_m"), "world transform");

            NSDictionary* qualification = (NSDictionary*)requiredValue(root, @"qualification");
            requireCondition([qualification isKindOfClass:[NSDictionary class]], "qualification is not an object");
            requireCondition(booleanValue(requiredValue(qualification, @"source_membership_and_hashes"), "source membership") &&
                             booleanValue(requiredValue(qualification, @"source_moments_recomputed"), "source moments") &&
                             booleanValue(requiredValue(qualification, @"source_to_world_frame_registered"), "world frame"),
                             "source registration qualification is incomplete");
            requireCondition(!booleanValue(requiredValue(qualification, @"body_link_registration"), "body link") &&
                             !booleanValue(requiredValue(qualification, @"tubular_vessel_field"), "tubular field") &&
                             !booleanValue(requiredValue(qualification, @"blood_mass_owner"), "blood mass") &&
                             !booleanValue(requiredValue(qualification, @"pressure_gradient_momentum_transfer"), "pressure momentum") &&
                             !booleanValue(requiredValue(qualification, @"two_way_tissue_exchange"), "tissue exchange") &&
                             !booleanValue(requiredValue(qualification, @"subject_calibration"), "subject calibration"),
                             "unqualified vessel mechanics were promoted");

            NSArray* rows = (NSArray*)requiredValue(root, @"bindings");
            requireCondition([rows isKindOfClass:[NSArray class]] && rows.count == 6u,
                             "vessel-registration must contain six bindings");
            const std::array<const char*, 6u> members{
                "FJ1932", "FJ3411", "FJ3413", "FJ3427", "FJ3441", "FJ3645"};
            const std::array<const char*, 6u> regions{
                "abdominal_aorta", "aortic_arch", "ascending_aorta", "descending_aorta",
                "inferior_vena_cava", "superior_vena_cava"};
            for (NSUInteger i = 0; i < rows.count; ++i) {
                requireCondition([rows[i] isKindOfClass:[NSDictionary class]], "binding is not an object");
                NSDictionary* row = (NSDictionary*)rows[i];
                requireCondition(stringValue(requiredValue(row, @"member_id"), "member_id") == members[i],
                                 "vessel source member order or identity changed");
                requireCondition(stringValue(requiredValue(row, @"region_id"), "region_id") == regions[i],
                                 "vessel region identity changed");
                requireCondition(!stringValue(requiredValue(row, @"source_name"), "source_name").empty(),
                                 "vessel source name is empty");
                requireCondition(!stringValue(requiredValue(row, @"myosim_body"), "myosim_body").empty(),
                                 "vessel MyoSim body is empty");
                requireCondition(stringValue(requiredValue(row, @"hierarchy"), "hierarchy") == "part_of",
                                 "vessel hierarchy changed");
                sha256Text(requiredValue(row, @"source_member_sha256"), "source_member_sha256");
                requireCondition(finiteNumber(requiredValue(row, @"source_surface_integral_volume_m3"),
                                              "source surface volume") > 0.0 &&
                                 finiteNumber(requiredValue(row, @"registered_world_surface_integral_volume_m3"),
                                              "world surface volume") > 0.0,
                                 "vessel source volume is not positive");
                finiteVector(requiredValue(row, @"source_centroid_m"), "source centroid", 3u);
                finiteVector(requiredValue(row, @"registered_world_centroid_m"), "world centroid", 3u);
                finiteMatrix3(requiredValue(row, @"registered_world_central_second_volume_moment_m5"),
                              "world central moment");
                requireCondition(booleanValue(requiredValue(row, @"world_frame_registration"), "world registration") &&
                                 !booleanValue(requiredValue(row, @"body_link_registration"), "body link") &&
                                 !booleanValue(requiredValue(row, @"tubular_field_registered"), "tubular field") &&
                                 !booleanValue(requiredValue(row, @"centreline_registered"), "centreline") &&
                                 !booleanValue(requiredValue(row, @"pressure_gradient_momentum_transfer"), "pressure momentum") &&
                                 !booleanValue(requiredValue(row, @"subject_calibration"), "subject calibration"),
                                 "vessel mechanics boundary changed");
                requireNull(requiredValue(row, @"cross_section_area_m2"), "cross_section_area_m2");
                requireNull(requiredValue(row, @"material_density_kg_per_m3"), "material_density_kg_per_m3");
                requireNull(requiredValue(row, @"mechanical_mass_owner"), "mechanical_mass_owner");
            }
            std::printf("numi_matter_vessel_registration=pass vessels=6 source_membership=pass source_moments=pass source_to_world=pass mechanics=unqualified calibration=unqualified\n");
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "numi_matter_vessel_registration=fail error=\"%s\"\n", error.what());
            return 1;
        }
    }
}
