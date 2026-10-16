#include "metalrobo/R2S2R.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace metalrobo {
namespace {

R2S2RResult invalid(std::string message) {
    return {R2S2RStatus::invalidManifest, std::move(message)};
}

NSString* stringValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    return [value isKindOfClass:[NSString class]]
        ? static_cast<NSString*>(value)
        : nil;
}

NSNumber* numberValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    return [value isKindOfClass:[NSNumber class]]
        ? static_cast<NSNumber*>(value)
        : nil;
}

NSArray* arrayValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    return [value isKindOfClass:[NSArray class]]
        ? static_cast<NSArray*>(value)
        : nil;
}

NSDictionary* dictionaryValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    return [value isKindOfClass:[NSDictionary class]]
        ? static_cast<NSDictionary*>(value)
        : nil;
}

std::string cppString(NSString* value) {
    return value == nil
        ? std::string{}
        : std::string{value.UTF8String};
}

bool parseHex64(NSString* text, std::uint64_t& output) {
    const std::string value = cppString(text);
    if (value.size() != 16u) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed =
        std::strtoull(value.c_str(), &end, 16);
    if (errno != 0 || end != value.c_str() + value.size()) {
        return false;
    }
    output = static_cast<std::uint64_t>(parsed);
    return true;
}

double finiteNumber(
    NSDictionary* object,
    NSString* key,
    const double fallback,
    bool& okay
) {
    NSNumber* number = numberValue(object, key);
    if (number == nil) {
        return fallback;
    }
    const double value = number.doubleValue;
    okay = okay && std::isfinite(value);
    return value;
}

std::optional<MREpisodeTermination> parseTermination(
    const std::string_view value
) {
    if (value == "success") {
        return MR_EPISODE_TERMINATION_SUCCESS;
    }
    if (value == "horizon") {
        return MR_EPISODE_TERMINATION_HORIZON;
    }
    if (value == "policy") {
        return MR_EPISODE_TERMINATION_POLICY;
    }
    if (value == "physics") {
        return MR_EPISODE_TERMINATION_PHYSICS;
    }
    if (value == "safety") {
        return MR_EPISODE_TERMINATION_SAFETY;
    }
    if (value == "external") {
        return MR_EPISODE_TERMINATION_EXTERNAL;
    }
    return std::nullopt;
}

} // namespace

R2S2RResult loadHardwareOutcomeManifestJSON(
    const std::filesystem::path& path,
    const ScenarioSchema& schema,
    const TaskOutcomeSchema& task,
    HardwareOutcomeManifest& output
) {
    @autoreleasepool {
        std::string reason;
        if (!schema.valid(&reason) || !task.valid(&reason)) {
            return {
                R2S2RStatus::invalidArgument,
                std::move(reason),
            };
        }
        NSError* error = nil;
        NSData* data = [
            NSData dataWithContentsOfFile:@(path.string().c_str())
                                  options:0
                                    error:&error
        ];
        if (data == nil) {
            return {
                R2S2RStatus::ioFailure,
                error == nil
                    ? "could not read hardware outcome manifest"
                    : cppString(error.localizedDescription),
            };
        }
        id rootObject = [
            NSJSONSerialization JSONObjectWithData:data
                                           options:0
                                             error:&error
        ];
        if (![rootObject isKindOfClass:[NSDictionary class]]) {
            return invalid(
                error == nil
                    ? "hardware outcome root must be a JSON object"
                    : cppString(error.localizedDescription)
            );
        }
        NSDictionary* root =
            static_cast<NSDictionary*>(rootObject);
        HardwareOutcomeManifest staged;
        NSNumber* version = numberValue(root, @"schema_version");
        if (version != nil) {
            staged.schemaVersion = version.unsignedIntValue;
        }
        if (staged.schemaVersion !=
            kHardwareOutcomeSchemaVersion) {
            return {
                R2S2RStatus::unsupportedVersion,
                "unsupported hardware outcome schema version",
            };
        }
        staged.scenarioSchemaId =
            cppString(stringValue(root, @"scenario_schema"));
        if (staged.scenarioSchemaId != schema.id) {
            return {
                R2S2RStatus::schemaMismatch,
                "hardware outcome references a different scenario schema",
            };
        }
        staged.policyId =
            cppString(stringValue(root, @"policy_id"));
        staged.robotId =
            cppString(stringValue(root, @"robot_id"));
        staged.taskId =
            cppString(stringValue(root, @"task_id"));
        if (staged.policyId.empty() || staged.robotId.empty() ||
            staged.taskId.empty()) {
            return invalid(
                "hardware outcome requires policy_id, robot_id, and task_id"
            );
        }

        EpisodeOutcome& outcome = staged.outcome;
        outcome.id = cppString(stringValue(root, @"id"));
        outcome.runId = cppString(stringValue(root, @"run_id"));
        outcome.source = MR_EPISODE_SOURCE_HARDWARE;
        if (!parseHex64(
                stringValue(root, @"family_fingerprint"),
                outcome.familyFingerprint
            ) ||
            !parseHex64(
                stringValue(root, @"policy_fingerprint"),
                outcome.policyFingerprint
            ) ||
            !parseHex64(
                stringValue(root, @"task_fingerprint"),
                outcome.taskFingerprint
            ) ||
            !parseHex64(
                stringValue(root, @"embodiment_fingerprint"),
                outcome.embodimentFingerprint
            )) {
            return invalid(
                "hardware outcome fingerprints must be 16-digit hex strings"
            );
        }
        NSString* scenarioKey =
            stringValue(root, @"scenario_key");
        if (scenarioKey != nil &&
            !parseHex64(scenarioKey, outcome.scenarioKey)) {
            return invalid("scenario_key must be a 16-digit hex string");
        }
        NSString* episodeCounter =
            stringValue(root, @"episode_counter");
        if (episodeCounter != nil &&
            !parseHex64(episodeCounter, outcome.episodeCounter)) {
            return invalid(
                "episode_counter must be a 16-digit hex string"
            );
        }
        NSString* alignment =
            stringValue(root, @"alignment_fingerprint");
        if (alignment != nil &&
            !parseHex64(alignment, outcome.alignmentFingerprint)) {
            return invalid(
                "alignment_fingerprint must be a 16-digit hex string"
            );
        }
        NSString* feedback =
            stringValue(root, @"feedback_fingerprint");
        if (feedback != nil &&
            !parseHex64(feedback, outcome.feedbackFingerprint)) {
            return invalid(
                "feedback_fingerprint must be a 16-digit hex string"
            );
        }
        if (outcome.taskFingerprint != task.fingerprint) {
            return {
                R2S2RStatus::schemaMismatch,
                "hardware outcome references a different task schema",
            };
        }

        const auto termination = parseTermination(
            cppString(stringValue(root, @"termination"))
        );
        NSNumber* success = numberValue(root, @"success");
        if (!termination.has_value() || success == nil) {
            return invalid(
                "hardware outcome requires termination and success"
            );
        }
        outcome.termination = *termination;
        outcome.success = success.boolValue;
        outcome.physicsStatus =
            numberValue(root, @"physics_status") == nil
            ? 0u
            : numberValue(root, @"physics_status").unsignedIntValue;
        outcome.stepCount =
            numberValue(root, @"step_count") == nil
            ? 0u
            : numberValue(root, @"step_count").unsignedIntValue;
        bool okay = true;
        outcome.episodeReturn =
            finiteNumber(root, @"episode_return", 0.0, okay);
        outcome.taskMargin =
            finiteNumber(root, @"task_margin", 0.0, okay);
        outcome.safetyMargin =
            finiteNumber(root, @"safety_margin", 0.0, okay);
        outcome.durationSeconds =
            finiteNumber(root, @"duration_seconds", 0.0, okay);
        outcome.minimumVisibility =
            finiteNumber(root, @"minimum_visibility", 0.0, okay);
        outcome.integratedContactLoad = finiteNumber(
            root,
            @"integrated_contact_load",
            0.0,
            okay
        );
        outcome.peakContactLoad =
            finiteNumber(root, @"peak_contact_load", 0.0, okay);
        if (!okay) {
            return invalid("hardware outcome metric is non-finite");
        }

        std::unordered_map<std::string, std::size_t> failureIndices;
        for (std::size_t index = 0u;
             index < task.failureTags.size();
             ++index) {
            failureIndices.emplace(task.failureTags[index], index);
        }
        NSArray* failures = arrayValue(root, @"failure_tags");
        for (id value in failures == nil ? @[] : failures) {
            if (![value isKindOfClass:[NSString class]]) {
                return invalid("failure_tags must contain strings");
            }
            const std::string tag =
                cppString(static_cast<NSString*>(value));
            const auto found = failureIndices.find(tag);
            if (found == failureIndices.end()) {
                return {
                    R2S2RStatus::schemaMismatch,
                    "hardware outcome contains an unknown failure tag: " +
                        tag,
                };
            }
            outcome.failureMask |= 1ull << found->second;
        }

        NSDictionary* values =
            dictionaryValue(root, @"scenario_values");
        if (values == nil) {
            return invalid("scenario_values must be an object");
        }
        outcome.scenarioValues.assign(
            schema.features.size(),
            0.0f
        );
        outcome.scenarioValueMask.assign(
            schema.features.size(),
            0u
        );
        staged.scenarioMissingMask.assign(
            schema.features.size(),
            0u
        );
        NSDictionary* missingValues =
            dictionaryValue(root, @"missing_value_mask");
        staged.scenarioFeatureIds.reserve(schema.features.size());
        for (std::size_t index = 0u;
             index < schema.features.size();
             ++index) {
            const std::string& featureId = schema.features[index].id;
            staged.scenarioFeatureIds.push_back(featureId);
            id value = values[@(featureId.c_str())];
            id missingValue =
                missingValues == nil
                ? nil
                : missingValues[@(featureId.c_str())];
            if (missingValue != nil &&
                ![missingValue isKindOfClass:[NSNumber class]]) {
                return invalid(
                    "missing_value_mask must contain booleans: " +
                        featureId
                );
            }
            const bool explicitlyMissing =
                missingValue != nil &&
                static_cast<NSNumber*>(missingValue).boolValue;
            if (value == nil || value == [NSNull null]) {
                staged.scenarioMissingMask[index] = 1u;
                if (missingValue != nil && !explicitlyMissing) {
                    return invalid(
                        "missing_value_mask conflicts with absent value: " +
                            featureId
                    );
                }
                continue;
            }
            if (explicitlyMissing) {
                return invalid(
                    "missing_value_mask conflicts with scenario value: " +
                        featureId
                );
            }
            if (![value isKindOfClass:[NSNumber class]]) {
                return invalid(
                    "scenario value must be numeric or null: " +
                        featureId
                );
            }
            const double measured =
                static_cast<NSNumber*>(value).doubleValue;
            if (!std::isfinite(measured)) {
                return invalid(
                    "scenario value is non-finite: " + featureId
                );
            }
            outcome.scenarioValues[index] =
                static_cast<float>(measured);
            outcome.scenarioValueMask[index] = 1u;
        }
        for (NSString* key in values) {
            if (schema.featureIndex(cppString(key)) ==
                MR_INVALID_INDEX) {
                return {
                    R2S2RStatus::schemaMismatch,
                    "hardware outcome contains an unknown scenario value: " +
                        cppString(key),
                };
            }
        }
        for (NSString* key in missingValues == nil ? @{} : missingValues) {
            if (schema.featureIndex(cppString(key)) ==
                MR_INVALID_INDEX) {
                return {
                    R2S2RStatus::schemaMismatch,
                    "hardware outcome contains an unknown missing mask: " +
                        cppString(key),
                };
            }
        }

        NSArray* artifacts = arrayValue(root, @"artifacts");
        for (id value in artifacts == nil ? @[] : artifacts) {
            if (![value isKindOfClass:[NSDictionary class]]) {
                return invalid("artifacts must contain objects");
            }
            NSDictionary* object =
                static_cast<NSDictionary*>(value);
            OutcomeArtifact artifact;
            artifact.kind =
                cppString(stringValue(object, @"kind"));
            artifact.uri =
                cppString(stringValue(object, @"uri"));
            artifact.contentHash =
                cppString(stringValue(object, @"content_hash"));
            outcome.artifacts.push_back(std::move(artifact));
        }
        if (!outcome.valid(&schema, &reason)) {
            return invalid(std::move(reason));
        }
        output = std::move(staged);
        return {};
    }
}

} // namespace metalrobo
