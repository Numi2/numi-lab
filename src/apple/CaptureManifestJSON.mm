#include "metalrobo/EpisodeTwinCompiler.hpp"

#include "metalrobo/FrankaWorld.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <optional>
#include <string_view>

namespace metalrobo {
namespace {

EpisodeTwinCompilerResult invalid(std::string message) {
    return {
        EpisodeTwinCompilerStatus::invalidManifest,
        std::move(message),
    };
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
    return [value isKindOfClass:[NSArray class]] ? static_cast<NSArray*>(value)
                                                 : nil;
}

NSDictionary* dictionaryValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    return [value isKindOfClass:[NSDictionary class]]
               ? static_cast<NSDictionary*>(value)
               : nil;
}

std::string cppString(NSString* value) {
    return value == nil ? std::string{} : std::string{value.UTF8String};
}

double finiteNumber(NSDictionary* object, NSString* key, const double fallback,
                    bool& okay) {
    NSNumber* value = numberValue(object, key);
    if (value == nil) {
        return fallback;
    }
    const double result = value.doubleValue;
    okay = okay && std::isfinite(result);
    return result;
}

std::optional<CaptureAdapterKind> parseAdapter(const std::string_view value) {
    if (value == "arkit") {
        return CaptureAdapterKind::arkit;
    }
    if (value == "rgbd_robot_telemetry") {
        return CaptureAdapterKind::rgbdRobotTelemetry;
    }
    if (value == "ros_bag") {
        return CaptureAdapterKind::rosBag;
    }
    if (value == "libfranka_log") {
        return CaptureAdapterKind::libfrankaLog;
    }
    if (value == "cad_urdf") {
        return CaptureAdapterKind::cadUrdf;
    }
    if (value == "video") {
        return CaptureAdapterKind::videoFallback;
    }
    return std::nullopt;
}

std::optional<CaptureStreamKind> parseStreamKind(const std::string_view value) {
    if (value == "rgb") {
        return CaptureStreamKind::rgb;
    }
    if (value == "depth") {
        return CaptureStreamKind::depth;
    }
    if (value == "rgbd") {
        return CaptureStreamKind::rgbd;
    }
    if (value == "video") {
        return CaptureStreamKind::video;
    }
    if (value == "camera_calibration") {
        return CaptureStreamKind::cameraCalibration;
    }
    if (value == "camera_poses") {
        return CaptureStreamKind::cameraPoses;
    }
    if (value == "robot_telemetry") {
        return CaptureStreamKind::robotTelemetry;
    }
    if (value == "robot_commands") {
        return CaptureStreamKind::robotCommands;
    }
    if (value == "gripper_state") {
        return CaptureStreamKind::gripperState;
    }
    if (value == "force_torque") {
        return CaptureStreamKind::forceTorque;
    }
    if (value == "cad") {
        return CaptureStreamKind::cad;
    }
    if (value == "urdf") {
        return CaptureStreamKind::urdf;
    }
    return std::nullopt;
}

std::optional<EpisodeTwinStage> parseStage(const std::string_view value) {
    if (value == "ingest") {
        return EpisodeTwinStage::ingest;
    }
    if (value == "select_frames") {
        return EpisodeTwinStage::selectFrames;
    }
    if (value == "discover_entities") {
        return EpisodeTwinStage::discoverEntities;
    }
    if (value == "segment") {
        return EpisodeTwinStage::segment;
    }
    if (value == "reconstruct_geometry") {
        return EpisodeTwinStage::reconstructGeometry;
    }
    if (value == "track_poses") {
        return EpisodeTwinStage::trackPoses;
    }
    if (value == "infer_physics") {
        return EpisodeTwinStage::inferPhysics;
    }
    if (value == "assemble_replay") {
        return EpisodeTwinStage::assembleReplay;
    }
    if (value == "align_replay") {
        return EpisodeTwinStage::alignReplay;
    }
    if (value == "publish") {
        return EpisodeTwinStage::publish;
    }
    return std::nullopt;
}

std::optional<EpisodeArtifactKind>
parseArtifactKind(const std::string_view value) {
    if (value == "capture") {
        return EpisodeArtifactKind::capture;
    }
    if (value == "camera_calibration") {
        return EpisodeArtifactKind::cameraCalibration;
    }
    if (value == "entity_graph") {
        return EpisodeArtifactKind::entityGraph;
    }
    if (value == "segmentation") {
        return EpisodeArtifactKind::segmentation;
    }
    if (value == "geometry") {
        return EpisodeArtifactKind::geometry;
    }
    if (value == "appearance") {
        return EpisodeArtifactKind::appearance;
    }
    if (value == "pose_track") {
        return EpisodeArtifactKind::poseTrack;
    }
    if (value == "physical_prior") {
        return EpisodeArtifactKind::physicalPrior;
    }
    if (value == "robot_trajectory") {
        return EpisodeArtifactKind::robotTrajectory;
    }
    if (value == "interaction_trace") {
        return EpisodeArtifactKind::interactionTrace;
    }
    if (value == "replay") {
        return EpisodeArtifactKind::replay;
    }
    return std::nullopt;
}

std::optional<EpisodeArtifactProducer>
parseProducer(const std::string_view value) {
    if (value == "measured") {
        return EpisodeArtifactProducer::measured;
    }
    if (value == "deterministic_tool") {
        return EpisodeArtifactProducer::deterministicTool;
    }
    if (value == "agent_decision") {
        return EpisodeArtifactProducer::agentDecision;
    }
    if (value == "authored") {
        return EpisodeArtifactProducer::authored;
    }
    return std::nullopt;
}

bool parseFloat4(NSArray* array, mr_float4& output, const bool exact) {
    if (array == nil || (exact && array.count != 4u) ||
        (!exact && array.count > 4u)) {
        return false;
    }
    float values[4]{};
    for (NSUInteger index = 0u; index < array.count; ++index) {
        id value = array[index];
        if (![value isKindOfClass:[NSNumber class]]) {
            return false;
        }
        const double number = static_cast<NSNumber*>(value).doubleValue;
        if (!std::isfinite(number)) {
            return false;
        }
        values[index] = static_cast<float>(number);
    }
    output = {values[0], values[1], values[2], values[3]};
    return true;
}

bool parseCalibration(NSDictionary* object, CaptureCalibration& output) {
    if (object == nil) {
        return true;
    }
    NSNumber* width = numberValue(object, @"width");
    NSNumber* height = numberValue(object, @"height");
    if ((width == nil) != (height == nil)) {
        return false;
    }
    if (width != nil && height != nil) {
        output.width = width.unsignedIntValue;
        output.height = height.unsignedIntValue;
        output.hasResolution = output.width != 0u && output.height != 0u;
        if (!output.hasResolution) {
            return false;
        }
    }
    NSArray* intrinsics = arrayValue(object, @"intrinsics");
    if (intrinsics != nil &&
        !parseFloat4(intrinsics, output.intrinsics, true)) {
        return false;
    }
    output.hasIntrinsics = intrinsics != nil;
    NSArray* distortion = arrayValue(object, @"distortion");
    if (distortion != nil &&
        !parseFloat4(distortion, output.distortion, false)) {
        return false;
    }
    output.hasDistortion = distortion != nil;
    NSArray* position = arrayValue(object, @"position");
    NSArray* orientation = arrayValue(object, @"orientation");
    if ((position == nil) != (orientation == nil)) {
        return false;
    }
    if (position != nil && orientation != nil) {
        if (position.count != 3u ||
            !parseFloat4(orientation, output.worldFromSensor.orientation,
                         true)) {
            return false;
        }
        mr_float4 parsed{};
        if (!parseFloat4(position, parsed, false)) {
            return false;
        }
        output.worldFromSensor.position = parsed;
        output.hasPose = true;
    }
    return true;
}

std::filesystem::path resolveSource(const std::filesystem::path& root,
                                    NSString* value) {
    std::filesystem::path source{cppString(value)};
    if (source.is_relative()) {
        source = root / source;
    }
    return std::filesystem::absolute(source).lexically_normal();
}

} // namespace

EpisodeTwinCompilerResult
loadCaptureManifestJSON(const std::filesystem::path& path,
                        CaptureManifest& output) {
    @autoreleasepool {
        NSError* error = nil;
        NSData* data = [NSData dataWithContentsOfFile:@(path.string().c_str())
                                              options:0
                                                error:&error];
        if (data == nil) {
            return {
                EpisodeTwinCompilerStatus::ioFailure,
                error == nil ? "could not read capture manifest"
                             : cppString(error.localizedDescription),
            };
        }
        id rootObject = [NSJSONSerialization JSONObjectWithData:data
                                                        options:0
                                                          error:&error];
        if (![rootObject isKindOfClass:[NSDictionary class]]) {
            return invalid(error == nil
                               ? "capture manifest root must be a JSON object"
                               : cppString(error.localizedDescription));
        }
        NSDictionary* root = static_cast<NSDictionary*>(rootObject);
        CaptureManifest candidate;
        NSNumber* schema = numberValue(root, @"schema_version");
        if (schema != nil) {
            candidate.schemaVersion = schema.unsignedIntValue;
        }
        candidate.id = cppString(stringValue(root, @"id"));
        candidate.coordinateConvention =
            cppString(stringValue(root, @"coordinate_convention"));
        if (candidate.coordinateConvention.empty()) {
            candidate.coordinateConvention = "x-forward,y-left,z-up";
        }
        candidate.engineModelId = cppString(stringValue(root, @"engine_model"));
        candidate.worldProgramId =
            cppString(stringValue(root, @"world_program"));

        const std::string adapterText =
            cppString(stringValue(root, @"adapter"));
        const auto adapter = parseAdapter(adapterText);
        if (!adapter.has_value()) {
            return invalid("capture manifest adapter is unsupported");
        }
        candidate.adapter = *adapter;

        const std::filesystem::path manifestDirectory =
            std::filesystem::absolute(path).parent_path();
        NSString* sourceRoot = stringValue(root, @"source_root");
        candidate.sourceRoot =
            sourceRoot == nil ? manifestDirectory
                              : resolveSource(manifestDirectory, sourceRoot);

        const std::string seedWorld =
            cppString(stringValue(root, @"seed_world"));
        if (seedWorld == "franka_pick_place") {
            candidate.seedEpisode = makeFrankaPickPlaceEpisodeTwin();
        } else {
            return invalid("seed_world must name a registered deterministic "
                           "world assembler");
        }

        NSArray* streams = arrayValue(root, @"streams");
        for (id value in streams == nil ? @[] : streams) {
            if (![value isKindOfClass:[NSDictionary class]]) {
                return invalid("capture stream must be a JSON object");
            }
            NSDictionary* object = static_cast<NSDictionary*>(value);
            CaptureStream stream;
            stream.id = cppString(stringValue(object, @"id"));
            NSString* pathValue = stringValue(object, @"path");
            const auto kind =
                parseStreamKind(cppString(stringValue(object, @"kind")));
            if (pathValue == nil || !kind.has_value()) {
                return invalid(
                    "capture stream path or kind is missing or unsupported");
            }
            stream.kind = *kind;
            stream.assetId = cppString(stringValue(object, @"asset_id"));
            stream.sensorId = cppString(stringValue(object, @"sensor_id"));
            stream.source = resolveSource(candidate.sourceRoot, pathValue);
            stream.expectedContentHash =
                cppString(stringValue(object, @"sha256"));
            const std::string timestampDomain =
                cppString(stringValue(object, @"timestamp_domain"));
            if (!timestampDomain.empty()) {
                stream.timestampDomain = timestampDomain;
            }
            bool numbersOkay = true;
            stream.startTimeSeconds =
                finiteNumber(object, @"start_seconds", 0.0, numbersOkay);
            stream.endTimeSeconds =
                finiteNumber(object, @"end_seconds", 0.0, numbersOkay);
            stream.nominalRateHz =
                finiteNumber(object, @"rate_hz", 0.0, numbersOkay);
            NSDictionary* calibration = dictionaryValue(object, @"calibration");
            if (!numbersOkay ||
                !parseCalibration(calibration, stream.calibration)) {
                return invalid(
                    "capture stream timing or calibration is invalid");
            }
            candidate.streams.push_back(std::move(stream));
        }

        NSArray* products = arrayValue(root, @"products");
        for (id value in products == nil ? @[] : products) {
            if (![value isKindOfClass:[NSDictionary class]]) {
                return invalid("capture product must be a JSON object");
            }
            NSDictionary* object = static_cast<NSDictionary*>(value);
            CaptureProduct product;
            product.id = cppString(stringValue(object, @"id"));
            NSString* pathValue = stringValue(object, @"path");
            const auto stage =
                parseStage(cppString(stringValue(object, @"stage")));
            const auto kind =
                parseArtifactKind(cppString(stringValue(object, @"kind")));
            const auto producer =
                parseProducer(cppString(stringValue(object, @"producer")));
            if (pathValue == nil || !stage.has_value() || !kind.has_value() ||
                !producer.has_value()) {
                return invalid("capture product stage, kind, producer, or path "
                               "is invalid");
            }
            product.stage = *stage;
            product.kind = *kind;
            product.producer = *producer;
            product.assetId = cppString(stringValue(object, @"asset_id"));
            product.source = resolveSource(candidate.sourceRoot, pathValue);
            product.expectedContentHash =
                cppString(stringValue(object, @"sha256"));
            bool numbersOkay = true;
            product.startTimeSeconds =
                finiteNumber(object, @"start_seconds", 0.0, numbersOkay);
            product.endTimeSeconds =
                finiteNumber(object, @"end_seconds", 0.0, numbersOkay);
            if (!numbersOkay) {
                return invalid("capture product timing is invalid");
            }
            candidate.products.push_back(std::move(product));
        }

        std::string reason;
        if (!candidate.valid(&reason)) {
            return invalid(std::move(reason));
        }
        output = std::move(candidate);
        return {};
    }
}

} // namespace metalrobo
