#include "metalrobo/EpisodeTwinCompiler.hpp"

#include "metalrobo/FrankaWorld.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct TemporaryDirectory {
    std::filesystem::path path;

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void writeFixture(const std::filesystem::path& path,
                  const std::string& contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << contents;
    if (!stream) {
        throw std::runtime_error("could not write probe fixture");
    }
}

metalrobo::CaptureProduct product(
    const std::string& id,
    const metalrobo::EpisodeTwinStage stage,
    const metalrobo::EpisodeArtifactKind artifactKind,
    const metalrobo::EpisodeTwinProductKind productKind,
    const std::filesystem::path& source) {
    metalrobo::CaptureProduct result;
    result.id = id;
    result.stage = stage;
    result.kind = artifactKind;
    result.producer =
        metalrobo::EpisodeArtifactProducer::deterministicTool;
    result.source = source;
    result.payload.kind = productKind;
    return result;
}

const metalrobo::WorldAsset& asset(
    const metalrobo::EpisodeTwin& episode,
    const std::string& id) {
    for (const metalrobo::WorldAsset& value : episode.assets) {
        if (value.id == id) {
            return value;
        }
    }
    throw std::runtime_error("compiled episode has no requested asset");
}

const metalrobo::SensorSpec& sensor(
    const metalrobo::EpisodeTwin& episode,
    const std::string& id) {
    for (const metalrobo::SensorSpec& value : episode.sensors) {
        if (value.id == id) {
            return value;
        }
    }
    throw std::runtime_error("compiled episode has no requested sensor");
}

} // namespace

int main() {
    try {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        TemporaryDirectory temporary{
            std::filesystem::temp_directory_path() /
            ("metalrobo-twin-product-probe-" + std::to_string(nonce))};
        std::filesystem::create_directories(temporary.path);
        const std::filesystem::path capture = temporary.path / "capture.bin";
        const std::filesystem::path trace = temporary.path / "trace.csv";
        const std::filesystem::path derived = temporary.path / "derived.bin";
        writeFixture(capture, "synchronized-rgbd");
        writeFixture(trace, "t,q,dq,command\n0,0,0,0\n");
        writeFixture(derived, "deterministic-derived-product");

        metalrobo::CaptureManifest manifest;
        manifest.schemaVersion = 2u;
        manifest.id = "franka_physical_twin_probe";
        manifest.adapter =
            metalrobo::CaptureAdapterKind::rgbdRobotTelemetry;
        manifest.profile = metalrobo::CaptureProfile::frankaFixedRGBD;
        manifest.engineModelId = "franka_pick_place";
        manifest.worldProgramId = "franka_pick_place";
        manifest.seedEpisode =
            metalrobo::makeFrankaPickPlaceEpisodeTwin();

        metalrobo::CaptureStream rgbd;
        rgbd.id = "fixed_rgbd_capture";
        rgbd.kind = metalrobo::CaptureStreamKind::rgbd;
        rgbd.sensorId = "fixed_rgbd";
        rgbd.source = capture;
        rgbd.nominalRateHz = 30.0;
        rgbd.calibration.hasResolution = true;
        rgbd.calibration.hasIntrinsics = true;
        rgbd.calibration.hasPose = true;
        rgbd.calibration.width = 640u;
        rgbd.calibration.height = 480u;
        rgbd.calibration.intrinsics =
            {610.0f, 609.0f, 320.0f, 240.0f};
        rgbd.calibration.worldFromSensor.position =
            {0.72f, -0.51f, 0.83f, 0.0f};

        metalrobo::CaptureStream state;
        state.id = "franka_state";
        state.kind = metalrobo::CaptureStreamKind::robotTelemetry;
        state.assetId = "franka";
        state.source = trace;
        state.nominalRateHz = 1000.0;

        metalrobo::CaptureStream commands = state;
        commands.id = "franka_commands";
        commands.kind = metalrobo::CaptureStreamKind::robotCommands;
        manifest.streams = {rgbd, state, commands};

        auto graph = product(
            "entity_support_graph",
            metalrobo::EpisodeTwinStage::discoverEntities,
            metalrobo::EpisodeArtifactKind::entityGraph,
            metalrobo::EpisodeTwinProductKind::semanticGraph,
            derived);
        graph.producer =
            metalrobo::EpisodeArtifactProducer::agentDecision;

        auto pose = product(
            "pick_object_pose_track",
            metalrobo::EpisodeTwinStage::trackPoses,
            metalrobo::EpisodeArtifactKind::poseTrack,
            metalrobo::EpisodeTwinProductKind::objectPoseTrack,
            derived);
        pose.assetId = "pick_object";
        pose.payload.targetId = "pick_object";
        pose.payload.hasWorldPose = true;
        pose.payload.worldPose.position =
            {0.57f, -0.04f, 0.031f, 0.0f};

        auto render = product(
            "pick_object_render_mesh",
            metalrobo::EpisodeTwinStage::reconstructGeometry,
            metalrobo::EpisodeArtifactKind::geometry,
            metalrobo::EpisodeTwinProductKind::renderGeometry,
            derived);
        render.assetId = "pick_object";
        render.payload.targetId = "pick_object";
        render.payload.renderRepresentation =
            MR_WORLD_RENDER_GAUSSIAN_FIELD;

        auto collision = product(
            "pick_object_collision",
            metalrobo::EpisodeTwinStage::reconstructGeometry,
            metalrobo::EpisodeArtifactKind::geometry,
            metalrobo::EpisodeTwinProductKind::collisionGeometry,
            derived);
        collision.assetId = "pick_object";
        collision.payload.targetId = "pick_object";
        collision.payload.collisionRepresentation =
            MR_WORLD_COLLISION_PRIMITIVES;
        collision.payload.hasCollisionBox = true;
        collision.payload.collisionBoxHalfExtents =
            {0.031f, 0.022f, 0.041f, 0.0f};
        manifest.products = {graph, render, collision, pose};

        metalrobo::EpisodeTwinCompiler compiler{
            {temporary.path / "store", true, false}};
        metalrobo::CompiledEpisodeTwin first;
        const auto compiled = compiler.compile(
            manifest,
            metalrobo::makeFrankaPickPlaceEngineModel(),
            metalrobo::makeFrankaPickPlaceWorldProgram(),
            first);
        if (!compiled.succeeded()) {
            throw std::runtime_error(
                "typed capture failed to compile: " + compiled.message);
        }
        const auto& object = asset(first.episode, "pick_object");
        const auto& camera = sensor(first.episode, "fixed_rgbd");
        if (object.initialPose.position.x != 0.57f ||
            object.render != MR_WORLD_RENDER_GAUSSIAN_FIELD ||
            object.collision != MR_WORLD_COLLISION_PRIMITIVES ||
            camera.width != 640u || camera.intrinsics.x != 610.0f ||
            first.worldTemplate.engineModel.shapes[32u].dimensions.x !=
                0.031f) {
            throw std::runtime_error(
                "typed products did not causally alter the compiled twin");
        }

        metalrobo::CompiledEpisodeTwin resumed;
        const auto resumedResult = compiler.compile(
            manifest,
            metalrobo::makeFrankaPickPlaceEngineModel(),
            metalrobo::makeFrankaPickPlaceWorldProgram(),
            resumed);
        if (!resumedResult.succeeded() ||
            asset(resumed.episode, "pick_object").initialPose.position.x !=
                0.57f) {
            throw std::runtime_error(
                "cached typed products did not reassemble the same twin");
        }

        metalrobo::CaptureManifest incomplete = manifest;
        incomplete.products.erase(incomplete.products.begin() + 2);
        metalrobo::CompiledEpisodeTwin rejected;
        const auto rejectedResult = compiler.compile(
            incomplete,
            metalrobo::makeFrankaPickPlaceEngineModel(),
            metalrobo::makeFrankaPickPlaceWorldProgram(),
            rejected);
        if (rejectedResult.status !=
            metalrobo::EpisodeTwinCompilerStatus::assemblyFailure) {
            throw std::runtime_error(
                "missing collision product did not fail publication");
        }

        std::size_t cacheHits = 0u;
        for (const auto& receipt : resumed.receipts) {
            cacheHits += receipt.cacheHit ? 1u : 0u;
        }
        std::cout
            << "typed_products=" << first.products.size()
            << " object_x=" << object.initialPose.position.x
            << " collision_half_x="
            << first.worldTemplate.engineModel.shapes[32u].dimensions.x
            << " camera_width=" << camera.width
            << " resumed_stages=" << cacheHits
            << " missing_collision_rejected=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_episode_twin_product_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
