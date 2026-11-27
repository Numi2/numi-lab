#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/WorldPack.hpp"
#include "ResidentArticulatedWorld.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metalrobo;

static void require(bool condition, const std::string& error) {
    if (!condition) throw std::runtime_error(error);
}

static std::vector<float> readActions(
    const std::filesystem::path& clip,
    const char* taskPath,
    const char* sensorsPath,
    const char* realityPath,
    const CompiledWorld& world,
    const RobotActuatorPack& actuators,
    MRResidentProgramGPU& program
) {
    TaskPack task;
    SensorProgramPack sensors;
    RealityProgramPack reality;
    const auto taskLoaded = readTaskPack(taskPath, task);
    require(taskLoaded.succeeded(), taskLoaded.message);
    const auto sensorsLoaded = readSensorProgramPack(sensorsPath, sensors);
    require(sensorsLoaded.succeeded(), sensorsLoaded.message);
    const auto realityLoaded = readRealityProgramPack(realityPath, reality);
    require(realityLoaded.succeeded(), realityLoaded.message);
    require(reality.reset.operators.empty() && reality.reset.maximumActionDelaySteps == 0 &&
            reality.program.variations.empty() && task.pushes.maximumVelocity == 0.0f &&
            task.pushes.projectileHorizontalSpeedUpper == 0.0f &&
            task.terrain.resetTranslations.empty(), "unsupported task reset, physical disturbance or world variation");
    CompiledTaskProgram bindings;
    const auto compiled = compileTaskProgram(task, actuators.actuators, sensors.observation,
                                              reality.reset, world, bindings);
    require(compiled.succeeded(), compiled.message);
    detail::bindResidentArticulationActions(
        bindings, program.schedule.y * program.gravityAndTimestep.w, program
    );
    std::ifstream in(clip, std::ios::ate | std::ios::binary);
    require(in.good(), "cannot open action clip");
    const auto bytes = in.tellg();
    require(bytes > 0 && bytes % (sizeof(float) * program.controls.x) == 0, "invalid float32 action clip size");
    std::vector<float> actions(static_cast<size_t>(bytes) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(actions.data()), bytes);
    require(in.good() && std::all_of(actions.begin(), actions.end(),
        [](float v) { return std::isfinite(v); }), "invalid action clip values");
    return actions;
}

static void writeTraces(const std::string& stem, const MRResidentProgramGPU& program,
                        const MRResidentFrameGPU* frames, const EngineModel& model) {
    std::ofstream state(stem + ".state.csv");
    std::ofstream force(stem + ".forces.csv");
    std::ofstream raw(stem + ".f32", std::ios::binary);
    std::ofstream target(stem + ".targets.csv");
    require(state && force && raw && target, "cannot write resident evidence");
    state << std::setprecision(10);
    force << std::setprecision(10);
    target << std::setprecision(10) << "tick,t0,t1,t2,t3,t4,t5,t6,t7,t8\n";
    state << "tick,q0,q1,q2,q3,q4,q5,q6,q7,q8";
    for (unsigned j = 0; j < program.sizes.w; ++j) {
        const unsigned body = j == 0 ? program.freeBody.x : program.freeBody.y;
        const auto& name = model.bodyNames[body];
        state << ',' << name << "_x," << name << "_y," << name << "_z";
    }
    for (unsigned j = 0; j < program.sizes.w; ++j) {
        const unsigned body = j == 0 ? program.freeBody.x : program.freeBody.y;
        const auto& name = model.bodyNames[body];
        state << ',' << name << "_qx," << name << "_qy," << name << "_qz," << name << "_qw";
    }
    state << '\n';
    force << "tick,collider_a,collider_b,body_a,body_b,separation,n,ix,iy,iz,px,py,pz,nx,ny,nz\n";
    for (unsigned i = 0; i < program.schedule.x; ++i) {
        const auto& frame = frames[i];
        raw.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
        if (frame.result.x != MR_STEP_SUCCESS || frame.result.w != i + 1) break;
        state << i + 1;
        for (const auto q : frame.q) state << ',' << q;
        for (unsigned j = 0; j < program.sizes.w; ++j) {
            const auto& pos = frame.freeBodies[j].position;
            state << ',' << pos.x << ',' << pos.y << ',' << pos.z;
        }
        for (unsigned j = 0; j < program.sizes.w; ++j) {
            const auto& q = frame.freeBodies[j].rotation;
            state << ',' << q.x << ',' << q.y << ',' << q.z << ',' << q.w;
        }
        state << '\n';
        target << i + 1;
        for (const auto t : frame.targets) target << ',' << t;
        target << '\n';
        for (unsigned j = 0; j < frame.result.y; ++j) {
            const auto& c = frame.contacts[j];
            force << i + 1 << ',' << c.pair.x << ',' << c.pair.y << ',' <<
                model.bodyNames[c.pair.z] << ',' << model.bodyNames[c.pair.w] <<
                ',' << c.pointAndSeparation.w << ',' << c.normalAndImpulse.w <<
                ',' << c.impulse.x << ',' << c.impulse.y << ',' << c.impulse.z <<
                ',' << c.pointAndSeparation.x << ',' << c.pointAndSeparation.y <<
                ',' << c.pointAndSeparation.z << ',' << c.normalAndImpulse.x <<
                ',' << c.normalAndImpulse.y << ',' << c.normalAndImpulse.z << '\n';
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 9 || argc > 12) {
        std::fprintf(stderr, "usage: metalrobo_resident_rollout METALLIB WORLD TASK ACTUATORS SENSORS "
            "REALITY ACTION_CLIP TRACE_PREFIX [SWEEPS=1] [TRIALS=7] [CONTACT_CAPACITY=31]\\n");
        return 2;
    }
    @autoreleasepool {
        try {
            MRWorldPack authored;
            const auto loaded = readWorldPack(argv[2], authored);
            require(loaded.succeeded(), loaded.message);
            require(authored.family.program.variations.empty(), "unsupported WorldPack variations");
            const auto materialized = makeWorldPackLocomotionWorld(authored);
            RobotActuatorPack actuators;
            const auto actuatorStatus = readRobotActuatorPack(argv[4], actuators);
            require(actuatorStatus.succeeded(), actuatorStatus.message);
            CompiledWorld compiled;
            const auto cooked = compileMetalWorld(materialized.model, materialized.articulationIndex, compiled);
            require(cooked.succeeded(), cooked.message);
            auto program = detail::compileResidentArticulation(compiled, materialized.sceneBodies);
            program.gravityAndTimestep.w = 0.005f;
            const auto sweeps = argc > 9 ? std::stoul(argv[9]) : 1;
            const auto trials = argc > 10 ? std::stoul(argv[10]) : 7;
            const auto capacity = argc > 11 ? std::stoul(argv[11]) : MR_RESIDENT_CONTACT_CAPACITY;
            require(sweeps > 0 && sweeps <= 64 && trials > 0 && trials <= 1000 &&
                    capacity > 0 && capacity <= MR_RESIDENT_CONTACT_CAPACITY, "invalid rollout budget");
            program.schedule = {0, 4, static_cast<unsigned>(sweeps), 0};
            program.controls.y = static_cast<unsigned>(capacity);
            const auto actions = readActions(argv[7], argv[3], argv[5], argv[6], compiled, actuators, program);
            const auto steps = actions.size() / program.controls.x;
            require(steps > 0 && steps <= std::numeric_limits<unsigned>::max(), "invalid clip length");
            program.schedule.x = static_cast<unsigned>(steps);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library != nil, error == nil ? "library load failed" : error.localizedDescription.UTF8String);
            id<MTLFunction> fn = [library newFunctionWithName:@"mr_resident_articulated_rollout"];
            require(fn != nil, "compact Metal entry point missing");
            MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = fn;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionNone reflection:nil error:&error];
            require(pipeline != nil, error == nil ? "pipeline creation failed" : error.localizedDescription.UTF8String);
            require(pipeline.threadExecutionWidth == 32 &&
                    pipeline.maxTotalThreadsPerThreadgroup >= 96 &&
                    pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength, "incompatible GPU");
            id<MTLBuffer> config = [device newBufferWithBytes:&program length:sizeof(program)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> input = [device newBufferWithBytes:actions.data() length:actions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            const size_t outputBytes = program.schedule.x * sizeof(MRResidentFrameGPU);
            id<MTLBuffer> output = [device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(config && input && output && queue, "Metal resource allocation failed");
            std::vector<unsigned char> replay(outputBytes);
            bool same = true;
            bool completed = true;
            std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (unsigned coordinate = 0; coordinate < program.sizes.y; ++coordinate) {
                const auto& dof = program.dofs[coordinate];
                std::cout << "binding=" << coordinate << " action=" << dof.topology.w
                          << " default=" << dof.initial.x << " scale=" << dof.action.x
                          << " lower=" << dof.action.y << " upper=" << dof.action.z
                          << " alpha=" << dof.action.w << '\n';
            }
            std::cout << "device=" << device.name.UTF8String << " bytes=" << outputBytes
                      << " scratch=" << pipeline.staticThreadgroupMemoryLength << " slop=" << program.stabilization.x
                      << " depenetration=" << program.stabilization.y << '\n';
            for (unsigned trial = 0; trial <= trials; ++trial) @autoreleasepool {
                std::memset(output.contents, 0, outputBytes);
                const auto start = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:config offset:0 atIndex:0];
                [encoder setBuffer:input offset:0 atIndex:1];
                [encoder setBuffer:output offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(96,1,1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                const auto stop = std::chrono::steady_clock::now();
                require(command.status == MTLCommandBufferStatusCompleted,
                        command.error == nil ? "GPU command failed" : command.error.localizedDescription.UTF8String);
                const auto* frames = static_cast<const MRResidentFrameGPU*>(output.contents);
                unsigned i = 0;
                for (; i < program.schedule.x && frames[i].result.w == i + 1 &&
                       frames[i].result.x == MR_STEP_SUCCESS; ++i) {}
                if (i < program.schedule.x) {
                    const auto result = frames[i].result;
                    std::cout << "failed accepted=" << i << " status=" << result.x <<
                        " count=" << result.y << " substep=" << result.z << '\n';
                    completed = false;
                }
                const double wall = std::chrono::duration<double, std::milli>(stop - start).count();
                const double gpu = 1000 * (command.GPUEndTime - command.GPUStartTime);
                std::cout << "trial=" << trial << " wall_ms=" << wall << " gpu_ms=" << gpu << " accepted=" << i << '\n';
                if (trial == 0) {
                    std::memcpy(replay.data(), output.contents, outputBytes);
                    writeTraces(argv[8], program, frames, materialized.model);
                } else {
                    same = same && std::memcmp(replay.data(), output.contents, outputBytes) == 0;
                }
                if (!completed) break;
            }
            std::cout << "bitwise_replay=" << same << '\n';
            return completed && same ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
