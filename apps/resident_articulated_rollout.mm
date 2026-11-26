#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/WorldPack.hpp"
#include "metalrobo/resident_articulated_world_types.h"

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

static mr_float4 normalized(mr_float4 value) {
    const float scale = 1.0f / std::sqrt(value.x * value.x + value.y * value.y +
                                         value.z * value.z + value.w * value.w);
    return {value.x * scale, value.y * scale, value.z * scale, value.w * scale};
}

static mr_float4 scaled(mr_float4 v, float s) {
    return {v.x * s, v.y * s, v.z * s, v.w * s};
}

static MRResidentProgramGPU compile(
    const LocomotionWorld& authored, const CompiledWorld& world
) {
    const EngineModel& model = authored.model;
    require(model.articulations.size() == 1 &&
            model.articulations.front().rootType == MR_ROOT_FIXED &&
            model.defaultQ.size() == model.defaultV.size() &&
            !model.defaultV.empty() && model.bodyNames.size() == model.bodies.size() &&
            model.defaultV.size() <= MR_RESIDENT_DOF_CAPACITY &&
            model.bodies.size() <= MR_RESIDENT_BODY_CAPACITY &&
            model.shapes.size() <= MR_RESIDENT_SHAPE_CAPACITY &&
            world.eligiblePairCount() <= MR_RESIDENT_PAIR_CAPACITY &&
            model.constraintProgram.blocks.empty(),
            "the cooked world exceeds the resident operator's capabilities");

    MRResidentProgramGPU result{};
    result.sizes = {static_cast<unsigned>(model.bodies.size()), static_cast<unsigned>(model.dofs.size()),
                    world.eligiblePairCount(), 0u};
    result.gravityAndTimestep = model.world.gravityAndTimestep;
    result.gravityAndTimestep.w = 0.005f;
    result.stabilization = {model.world.solverScales.w, model.world.solverScales.z, 0, 0};
    std::copy(model.shapes.begin(), model.shapes.end(), result.shapes);
    const unsigned root = model.articulations.front().rootBody;
    for (unsigned i = 0; i < result.sizes.x; ++i) {
        const auto& src = model.bodies[i];
        auto& dst = result.bodies[i];
        auto& joint = result.joints[i];
        const bool articulated = src.articulationIndex != MR_INVALID_INDEX;
        dst.topology.x = articulated;
        dst.topology.y = MR_INVALID_INDEX;
        dst.massAndDamping = {src.massAndInverseMass.x, src.massAndInverseMass.y,
                              src.dampingAndSpeedLimits.x, src.dampingAndSpeedLimits.y};
        dst.inertia[0] = src.inertiaRow0;
        dst.inertia[1] = src.inertiaRow1;
        dst.inertia[2] = src.inertiaRow2;
        dst.inverseInertia[0] = src.inverseInertiaRow0;
        dst.inverseInertia[1] = src.inverseInertiaRow1;
        dst.inverseInertia[2] = src.inverseInertiaRow2;
        dst.initialRotation.w = 1;
        joint.topology = {i, MR_INVALID_INDEX, MR_JOINT_FIXED, 0};
        if (articulated && i != root) {
            const auto& edge = model.joints.at(src.inboundJoint);
            require(edge.jointType == MR_JOINT_REVOLUTE || edge.jointType == MR_JOINT_PRISMATIC ||
                    edge.jointType == MR_JOINT_FIXED, "unsupported joint family");
            joint.topology.x = edge.parentBody;
            joint.topology.y = edge.nv == 0 ? MR_INVALID_INDEX : edge.vOffset;
            joint.topology.z = edge.jointType;
            joint.axis = edge.nv == 0 ? mr_float4{} : normalized(edge.axis0);
            joint.parentAnchor = edge.parentAnchor;
            joint.childAnchor = edge.childAnchor;
            joint.parentRotation = normalized(edge.parentRotation);
            joint.inverseChildRotation = normalized({-edge.childRotation.x, -edge.childRotation.y,
                                                     -edge.childRotation.z, edge.childRotation.w});
        }
    }

    for (unsigned n = 0; n < world.sceneBodyCount(); ++n) {
        const unsigned i = world.sceneBodyIndices()[n];
        const auto& state = authored.sceneBodies.at(n);
        auto& dst = result.bodies[i];
        dst.initialPosition = state.position;
        dst.initialRotation = normalized(state.orientation);
        dst.initialLinear = state.linearVelocityAndInverseMass;
        dst.initialAngular = state.angularVelocity;
        if (state.flagsAndIndices[0] == MR_MOTION_DYNAMIC) {
            require(result.sizes.w < MR_RESIDENT_FREE_CAPACITY, "free-body capacity exceeded");
            require(state.linearVelocityAndInverseMass.w > 0.0f, "invalid inverse mass");
            dst.topology.y = result.sizes.w;
            if (result.sizes.w == 0) result.freeBody.x = i;
            else result.freeBody.y = i;
            ++result.sizes.w;
            const float originalInverseMass = dst.massAndDamping.y;
            const float inverseMass = state.linearVelocityAndInverseMass.w;
            const float scale = originalInverseMass / inverseMass;
            dst.massAndDamping.x *= scale;
            dst.massAndDamping.y = inverseMass;
            for (unsigned j = 0; j < 3; ++j) {
                dst.inertia[j] = scaled(dst.inertia[j], scale);
                dst.inverseInertia[j] = scaled(dst.inverseInertia[j], 1.0f / scale);
            }
        } else {
            require(state.flagsAndIndices[0] == MR_MOTION_STATIC, "unsupported scene body motion");
        }
    }

    for (unsigned i = 0; i < result.sizes.y; ++i) {
        const auto& src = model.dofs[i];
        auto& dst = result.dofs[i];
        require(src.qIndex == i && src.vIndex == i, "unsupported coordinate ordering");
        dst.topology = {model.joints[src.jointIndex].childBody, 0, src.flags, 0};
        dst.drive = src.drive;
        dst.limits = src.limits;
        dst.initial = {model.defaultQ[i], model.defaultV[i], 0, 0};
        if (!model.actuatorProfiles.empty()) {
            const auto& actuator = model.actuatorProfiles[i];
            const bool active = (actuator.identity.y & MR_ACTUATOR_PROFILE_ACTIVE) != 0;
            require(!active || (actuator.transmissionAndEnvelope.x == 0 &&
                                actuator.transmissionAndEnvelope.y == 0), "unsupported actuator delay or play");
            dst.initial.z = active;
            dst.motorAndSpeed = actuator.motorAndSpeed;
            dst.transmissionAndEnvelope = actuator.transmissionAndEnvelope;
        }
    }

    for (unsigned body = 0; body < result.sizes.x; ++body) {
        if (result.bodies[body].topology.x == 0) continue;
        unsigned cursor = body;
        for (unsigned depth = 0; cursor != root; ++depth) {
            require(depth < result.sizes.x, "cyclic articulation");
            const auto& joint = result.joints[cursor];
            if (joint.topology.y != MR_INVALID_INDEX) {
                result.joints[body].topology.w |= 1u << joint.topology.y;
                result.dofs[joint.topology.y].topology.y |= 1u << body;
            }
            cursor = joint.topology.x;
        }
    }

    unsigned index = 0;
    for (const auto& src : world.eligiblePairs()) {
        const auto& a = model.shapes[src.colliderA];
        const auto& b = model.shapes[src.colliderB];
        const auto allowed = [](const MRShapeGPU& shape) {
            return shape.shapeType <= MR_SHAPE_PLANE && shape.flags == 0 && shape.geometryCount == 0;
        };
        require(allowed(a) && allowed(b), "unsupported collision geometry");
        const auto& ma = model.materials[a.materialIndex];
        const auto& mb = model.materials[b.materialIndex];
        require(ma.geometry.y == 0 && mb.geometry.y == 0 &&
                ma.friction.z == 0 && mb.friction.z == 0 &&
                ma.friction.w == 0 && mb.friction.w == 0, "unsupported contact material");
        auto& dst = result.pairs[index++];
        dst.geometry = {src.colliderA, src.colliderB, src.pairClass, 0};
        dst.frictionAndRestitution = {
            std::sqrt(ma.friction.x * mb.friction.x), std::sqrt(ma.friction.y * mb.friction.y),
            std::max(ma.response.x, mb.response.x), std::max(ma.response.y, mb.response.y)
        };
        dst.response = {ma.response.z + mb.response.z, ma.response.w + mb.response.w,
                        a.contactRestAndBoundingRadius.y + b.contactRestAndBoundingRadius.y, 0};
    }
    std::stable_partition(result.pairs, result.pairs + result.sizes.z,
        [&](const MRResidentPairGPU& pair) {
            return result.shapes[pair.geometry.x].shapeType == MR_SHAPE_PLANE ||
                result.shapes[pair.geometry.y].shapeType == MR_SHAPE_PLANE;
        });
    return result;
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
    require(bindings.actionBindings().size() == program.sizes.y, "incompatible position-action binding");
    program.controls.x = bindings.actionBindings().size();
    unsigned mask = 0;
    for (const auto& binding : bindings.actionBindings()) {
        const auto& i = binding.indices;
        require((binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION ||
                 binding.actuator.x == MR_TASK_ACTUATOR_GRIPPER_POSITION) &&
                i.x < program.controls.x && i.z == i.w && i.w < program.sizes.y &&
                (mask & (1u << i.w)) == 0, "resident input requires distinct scalar position actions");
        mask |= 1u << i.w;
        auto& dof = program.dofs[i.w];
        dof.topology.w = i.x;
        dof.action = binding.parameters;
        if (binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION) {
            dof.drive.x = binding.drive.x;
            dof.drive.y = binding.drive.y;
            if (dof.drive.x > 0.0f || dof.drive.y > 0.0f) dof.topology.z |= MR_DOF_FLAG_DRIVE;
        }
        const float dt = program.schedule.y * program.gravityAndTimestep.w;
        const float tau = dof.action.w;
        dof.action.w = tau > 0 ? std::clamp(1.0f - std::exp(-dt / tau), 0.0f, 1.0f) : 1.0f;
    }
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
            auto program = compile(materialized, compiled);
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
