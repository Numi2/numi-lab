#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/FreeBodyDynamics.hpp"
#include "metalrobo/Multicopter.hpp"
#include "metalrobo/PX4X500.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {
constexpr std::uint32_t kEnvironments = 1024u;
constexpr std::uint32_t kSteps = 2000u;
constexpr float kDt = 0.001f;

mr_float4 f4(float x, float y, float z, float w = 0.0f) { return {x, y, z, w}; }
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
std::string string(NSString* value) { return value == nil || value.UTF8String == nullptr ? "" : std::string(value.UTF8String); }
std::string errorString(NSError* error) { return error == nil ? "unknown Metal error" : string(error.localizedDescription); }

MRBodyStateGPU initialState(const MRBodyPropertiesGPU& properties, std::uint32_t index) {
    MRBodyStateGPU state{};
    state.position = f4(float(index % 32u) * 3.0f, float(index / 32u) * 3.0f, 2.0f, 1.0f);
    state.orientation = f4(0, 0, 0, 1);
    state.linearVelocityAndInverseMass = f4(0, 0, 0, properties.massAndInverseMass.y);
    state.inverseInertiaWorldRow0 = properties.inverseInertiaRow0;
    state.inverseInertiaWorldRow1 = properties.inverseInertiaRow1;
    state.inverseInertiaWorldRow2 = properties.inverseInertiaRow2;
    state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = index;
    return state;
}

template <typename T>
id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values, NSString* label) {
    require(!values.empty(), "empty Metal buffer");
    id<MTLBuffer> result = [device newBufferWithBytes:values.data() length:values.size() * sizeof(T) options:MTLResourceStorageModeShared];
    require(result != nil, "failed to allocate " + string(label));
    result.label = label;
    return result;
}

template <typename T>
id<MTLBuffer> scalarBuffer(id<MTLDevice> device, const T& value, NSString* label) {
    const std::vector<T> values{value};
    return buffer(device, values, label);
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil, "missing Metal function " + string(name));
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, "failed pipeline " + string(name) + ": " + errorString(error));
    return result;
}

void encodeMulticopter(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe, id<MTLBuffer> rotors, id<MTLBuffer> model, id<MTLBuffer> motorState, id<MTLBuffer> commands, id<MTLBuffer> bodies, id<MTLBuffer> wrenches, id<MTLBuffer> dispatch, std::uint32_t count) {
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:rotors offset:0 atIndex:0]; [encoder setBuffer:model offset:0 atIndex:1];
    [encoder setBuffer:motorState offset:0 atIndex:2]; [encoder setBuffer:commands offset:0 atIndex:3];
    [encoder setBuffer:bodies offset:0 atIndex:4]; [encoder setBuffer:wrenches offset:0 atIndex:5];
    [encoder setBuffer:dispatch offset:0 atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void encodeFreeBody(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe, id<MTLBuffer> properties, id<MTLBuffer> bodies, id<MTLBuffer> wrenches, id<MTLBuffer> batch, id<MTLBuffer> statuses, std::uint32_t count) {
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:properties offset:0 atIndex:0]; [encoder setBuffer:bodies offset:0 atIndex:1];
    [encoder setBuffer:wrenches offset:0 atIndex:2]; [encoder setBuffer:batch offset:0 atIndex:3]; [encoder setBuffer:statuses offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
void encodeMixer(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe, id<MTLBuffer> rotors, id<MTLBuffer> model, id<MTLBuffer> actions, id<MTLBuffer> commands, id<MTLBuffer> mixer, id<MTLBuffer> dispatch, std::uint32_t count) {
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:rotors offset:0 atIndex:0]; [encoder setBuffer:model offset:0 atIndex:1];
    [encoder setBuffer:actions offset:0 atIndex:2]; [encoder setBuffer:commands offset:0 atIndex:3];
    [encoder setBuffer:mixer offset:0 atIndex:4]; [encoder setBuffer:dispatch offset:0 atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
void encodeTask(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe, id<MTLBuffer> model, id<MTLBuffer> motors, id<MTLBuffer> bodies, id<MTLBuffer> transitions, id<MTLBuffer> task, id<MTLBuffer> dispatch, std::uint32_t count) {
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:model offset:0 atIndex:0]; [encoder setBuffer:motors offset:0 atIndex:1]; [encoder setBuffer:bodies offset:0 atIndex:2];
    [encoder setBuffer:transitions offset:0 atIndex:3]; [encoder setBuffer:task offset:0 atIndex:4]; [encoder setBuffer:dispatch offset:0 atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
void encodeReset(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe, id<MTLBuffer> bodies, id<MTLBuffer> motors, id<MTLBuffer> transitions, id<MTLBuffer> resetBodies, id<MTLBuffer> resetMotors, id<MTLBuffer> dispatch, std::uint32_t count) {
    [encoder setComputePipelineState:pipe];
    [encoder setBuffer:bodies offset:0 atIndex:0]; [encoder setBuffer:motors offset:0 atIndex:1]; [encoder setBuffer:transitions offset:0 atIndex:2];
    [encoder setBuffer:resetBodies offset:0 atIndex:3]; [encoder setBuffer:resetMotors offset:0 atIndex:4]; [encoder setBuffer:dispatch offset:0 atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
mr_float4 requestedAction() {
    const char* text = std::getenv("PX4_X500_ACTION");
    if (text == nullptr || text[0] == '\0') return f4(0, 0, 0, 0);
    std::istringstream input(text); mr_float4 result{};
    if (!(input >> result.x >> result.y >> result.z >> result.w) || (input >> std::ws && !input.eof())) throw std::runtime_error("PX4_X500_ACTION must contain collective roll pitch yaw");
    return result;
}
} // namespace

int main() {
    @autoreleasepool {
        try {
            const auto model = metalrobo::makePX4X500MulticopterModel(kDt);
            const auto rotorArray = metalrobo::makePX4X500Rotors();
            const auto propertiesOne = metalrobo::makePX4X500BodyProperties();
            const float hover = std::sqrt(propertiesOne.massAndInverseMass.x * 9.81f / (4.0f * model.coefficients.x));
            require(hover < model.motorAndTimestep.z, "X500 source motor limit cannot hover the sourced mass");
            std::vector<MRBodyPropertiesGPU> properties(kEnvironments, propertiesOne);
            std::vector<MRBodyStateGPU> states; states.reserve(kEnvironments);
            for (std::uint32_t index = 0; index < kEnvironments; ++index) states.push_back(initialState(propertiesOne, index));
            std::vector<MRMulticopterStateGPU> motors(kEnvironments);
            for (auto& motor : motors) { motor.rotorSpeed01 = f4(hover, hover, hover, hover); }
            const auto resetStates = states;
            const auto resetMotors = motors;
            const bool autoReset = std::getenv("PX4_X500_AUTO_RESET") != nullptr;
            std::vector<float> commands(kEnvironments * 4u, hover);
            std::vector<MRMulticopterActionGPU> actions(kEnvironments, {requestedAction()});
            MRMulticopterMixerGPU mixer{}; mixer.hoverAndScales = f4(hover, 120.0f, 35.0f, 12.0f);
            MRMulticopterFlightTaskGPU task{}; task.targetPositionAndMinimumHeight = f4(0, 0, 2.0f, 0.20f); task.maximumHeightTiltAndScales = f4(8.0f, 0.90f, 0.45f, 2.0f);
            std::vector<MRMulticopterFlightTransitionGPU> transitions(kEnvironments);
            std::vector<MRBodyWrenchGPU> wrenches(kEnvironments);
            std::vector<MRFreeBodyStatusGPU> statuses(kEnvironments);
            std::vector<MRMulticopterRotorGPU> rotors(rotorArray.begin(), rotorArray.end());
            MRMulticopterStateGPU oracleMotors{};
            oracleMotors.rotorSpeed01 = f4(hover, hover, hover, hover);
            const std::array<float, MR_MULTICOPTER_MAX_ROTORS> oracleCommands{
                hover, hover, hover, hover, 0, 0, 0, 0,
            };
            const auto oracle = metalrobo::stepMulticopter(
                model, rotorArray, oracleMotors, oracleCommands, states.front()
            );
            require(oracle.succeeded(), "FP64 X500 actuator oracle rejected hover");
            MRMulticopterDispatchGPU multicopterDispatch{}; multicopterDispatch.environmentCount = kEnvironments; multicopterDispatch.bodyStride = 1u;
            MRFreeBodyBatchGPU freeBodyBatch{}; freeBodyBatch.bodyCount = kEnvironments; freeBodyBatch.integratorType = MR_FREE_BODY_IMPLICIT_MIDPOINT; freeBodyBatch.nonlinearIterations = 12u; freeBodyBatch.gravityAndTimestep = f4(0, 0, -9.81f, kDt); freeBodyBatch.convergence = f4(2.0e-6f, 0, 0);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue]; require(queue != nil, "no Metal queue");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:METALROBO_DEFAULT_METALLIB]] error:&error];
            require(library != nil, "failed metallib: " + errorString(error));
            const auto multicopterPipe = pipeline(device, library, @"mr_step_multicopters");
            const auto mixerPipe = pipeline(device, library, @"mr_mix_multicopter_actions");
            const auto taskPipe = pipeline(device, library, @"mr_evaluate_multicopter_flight_task");
            const auto resetPipe = pipeline(device, library, @"mr_reset_multicopter_flights");
            const auto freeBodyPipe = pipeline(device, library, @"mr_integrate_free_bodies");
            const auto rotorsBuffer = buffer(device, rotors, @"x500 rotors"); const auto modelBuffer = scalarBuffer(device, model, @"x500 model");
            const auto motorBuffer = buffer(device, motors, @"x500 motors"); const auto commandBuffer = buffer(device, commands, @"x500 commands");
            const auto resetBodyBuffer = buffer(device, resetStates, @"x500 task reset bodies"); const auto resetMotorBuffer = buffer(device, resetMotors, @"x500 task reset motors");
            const auto actionBuffer = buffer(device, actions, @"x500 policy actions"); const auto mixerBuffer = scalarBuffer(device, mixer, @"x500 mixer");
            const auto transitionBuffer = buffer(device, transitions, @"x500 flight transitions"); const auto taskBuffer = scalarBuffer(device, task, @"x500 flight task");
            const auto propertyBuffer = buffer(device, properties, @"x500 properties"); const auto bodyBuffer = buffer(device, states, @"x500 bodies");
            const auto wrenchBuffer = buffer(device, wrenches, @"x500 wrenches"); const auto statusBuffer = buffer(device, statuses, @"x500 statuses");
            const auto multicopterDispatchBuffer = scalarBuffer(device, multicopterDispatch, @"x500 actuator dispatch"); const auto freeBodyBatchBuffer = scalarBuffer(device, freeBodyBatch, @"x500 free body batch");
            id<MTLCommandBuffer> command = [queue commandBuffer]; require(command != nil, "no Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "no Metal encoder");
            for (std::uint32_t step = 0; step < kSteps; ++step) {
                encodeMixer(encoder, mixerPipe, rotorsBuffer, modelBuffer, actionBuffer, commandBuffer, mixerBuffer, multicopterDispatchBuffer, kEnvironments);
                encodeMulticopter(encoder, multicopterPipe, rotorsBuffer, modelBuffer, motorBuffer, commandBuffer, bodyBuffer, wrenchBuffer, multicopterDispatchBuffer, kEnvironments);
                encodeFreeBody(encoder, freeBodyPipe, propertyBuffer, bodyBuffer, wrenchBuffer, freeBodyBatchBuffer, statusBuffer, kEnvironments);
                encodeTask(encoder, taskPipe, modelBuffer, motorBuffer, bodyBuffer, transitionBuffer, taskBuffer, multicopterDispatchBuffer, kEnvironments);
                if (autoReset) encodeReset(encoder, resetPipe, bodyBuffer, motorBuffer, transitionBuffer, resetBodyBuffer, resetMotorBuffer, multicopterDispatchBuffer, kEnvironments);
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted, "X500 Metal command failed: " + errorString(command.error));
            const auto* finalStates = static_cast<const MRBodyStateGPU*>(bodyBuffer.contents);
            const auto* finalStatuses = static_cast<const MRFreeBodyStatusGPU*>(statusBuffer.contents);
            const auto* finalWrenches = static_cast<const MRBodyWrenchGPU*>(wrenchBuffer.contents);
            const auto* finalTransitions = static_cast<const MRMulticopterFlightTransitionGPU*>(transitionBuffer.contents);
            float maximumAltitudeError = 0.0f;
            for (std::uint32_t index = 0; index < kEnvironments; ++index) {
                require(finalStatuses[index].code == MR_STEP_SUCCESS, "X500 free-body step failed");
                maximumAltitudeError = std::max(maximumAltitudeError, std::abs(finalStates[index].position.z - 2.0f));
            }
            const auto action = actions.front().collectiveRollPitchYaw;
            const bool neutralAction = action.x == 0.0f && action.y == 0.0f && action.z == 0.0f && action.w == 0.0f;
            if (neutralAction) require(maximumAltitudeError < 2.0e-3f, "X500 source hover drift exceeds gate");
            const float wrenchParity = std::max({
                std::abs(finalWrenches[0].force.x - oracle.wrench.force.x),
                std::abs(finalWrenches[0].force.y - oracle.wrench.force.y),
                std::abs(finalWrenches[0].force.z - oracle.wrench.force.z),
                std::abs(finalWrenches[0].torque.x - oracle.wrench.torque.x),
                std::abs(finalWrenches[0].torque.y - oracle.wrench.torque.y),
                std::abs(finalWrenches[0].torque.z - oracle.wrench.torque.z),
            });
            if (neutralAction) require(wrenchParity < 2.0e-4f, "X500 CPU/Metal actuator wrench parity exceeded");
            const char* tracePath = std::getenv("PX4_X500_TRACE");
            if (tracePath != nullptr && tracePath[0] != '\0') {
                std::ofstream trace(tracePath);
                require(trace.good(), "cannot write X500 trace");
                trace << "time_s,x_m,y_m,z_m,qx,qy,qz,qw\n";
                MRBodyStateGPU traceState = initialState(propertiesOne, 0u);
                traceState.position.z = 0.60f;
                std::memcpy(bodyBuffer.contents, &traceState, sizeof(traceState));
                MRMulticopterStateGPU traceMotors{};
                traceMotors.rotorSpeed01 = f4(hover, hover, hover, hover);
                std::memcpy(motorBuffer.contents, &traceMotors, sizeof(traceMotors));
                auto* traceCommands = static_cast<float*>(commandBuffer.contents);
                for (std::uint32_t rotor = 0; rotor < 4u; ++rotor) traceCommands[rotor] = 810.0f;
                MRMulticopterDispatchGPU traceDispatch{};
                traceDispatch.environmentCount = 1u;
                traceDispatch.bodyStride = 1u;
                std::memcpy(multicopterDispatchBuffer.contents, &traceDispatch, sizeof(traceDispatch));
                MRFreeBodyBatchGPU traceBatch = freeBodyBatch;
                traceBatch.bodyCount = 1u;
                std::memcpy(freeBodyBatchBuffer.contents, &traceBatch, sizeof(traceBatch));
                for (std::uint32_t step = 0; step < kSteps; ++step) {
                    id<MTLCommandBuffer> traceCommand = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> traceEncoder = [traceCommand computeCommandEncoder];
                    require(traceCommand != nil && traceEncoder != nil, "cannot encode X500 trace step");
                    encodeMulticopter(traceEncoder, multicopterPipe, rotorsBuffer, modelBuffer, motorBuffer, commandBuffer, bodyBuffer, wrenchBuffer, multicopterDispatchBuffer, 1u);
                    encodeFreeBody(traceEncoder, freeBodyPipe, propertyBuffer, bodyBuffer, wrenchBuffer, freeBodyBatchBuffer, statusBuffer, 1u);
                    [traceEncoder endEncoding]; [traceCommand commit]; [traceCommand waitUntilCompleted];
                    require(traceCommand.status == MTLCommandBufferStatusCompleted, "X500 trace GPU step failed");
                    const auto* current = static_cast<const MRBodyStateGPU*>(bodyBuffer.contents);
                    const auto* status = static_cast<const MRFreeBodyStatusGPU*>(statusBuffer.contents);
                    require(status[0].code == MR_STEP_SUCCESS, "X500 trace physics step failed");
                    if (step % 10u == 0u) {
                        trace << std::fixed << std::setprecision(7) << (step + 1u) * kDt << ','
                              << current[0].position.x << ',' << current[0].position.y << ',' << current[0].position.z << ','
                              << current[0].orientation.x << ',' << current[0].orientation.y << ',' << current[0].orientation.z << ',' << current[0].orientation.w << '\n';
                    }
                }
                require(trace.good(), "X500 trace write failed");
            }
            std::cout << std::fixed << std::setprecision(7) << "robot=px4_x500 source_revision=e00d3b9cde682dbcb3bf6f30a2f2b8ef4325dae8 device=" << string(device.name) << " environments=" << kEnvironments << " steps=" << kSteps << " auto_reset=" << autoReset << " hover_rad_s=" << hover << " policy_action=" << action.x << ',' << action.y << ',' << action.z << ',' << action.w << " final_position_m=" << finalStates[0].position.x << ',' << finalStates[0].position.y << ',' << finalStates[0].position.z << " task_reward=" << finalTransitions[0].rewardAndDone.x << " task_done=" << finalTransitions[0].rewardAndDone.y << " task_tilt_rad=" << finalTransitions[0].rewardAndDone.z << " task_target_distance_m=" << finalTransitions[0].rewardAndDone.w << " maximum_altitude_error_m=" << maximumAltitudeError << " actuator_wrench_parity=" << wrenchParity << " failed_steps=0 status=pass\n";
            return 0;
        } catch (const std::exception& exception) { std::cerr << "metalrobo_px4_x500_gpu_probe: " << exception.what() << '\n'; return 1; }
    }
}
