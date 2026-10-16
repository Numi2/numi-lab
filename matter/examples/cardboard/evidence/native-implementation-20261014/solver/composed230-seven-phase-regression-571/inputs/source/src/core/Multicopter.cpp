#include "metalrobo/Multicopter.hpp"

#include <algorithm>
#include <cmath>

namespace metalrobo {
namespace {

struct Vec3 { double x; double y; double z; };
[[nodiscard]] Vec3 xyz(const mr_float4 value) { return {value.x, value.y, value.z}; }
[[nodiscard]] mr_float4 f4(const Vec3 value) { return {float(value.x), float(value.y), float(value.z), 0.0f}; }
[[nodiscard]] Vec3 operator+(const Vec3 a, const Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
[[nodiscard]] Vec3 operator-(const Vec3 a, const Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
[[nodiscard]] Vec3 operator*(const Vec3 a, const double b) { return {a.x * b, a.y * b, a.z * b}; }
[[nodiscard]] double dot(const Vec3 a, const Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
[[nodiscard]] Vec3 cross(const Vec3 a, const Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
[[nodiscard]] bool finite(const Vec3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }

struct Mat3 { double m[3][3]; };
[[nodiscard]] Mat3 rotation(const mr_float4 q) {
    const double length = std::sqrt(double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w);
    if (!(length > 1.0e-12) || !std::isfinite(length)) return {};
    const double x=q.x/length, y=q.y/length, z=q.z/length, w=q.w/length;
    return {{{1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)}, {2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)}, {2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)}}};
}
[[nodiscard]] Vec3 multiply(const Mat3& m, const Vec3 v) { return {m.m[0][0]*v.x+m.m[0][1]*v.y+m.m[0][2]*v.z, m.m[1][0]*v.x+m.m[1][1]*v.y+m.m[1][2]*v.z, m.m[2][0]*v.x+m.m[2][1]*v.y+m.m[2][2]*v.z}; }

float& speed(MRMulticopterStateGPU& state, const std::size_t index) { return index < 4u ? (&state.rotorSpeed01.x)[index] : (&state.rotorSpeed45.x)[index - 4u]; }

bool valid(const MRMulticopterModelGPU& model) {
    return model.rotorCount > 0u && model.rotorCount <= MR_MULTICOPTER_MAX_ROTORS &&
        std::isfinite(model.coefficients.x) && model.coefficients.x > 0.0f &&
        std::isfinite(model.coefficients.y) && model.coefficients.y >= 0.0f &&
        std::isfinite(model.coefficients.z) && model.coefficients.z >= 0.0f &&
        std::isfinite(model.coefficients.w) && model.coefficients.w >= 0.0f &&
        std::isfinite(model.motorAndTimestep.x) && model.motorAndTimestep.x > 0.0f &&
        std::isfinite(model.motorAndTimestep.y) && model.motorAndTimestep.y > 0.0f &&
        std::isfinite(model.motorAndTimestep.z) && model.motorAndTimestep.z > 0.0f &&
        std::isfinite(model.motorAndTimestep.w) && model.motorAndTimestep.w > 0.0f;
}
} // namespace

MulticopterStepResult stepMulticopter(
    const MRMulticopterModelGPU& model,
    const std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS>& rotors,
    MRMulticopterStateGPU& state,
    const std::array<float, MR_MULTICOPTER_MAX_ROTORS>& commands,
    const MRBodyStateGPU& body,
    const mr_float4 windVelocity
) {
    MulticopterStepResult result;
    if (!valid(model)) { result.status = MulticopterStatus::invalidModel; return result; }
    const Mat3 bodyToWorld = rotation(body.orientation);
    const double q2 = double(body.orientation.x)*body.orientation.x + double(body.orientation.y)*body.orientation.y + double(body.orientation.z)*body.orientation.z + double(body.orientation.w)*body.orientation.w;
    if (!(q2 > 1.0e-24) || !std::isfinite(q2) || !finite(xyz(windVelocity)) ||
        !finite(xyz(body.linearVelocityAndInverseMass))) { result.status = MulticopterStatus::invalidInput; return result; }
    MRMulticopterStateGPU candidate = state;
    Vec3 forceBody{};
    Vec3 torqueBody{};
    for (std::size_t index = 0; index < model.rotorCount; ++index) {
        if (!std::isfinite(commands[index]) || !std::isfinite(speed(state, index)) || speed(state, index) < 0.0f || !finite(xyz(rotors[index].positionAndReactionSign)) || !std::isfinite(rotors[index].positionAndReactionSign.w)) { result.status = MulticopterStatus::invalidInput; return result; }
        const float target = std::clamp(commands[index], 0.0f, model.motorAndTimestep.z);
        const float tau = target > speed(state, index) ? model.motorAndTimestep.x : model.motorAndTimestep.y;
        speed(candidate, index) += -std::expm1(-model.motorAndTimestep.w / tau) * (target - speed(candidate, index));
        const double squared = double(speed(candidate, index)) * speed(candidate, index);
        const Vec3 thrust{0.0, 0.0, model.coefficients.x * squared};
        forceBody = forceBody + thrust;
        torqueBody = torqueBody + cross(xyz(rotors[index].positionAndReactionSign), thrust);
        torqueBody.z += rotors[index].positionAndReactionSign.w * model.coefficients.x * model.coefficients.y * squared;
    }
    const Vec3 axisWorld = multiply(bodyToWorld, {0.0, 0.0, 1.0});
    const Vec3 relativeVelocity = xyz(body.linearVelocityAndInverseMass) - xyz(windVelocity);
    const Vec3 perpendicularVelocity = relativeVelocity - axisWorld * dot(relativeVelocity, axisWorld);
    double rotorSpeedSum = 0.0;
    for (std::size_t index = 0; index < model.rotorCount; ++index) rotorSpeedSum += std::abs(speed(candidate, index));
    const Vec3 rotorDrag = perpendicularVelocity * (-rotorSpeedSum * model.coefficients.z);
    const Vec3 rollingMoment = perpendicularVelocity * (-rotorSpeedSum * model.coefficients.w);
    const Vec3 forceWorld = multiply(bodyToWorld, forceBody) + rotorDrag;
    const Vec3 torqueWorld = multiply(bodyToWorld, torqueBody) + rollingMoment;
    if (!finite(forceWorld) || !finite(torqueWorld)) { result.status = MulticopterStatus::invalidInput; return result; }
    state = candidate;
    result.wrench.force = f4(forceWorld);
    result.wrench.torque = f4(torqueWorld);
    return result;
}
} // namespace metalrobo
