#include "numi/matter/human_support_precision_gpu.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <initializer_list>

namespace {
using Vector = std::array<double, 3>;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
Vector xyz(mr_float4 v) { return {v.x, v.y, v.z}; }
Vector add(Vector a, Vector b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
Vector scale(Vector a, double k) { return {a[0]*k,a[1]*k,a[2]*k}; }
Vector cross(Vector a, Vector b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double dot(Vector a, Vector b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vector rotate(mr_float4 quaternion, Vector v) {
    const auto axis = xyz(quaternion);
    const auto t = scale(cross(axis, v), 2.0);
    return add(v, add(scale(t, quaternion.w), cross(axis, t)));
}
Vector physical(MRCompensatedPositionGPU p) { return add(xyz(p.high),xyz(p.low)); }
Vector offsetOracle(mr_float4 q, mr_float4 local, mr_float4 radii,
                    mr_float4 normal, unsigned kind) {
    // Identity support frame deliberately keeps the independent oracle free of
    // the provider's intermediate FP32 quaternion-composition expression.
    auto result = rotate(q, xyz(local));
    if (kind != 2) return add(result, scale(xyz(normal), -double(normal.w)));
    const auto direction = rotate({-q.x,-q.y,-q.z,q.w},xyz(normal));
    Vector scaled{}, surface{};
    const auto radius = xyz(radii);
    for (unsigned i=0;i<3;++i) scaled[i]=radius[i]*direction[i];
    const double magnitude = std::sqrt(dot(scaled,scaled));
    for (unsigned i=0;i<3;++i) surface[i]=radius[i]*scaled[i]/magnitude;
    return add(result,scale(rotate(q,surface),-1.0));
}
}

int main() {
    try {
        const std::array<mr_float4,3> rotations{{
            {0,0,0,1}, {0.26726124f,0.53452248f,0,0.80178374f},
            {0.1f,0.2f,0.3f,0.92736197f}}};
        const std::array<mr_float4,2> normals{{{0,0,1,0},{0.6f,0,0.8f,0}}};
        const mr_float4 local{0.131f,-0.043f,-0.077f,0};
        const mr_float4 radii{0.12f,0.025f,0.09f,0};
        const mr_float4 supportRotation{0,0,0,1};
        const mr_float4 ground{1.0f,-2.0f,0.875f,0};
        unsigned cases=0, checks=0, discriminatingProjectionControls=0;
        double maximumOffsetError=0, maximumSeparationError=0, maximumSecantError=0;
        for (auto rotation : rotations) for (auto normal : normals) for(unsigned kind=0;kind<3;++kind) {
            auto normalAndRadius=normal;
            normalAndRadius.w=kind==1 ? 0.017f : 0.0f;
            if (kind==0) normalAndRadius={0,0,0,0};
            const auto offset=mrCompensatedSupportOffset(rotation,local,supportRotation,
                radii,normalAndRadius,kind);
            const auto expectedOffset=offsetOracle(rotation,local,radii,normalAndRadius,kind);
            const auto actualOffset=physical(offset);
            for(unsigned axis=0;axis<3;++axis) {
                maximumOffsetError=std::max(maximumOffsetError,std::abs(actualOffset[axis]-expectedOffset[axis]));
                require(std::abs(actualOffset[axis]-expectedOffset[axis])<2e-13,
                    "paired support offset differs from independent FP64 source geometry");
                ++checks;
            }
            const auto body=mrCompensatedVectorAdd(mrCompensatedVector(ground),mrCompensatedVectorNegate(offset));
            const auto initial=nmHumanSupportPairedSeparation(body,offset,ground,normal);
            double expectedSeparation=0;
            for(unsigned axis=0;axis<3;++axis) {
                // Cancel the authored reference before adding small terms in
                // FP64; summing large world positions first weakens the oracle.
                expectedSeparation+=((xyz(body.high)[axis]-xyz(ground)[axis])+
                    xyz(body.low)[axis]+expectedOffset[axis])*xyz(normal)[axis];
            }
            const double separation=double(initial.high)+initial.low;
            maximumSeparationError=std::max(maximumSeparationError,std::abs(separation-expectedSeparation));
            require(std::abs(separation-expectedSeparation)<2e-13,
                "paired plane separation discarded an offset or body residual");
            ++checks;
            for (float sign : {-1.0f,1.0f}) {
                constexpr float dt=0.000025f;
                const mr_float4 velocity{sign*0x1p-18f,-sign*0x1p-20f,sign*0x1p-17f,0};
                const auto delta=mrCompensatedVectorPack(mrCompensatedProduct(dt,velocity.x),
                    mrCompensatedProduct(dt,velocity.y),mrCompensatedProduct(dt,velocity.z));
                const auto moved=mrCompensatedVectorAdd(body,delta);
                const auto final=nmHumanSupportPairedSeparation(moved,offset,ground,normal);
                // Exercise the exact FP32 scalar consumed by contact assembly.
                const double observed=(double(final.high)-initial.high)/dt;
                const double expected=dot(xyz(velocity),xyz(normal));
                maximumSecantError=std::max(maximumSecantError,std::abs(observed-expected));
                require(std::abs(observed-expected)<2e-10,
                    "contact normal secant disagrees with translation Jacobian action");
                const auto rounded=mrCompensatedVectorAdd(moved,offset).high;
                const double roundedGap=dot(add(xyz(rounded),scale(xyz(ground),-1)),xyz(normal));
                if (std::abs(roundedGap/dt-expected)>2e-10) ++discriminatingProjectionControls;
                ++checks;
            }
            ++cases;
        }
        require(discriminatingProjectionControls>0,"oracle does not distinguish early world-position projection");
        std::printf("{\"mode\":\"cpu_support_precision_oracle\",\"passed\":true,\"cases\":%u,\"checks\":%u,\"early_projection_failures\":%u,\"maximum_offset_error_m\":%.17g,\"maximum_separation_error_m\":%.17g,\"maximum_secant_error_mps\":%.17g}\n",
            cases,checks,discriminatingProjectionControls,maximumOffsetError,maximumSeparationError,maximumSecantError);
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"support precision check failed: %s\n",error.what());return 1;
    }
}
