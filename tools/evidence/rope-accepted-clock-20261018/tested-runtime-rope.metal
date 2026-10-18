#include <metal_stdlib>
using namespace metal;
#define MR_ALIGN16 alignas(16)
using mr_u32=uint;using mr_uint4=uint4;using mr_float4=float4;

typedef struct MR_ALIGN16 MRBodyStateGPU {
    // xyz = COM position in world coordinates.
    mr_float4 position;
    // Normalizable quaternion (x, y, z, w), body-to-world. Collision
    // canonicalizes it under the shared component-domain contract.
    mr_float4 orientation;
    // xyz = world linear velocity at COM; w = inverse mass.
    mr_float4 linearVelocityAndInverseMass;
    // xyz = world angular velocity.
    mr_float4 angularVelocity;
    // Inverse inertia about COM, already rotated into world coordinates.
    mr_float4 inverseInertiaWorldRow0;
    mr_float4 inverseInertiaWorldRow1;
    mr_float4 inverseInertiaWorldRow2;
    // x = MRMotionType, y = articulation, z = link, w = MRBodyStateFlags.
    mr_u32 flagsAndIndices[4];
} MRBodyStateGPU;
typedef struct MR_ALIGN16 MRVisualVertexGPUV2 {
    // xyz position in the primitive's local frame, w homogeneous one.
    mr_float4 position;
    // xyz unit normal, w tangent handedness (+1 or -1).
    mr_float4 normalAndTangentSign;
    // xyz unit tangent, w reserved.
    mr_float4 tangent;
    // UV0.xy and UV1.xy.
    mr_float4 texcoord01;
    // Linear RGBA vertex color.
    mr_float4 color;
} MRVisualVertexGPUV2;
typedef struct MR_ALIGN16 MRHumanRestingRopeVisualGPU {
    // First world-bound mesh vertex, rope rings, sides per ring, harness ring start.
    mr_uint4 ranges;
    // Rope vertex count, harness major segments, minor segments, harness vertex count.
    mr_uint4 counts;
    // Rope radius, harness major radius, harness tube radius, visual-only slack sag cap.
    mr_float4 radiiAndSag;
} MRHumanRestingRopeVisualGPU;
typedef struct MR_ALIGN16 MRArticulatedBodyPoseGPU {
    // xyz = body COM world position.
    mr_float4 position;
    // Normalized body-to-world quaternion xyzw.
    mr_float4 orientation;
} MRArticulatedBodyPoseGPU;
typedef struct MR_ALIGN16 MRArticulatedPointWorldGPU {
    // xyz = queried point in world coordinates.
    mr_float4 position;
} MRArticulatedPointWorldGPU;
typedef struct MR_ALIGN16 MRNumiHumanStandContactGPU {
    mr_u32 bodyIndex;
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;

    // x = Coulomb friction, y = source activation and warm-start slop
    // metres (operator-level speculative admission may be wider),
    // z = normal stabilization fraction, w = source static normal
    // support force in N. A zero force retains cold-start contact.
    mr_float4 frictionSlopAndStabilization;

    // Dynamic fixed-world support plane written by the registered support
    // geometry callback before the stand solve. xyz is the plane point in m
    // and the unit normal; w is reserved and must remain zero. In global
    // plane mode these appended lanes are ignored.
    mr_float4 planePoint;
    mr_float4 planeNormal;
} MRNumiHumanStandContactGPU;
typedef struct MR_ALIGN16 MRHumanRestingRopeGPU {
    mr_uint4 indices; // contact, query, global body, articulation first body
    mr_uint4 strides; // body pose, first body probe, point world, point Jacobian
    mr_uint4 counts; // environments, bodies, DOFs, ABI=1
    mr_float4 anchorAndLength; // fixed world anchor xyz, maximum length metres
    mr_float4 bodyLocalSite; // COM-relative body-local harness point xyz, w=0
} MRHumanRestingRopeGPU;
inline float3 nmRopeSwingOffset(float4 motion) {
    const float u=clamp(motion.x/motion.w,0.0f,1.0f);
    const float ramp=u*u*u*(10.0f+u*(-15.0f+6.0f*u));
    const float phase=6.28318530718f*motion.x/motion.z;
    return motion.y*ramp*float3(sin(phase),0.5f*(cos(phase)-1.0f),0.0f);
}
// BEGIN resting rope support
// A moving tangent plane of the sphere |p-anchor|<=L supplies exactly the
// unilateral gap L-|p-anchor| and its gradient. The existing articulated
// contact solver owns all impulses; mu=0 admits no artificial tangential load.
kernel void nm_human_resting_rope_support(
    constant MRHumanRestingRopeGPU& cfg [[buffer(0)]],
    device const MRArticulatedBodyPoseGPU* bodies [[buffer(1)]],
    device const float4* bodyLow [[buffer(2)]],
    device MRArticulatedPointWorldGPU* points [[buffer(3)]],
    device float4* pointLow [[buffer(4)]],
    device float* jacobians [[buffer(5)]],
    device MRNumiHumanStandContactGPU* contacts [[buffer(6)]],
    constant float4& motion [[buffer(7)]],
    uint env [[thread_position_in_grid]]) {
    if (env>=cfg.counts.x) return;
    // ABI2 per-contact planes are currently restricted to one environment.
    const uint body=cfg.indices.z-cfg.indices.w;
    const uint query=cfg.indices.y;
    const uint pointIndex=env*cfg.strides.z+query;
    const uint poseIndex=env*cfg.strides.x+body;
    const uint dofs=cfg.counts.z;
    const uint probe=cfg.strides.y+4u*body;
    const auto pose=bodies[poseIndex];
    const auto contact=contacts[cfg.indices.x];
    const float qnorm=dot(pose.orientation,pose.orientation);
    bool valid=cfg.counts.w==1u && cfg.counts.x==1u &&
        cfg.indices.z>=cfg.indices.w && body<cfg.counts.y &&
        cfg.anchorAndLength.w>0.0f && all(isfinite(cfg.anchorAndLength)) &&
        all(isfinite(cfg.bodyLocalSite)) && cfg.bodyLocalSite.w==0.0f &&
        all(isfinite(pose.position)) && all(isfinite(pose.orientation)) &&
        all(isfinite(bodyLow[poseIndex])) && qnorm>1e-12f &&
        contact.bodyIndex==cfg.indices.z && contact.pointQueryIndex==query &&
        contact.frictionSlopAndStabilization.x==0.0f &&
        contact.frictionSlopAndStabilization.w==0.0f;
    const float4 q=pose.orientation*rsqrt(qnorm);
    const float3 local=cfg.bodyLocalSite.xyz;
    const float3 offset=local+2.0f*cross(q.xyz,cross(q.xyz,local)+q.w*local);
    // Error-free two-sum for body COM plus rigid site, retaining body low bits.
    const float3 high=pose.position.xyz+offset;
    const float3 virtualOffset=high-pose.position.xyz;
    const float3 low=(pose.position.xyz-(high-virtualOffset))+
        (offset-virtualOffset)+bodyLow[poseIndex].xyz;
    const float3 drivenAnchor=cfg.anchorAndLength.xyz+nmRopeSwingOffset(motion);
    const float3 toward=(drivenAnchor-high)-low;
    const float radius=length(toward);
    valid=valid && all(isfinite(high)) && all(isfinite(low)) && radius>1e-8f;
    const float3 normal=toward/radius;
    const float3 plane=drivenAnchor-cfg.anchorAndLength.w*normal;
    points[pointIndex].position=valid?float4(high,1):float4(NAN,NAN,NAN,1);
    pointLow[pointIndex]=valid?float4(low,0):float4(NAN,NAN,NAN,0);
    contacts[cfg.indices.x].planePoint=valid?float4(plane,0):float4(NAN);
    contacts[cfg.indices.x].planeNormal=valid?float4(normal,0):float4(NAN);
    for(uint axis=0;axis<3u;++axis) for(uint dof=0;dof<dofs;++dof) {
        const uint base=env*cfg.strides.w;
        const float j0=jacobians[base+(3u*probe+axis)*dofs+dof];
        const float jx=jacobians[base+(3u*(probe+1u)+axis)*dofs+dof];
        const float jy=jacobians[base+(3u*(probe+2u)+axis)*dofs+dof];
        const float jz=jacobians[base+(3u*(probe+3u)+axis)*dofs+dof];
        const float value=j0+local.x*(jx-j0)+local.y*(jy-j0)+local.z*(jz-j0);
        jacobians[base+(3u*query+axis)*dofs+dof]=valid?value:NAN;
    }
}

inline float3 nmRestingRopeRotate(float4 q, float3 v) {
    return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

inline void nmRestingRopeFrame(float3 tangent, thread float3& u, thread float3& v) {
    const float3 reference = fabs(tangent.z) < 0.88f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(1.0f, 0.0f, 0.0f);
    u = normalize(cross(reference, tangent));
    v = normalize(cross(tangent, u));
}

kernel void nm_human_resting_rope_visual(
    constant MRHumanRestingRopeGPU& rope [[buffer(0)]],
    constant MRHumanRestingRopeVisualGPU& visual [[buffer(1)]],
    device const MRBodyStateGPU* bodies [[buffer(2)]],
    device MRVisualVertexGPUV2* vertices [[buffer(3)]],
    constant uint4& acceptedStatus [[buffer(4)]],
    constant float4& motionParameters [[buffer(5)]],
    uint vertexIndex [[thread_position_in_grid]]) {
    float4 motion=motionParameters;
    motion.x=float(acceptedStatus.x)*motionParameters.x;
    const uint ropeVertexCount = visual.counts.x;
    const uint harnessVertexCount = visual.counts.w;
    if (vertexIndex >= ropeVertexCount + harnessVertexCount + 164u) return;
    const uint bodyIndex = rope.indices.z;
    const MRBodyStateGPU body = bodies[bodyIndex];
    const float q2 = dot(body.orientation, body.orientation);
    if (!(q2 > 1.0e-12f) || !isfinite(q2) ||
        !all(isfinite(body.position)) || !all(isfinite(body.orientation)) ||
        !all(isfinite(rope.anchorAndLength)) ||
        !(rope.anchorAndLength.w > 0.0f)) return;

    const float4 q = body.orientation * rsqrt(q2);
    const float3 anchor = rope.anchorAndLength.xyz+nmRopeSwingOffset(motion);
    const float3 endpoint = body.position.xyz +
        nmRestingRopeRotate(q, rope.bodyLocalSite.xyz);
    if(vertexIndex >= ropeVertexCount + harnessVertexCount) {
        const uint hardware=vertexIndex-ropeVertexCount-harnessVertexCount;
        auto out=vertices[visual.ranges.x+vertexIndex];
        float3 p;
        if(hardware<160u) {
            const float theta=6.28318530718f*float(hardware/8u)/20.0f;
            const float phi=6.28318530718f*float(hardware%8u)/8.0f;
            const float r=.018f+.003f*cos(phi);
            p=anchor+float3(r*cos(theta),r*sin(theta),.003f*sin(phi));
        } else {
            const uint k=hardware-160u;
            p=anchor+float3((k==0u||k==3u)?-.07f:.07f,k<2u?-.07f:.07f,.012f);
        }
        out.position=float4(p,1);vertices[visual.ranges.x+vertexIndex]=out;return;
    }
    const float3 chord = endpoint - anchor;
    const float chordLength = length(chord);
    if (!(chordLength > 1.0e-7f) || !isfinite(chordLength)) return;
    const float3 chordDirection = chord / chordLength;
    float3 sagDirection = float3(0.0f, 0.0f, -1.0f) -
        chordDirection * dot(float3(0.0f, 0.0f, -1.0f), chordDirection);
    if (dot(sagDirection, sagDirection) < 1.0e-8f) {
        sagDirection = cross(chordDirection, float3(1.0f, 0.0f, 0.0f));
        if (dot(sagDirection, sagDirection) < 1.0e-8f)
            sagDirection = cross(chordDirection, float3(0.0f, 1.0f, 0.0f));
    }
    sagDirection = normalize(sagDirection);
    const float sag = min(visual.radiiAndSag.w,
        0.35f * sqrt(max(rope.anchorAndLength.w * rope.anchorAndLength.w -
                          chordLength * chordLength, 0.0f)));

    if (vertexIndex < ropeVertexCount) {
        const uint sides = visual.ranges.z;
        if (sides < 3u || visual.ranges.y < 2u) return;
        const uint ring = vertexIndex / sides;
        const uint side = vertexIndex % sides;
        const float t = float(ring) / float(visual.ranges.y - 1u);
        const float theta = 6.283185307179586f * float(side) / float(sides);
        const float along = 4.0f * sag * t * (1.0f - t);
        const float3 center = anchor + chord * t + sagDirection * along;
        const float3 derivative = chord + sagDirection *
            (4.0f * sag * (1.0f - 2.0f * t));
        const float derivativeLength = length(derivative);
        if (!(derivativeLength > 1.0e-8f)) return;
        const float3 tangent = derivative / derivativeLength;
        float3 u, v;
        nmRestingRopeFrame(tangent, u, v);
        const float3 radial = cos(theta) * u + sin(theta) * v;
        auto out = vertices[visual.ranges.x + vertexIndex];
        out.position = float4(center + visual.radiiAndSag.x * radial, 1.0f);
        out.normalAndTangentSign = float4(radial, 1.0f);
        out.tangent = float4(tangent, 0.0f);
        vertices[visual.ranges.x + vertexIndex] = out;
        return;
    }

    const uint local = vertexIndex - ropeVertexCount;
    const uint majorSegments = visual.counts.y;
    const uint minorSegments = visual.counts.z;
    if (majorSegments < 3u || minorSegments < 3u ||
        local >= majorSegments * minorSegments) return;
    float3 u, v;
    nmRestingRopeFrame(chordDirection, u, v);
    const uint major = local / minorSegments;
    const uint minor = local % minorSegments;
    const float theta = 6.283185307179586f * float(major) / float(majorSegments);
    const float phi = 6.283185307179586f * float(minor) / float(minorSegments);
    const float3 majorDirection = cos(theta) * u + sin(theta) * v;
    const float3 around = -sin(theta) * u + cos(theta) * v;
    const float3 normal = cos(phi) * majorDirection + sin(phi) * chordDirection;
    const float3 position = endpoint +
        (visual.radiiAndSag.y + visual.radiiAndSag.z * cos(phi)) * majorDirection +
        visual.radiiAndSag.z * sin(phi) * chordDirection;
    const uint outIndex = visual.ranges.w + local;
    auto out = vertices[outIndex];
    out.position = float4(position, 1.0f);
    out.normalAndTangentSign = float4(normal, 1.0f);
    out.tangent = float4(around, 0.0f);
    vertices[outIndex] = out;
}

// An authored ideal separation interface. Six unilateral rows implement three
// bilateral relative-position constraints before the release step. Every row
// uses J_B-J_A, so these are internal reactions, never fixed-world support.
// After release all six rows have a strictly inactive gap. Inputs are regenerated
// from accepted pose and authoritative step; no independent event history exists.
struct MRHumanSeparationGPU {
    uint4 bodies; // A global, B global, first body, first body probe
    uint4 strides; // body poses, point world, point Jacobians, DOFs
    float4 parentAnchor; // parent-local interface relative to COM
    uint4 event; // release step (first released attempt), current step, ABI=1, 0
    uint4 rows[6]; // contact, dedicated point query, axis, negative-normal flag
};
kernel void nm_human_separation_constraints(
    constant MRHumanSeparationGPU& cfg [[buffer(0)]],
    device const MRArticulatedBodyPoseGPU* bodies [[buffer(1)]],
    device const float4* bodyLow [[buffer(2)]],
    device MRArticulatedPointWorldGPU* points [[buffer(3)]],
    device float4* pointLow [[buffer(4)]],
    device float* jacobians [[buffer(5)]],
    device MRNumiHumanStandContactGPU* contacts [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row>=6) return;
    uint a=cfg.bodies.x-cfg.bodies.z, b=cfg.bodies.y-cfg.bodies.z;
    uint4 entry=cfg.rows[row];
    float4 qa=bodies[a].orientation;
    float3 anchor=cfg.parentAnchor.xyz;
    float3 offset=anchor+2.f*cross(qa.xyz,cross(qa.xyz,anchor)+qa.w*anchor);
    float3 delta=bodies[b].position.xyz-bodies[a].position.xyz;
    float3 low=((bodies[b].position.xyz-delta)-bodies[a].position.xyz)
        +bodyLow[b].xyz-bodyLow[a].xyz-offset;
    float3 normal=float3(0);
    normal[entry.z]=entry.w==0?1.0f:-1.0f;
    bool valid=cfg.event.z==1 && cfg.event.w==0 &&
        all(isfinite(delta)) && all(isfinite(low)) &&
        contacts[entry.x].pointQueryIndex==entry.y &&
        contacts[entry.x].frictionSlopAndStabilization.x==0;
    points[entry.y].position=valid?float4(delta,1):float4(NAN);
    pointLow[entry.y]=valid?float4(low,0):float4(NAN);
    bool released=cfg.event.y>=cfg.event.x;
    // A remote inactive plane avoids division by a zero contact response.
    contacts[entry.x].planePoint=valid?
        float4(released?delta+low-1.e6f*normal:float3(0),0):float4(NAN);
    contacts[entry.x].planeNormal=valid?float4(normal,0):float4(NAN);
    for(uint axis=0;axis<3;++axis)for(uint dof=0;dof<cfg.strides.w;++dof) {
        uint pa=cfg.bodies.w+4*a, pb=cfg.bodies.w+4*b;
        float ja=jacobians[(3*pa+axis)*cfg.strides.w+dof];
        for(uint component=0;component<3;++component)
            ja+=cfg.parentAnchor[component]*(jacobians[(3*(pa+1+component)+axis)*cfg.strides.w+dof]
                -jacobians[(3*pa+axis)*cfg.strides.w+dof]);
        float jb=jacobians[(3*pb+axis)*cfg.strides.w+dof];
        jacobians[(3*entry.y+axis)*cfg.strides.w+dof]=valid?jb-ja:NAN;
    }
}

// BEGIN NumiHumanSeparationForce kernel abi_header_sha256=1ab91d0618a17981e598d1b16774efc94c05b53d3800d00da67437fa93318058 kernel_source_sha256=dcc2887a8f762d640fda700f048c637e8d27815275e91df8220e6fb8147c9a83
typedef struct MR_ALIGN16 MRNumiHumanTendonBindingGPU {
    mr_u32 muscleIndex;
    mr_u32 endpointOrdinal;
    mr_u32 bodyIndex;
    mr_u32 mode;
    mr_u32 envelopeIndex;
    mr_u32 boneStableId;
    mr_u32 reserved0;
    mr_u32 reserved1;
    // Authored MyoSim endpoint in COM-relative Core body coordinates.
    mr_float4 sourceLocalPoint;
} MRNumiHumanTendonBindingGPU;

typedef struct MR_ALIGN16 MRNumiHumanTendonTransferResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 bindingIndex;
    mr_u32 envelopeIndex;
    mr_float4 terminalWorldForce;
    mr_float4 nodalWorldForces[4];
    // x force residual N; y source-point moment residual Nm;
    // z maximum absolute generalized correction; w represented actuator force.
    mr_float4 residualsAndForce;
} MRNumiHumanTendonTransferResultGPU;

// Include this source fragment after metal_stdlib and
// metalrobo/numi_human_tendon_gpu.h in the source-built Human Metal library.
// The owning callback binds the exact source library once and supplies the
// explicit crossing-muscle set as a resident buffer.

kernel void mr_numi_human_separation_release_crossing_forces(
    constant uint4& shape [[buffer(0)]],
    constant uint4& forceLayout [[buffer(1)]],
    constant uint& crossingMuscleCount [[buffer(2)]],
    device const uint* crossingMuscleIndices [[buffer(3)]],
    device const MRNumiHumanTendonBindingGPU* bindings [[buffer(4)]],
    device MRNumiHumanTendonTransferResultGPU* transfers [[buffer(5)]],
    device float* generalizedCorrections [[buffer(6)]],
    device float* generalizedForces [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    const uint environmentCount = shape.x;
    const uint muscleCount = shape.y;
    const uint endpointCount = shape.z;
    const uint dofCount = shape.w;
    const uint generalizedForceOffset = forceLayout.x;
    const uint generalizedForceStride = forceLayout.y;
    const uint stepIndex = forceLayout.z;
    const uint releaseStep = forceLayout.w;
    const uint forceWorkCount = environmentCount * dofCount;
    const uint endpointWorkCount = environmentCount * endpointCount;
    const uint workCount = max(forceWorkCount, endpointWorkCount);
    if (index >= workCount || stepIndex < releaseStep ||
        environmentCount == 0u || muscleCount == 0u || endpointCount == 0u ||
        dofCount == 0u || generalizedForceStride != dofCount ||
        crossingMuscleCount == 0u) {
        return;
    }

    // Each environment/DoF lane removes the declared source rows exactly once
    // from the existing reduction, then clears their provenance rows. No new
    // force is accumulated, and unrelated muscle rows are left untouched.
    if (index < forceWorkCount) {
        const uint environment = index / dofCount;
        const uint dof = index - environment * dofCount;
        float releasedForce = 0.0f;
        for (uint crossing = 0u; crossing < crossingMuscleCount; ++crossing) {
            const uint muscle = crossingMuscleIndices[crossing];
            if (muscle >= muscleCount) continue;
            const uint muscleRow =
                (environment * muscleCount + muscle) * dofCount + dof;
            const float sourceForce = generalizedForces[muscleRow];
            generalizedForces[muscleRow] = 0.0f;
            releasedForce += sourceForce;
        }
        const uint reducedRow = environment * generalizedForceStride +
            generalizedForceOffset + dof;
        generalizedForces[reducedRow] -= releasedForce;
    }

    // Keep transfer status and residual diagnostics as authored by the tendon
    // solver. Remove only the released endpoint wrench and its generalized
    // correction so consumers cannot apply the cut muscle's terminal load.
    if (index < endpointWorkCount) {
        const uint environment = index / endpointCount;
        const uint endpoint = index - environment * endpointCount;
        const MRNumiHumanTendonBindingGPU binding = bindings[endpoint];
        bool crossingMuscle = false;
        for (uint crossing = 0u; crossing < crossingMuscleCount; ++crossing) {
            crossingMuscle = crossingMuscle ||
                crossingMuscleIndices[crossing] == binding.muscleIndex;
        }
        if (!crossingMuscle) return;
        const uint transferIndex = environment * endpointCount + endpoint;
        MRNumiHumanTendonTransferResultGPU transfer = transfers[transferIndex];
        transfer.terminalWorldForce = float4(0.0f);
        transfer.residualsAndForce.w = 0.0f;
        for (uint node = 0u; node < 4u; ++node)
            transfer.nodalWorldForces[node] = float4(0.0f);
        transfers[transferIndex] = transfer;
        const uint correctionBase = transferIndex * dofCount;
        for (uint dof = 0u; dof < dofCount; ++dof)
            generalizedCorrections[correctionBase + dof] = 0.0f;
    }
}
// END NumiHumanSeparationForce kernel
