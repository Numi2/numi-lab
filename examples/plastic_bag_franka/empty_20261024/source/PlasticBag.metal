#include "franka_grip_gpu.h"
#include <metal_stdlib>
#include "plastic_bag_gpu.h"

using namespace metal;

#define PB_COIN_PAIR_SAMPLE_COUNT 66u
#define PB_SEAM_RESTITUTION 0.03f

// Maximum-dissipation projection in the contact effective-mass metric.
// Euclidean radial clipping of an unconstrained 2D impulse can add energy
// when rotational inertia makes K anisotropic. Solve its KKT multiplier.
inline float2 pb_friction_disk(const float kxx,const float kzz,const float kxz,
 const float2 old,const float2 velocity,const float radius){
 if(radius<=0.0f)return float2(0);
 const float2 rhs=float2(kxx*old.x+kxz*old.y,kxz*old.x+kzz*old.y)-velocity;
 const float det=max(fma(kxx,kzz,-kxz*kxz),1.e-20f);
 const float2 freeImpulse=float2(kzz*rhs.x-kxz*rhs.y,kxx*rhs.y-kxz*rhs.x)/det;
 if(dot(freeImpulse,freeImpulse)<=radius*radius)return freeImpulse;
 float lo=0.0f,hi=length(rhs)/radius;
 for(uint iteration=0u;iteration<28u;++iteration){
  const float eta=.5f*(lo+hi),a=kxx+eta,d=kzz+eta;
  const float denom=max(fma(a,d,-kxz*kxz),1.e-20f);
  const float2 impulse=float2(d*rhs.x-kxz*rhs.y,a*rhs.y-kxz*rhs.x)/denom;
  if(dot(impulse,impulse)>radius*radius)lo=eta;else hi=eta;
 }
 const float a=kxx+hi,d=kzz+hi,denom=max(fma(a,d,-kxz*kxz),1.e-20f);
 return float2(d*rhs.x-kxz*rhs.y,a*rhs.y-kxz*rhs.x)/denom;
}
inline int pb_pair_mpr(const PlasticBagCoinGPU a,const PlasticBagCoinGPU b,const float R,const float h,thread float& depth,thread float3& normal,thread float3& armA,thread float3& armB);
inline bool pb_pair_certified_separated(const PlasticBagCoinGPU a,const PlasticBagCoinGPU b,const float R,const float h);


// Use the same physical lumped mass for contact and membrane integration.
// Both film and rigid velocities are reconstructed consistently from XPBD poses.

inline float3 pb_xyz(const mr_float4 v) { return float3(v.x, v.y, v.z); }
inline mr_float4 pb_f4(const float3 v, const float w = 0.0f) {
    return mr_float4(v.x, v.y, v.z, w);
}

// Integrate constraint increments separately from world-space poses. Subtracting
// rounded poses loses small reciprocal impulses at short physical timesteps.
#pragma clang fp reassociate(off)
#define PB_NODE_DELTA(AS) \
inline void pb_node_advance(AS PlasticBagNodeGPU& n,const float3 d){ \
 const float3 y=d-n.positionRemainder.xyz,t=n.positionAndInverseMass.xyz+y; \
 n.positionRemainder.xyz=(t-n.positionAndInverseMass.xyz)-y;n.positionAndInverseMass.xyz=t;} \
inline void pb_node_delta(AS PlasticBagNodeGPU& n,const float3 d){ \
 pb_node_advance(n,d);n.contactPositionCorrection.xyz+=d;}
PB_NODE_DELTA(thread)
PB_NODE_DELTA(device)
#undef PB_NODE_DELTA
inline float3 pb_node_difference(const PlasticBagNodeGPU a,const PlasticBagNodeGPU b){
 return (a.positionAndInverseMass.xyz-b.positionAndInverseMass.xyz)-a.positionRemainder.xyz+b.positionRemainder.xyz;
}
inline float3 pb_node_coin_difference(const PlasticBagNodeGPU a,const PlasticBagCoinGPU b){
 return (a.positionAndInverseMass.xyz-b.positionAndInverseMass.xyz)-a.positionRemainder.xyz+b.positionRemainder.xyz;
}
inline float3 pb_coin_difference(const PlasticBagCoinGPU a,const PlasticBagCoinGPU b){
 return (a.positionAndInverseMass.xyz-b.positionAndInverseMass.xyz)-a.positionRemainder.xyz+b.positionRemainder.xyz;
}
#pragma clang fp reassociate(on)

// Unilateral floor multiplier for the current physical substep. The rest
// position xyz stays immutable; its previously unused w stores this multiplier.
// Unload prior support when a material/contact correction lifts a vertex.
inline void pb_project_film_floor(thread PlasticBagNodeGPU& node,const float floor){
 const float w=node.positionAndInverseMass.w;if(w<=0.0f)return;
 const float old=node.restPosition.w;
 const float next=max(0.0f,old+(floor-node.positionAndInverseMass.y+node.positionRemainder.y)/w);
 pb_node_delta(node,float3(0,w*(next-old),0));node.restPosition.w=next;
}

inline float3 pb_candidate_closest_triangle_point(
    const float3 p,const float3 a,const float3 b,const float3 c) {
    const float3 ab=b-a,ac=c-a,ap=p-a;
    const float d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0.0f&&d2<=0.0f)return a;
    const float3 bp=p-b;
    const float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0.0f&&d4<=d3)return b;
    const float vc=d1*d4-d3*d2;
    if(vc<=0.0f&&d1>=0.0f&&d3<=0.0f){
        const float v=d1/max(d1-d3,1.0e-20f);
        return a+v*ab;
    }
    const float3 cp=p-c;
    const float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0.0f&&d5<=d6)return c;
    const float vb=d5*d2-d1*d6;
    if(vb<=0.0f&&d2>=0.0f&&d6<=0.0f){
        const float w=d2/max(d2-d6,1.0e-20f);
        return a+w*ac;
    }
    const float va=d3*d6-d5*d4;
    if(va<=0.0f&&(d4-d3)>=0.0f&&(d5-d6)>=0.0f){
        const float w=(d4-d3)/max((d4-d3)+(d5-d6),1.0e-20f);
        return b+w*(c-b);
    }
    const float denominator=1.0f/max(va+vb+vc,1.0e-20f);
    const float v=vb*denominator,w=vc*denominator;
    return a+ab*v+ac*w;
}

inline float4 pb_quat_normalize(const float4 q) {
    return q * rsqrt(max(dot(q, q), 1.0e-20f));
}

inline float4 pb_quat_conjugate(const float4 q) {
    return float4(-q.xyz, q.w);
}

inline float4 pb_quat_multiply(const float4 a, const float4 b) {
    return float4(a.w*b.xyz + b.w*a.xyz + cross(a.xyz,b.xyz),
                  a.w*b.w - dot(a.xyz,b.xyz));
}

inline float3 pb_rotate(const float4 q0, const float3 v) {
    const float4 q=pb_quat_normalize(q0);
    return v + 2.0f*cross(q.xyz,cross(q.xyz,v)+q.w*v);
}

inline float4 pb_apply_rotation(const float4 q0, const float3 rotationVector) {
    const float angle=length(rotationVector);
    float4 delta;
    if(angle>1.0e-8f){
        const float halfAngle=0.5f*angle;
        delta=float4(rotationVector*(sin(halfAngle)/angle),cos(halfAngle));
    }else delta=float4(0.5f*rotationVector,1.0f);
    return pb_quat_normalize(pb_quat_multiply(delta,q0));
}

#pragma clang fp reassociate(off)
#define PB_COIN_DELTA(AS) \
inline void pb_coin_advance(AS PlasticBagCoinGPU& c,const float3 d){ \
 const float3 y=d-c.positionRemainder.xyz,t=c.positionAndInverseMass.xyz+y; \
 c.positionRemainder.xyz=(t-c.positionAndInverseMass.xyz)-y;c.positionAndInverseMass.xyz=t;} \
inline void pb_coin_translation(AS PlasticBagCoinGPU& c,const float3 d){ \
 pb_coin_advance(c,d);c.predictedPosition.xyz+=d;} \
inline void pb_coin_rotation(AS PlasticBagCoinGPU& c,const float3 r){ \
 if(length_squared(r)==0.0f)return; \
 const float4 dq=pb_apply_rotation(float4(0,0,0,1),r); \
 c.orientation=pb_quat_normalize(pb_quat_multiply(dq,float4(c.orientation))); \
 c.predictedOrientation=pb_quat_normalize(pb_quat_multiply(dq,float4(c.predictedOrientation)));}
PB_COIN_DELTA(thread)
PB_COIN_DELTA(device)
PB_COIN_DELTA(threadgroup)
#undef PB_COIN_DELTA
#pragma clang fp reassociate(on)

inline void pb_free_coin_rotation(thread PlasticBagCoinGPU& c,const float dt,const float R,const float h){
 const float mass=c.velocityAndMass.w,parallel=.5f*mass*R*R,perpendicular=mass*(.25f*R*R+h*h/3.0f);
 const float3 axis=pb_rotate(float4(c.orientation),float3(0,1,0));
 const float3 L=perpendicular*c.angularVelocity.xyz+(parallel-perpendicular)*dot(c.angularVelocity.xyz,axis)*axis;
 const float spin=(1.0f/parallel-1.0f/perpendicular)*dot(L,axis)*dt;
 c.orientation=pb_apply_rotation(float4(c.orientation),L*(dt/perpendicular));
 float3 after=pb_rotate(float4(c.orientation),float3(0,1,0));
 c.orientation=pb_apply_rotation(float4(c.orientation),after*spin);
 after=pb_rotate(float4(c.orientation),float3(0,1,0));
 c.angularVelocity.xyz=L/perpendicular+(1.0f/parallel-1.0f/perpendicular)*dot(L,after)*after;
}

inline float3 pb_apply_inverse_inertia(const PlasticBagCoinGPU coin,
    const float3 worldVector,const float radius,const float halfThickness) {
    const float mass=max(coin.velocityAndMass.w,1.0e-9f);
    const float axial=0.5f*mass*radius*radius;
    const float radial=0.25f*mass*radius*radius+
        (1.0f/3.0f)*mass*halfThickness*halfThickness;
    const float3 local=pb_rotate(pb_quat_conjugate(coin.orientation),worldVector);
    return pb_rotate(coin.orientation,float3(local.x/max(radial,1.0e-12f),
        local.y/max(axial,1.0e-12f),local.z/max(radial,1.0e-12f)));
}

inline float3 pb_coin_support_offset(const PlasticBagCoinGPU coin,
    const float3 direction,const float radius,const float halfThickness) {
    const float3 axis=normalize(pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f)));
    const float axial=dot(direction,axis);
    const float3 radial=direction-axial*axis;
    const float radialLength=length(radial);
    // Near an axis-aligned query, float round-off can leave a tiny radial
    // vector. Normalizing it would add a full disk radius to the support point.
    // At zero axial projection the support set spans both coin faces. Choosing
    // either face edge arbitrarily invents a torque for a symmetric impact.
    const float axialOffset=axial>1.0e-6f ? halfThickness :
        axial< -1.0e-6f ? -halfThickness : 0.0f;
    return (radialLength>1.0e-6f ? radius*radial/radialLength : float3(0.0f))+
        axialOffset*axis;
}

inline float3 pb_limit_angular_velocity(const float3 angularVelocity,
    const float maximumSpeed) {
    const float speed=length(angularVelocity);
    return speed>maximumSpeed
        ? angularVelocity*(maximumSpeed/speed) : angularVelocity;
}

inline float pb_coin_vertical_extent(const PlasticBagCoinGPU coin,
    const float radius,const float halfThickness) {
    const float3 axis=pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f));
    return radius*length(axis.xz)+
        halfThickness*abs(axis.y);
}

// Finite-cylinder/table position constraint, including rotational inertia.
// Raising the center alone incorrectly converts rotational penetration into
// an upward launch. Solve the actual support-point Jacobian instead.
inline void pb_table_project(thread PlasticBagCoinGPU& coin,
 constant PlasticBagConfigGPU& c){
 for(uint iteration=0u;iteration<8u;++iteration){
  const float depth=(c.geometry.y-coin.positionAndInverseMass.y)+pb_coin_vertical_extent(coin,c.scales.y,c.scales.z)+coin.positionRemainder.y;
  if(depth<=1.e-9f)break;
  const float3 arm=pb_coin_support_offset(coin,float3(0,-1,0),c.scales.y,c.scales.z);
  const float3 rn=cross(arm,float3(0,1,0));
  const float3 invRn=pb_apply_inverse_inertia(coin,rn,c.scales.y,c.scales.z);
  const float lambda=depth/(coin.positionAndInverseMass.w+dot(rn,invRn));
  pb_coin_translation(coin,float3(0,coin.positionAndInverseMass.w*lambda,0));
  pb_coin_rotation(coin,lambda*invRn);
 }
}

inline uint pb_edge_count(const constant PlasticBagConfigGPU& c) { return c.counts.y; }

inline float3 pb_coin_sample_offset(const uint sampleIndex,const float radius,
    const float halfThickness,const float4 orientation) {
    if(sampleIndex<2u)
        return pb_rotate(orientation,float3(0.0f,sampleIndex==0u?
            halfThickness:-halfThickness,0.0f));
    const float radialRadius=0.99f*radius;
    float angle=0.0f;
    float plane=0.0f;
    if(sampleIndex<34u){
        const uint ringIndex=(sampleIndex-2u)%16u;
        angle=(float(ringIndex)+0.5f)*0.39269908169f;
        plane=sampleIndex<18u?halfThickness:-halfThickness;
    }else{
        const uint sideIndex=sampleIndex-34u;
        angle=(float(sideIndex)+0.5f)*0.19634954084f;
    }
    const float3 local=float3(radialRadius*cos(angle),plane,radialRadius*sin(angle));
    return pb_rotate(orientation,local);
}

inline float3 pb_coin_pair_sample_offset(const uint sampleIndex,const float radius,
    const float halfThickness,const float4 orientation) {
    if(sampleIndex<50u)
        return pb_coin_sample_offset(sampleIndex,radius,halfThickness,orientation);
    const uint innerIndex=sampleIndex-50u;
    const uint ringIndex=innerIndex&7u;
    const float angle=(float(ringIndex)+0.5f)*0.78539816339f;
    const float plane=innerIndex<8u?halfThickness:-halfThickness;
    const float radialRadius=0.55f*radius;
    return pb_rotate(orientation,float3(radialRadius*cos(angle),plane,
        radialRadius*sin(angle)));
}


// Signed dihedral bending. Scalar angle and its four gradients are invariant
// under rigid translation/rotation; no fixed world-space rest normal exists.
inline float pb_hinge_angle(const float3 p0,const float3 p1,const float3 p2,
 const float3 p3,thread float3* g) {
 const float3 e=p1-p0;const float l=length(e),l2=dot(e,e);
 const float3 a=cross(e,p2-p0),b=cross(p3-p0,e);
 const float aa=dot(a,a),bb=dot(b,b);
 if(l2<1.e-16f||aa<1.e-18f||bb<1.e-18f){for(uint k=0;k<4u;++k)g[k]=float3(0);return 0;}
 const float3 n1=a*rsqrt(aa),n2=b*rsqrt(bb);
 g[2]=-l/aa*a;g[3]=-l/bb*b;
 g[1]=-dot(p2-p0,e)/l2*g[2]-dot(p3-p0,e)/l2*g[3];
 g[0]=-g[1]-g[2]-g[3];
 return atan2(dot(cross(n1,n2),e/l),clamp(dot(n1,n2),-1.0f,1.0f));
}
kernel void pb_hinge_colored_solve(
 constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device PlasticBagHingeGPU* hinges [[buffer(2)]],
 device const uint* order [[buffer(3)]],
 device const mr_uint4* spans [[buffer(4)]],
 constant uint& reset [[buffer(5)]],
 uint lane [[thread_index_in_threadgroup]]) {
 const uint colors=spans[0].w;
 for(uint repeat=0u;repeat<(reset?1u:c.iterations.y);++repeat){
 for(uint color=0u;color<colors;++color){
  const uint begin=spans[color].x,end=begin+spans[color].y;
  for(uint j=begin+lane;j<end;j+=128u){
   const uint hi=order[j];PlasticBagHingeGPU h=hinges[hi];
   if(reset){h.material.z=0.0f;hinges[hi]=h;continue;}
   float3 g[4];const float angle=pb_hinge_angle(
    float3(0),pb_node_difference(nodes[h.nodes.y],nodes[h.nodes.x]),
    pb_node_difference(nodes[h.nodes.z],nodes[h.nodes.x]),pb_node_difference(nodes[h.nodes.w],nodes[h.nodes.x]),g);
   const float C=atan2(sin(angle-h.material.x),cos(angle-h.material.x));
   float3 previousGradient[4];
   const float previousAngle=pb_hinge_angle(
    nodes[h.nodes.x].previousAndFixed.xyz,nodes[h.nodes.y].previousAndFixed.xyz,
    nodes[h.nodes.z].previousAndFixed.xyz,nodes[h.nodes.w].previousAndFixed.xyz,previousGradient);
   float rate=0.0f;for(uint k=0u;k<4u;++k){const auto nd=nodes[h.nodes[k]];rate+=dot(g[k],c.scales.x*nd.velocityAndMass.xyz+nd.contactPositionCorrection.xyz);}
   float w=0.0f;
   for(uint k=0;k<4u;++k){const auto n=nodes[h.nodes[k]];
    w+=n.positionAndInverseMass.w*dot(g[k],g[k]);}
   const float alpha=h.material.y/(c.scales.x*c.scales.x),gamma=h.material.w/c.scales.x;
   const float dl=-(C+alpha*h.material.z+gamma*rate)/max((1.0f+gamma)*w+alpha,1.e-16f);
   for(uint k=0;k<4u;++k){const uint ni=h.nodes[k];
    pb_node_delta(nodes[ni],nodes[ni].positionAndInverseMass.w*dl*g[k]);}
   h.material.z+=dl;hinges[hi]=h;
  }
  threadgroup_barrier(mem_flags::mem_device);
 }
}
}

kernel void pb_integrate_nodes(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if (nodeIndex >= config.counts.x) return;
    PlasticBagNodeGPU node = nodes[nodeIndex];
    const float inverseMass = node.positionAndInverseMass.w;
    const float3 rest = pb_xyz(node.restPosition);
    if (!(inverseMass > 0.0f)) {
        // Kinematic grip vertices are advanced by the host once per 240 Hz
        // frame. Preserve that frame's swept previous position and velocity
        // through all solver substeps so impacts see the hand's motion.
        node.positionAndInverseMass = pb_f4(rest, 0.0f);
        nodes[nodeIndex] = node;
        return;
    }
    const float3 position = pb_xyz(node.positionAndInverseMass);
    node.previousAndFixed = pb_f4(position, 0.0f);
    node.contactPositionCorrection=mr_float4(0.0f);
    float3 velocity = node.velocityAndMass.xyz +
        config.gravityAndBending.xyz * config.scales.x;
    float3 next = position + velocity * config.scales.x;
    const float incomingNormalSpeed=velocity.y;
    const float shellFloor = config.geometry.y + config.geometry.x;
    // Keep the unconstrained predictor for consistent pose-to-velocity work.
    // Floor support is solved with the material/contact constraints below.
    node.restPosition.w=0.0f;
    node.velocityAndMass.xyz = velocity;
    node.contactPositionCorrection=pb_f4(float3(0),incomingNormalSpeed);
    pb_node_advance(node,velocity*config.scales.x);
    nodes[nodeIndex] = node;
}

kernel void pb_reset_edge_lambdas(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagEdgeGPU* edges [[buffer(1)]],
    uint edgeIndex [[thread_position_in_grid]]) {
    if (edgeIndex >= config.counts.y) return;
    PlasticBagEdgeGPU edge=edges[edgeIndex];
    edge.material.z=0.0f;
    edges[edgeIndex]=edge;
}

kernel void pb_update_triangle_bounds(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device const PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagTriangleGPU* triangles [[buffer(2)]],
    uint triangleIndex [[thread_position_in_grid]]) {
    if(triangleIndex>=config.counts.z)return;
    PlasticBagTriangleGPU triangle=triangles[triangleIndex];
    const float3 a=pb_xyz(nodes[triangle.nodes.x].positionAndInverseMass);
    const float3 b=pb_xyz(nodes[triangle.nodes.y].positionAndInverseMass);
    const float3 c=pb_xyz(nodes[triangle.nodes.z].positionAndInverseMass);
    const float3 center=(a+b+c)/3.0f;
    const float radius=max(length(a-center),max(length(b-center),length(c-center)));
    const float3 pa=pb_xyz(nodes[triangle.nodes.x].previousAndFixed);
    const float3 pb=pb_xyz(nodes[triangle.nodes.y].previousAndFixed);
    const float3 pc=pb_xyz(nodes[triangle.nodes.z].previousAndFixed);
    const float3 previousCenter=(pa+pb+pc)/3.0f;
    const float previousRadius=max(length(pa-previousCenter),
        max(length(pb-previousCenter),length(pc-previousCenter)));
    triangle.centerAndRadius=pb_f4(center,radius);
    triangle.previousCenterAndRadius=pb_f4(previousCenter,previousRadius);
    triangles[triangleIndex]=triangle;
}

kernel void pb_build_coin_candidates(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(1)]],
    device const PlasticBagNodeGPU* nodes [[buffer(2)]],
    device const PlasticBagCoinGPU* coins [[buffer(3)]],
    device uint* candidateTriangles [[buffer(4)]],
    device uint* candidateCounts [[buffer(5)]],
    device uint* candidateOverflow [[buffer(6)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if (coinIndex >= config.counts.w) return;
    const uint capacity=uint(config.geometry.z);
    const PlasticBagCoinGPU coin=coins[coinIndex];
    const float3 currentCoin=pb_xyz(coin.positionAndInverseMass);
    const float3 previousCoin=pb_xyz(coin.previousPosition);
    const float coinBound=length(float2(config.scales.y,config.scales.z));
    uint count=0u;
    uint overflow=0u;
    for(uint triangleIndex=0u;triangleIndex<config.counts.z;++triangleIndex){
        const PlasticBagTriangleGPU triangle=triangles[triangleIndex];
        const float reach=coinBound+config.geometry.x+
            max(triangle.centerAndRadius.w,triangle.previousCenterAndRadius.w)+0.001f;
        const float3 relativeStart=previousCoin-triangle.previousCenterAndRadius.xyz;
        const float3 relativeEnd=currentCoin-triangle.centerAndRadius.xyz;
        const float3 relativeMotion=relativeEnd-relativeStart;
        const float motionSquared=dot(relativeMotion,relativeMotion);
        const float sweepT=motionSquared>1.0e-16f
            ? clamp(-dot(relativeStart,relativeMotion)/motionSquared,0.0f,1.0f):0.0f;
        const float3 closestRelative=relativeStart+sweepT*relativeMotion;
        if(dot(closestRelative,closestRelative)<=reach*reach){
            if(count<capacity)
                candidateTriangles[coinIndex*capacity+count]=triangleIndex;
            else
                overflow=1u;
            ++count;
        }
    }
    candidateCounts[coinIndex]=min(count,capacity);
    candidateOverflow[coinIndex]=overflow;
}

kernel void pb_integrate_coins(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if (coinIndex >= config.counts.w) return;
    PlasticBagCoinGPU coin = coins[coinIndex];
    const float3 position = pb_xyz(coin.positionAndInverseMass);
    coin.previousPosition = pb_f4(position, 0.0f);
    coin.previousOrientation=mr_float4(coin.orientation);
    float3 velocity = coin.velocityAndMass.xyz +
        config.gravityAndBending.xyz * config.scales.x;
    float3 next = position + velocity * config.scales.x;
    pb_free_coin_rotation(coin,config.scales.x,config.scales.y,config.scales.z);
    coin.predictedPosition=pb_f4(float3(0));
    coin.predictedOrientation=mr_float4(0,0,0,1);
    coin.predictedVelocity=pb_f4(velocity);
    coin.predictedAngularVelocity=coin.angularVelocity;
    coin.velocityAndMass.xyz=velocity;
    pb_coin_advance(coin,velocity*config.scales.x);
    pb_table_project(coin,config);
    coins[coinIndex] = coin;
}

kernel void pb_edge_solve(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagEdgeGPU* edges [[buffer(2)]],
    device PlasticBagEdgeScratchGPU* scratch [[buffer(3)]],
    uint edgeIndex [[thread_position_in_grid]]) {
    if (edgeIndex >= pb_edge_count(config)) return;
    PlasticBagEdgeGPU edge = edges[edgeIndex];
    const uint a = edge.nodes.x;
    const uint b = edge.nodes.y;
    const PlasticBagNodeGPU nodeA = nodes[a];
    const PlasticBagNodeGPU nodeB = nodes[b];
    const float wa = nodeA.positionAndInverseMass.w;
    const float wb = nodeB.positionAndInverseMass.w;
    const float3 delta = pb_xyz(nodeB.positionAndInverseMass) -
        pb_xyz(nodeA.positionAndInverseMass);
    const float lengthNow = length(delta);
    const float alpha = edge.material.y /
        (config.scales.x * config.scales.x);
    const float denominator = wa + wb + alpha;
    PlasticBagEdgeScratchGPU result{};
    if (lengthNow > 1.0e-8f && denominator > 1.0e-12f) {
        const float constraint = lengthNow - edge.material.x;
        const float deltaLambda =
            (-constraint - alpha * edge.material.z) / denominator *
            config.solver.y;
        const float3 direction = delta / lengthNow;
        result.correctionA = pb_f4(-wa * deltaLambda * direction);
        result.correctionB = pb_f4( wb * deltaLambda * direction);
        edge.material.z += deltaLambda;
    }
    edges[edgeIndex] = edge;
    scratch[edgeIndex] = result;
}

kernel void pb_edge_apply(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagEdgeScratchGPU* scratch [[buffer(2)]],
    device const PlasticBagNodeEdgeRefGPU* refs [[buffer(3)]],
    device const PlasticBagNodeSpanGPU* spans [[buffer(4)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if (nodeIndex >= config.counts.x) return;
    PlasticBagNodeGPU node = nodes[nodeIndex];
    if (!(node.positionAndInverseMass.w > 0.0f)) return;
    const uint begin = spans[nodeIndex].range.x;
    const uint count = spans[nodeIndex].range.y;
    float3 correction = float3(0.0f);
    for (uint r = 0u; r < count; ++r) {
        const PlasticBagNodeEdgeRefGPU ref = refs[begin + r];
        const PlasticBagEdgeScratchGPU edge = scratch[ref.identity.x];
        correction += ref.identity.y == 0u
            ? edge.correctionA.xyz : edge.correctionB.xyz;
    }
    node.positionAndInverseMass.xyz += correction;
    node.positionAndInverseMass.y=max(node.positionAndInverseMass.y,config.geometry.y+config.geometry.x);
    nodes[nodeIndex] = node;
}

inline float3 pb_closest_triangle_point(
    const float3 p, const float3 a, const float3 b, const float3 c,
    thread float3& barycentric) {
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { barycentric = float3(1,0,0); return a; }
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { barycentric = float3(0,1,0); return b; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / max(d1 - d3, 1.0e-20f);
        barycentric = float3(1.0f-v,v,0.0f);
        return a + v * ab;
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { barycentric = float3(0,0,1); return c; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / max(d2 - d6, 1.0e-20f);
        barycentric = float3(1.0f-w,0.0f,w);
        return a + w * ac;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4-d3) >= 0.0f && (d5-d6) >= 0.0f) {
        const float w = (d4-d3) / max((d4-d3)+(d5-d6), 1.0e-20f);
        barycentric = float3(0.0f,1.0f-w,w);
        return b + w * (c-b);
    }
    const float denominator = 1.0f / max(va + vb + vc, 1.0e-20f);
    const float v = vb * denominator;
    const float w = vc * denominator;
    barycentric = float3(1.0f-v-w,v,w);
    return a + ab * v + ac * w;
}

// Cubic moving point / moving triangle CCD. All four vertices follow the
// linear substep trajectories. Stationary points partition the cubic into
// monotone intervals; each crossing is checked on the finite triangle.
inline float pb_poly3(const float4 c,const float t){return ((c.w*t+c.z)*t+c.y)*t+c.x;}
inline bool pb_vt_crossing(const float3 p0,const float3 p,
 const float3 a0,const float3 a,const float3 b0,const float3 b,
 const float3 c0,const float3 c,thread float3& bary,thread float3& normal){
 const float3 v0=p0-a0,v1=(p-p0)-(a-a0);
 const float3 e0=b0-a0,e1=(b-b0)-(a-a0);
 const float3 f0=c0-a0,f1=(c-c0)-(a-a0);
 const float3 n0=cross(e0,f0),n1=cross(e1,f0)+cross(e0,f1),n2=cross(e1,f1);
 const float4 coeff(dot(v0,n0),dot(v1,n0)+dot(v0,n1),dot(v1,n1)+dot(v0,n2),dot(v1,n2));
 float cuts[4]={0.0f,1.0f,1.0f,1.0f};uint count=2u;
 const float aa=3.0f*coeff.w,bb=2.0f*coeff.z,cc=coeff.y;
 if(abs(aa)>1.e-16f){
  const float disc=bb*bb-4.0f*aa*cc;
  if(disc>=0.0f){
   const float sq=sqrt(disc);float t1=(-bb-sq)/(2.0f*aa),t2=(-bb+sq)/(2.0f*aa);
   if(t1>t2){float tmp=t1;t1=t2;t2=tmp;}
   if(t1>0.0f&&t1<1.0f)cuts[count++]=t1;
   if(t2>0.0f&&t2<1.0f)cuts[count++]=t2;
  }
 }else if(abs(bb)>1.e-16f){float t=-cc/bb;if(t>0.0f&&t<1.0f)cuts[count++]=t;}
 for(uint i=1u;i<count;++i)for(uint j=i;j>0u&&cuts[j]<cuts[j-1u];--j){float tmp=cuts[j];cuts[j]=cuts[j-1u];cuts[j-1u]=tmp;}
 for(uint i=0u;i+1u<count;++i){
  float lo=cuts[i],hi=cuts[i+1u];float flo=pb_poly3(coeff,lo),fhi=pb_poly3(coeff,hi);
  if(flo*fhi>=0.0f||abs(flo)<1.e-18f||abs(fhi)<1.e-18f)continue;
  const float side=flo>0.0f?1.0f:-1.0f;
  for(uint k=0u;k<24u;++k){const float mid=0.5f*(lo+hi);float fm=pb_poly3(coeff,mid);if(fm*flo>0.0f){lo=mid;flo=fm;}else hi=mid;}
  const float t=0.5f*(lo+hi);if(t<=1.e-6f)continue;
  const float3 at=mix(a0,a,t),bt=mix(b0,b,t),ct=mix(c0,c,t),pt=mix(p0,p,t);
  float3 btHit;const float3 qt=pb_closest_triangle_point(pt,at,bt,ct,btHit);
  if(min(btHit.x,min(btHit.y,btHit.z))<1.e-5f||distance(pt,qt)>2.e-6f)continue;
  const float3 nn=cross(b-a,c-a);const float len=length(nn);if(len<1.e-10f)continue;
  bary=btHit;normal=side*nn/len;return true;
 }
 return false;
}

kernel void pb_shell_self_contact_solve(
 constant PlasticBagConfigGPU& config [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const PlasticBagTriangleGPU* triangles [[buffer(2)]],
 device PlasticBagContactGPU* contacts [[buffer(3)]],
 device const PlasticBagNodeEdgeRefGPU* edgeRefs [[buffer(4)]],
 device const PlasticBagNodeSpanGPU* edgeSpans [[buffer(5)]],
 device const PlasticBagEdgeGPU* edges [[buffer(6)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=config.counts.x)return;
 PlasticBagContactGPU hit{};
 const PlasticBagNodeGPU node=nodes[i];
 if(node.positionAndInverseMass.w<=0.0f){contacts[i]=hit;return;}
 const float3 p=node.positionAndInverseMass.xyz,p0=node.previousAndFixed.xyz;
 const float gap=2.0f*config.geometry.x;
 float best=0.0f;
 for(uint ti=0u;ti<config.counts.z;++ti){
  const PlasticBagTriangleGPU tri=triangles[ti];
  const uint ia=tri.nodes.x,ib=tri.nodes.y,ic=tri.nodes.z;
  if(i==ia||i==ib||i==ic)continue;
  const float3 rs=p0-tri.previousCenterAndRadius.xyz,re=p-tri.centerAndRadius.xyz,dd=re-rs;
  const float t=clamp(-dot(rs,dd)/max(dot(dd,dd),1.e-20f),0.0f,1.0f);
  const float reach=max(tri.centerAndRadius.w,tri.previousCenterAndRadius.w)+gap;
  if(dot(rs+t*dd,rs+t*dd)>reach*reach)continue;
  // Exclude the actual topological one-ring, including the shared weld.
  bool local=false;
  for(uint j=0u;j<edgeSpans[i].range.y;++j){
   const uint ei=edgeRefs[edgeSpans[i].range.x+j].identity.x;
   const uint other=edges[ei].nodes.x==i?edges[ei].nodes.y:edges[ei].nodes.x;
   if(other==ia||other==ib||other==ic){local=true;break;}
  }
  if(local)continue;
  const float3 a=nodes[ia].positionAndInverseMass.xyz,b=nodes[ib].positionAndInverseMass.xyz,c=nodes[ic].positionAndInverseMass.xyz;
  float3 bary,n;const float3 q=pb_closest_triangle_point(p,a,b,c,bary);
  const float dist=distance(p,q);float depth=0.0f;
  if(dist<gap){
   n=dist>1.e-9f?(p-q)/dist:normalize(cross(b-a,c-a));
   depth=gap-dist;
  }
  float3 sweptBary,sweptNormal;
  if(pb_vt_crossing(p0,p,nodes[ia].previousAndFixed.xyz,a,
      nodes[ib].previousAndFixed.xyz,b,nodes[ic].previousAndFixed.xyz,c,sweptBary,sweptNormal)){
   const float sweptDepth=gap-dot(p-(a*sweptBary.x+b*sweptBary.y+c*sweptBary.z),sweptNormal);
   if(sweptDepth>depth){bary=sweptBary;n=sweptNormal;depth=sweptDepth;}
  }
  if(depth>best){best=depth;hit.normalAndActive=mr_float4(n,1.0f);hit.barycentricAndDepth=mr_float4(bary,depth);hit.identity=mr_uint4(ti,i,0u,0u);}
 }
 contacts[i]=hit;
}

// One owner applies the sparse contact set in place. The constraint gradient
// includes the query vertex and all three triangle vertices. Equal and opposite
// mass-weighted corrections conserve linear momentum; there is no split motion.
kernel void pb_shell_self_contact_apply(
 constant PlasticBagConfigGPU& config [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const PlasticBagTriangleGPU* triangles [[buffer(2)]],
 device const PlasticBagContactGPU* contacts [[buffer(3)]],
 uint owner [[thread_position_in_grid]]) {
 if(owner)return;
 for(uint i=0u;i<config.counts.x;++i){
  const PlasticBagContactGPU ct=contacts[i];if(ct.normalAndActive.w==0.0f)continue;
  const uint3 ids=triangles[ct.identity.x].nodes.xyz;
  const float3 bary=ct.barycentricAndDepth.xyz,n=ct.normalAndActive.xyz;
  const float3 q=nodes[ids.x].positionAndInverseMass.xyz*bary.x+nodes[ids.y].positionAndInverseMass.xyz*bary.y+nodes[ids.z].positionAndInverseMass.xyz*bary.z;
  const float depth=2.0f*config.geometry.x-dot(nodes[i].positionAndInverseMass.xyz-q,n);
  if(depth<=0.0f)continue;
  const float wi=nodes[i].positionAndInverseMass.w;
  const float wa=nodes[ids.x].positionAndInverseMass.w,wb=nodes[ids.y].positionAndInverseMass.w,wc=nodes[ids.z].positionAndInverseMass.w;
  const float denom=wi+wa*bary.x*bary.x+wb*bary.y*bary.y+wc*bary.z*bary.z;
  const float lambda=depth/max(denom,1.e-20f);
  pb_node_delta(nodes[i],wi*lambda*n);
  pb_node_delta(nodes[ids.x],-wa*bary.x*lambda*n);
  pb_node_delta(nodes[ids.y],-wb*bary.y*lambda*n);
  pb_node_delta(nodes[ids.z],-wc*bary.z*lambda*n);
 }
}

inline float pb_cylinder_sdf(
    const float3 pointFromCenter,const float radius,const float halfThickness,
    const float4 orientation,thread float3& normal) {
    const float3 localPoint=pb_rotate(pb_quat_conjugate(orientation),pointFromCenter);
    const float radial = length(localPoint.xz);
    const float2 q = float2(radial - radius,
                            abs(localPoint.y) - halfThickness);
    const float2 outside = max(q, float2(0.0f));
    const float distance = length(outside) + min(max(q.x, q.y), 0.0f);
    const float2 radialNormal = radial > 1.0e-8f
        ? localPoint.xz / radial : float2(1.0f, 0.0f);
    if (any(outside > 0.0f)) {
        const float2 n2 = normalize(outside);
        normal = normalize(float3(radialNormal.x*n2.x,
            copysign(n2.y, localPoint.y), radialNormal.y*n2.x));
    } else if (q.x > q.y) {
        normal = float3(radialNormal.x, 0.0f, radialNormal.y);
    } else {
        normal = float3(0.0f, copysign(1.0f, localPoint.y), 0.0f);
    }
    normal=pb_rotate(orientation,normal);
    return distance;
}

// A thin membrane can pass through the middle of a coin even when the
// nearest-cylinder SDF gradient points across the coin's thickness. Resolve
// the collision along the membrane normal, using the coin's support extent in
// that direction. Previous-side information keeps the two-sided contact from
// arbitrarily flipping when the coin center lies on the triangle plane.
inline float pb_contact_node_inverse_mass(
    const PlasticBagNodeGPU node,const float3 contactNormal,const float shellFloor) {
    const float inverseMass=node.positionAndInverseMass.w;
    const bool blockedByGround=contactNormal.y < -1.0e-6f &&
        node.positionAndInverseMass.y <= shellFloor+1.0e-6f;
    return inverseMass>0.0f&&!blockedByGround
        ? inverseMass:0.0f;
}

inline float pb_coin_film_support_penetration(
    const float3 trianglePoint,const PlasticBagTriangleGPU triangle,
    const PlasticBagCoinGPU coin,device const PlasticBagNodeGPU* nodes,
    const float radius,const float halfThickness,const float filmHalfThickness,
    thread float3& contactNormal) {
    const uint a=triangle.nodes.x,b=triangle.nodes.y,c=triangle.nodes.z;
    const float3 aNow=pb_xyz(nodes[a].positionAndInverseMass);
    const float3 bNow=pb_xyz(nodes[b].positionAndInverseMass);
    const float3 cNow=pb_xyz(nodes[c].positionAndInverseMass);
    float3 planeNormal=cross(bNow-aNow,cNow-aNow);
    const float normalLength=length(planeNormal);
    if(normalLength<1.0e-8f){contactNormal=float3(0.0f,1.0f,0.0f);return 0.0f;}
    planeNormal/=normalLength;
    const float3 center=pb_xyz(coin.positionAndInverseMass);
    const float3 pa=nodes[a].previousAndFixed.xyz,pb=nodes[b].previousAndFixed.xyz,pc=nodes[c].previousAndFixed.xyz;
    const float3 previousNormal=cross(pb-pa,pc-pa);
    const float previousSide=dot(coin.previousPosition.xyz-pa,previousNormal);
    const float currentSide=dot(center-aNow,planeNormal);
    const float side=abs(previousSide)>1.e-12f ? (previousSide<0.0f?1.0f:-1.0f)
        : (currentSide<0.0f?1.0f:-1.0f);
    contactNormal=side*planeNormal;
    const float3 diskAxis=pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f));
    const float axisProjection=dot(diskAxis,planeNormal);
    const float supportExtent=radius*length(cross(diskAxis,planeNormal))+
        halfThickness*abs(axisProjection);
    // Use a cross product for the radial projection: sqrt(1-dot^2) loses
    // the entire small angle near parallel cap normals in float precision.
    // The witness normal points from the pouch interior toward the film.
    // Keep penetration signed so a coin remains detectable after crossing it.
    return max(0.0f,supportExtent+filmHalfThickness+
        dot(center-trianglePoint,contactNormal));
}

kernel void pb_contact_solve(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(2)]],
    device PlasticBagCoinGPU* coins [[buffer(3)]],
    device PlasticBagContactGPU* contacts [[buffer(4)]],
    device const uint* candidateTriangles [[buffer(5)]],
    device const uint* candidateCounts [[buffer(6)]],
    uint contactIndex [[thread_position_in_grid]]) {
    const uint sampleCount=uint(config.scales.w);
    const uint contactCount=config.counts.w*sampleCount;
    if(contactIndex>=contactCount)return;
    const uint coinIndex=contactIndex/sampleCount;
    const uint sampleIndex=contactIndex-coinIndex*sampleCount;
    const uint maxCandidates=uint(config.geometry.z);
    const PlasticBagCoinGPU coin=coins[coinIndex];
    const float radius=config.scales.y;
    const float3 offset=pb_coin_sample_offset(sampleIndex,radius,config.scales.z,
        float4(coin.orientation));
    const float3 queryPosition=pb_xyz(coin.positionAndInverseMass)+offset;
    float bestDepth=0.0f;
    uint bestTriangle=0u;
    bool bestCenterWitness=false;
    for(uint candidateSlot=0u;candidateSlot<candidateCounts[coinIndex];++candidateSlot){
        const uint triangleIndex=candidateTriangles[coinIndex*maxCandidates+candidateSlot];
        const PlasticBagTriangleGPU triangle=triangles[triangleIndex];
        float3 centerDelta=queryPosition-triangle.centerAndRadius.xyz;
        if(sampleIndex==0u){
            const float3 relativeStart=pb_xyz(coin.previousPosition)-
                triangle.previousCenterAndRadius.xyz;
            const float3 relativeEnd=pb_xyz(coin.positionAndInverseMass)-
                triangle.centerAndRadius.xyz;
            const float3 relativeMotion=relativeEnd-relativeStart;
            const float motionSquared=dot(relativeMotion,relativeMotion);
            const float sweepT=motionSquared>1.0e-16f
                ? clamp(-dot(relativeStart,relativeMotion)/motionSquared,0.0f,1.0f):0.0f;
            centerDelta=relativeStart+sweepT*relativeMotion;
        }
        const float reach=radius+config.geometry.x+triangle.centerAndRadius.w;
        if(dot(centerDelta,centerDelta)>reach*reach)continue;
        if(sampleIndex==0u){
            float3 centerBary;
            const float3 coinCenter=pb_xyz(coin.positionAndInverseMass);
            const float3 centerPoint=pb_closest_triangle_point(coinCenter,
                pb_xyz(nodes[triangle.nodes.x].positionAndInverseMass),
                pb_xyz(nodes[triangle.nodes.y].positionAndInverseMass),
                pb_xyz(nodes[triangle.nodes.z].positionAndInverseMass),centerBary);
            float3 centerNormal;
            const float centerPenetration=pb_coin_film_support_penetration(
                centerPoint,triangle,coin,nodes,radius,config.scales.z,
                config.geometry.x,centerNormal);
            const float3 centerDelta=coinCenter-centerPoint;
            const float3 tangentDelta=centerDelta-
                dot(centerDelta,centerNormal)*centerNormal;
            float3 witnessNormal;
            const float witnessDistance=pb_cylinder_sdf(centerPoint-coinCenter,
                radius,config.scales.z,float4(coin.orientation),witnessNormal);
            if(centerPenetration>0.0f&&witnessDistance<=config.geometry.x&&
                dot(tangentDelta,tangentDelta)<=radius*radius){
                if(centerPenetration>bestDepth){
                    bestDepth=centerPenetration;
                    bestTriangle=triangleIndex;
                    bestCenterWitness=true;
                }
                continue;
            }
        }
        float3 bary;
        const float3 point=pb_closest_triangle_point(queryPosition,
            pb_xyz(nodes[triangle.nodes.x].positionAndInverseMass),
            pb_xyz(nodes[triangle.nodes.y].positionAndInverseMass),
            pb_xyz(nodes[triangle.nodes.z].positionAndInverseMass),bary);
        float3 sdfNormal;
        const float signedDistance=pb_cylinder_sdf(point-
            pb_xyz(coin.positionAndInverseMass),radius,config.scales.z,
            float4(coin.orientation),sdfNormal);
        if(signedDistance>config.geometry.x)continue;
        float3 filmNormal;
        const float planePenetration=pb_coin_film_support_penetration(point,triangle,coin,
            nodes,radius,config.scales.z,config.geometry.x,filmNormal);
        const float sampledPenetration=max(0.0f,config.geometry.x-signedDistance);
        const float correctionLimit=1.5f*sampledPenetration+0.00002f;
        const float depth=min(planePenetration,correctionLimit);
        if(depth>bestDepth){
            bestDepth=depth;
            bestTriangle=triangleIndex;
            bestCenterWitness=false;
        }
    }
    PlasticBagContactGPU seed{};
    if(bestDepth>0.0f){
        seed.barycentricAndDepth.w=bestDepth;
        seed.identity=mr_uint4(bestTriangle,sampleIndex,bestCenterWitness?1u:0u,0u);
    }
    contacts[contactIndex]=seed;
}

inline void pb_contact_reduce_one(
 constant PlasticBagConfigGPU& config,device PlasticBagContactGPU* contacts,
 device PlasticBagNodeGPU* nodes,device const PlasticBagTriangleGPU* triangles,
 device PlasticBagCoinGPU* coins,device PlasticBagContactGPU* loads,device mr_float4* floorLoads,uint coinIndex) {
    if (coinIndex >= config.counts.w) return;
    const uint sampleCount=uint(config.scales.w);
    PlasticBagContactGPU selected[PB_COIN_PAIR_SAMPLE_COUNT];
    float nodeWeights[PB_COIN_PAIR_SAMPLE_COUNT]={0.0f};
    const PlasticBagCoinGPU coin=coins[coinIndex];
    const float radius=config.scales.y;

    // Keep the deepest triangle witness at each point on the finite cylinder.
    for(uint sampleIndex=0u;sampleIndex<sampleCount;++sampleIndex){
        const PlasticBagContactGPU seed=contacts[coinIndex*sampleCount+sampleIndex];
        const float sampleDepth=seed.barycentricAndDepth.w;
        const uint selectedTriangle=seed.identity.x;
        const bool centerWitness=seed.identity.z!=0u;
        const float3 offset=pb_coin_sample_offset(sampleIndex,radius,config.scales.z,
            float4(coin.orientation));
        const float3 queryPosition=pb_xyz(coin.positionAndInverseMass)+
            (centerWitness?float3(0.0f):offset);
        PlasticBagContactGPU contact{};
        if(sampleDepth>0.0f){
            const PlasticBagTriangleGPU triangle=triangles[selectedTriangle];
            const uint a=triangle.nodes.x,b=triangle.nodes.y,c=triangle.nodes.z;
            float3 bary;
            const float3 point=pb_closest_triangle_point(queryPosition,
                pb_xyz(nodes[a].positionAndInverseMass),pb_xyz(nodes[b].positionAndInverseMass),
                pb_xyz(nodes[c].positionAndInverseMass),bary);
            float3 sdfNormal;
            const float signedDistance=pb_cylinder_sdf(point-
                pb_xyz(coin.positionAndInverseMass),radius,config.scales.z,
                float4(coin.orientation),sdfNormal);
            float3 normal;
            const float planePenetration=pb_coin_film_support_penetration(point,triangle,coin,
                nodes,radius,config.scales.z,config.geometry.x,normal);
            const float sampledPenetration=max(0.0f,config.geometry.x-signedDistance);
            const float correctionLimit=1.5f*sampledPenetration+0.00002f;
            // Keep the center witness for side-stable CCD after a crossing,
            // but bound its positional cleanup by the same local support depth
            // as a sampled witness. Never turn an entire outside distance into
            // a single large pose correction.
            const float centerWitnessCorrectionLimit=0.0015f;
            const float boundedPenetration=centerWitness
                ? min(min(planePenetration,sampleDepth),min(correctionLimit,centerWitnessCorrectionLimit))
                : min(planePenetration,correctionLimit);
            if(signedDistance<=config.geometry.x&&
                boundedPenetration>0.0f){
                contact.normalAndActive=pb_f4(normal,1.0f);
                contact.barycentricAndDepth=pb_f4(bary,boundedPenetration);
                contact.identity=mr_uint4(selectedTriangle,sampleIndex,
                    centerWitness?1u:0u,0u);
                const float shellFloor=config.geometry.y+config.geometry.x;
                const float wa=pb_contact_node_inverse_mass(nodes[a],normal,shellFloor);
                const float wb=pb_contact_node_inverse_mass(nodes[b],normal,shellFloor);
                const float wc=pb_contact_node_inverse_mass(nodes[c],normal,shellFloor);
                nodeWeights[sampleIndex]=wa*bary.x*bary.x+
                    wb*bary.y*bary.y+wc*bary.z*bary.z;
            }
        }
        selected[sampleIndex]=contact;
    }

    // Per-coin projected Gauss-Seidel: solve each unilateral witness against
    // corrections already assigned to this coin and its shared film nodes.
    // The prior max-depth/equal-share Jacobi update let opposite panel normals
    // cancel and left coins straddling the film after every coupled pass.
    float3 coinCorrectionAccum=float3(0.0f),coinRotationAccum=float3(0.0f);
    for(uint sampleIndex=0u;sampleIndex<sampleCount;++sampleIndex){
        PlasticBagContactGPU contact=selected[sampleIndex];
        if(contact.normalAndActive.w>0.0f){
            const PlasticBagTriangleGPU triangle=triangles[contact.identity.x];
            const uint a=triangle.nodes.x,b=triangle.nodes.y,c=triangle.nodes.z;
            const float3 bary=contact.barycentricAndDepth.xyz;
            const float3 normal=contact.normalAndActive.xyz;
            float3 priorA=float3(0.0f),priorB=float3(0.0f),priorC=float3(0.0f);
            for(uint priorIndex=0u;priorIndex<sampleIndex;++priorIndex){
                const PlasticBagContactGPU prior=selected[priorIndex];
                if(prior.normalAndActive.w==0.0f)continue;
                const PlasticBagTriangleGPU priorTriangle=triangles[prior.identity.x];
                if(a==priorTriangle.nodes.x)priorA+=prior.correctionA.xyz;
                else if(a==priorTriangle.nodes.y)priorA+=prior.correctionB.xyz;
                else if(a==priorTriangle.nodes.z)priorA+=prior.correctionC.xyz;
                if(b==priorTriangle.nodes.x)priorB+=prior.correctionA.xyz;
                else if(b==priorTriangle.nodes.y)priorB+=prior.correctionB.xyz;
                else if(b==priorTriangle.nodes.z)priorB+=prior.correctionC.xyz;
                if(c==priorTriangle.nodes.x)priorC+=prior.correctionA.xyz;
                else if(c==priorTriangle.nodes.y)priorC+=prior.correctionB.xyz;
                else if(c==priorTriangle.nodes.z)priorC+=prior.correctionC.xyz;
            }
            const float3 priorSurface=bary.x*priorA+bary.y*priorB+bary.z*priorC;
            const float3 hitOffset=contact.identity.z!=0u
                ? pb_coin_support_offset(coin,normal,config.scales.y,config.scales.z)
                : pb_coin_sample_offset(contact.identity.y,config.scales.y,
                    config.scales.z,float4(coin.orientation));
            const float3 rn=cross(hitOffset,normal);
            const float3 inverseRn=pb_apply_inverse_inertia(coin,rn,config.scales.y,config.scales.z);
            const float achievedSeparation=dot(priorSurface-coinCorrectionAccum-
                cross(coinRotationAccum,hitOffset),normal);
            const float remainingDepth=max(0.0f,
                contact.barycentricAndDepth.w-achievedSeparation);
            const float denominator=coin.positionAndInverseMass.w+nodeWeights[sampleIndex]+dot(rn,inverseRn);
            const float lambdaNormal=remainingDepth/max(denominator,1.0e-12f)*config.solver.z;

            const float shellFloor=config.geometry.y+config.geometry.x;
            const float wa=pb_contact_node_inverse_mass(nodes[a],normal,shellFloor);
            const float wb=pb_contact_node_inverse_mass(nodes[b],normal,shellFloor);
            const float wc=pb_contact_node_inverse_mass(nodes[c],normal,shellFloor);
            contact.correctionA=pb_f4(wa*bary.x*
                (lambdaNormal*normal));
            contact.correctionB=pb_f4(wb*bary.y*
                (lambdaNormal*normal));
            contact.correctionC=pb_f4(wc*bary.z*
                (lambdaNormal*normal));
            contact.correctionCoin=pb_f4(-coin.positionAndInverseMass.w*
                (lambdaNormal*normal));
            contact.angularVelocityChangeCoin=pb_f4(-lambdaNormal*inverseRn);
            coinRotationAccum+=contact.angularVelocityChangeCoin.xyz;
            // A ground-blocked film reaction is carried by the table. Retain
            // its normal impulse for friction even though the node cannot move.
            if(normal.y<0.0f){
                if(wa==0.0f){const float dl=-normal.y*bary.x*lambdaNormal;floorLoads[a].x+=dl/config.scales.x;nodes[a].restPosition.w+=dl;}
                if(wb==0.0f){const float dl=-normal.y*bary.y*lambdaNormal;floorLoads[b].x+=dl/config.scales.x;nodes[b].restPosition.w+=dl;}
                if(wc==0.0f){const float dl=-normal.y*bary.z*lambdaNormal;floorLoads[c].x+=dl/config.scales.x;nodes[c].restPosition.w+=dl;}
            }
            if(lambdaNormal>0.0f){
                const uint li=coinIndex*sampleCount+sampleIndex;
                PlasticBagContactGPU load=loads[li];
                const bool same=load.normalAndActive.w>0.0f&&load.identity.x==contact.identity.x&&
                    load.identity.z==contact.identity.z&&dot(load.normalAndActive.xyz,normal)>0.9f;
                const float budget=same?load.barycentricAndDepth.w:0.0f;
                const float3 savedStatic=same?load.correctionCoin.xyz:float3(0);
                load=PlasticBagContactGPU{};
                load.normalAndActive=contact.normalAndActive;load.identity=contact.identity;
                load.barycentricAndDepth=pb_f4(bary,budget+lambdaNormal/config.scales.x);
                load.correctionCoin=pb_f4(savedStatic);load.correctionA=pb_f4(savedStatic/config.scales.x);
                loads[li]=load;
            }
            coinCorrectionAccum+=contact.correctionCoin.xyz;
        }
        selected[sampleIndex]=contact;
        contacts[coinIndex*sampleCount+sampleIndex]=contact;
    }
}

kernel void pb_contact_reduce(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagContactGPU* contacts [[buffer(1)]],
    device PlasticBagNodeGPU* nodes [[buffer(2)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(3)]],
    device PlasticBagCoinGPU* coins [[buffer(4)]],
    device PlasticBagContactGPU* loads [[buffer(5)]],
    device mr_float4* floorLoads [[buffer(6)]],
    uint coinIndex [[thread_position_in_grid]]) { pb_contact_reduce_one(config,contacts,nodes,triangles,coins,loads,floorLoads,coinIndex); }

kernel void pb_contact_apply_nodes(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(2)]],
    device const PlasticBagContactGPU* contacts [[buffer(3)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if (nodeIndex >= config.counts.x) return;
    PlasticBagNodeGPU node = nodes[nodeIndex];
    if (!(node.positionAndInverseMass.w > 0.0f)) return;
    float3 correction = float3(0.0f);
    const uint sampleCount=uint(config.scales.w);
    for(uint contactIndex=0u;contactIndex<config.counts.w*sampleCount;++contactIndex){
        const PlasticBagContactGPU contact=contacts[contactIndex];
        if(contact.normalAndActive.w==0.0f)continue;
        const PlasticBagTriangleGPU triangle=triangles[contact.identity.x];
        correction+=nodeIndex==triangle.nodes.x ? contact.correctionA.xyz :
            nodeIndex==triangle.nodes.y ? contact.correctionB.xyz :
            nodeIndex==triangle.nodes.z ? contact.correctionC.xyz : float3(0.0f);
    }
    node.positionAndInverseMass.xyz += correction;

    nodes[nodeIndex] = node;
}

kernel void pb_contact_apply_coins(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagContactGPU* contacts [[buffer(2)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if (coinIndex >= config.counts.w) return;
    PlasticBagCoinGPU coin = coins[coinIndex];
    float3 correction=float3(0.0f);
    float3 angularVelocityChange=float3(0.0f);
    const uint sampleCount=uint(config.scales.w);
    for(uint sampleIndex=0u;sampleIndex<sampleCount;++sampleIndex){
        const PlasticBagContactGPU contact=contacts[coinIndex*sampleCount+sampleIndex];
        correction+=contact.correctionCoin.xyz;
        angularVelocityChange+=contact.angularVelocityChangeCoin.xyz;
    }
    coin.positionAndInverseMass.xyz += correction;
    // Stored rotation is a position-level inertia correction. Finalize derives
    // angular velocity from this corrected pose, exactly as for table contact.
    coin.orientation=mr_float4(pb_apply_rotation(float4(coin.orientation),angularVelocityChange));
    coins[coinIndex] = coin;
}

kernel void pb_coin_pair_solve(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device const PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagCoinPairGPU* pairs [[buffer(2)]],
    device PlasticBagCoinPairScratchGPU* scratch [[buffer(3)]],
    device PlasticBagPairLoadGPU* pairLoads [[buffer(4)]],
    uint pairIndex [[thread_position_in_grid]]) {
    const uint pairCount=config.counts.w*(config.counts.w-1u)/2u;
    if(pairIndex>=pairCount)return;
    const PlasticBagCoinPairGPU pair=pairs[pairIndex];
    const PlasticBagCoinGPU coinA=coins[pair.coins.x];
    const PlasticBagCoinGPU coinB=coins[pair.coins.y];
    PlasticBagCoinPairScratchGPU result{};
    const float3 centerA=pb_xyz(coinA.positionAndInverseMass);
    const float3 centerB=pb_xyz(coinB.positionAndInverseMass);
    const float3 centerDelta=centerA-centerB;
    const float radius=config.scales.y,halfThickness=config.scales.z;
    const float pairBound=2.0f*sqrt(radius*radius+halfThickness*halfThickness);
    if(dot(centerDelta,centerDelta)>pairBound*pairBound){
        scratch[pairIndex]=result;
        return;
    }
    const float3 axisA=normalize(pb_rotate(coinA.orientation,float3(0.0f,1.0f,0.0f)));
    const float3 axisB=normalize(pb_rotate(coinB.orientation,float3(0.0f,1.0f,0.0f)));
    const float3 candidateAxes[8]={axisA,axisB,centerDelta,cross(axisA,axisB),
        centerDelta-dot(centerDelta,axisA)*axisA,
        centerDelta-dot(centerDelta,axisB)*axisB,
        cross(axisA,centerDelta),cross(axisB,centerDelta)};
    float leastPenetration=1.0e10f;
    float3 bestNormal=float3(1.0f,0.0f,0.0f);
    bool separated=false;
    for(uint axisIndex=0u;axisIndex<8u;++axisIndex){
        const float axisLength=length(candidateAxes[axisIndex]);
        if(axisLength<1.0e-7f)continue;
        float3 normal=candidateAxes[axisIndex]/axisLength;
        float signedCenterDistance=dot(centerDelta,normal);
        if(signedCenterDistance<0.0f){
            normal=-normal;
            signedCenterDistance=-signedCenterDistance;
        }else if(abs(signedCenterDistance)<1.0e-7f&&(pairIndex&1u)!=0u){
            normal=-normal;
        }
        const float axisProjectionA=dot(axisA,normal);
        const float axisProjectionB=dot(axisB,normal);
        const float supportA=radius*length(cross(axisA,normal))+
            halfThickness*abs(axisProjectionA);
        const float supportB=radius*length(cross(axisB,normal))+
            halfThickness*abs(axisProjectionB);
        const float gap=signedCenterDistance-supportA-supportB;
        if(gap>0.00002f){
            separated=true;
            break;
        }
        const float penetration=-gap;
        if(penetration<leastPenetration){
            leastPenetration=penetration;
            bestNormal=normal;
        }
    }
    if(!separated&&leastPenetration>0.0f&&!pb_pair_certified_separated(coinA,coinB,radius,halfThickness)){
        // Support projections give the rigid-cylinder separating translation
        // even when thin coplanar discs make the sampled SDF report only half
        // a coin gauge. Relax each Jacobi sweep and cap it to control clusters.
        const float correction=min(0.80f*leastPenetration,0.003f);
        const float invMassA=coinA.positionAndInverseMass.w;
        const float invMassB=coinB.positionAndInverseMass.w;
        const float denominator=invMassA+invMassB;
        if(denominator>1.0e-12f){
            const float lambda=correction/denominator;
            pairLoads[pairIndex].normalImpulse.xyz+=bestNormal*(lambda/config.scales.x);
            pairLoads[pairIndex].normalImpulse.w+=lambda/config.scales.x;
            result.correctionA=pb_f4(invMassA*lambda*bestNormal);
            result.correctionB=pb_f4(-invMassB*lambda*bestNormal);
        }
    }
    scratch[pairIndex]=result;
}

struct PlasticBagFrameGPU {
    float3 origin;
    float3 widthAxis;
    float3 upAxis;
    float3 depthAxis;
    float width;
    float height;
};

inline float3 pb_panel_grid_point(device const PlasticBagNodeGPU* nodes,
    device const uint4* panelPairs,const float u,const float v,const bool back) {
    constexpr uint width=25u,height=23u;
    const float gx=clamp(u,0.0f,1.0f)*float(width-1u);
    const float gy=clamp(v,0.0f,1.0f)*float(height-1u);
    const uint x0=min(uint(gx),width-2u),y0=min(uint(gy),height-2u);
    const float tx=gx-float(x0),ty=gy-float(y0);
    const uint i00=y0*width+x0,i10=i00+1u,i01=i00+width,i11=i01+1u;
    const uint n00=back?panelPairs[i00].y:panelPairs[i00].x;
    const uint n10=back?panelPairs[i10].y:panelPairs[i10].x;
    const uint n01=back?panelPairs[i01].y:panelPairs[i01].x;
    const uint n11=back?panelPairs[i11].y:panelPairs[i11].x;
    const float3 p00=pb_xyz(nodes[n00].positionAndInverseMass);
    const float3 p10=pb_xyz(nodes[n10].positionAndInverseMass);
    const float3 p01=pb_xyz(nodes[n01].positionAndInverseMass);
    const float3 p11=pb_xyz(nodes[n11].positionAndInverseMass);
    return mix(mix(p00,p10,tx),mix(p01,p11,tx),ty);
}

inline PlasticBagFrameGPU pb_bag_frame(device const PlasticBagNodeGPU* nodes,
    device const uint4* panelPairs) {
    constexpr uint width=25u,height=23u;
    float3 bottom=float3(0.0f),top=float3(0.0f);
    float3 left=float3(0.0f),right=float3(0.0f);
    for(uint u=0u;u<width;++u){
        const uint bottomIndex=u,topIndex=(height-1u)*width+u;
        bottom+=0.5f*(pb_xyz(nodes[panelPairs[bottomIndex].x].positionAndInverseMass)+
            pb_xyz(nodes[panelPairs[bottomIndex].y].positionAndInverseMass));
        top+=0.5f*(pb_xyz(nodes[panelPairs[topIndex].x].positionAndInverseMass)+
            pb_xyz(nodes[panelPairs[topIndex].y].positionAndInverseMass));
    }
    bottom/=float(width);top/=float(width);
    for(uint v=0u;v<height;++v){
        const uint li=v*width,ri=li+width-1u;
        left+=0.5f*(pb_xyz(nodes[panelPairs[li].x].positionAndInverseMass)+
            pb_xyz(nodes[panelPairs[li].y].positionAndInverseMass));
        right+=0.5f*(pb_xyz(nodes[panelPairs[ri].x].positionAndInverseMass)+
            pb_xyz(nodes[panelPairs[ri].y].positionAndInverseMass));
    }
    left/=float(height);right/=float(height);
    const float3 upDelta=top-bottom;
    const float upLength=length(upDelta);
    const float3 upAxis=upLength>1.0e-8f?upDelta/upLength:float3(0.0f,1.0f,0.0f);
    float3 widthDelta=right-left;
    widthDelta-=upAxis*dot(widthDelta,upAxis);
    const float widthLength=length(widthDelta);
    const float3 widthAxis=widthLength>1.0e-8f?widthDelta/widthLength:float3(1.0f,0.0f,0.0f);
    float3 depthAxis=cross(widthAxis,upAxis);
    const float depthLength=length(depthAxis);
    depthAxis=depthLength>1.0e-8f?depthAxis/depthLength:float3(0.0f,0.0f,1.0f);
    const uint mid=(height/2u)*width+width/2u;
    const float3 front=pb_xyz(nodes[panelPairs[mid].x].positionAndInverseMass);
    const float3 back=pb_xyz(nodes[panelPairs[mid].y].positionAndInverseMass);
    if(dot(back-front,depthAxis)<0.0f)depthAxis=-depthAxis;
    return {bottom,widthAxis,upAxis,depthAxis,max(dot(right-left,widthAxis),1.0e-6f),
        max(dot(top-bottom,upAxis),1.0e-6f)};
}

inline void pb_constrain_coin_to_panel_gap(
    thread PlasticBagCoinGPU& coin,const float3 position,
    constant PlasticBagConfigGPU& config,
    device const PlasticBagNodeGPU* nodes,
    device const uint4* panelPairs,
    const PlasticBagFrameGPU frame) {
    const float3 local=position-frame.origin;
    const float x=dot(local,frame.widthAxis),y=dot(local,frame.upAxis);
    if(y<0.0f||y>frame.height)return;
    const float u=clamp(x/frame.width+0.5f,0.0f,1.0f);
    const float v=clamp(y/frame.height,0.0f,1.0f);
    const float zFront=dot(pb_panel_grid_point(nodes,panelPairs,u,v,false)-
        frame.origin,frame.depthAxis);
    const float zBack=dot(pb_panel_grid_point(nodes,panelPairs,u,v,true)-
        frame.origin,frame.depthAxis);
    const float availableHalfDepth=0.5f*abs(zBack-zFront)-
        config.geometry.x-2.0e-5f;
    if(availableHalfDepth<config.scales.z)return;
    const float3 axis=pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f));
    const float axisSign=dot(axis,frame.depthAxis)<0.0f?-1.0f:1.0f;
    const float currentProjection=abs(dot(axis,frame.depthAxis));
    const float currentExtent=config.scales.y*sqrt(max(0.0f,
        1.0f-currentProjection*currentProjection))+config.scales.z*currentProjection;
    if(currentExtent<=availableHalfDepth)return;
    float lo=0.0f,hi=1.0f;
    for(uint iteration=0u;iteration<20u;++iteration){
        const float mid=0.5f*(lo+hi);
        const float extent=config.scales.y*sqrt(max(0.0f,1.0f-mid*mid))+
            config.scales.z*mid;
        if(extent>availableHalfDepth)lo=mid;else hi=mid;
    }
    const float3 targetAxis=axisSign*frame.depthAxis;
    const float3 rotationAxis=cross(axis,targetAxis);
    const float sine=length(rotationAxis);
    const float cosine=clamp(dot(axis,targetAxis),-1.0f,1.0f);
    if(sine>1.0e-7f){
        const float angle=atan2(sine,cosine);
        coin.orientation=mr_float4(pb_apply_rotation(float4(coin.orientation),
            rotationAxis*(angle/sine)));
    }
    const float3 angularVelocity=coin.angularVelocity.xyz;
    coin.angularVelocity=mr_float4(dot(angularVelocity,targetAxis)*targetAxis,0.0f);
}

inline float pb_pouch_depth_clearance(
    const PlasticBagFrameGPU frame,const float x,const float y,
    const float extentDepth,constant PlasticBagConfigGPU& config,
    device const PlasticBagNodeGPU* nodes,
    device const uint4* panelPairs) {
    const float u=clamp(x/frame.width+0.5f,0.0f,1.0f);
    const float v=clamp(y/frame.height,0.0f,1.0f);
    const float front=dot(pb_panel_grid_point(nodes,panelPairs,u,v,false)-
        frame.origin,frame.depthAxis);
    const float back=dot(pb_panel_grid_point(nodes,panelPairs,u,v,true)-
        frame.origin,frame.depthAxis);
    return 0.5f*abs(back-front)-extentDepth-config.geometry.x-2.0e-5f;
}

inline bool pb_pouch_footprint_depth_interval(
    const PlasticBagFrameGPU frame,const float x,const float y,const float z,
    const PlasticBagCoinGPU coin,constant PlasticBagConfigGPU& config,
    device const PlasticBagNodeGPU* nodes,device const uint4* panelPairs,
    thread float& low,thread float& high) {
    low=-INFINITY;
    high=INFINITY;
    const float3 center=frame.origin+x*frame.widthAxis+y*frame.upAxis+z*frame.depthAxis;
    const float clearance=config.geometry.x+2.0e-5f;
    for(uint sampleIndex=0u;sampleIndex<PB_COIN_PAIR_SAMPLE_COUNT;++sampleIndex){
        const float3 offset=pb_coin_sample_offset(sampleIndex,config.scales.y,
            config.scales.z,float4(coin.orientation));
        const float3 sampleLocal=center+offset-frame.origin;
        const float sampleX=dot(sampleLocal,frame.widthAxis);
        const float sampleY=dot(sampleLocal,frame.upAxis);
        if(sampleY<0.001f||sampleY>frame.height||abs(sampleX)>0.5f*frame.width)
            continue;
        const float u=clamp(sampleX/frame.width+0.5f,0.0f,1.0f);
        const float v=clamp(sampleY/frame.height,0.0f,1.0f);
        const float front=dot(pb_panel_grid_point(nodes,panelPairs,u,v,false)-
            frame.origin,frame.depthAxis);
        const float back=dot(pb_panel_grid_point(nodes,panelPairs,u,v,true)-
            frame.origin,frame.depthAxis);
        const float offsetDepth=dot(offset,frame.depthAxis);
        low=max(low,min(front,back)+clearance-offsetDepth);
        high=min(high,max(front,back)-clearance-offsetDepth);
        if(low>high)return false;
    }
    return high>=low;
}

inline float3 pb_project_pouch_seam_position(
    const float3 inputPosition,const PlasticBagCoinGPU coin,
    constant PlasticBagConfigGPU& config,
    device const PlasticBagNodeGPU* nodes,
    device const uint4* panelPairs,const bool searchForFeasibleCenter) {
    const PlasticBagFrameGPU frame=pb_bag_frame(nodes,panelPairs);
    const float3 diskAxis=pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f));
    const float projectionX=dot(diskAxis,frame.widthAxis);
    const float projectionY=dot(diskAxis,frame.upAxis);
    const float projectionZ=dot(diskAxis,frame.depthAxis);
    const float extentX=config.scales.y*sqrt(max(0.0f,1.0f-projectionX*projectionX))+
        config.scales.z*abs(projectionX);
    const float extentY=config.scales.y*sqrt(max(0.0f,1.0f-projectionY*projectionY))+
        config.scales.z*abs(projectionY);
    const float extentDepth=config.scales.y*sqrt(max(0.0f,1.0f-projectionZ*projectionZ))+
        config.scales.z*abs(projectionZ);
    float3 local=inputPosition-frame.origin;
    float x=dot(local,frame.widthAxis),y=dot(local,frame.upAxis),z=dot(local,frame.depthAxis);
    x=clamp(x,-0.5f*frame.width+extentX,0.5f*frame.width-extentX);
    // The bottom weld is closed; maintain support clearance above it.
    y=max(y,extentY+config.geometry.x+2.0e-5f);
    if(y<=frame.height&&y>=0.0f){
        float low=0.0f,high=0.0f;
        bool footprintFits=searchForFeasibleCenter&&pb_pouch_footprint_depth_interval(
            frame,x,y,z,coin,config,nodes,panelPairs,low,high);
        if(searchForFeasibleCenter&&!footprintFits){
            bool found=false;
            const float inwardSign=x>0.0f?-1.0f:1.0f;
            for(uint step=1u;step<=48u&&!found;++step){
                const float distance=0.001f*float(step);
                float candidateLow=0.0f,candidateHigh=0.0f;
                const float candidateY=y+distance;
                if(candidateY<=frame.height-extentY&&
                    pb_pouch_footprint_depth_interval(frame,x,candidateY,z,coin,
                        config,nodes,panelPairs,candidateLow,candidateHigh)){
                    y=candidateY;
                    low=candidateLow;
                    high=candidateHigh;
                    found=true;
                }
                const float candidateX=abs(x)>1.0e-7f
                    ? clamp(x+inwardSign*distance,-0.5f*frame.width+extentX,
                        0.5f*frame.width-extentX) : x;
                if(!found&&abs(candidateX-x)>1.0e-7f&&
                    pb_pouch_footprint_depth_interval(frame,candidateX,y,z,coin,
                        config,nodes,panelPairs,candidateLow,candidateHigh)){
                    x=candidateX;
                    low=candidateLow;
                    high=candidateHigh;
                    found=true;
                }
                if(!found){
                    const float diagonalX=abs(x)>1.0e-7f
                        ? clamp(x+inwardSign*distance,-0.5f*frame.width+extentX,
                            0.5f*frame.width-extentX) : x;
                    const float diagonalY=y+distance;
                    if(diagonalY<=frame.height-extentY&&
                        pb_pouch_footprint_depth_interval(frame,diagonalX,diagonalY,z,
                            coin,config,nodes,panelPairs,candidateLow,candidateHigh)){
                        x=diagonalX;
                        y=diagonalY;
                        low=candidateLow;
                        high=candidateHigh;
                        found=true;
                    }
                }
            }
            footprintFits=found;
        }
        if(footprintFits){
            z=clamp(z,low,high);
        }else{
            // If the whole footprint cannot fit, at least keep its center
            // between the local films while the bounded x/y search continues.
            const float available=pb_pouch_depth_clearance(frame,x,y,extentDepth,
                config,nodes,panelPairs);
            if(available>=0.0f){
                const float u=clamp(x/frame.width+0.5f,0.0f,1.0f);
                const float v=clamp(y/frame.height,0.0f,1.0f);
                const float front=dot(pb_panel_grid_point(nodes,panelPairs,u,v,false)-
                    frame.origin,frame.depthAxis);
                const float back=dot(pb_panel_grid_point(nodes,panelPairs,u,v,true)-
                    frame.origin,frame.depthAxis);
                const float center=0.5f*(front+back);
                z=clamp(z,center-available,center+available);
            }
        }
    }
    return frame.origin+x*frame.widthAxis+y*frame.upAxis+z*frame.depthAxis;
}

kernel void pb_coin_pair_ordered_cleanup(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagNodeGPU* nodes [[buffer(2)]],
    device const uint4* panelPairs [[buffer(3)]],
    device PlasticBagPairLoadGPU* pairLoads [[buffer(4)]],
    device mr_float4* diagnostics [[buffer(5)]],
    uint coinIndex [[thread_position_in_threadgroup]]) {
    constexpr uint coinCount=PLASTIC_BAG_COIN_COUNT;
    constexpr uint ringCount=coinCount-1u;
    threadgroup PlasticBagCoinGPU localCoins[coinCount];
    threadgroup float4 corrections[coinCount];
    threadgroup float4 rotations[coinCount];
    if(coinIndex>=coinCount)return;
    localCoins[coinIndex]=coins[coinIndex];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint orderedSweepCount=36u;
    for(uint sweep=0u;sweep<orderedSweepCount;++sweep){

        for(uint round=0u;round<ringCount;++round){
            uint partner=0u;
            if(coinIndex==ringCount)partner=round;
            else if(coinIndex==round)partner=ringCount;
            else partner=(2u*round+ringCount-coinIndex)%ringCount;
            corrections[coinIndex]=float4(0.0f);rotations[coinIndex]=float4(0.0f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(coinIndex<partner){
                const PlasticBagCoinGPU coinA=localCoins[coinIndex];
                const PlasticBagCoinGPU coinB=localCoins[partner];
                const float3 centerA=pb_xyz(coinA.positionAndInverseMass);
                const float3 centerB=pb_xyz(coinB.positionAndInverseMass);
                const float3 centerDelta=centerA-centerB;
                const float radius=config.scales.y,halfThickness=config.scales.z;
                const float pairBound=2.0f*sqrt(radius*radius+halfThickness*halfThickness);
                float3 correctionA=float3(0.0f),correctionB=float3(0.0f),rotationA=float3(0),rotationB=float3(0);
                if(dot(centerDelta,centerDelta)<=pairBound*pairBound){
                    float depth;float3 n,ra,rb;
                    const int status=pb_pair_mpr(coinA,coinB,radius,halfThickness,depth,n,ra,rb);
                    if(status==0&&atomic_fetch_or_explicit(((device atomic_uint*)diagnostics)+7,1u,memory_order_relaxed)==0u){
 device PlasticBagCoinGPU* debug=(device PlasticBagCoinGPU*)(diagnostics+3);debug[0]=coinA;debug[1]=coinB;
 }
                    if(status==1&&depth>5.e-8f){
                        const float wa=coinA.positionAndInverseMass.w,wb=coinB.positionAndInverseMass.w;
                        const float3 an=cross(ra,n),bn=cross(rb,n);
                        const float3 ia=pb_apply_inverse_inertia(coinA,an,radius,halfThickness),ib=pb_apply_inverse_inertia(coinB,bn,radius,halfThickness);
                        const float denominator=wa+wb+dot(an,ia)+dot(bn,ib);
                        const float lambda=(depth-5.e-8f)/max(denominator,1.e-12f);
                        const uint pi=coinIndex*(2u*coinCount-coinIndex-1u)/2u+(partner-coinIndex-1u);
                        pairLoads[pi].normalImpulse.xyz+=n*(lambda/config.scales.x);
                        pairLoads[pi].normalImpulse.w+=lambda/config.scales.x;
                        correctionA=wa*lambda*n;correctionB=-wb*lambda*n;
                        rotationA=lambda*ia;rotationB=-lambda*ib;
                    }
                }
                corrections[coinIndex]=float4(correctionA,0.0f);
                corrections[partner]=float4(correctionB,0.0f);
                rotations[coinIndex]=float4(rotationA,0);rotations[partner]=float4(rotationB,0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pb_coin_translation(localCoins[coinIndex],corrections[coinIndex].xyz);
            pb_coin_rotation(localCoins[coinIndex],rotations[coinIndex].xyz);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    }
    coins[coinIndex]=localCoins[coinIndex];
}

kernel void pb_coin_pair_apply(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagCoinPairScratchGPU* scratch [[buffer(2)]],
    device const PlasticBagCoinPairRefGPU* refs [[buffer(3)]],
    device const PlasticBagNodeSpanGPU* spans [[buffer(4)]],
    device const PlasticBagNodeGPU* nodes [[buffer(5)]],
    device const uint4* panelPairs [[buffer(6)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if (coinIndex>=config.counts.w) return;
    PlasticBagCoinGPU coin=coins[coinIndex];
    const uint begin=spans[coinIndex].range.x;
    const uint count=spans[coinIndex].range.y;
    float3 correction=float3(0.0f);
    float3 angularCorrection=float3(0.0f);
    for(uint r=0u;r<count;++r){
        const PlasticBagCoinPairRefGPU ref=refs[begin+r];
        const uint pairIndex=ref.identity.x;
        const PlasticBagCoinPairScratchGPU pair=scratch[pairIndex];
        correction+=ref.identity.y==0u ? pair.correctionA.xyz : pair.correctionB.xyz;
        angularCorrection+=ref.identity.y==0u ?
            pair.angularVelocityChangeA.xyz : pair.angularVelocityChangeB.xyz;
    }
    coin.positionAndInverseMass.xyz+=correction;
    float4 correctedOrientation=pb_apply_rotation(float4(coin.orientation),
        angularCorrection);
    coin.orientation=mr_float4(correctedOrientation);
    coins[coinIndex]=coin;
}

inline void pb_resolve_coin_pair_velocity(
    threadgroup PlasticBagCoinGPU& coinA,threadgroup PlasticBagCoinGPU& coinB,
    constant PlasticBagConfigGPU& config,const uint pairIndex,device PlasticBagPairLoadGPU* pairLoads,device mr_float4* diagnostics,device PlasticBagPairVelocityGeometryGPU* geometry) {
    const float3 centerA=pb_xyz(coinA.positionAndInverseMass);
    const float3 centerB=pb_xyz(coinB.positionAndInverseMass);
    const float3 centerDelta=centerA-centerB;
    const float velocityPairBound=2.0f*sqrt(config.scales.y*config.scales.y+
        config.scales.z*config.scales.z)+0.0007f;
    if(dot(centerDelta,centerDelta)>velocityPairBound*velocityPairBound)return;
    float bestDistance;float3 n,t1,t2;float3 points[4];float gaps[4];uint count=0u;
    if(geometry[pairIndex].validAndCount.x!=0u){
        const auto cached=geometry[pairIndex];count=cached.validAndCount.y;
        if(count==0u)return;
        n=cached.normal.xyz;bestDistance=cached.normal.w;
        t1=cached.tangent1.xyz;t2=cached.tangent2.xyz;
        for(uint j=0u;j<count;++j){points[j]=cached.pointsAndGap[j].xyz;gaps[j]=cached.pointsAndGap[j].w;}
    }else{
    bestDistance=1.0e10f;
    float3 bestNormal=float3(1.0f,0.0f,0.0f);
    float3 offsetA=float3(0.0f);
    float3 pointA=centerA;
    for(uint sampleIndex=0u;sampleIndex<PB_COIN_PAIR_SAMPLE_COUNT;++sampleIndex){
        const float3 sampleOffset=pb_coin_pair_sample_offset(sampleIndex,config.scales.y,
            config.scales.z,float4(coinA.orientation));
        const float3 samplePoint=centerA+sampleOffset;
        float3 normal;
        const float distance=pb_cylinder_sdf(samplePoint-centerB,config.scales.y,
            config.scales.z,float4(coinB.orientation),normal);
        if(distance<bestDistance){
            bestDistance=distance;
            bestNormal=normal;
            offsetA=sampleOffset;
            pointA=samplePoint;
        }
        const float3 sampleOffsetB=pb_coin_pair_sample_offset(sampleIndex,config.scales.y,
            config.scales.z,float4(coinB.orientation));
        const float3 samplePointB=centerB+sampleOffsetB;
        float3 reverseNormal;
        const float reverseDistance=pb_cylinder_sdf(samplePointB-centerA,
            config.scales.y,config.scales.z,float4(coinA.orientation),reverseNormal);
        if(reverseDistance<bestDistance){
            bestDistance=reverseDistance;
            bestNormal=-reverseNormal;
            pointA=samplePointB-reverseDistance*reverseNormal;
            offsetA=pointA-centerA;
        }
    }
    const float velocityNormalSide=dot(centerDelta,bestNormal);
    if(velocityNormalSide< -1.0e-7f||
        (abs(velocityNormalSide)<=1.0e-7f&&(pairIndex&1u)!=0u))
        bestNormal=-bestNormal;
    float penetration;float3 contactNormal,armA,armB;
    const int status=pb_pair_mpr(coinA,coinB,config.scales.y,config.scales.z,penetration,contactNormal,armA,armB);
    if(status==0&&atomic_fetch_or_explicit(((device atomic_uint*)diagnostics)+7,1u,memory_order_relaxed)==0u){
 device PlasticBagCoinGPU* debug=(device PlasticBagCoinGPU*)(diagnostics+3);debug[0]=coinA;debug[1]=coinB;
 }
    const bool overlap=status==1&&penetration>0.0f;
    if(overlap){bestDistance=-penetration;bestNormal=contactNormal;}
    else if(status==-1)bestDistance=max(0.0f,bestDistance);
    if(bestDistance>0.00035f){geometry[pairIndex].validAndCount=mr_uint4(1u,0u,0u,0u);return;}
    float3 contactOffsetA=offsetA,contactOffsetB=pointA-centerB-bestDistance*bestNormal;
    if(overlap){
        const float3 midpoint=.5f*(centerA+armA+centerB+armB);
        contactOffsetA=midpoint-centerA;contactOffsetB=midpoint-centerB;
    }
    // Build a four-point patch from the actual nearby sampled surfaces.
    // Extremal witnesses span a flat overlap; tilted/rim contact naturally
    // collapses to fewer distinct points. All impulses are reciprocal.
    n=bestNormal;
    const float3 reference=abs(n.x)<.8f?float3(1,0,0):float3(0,0,1);
    t1=normalize(cross(n,reference));t2=cross(n,t1);
    float3 extrema[4];float extGap[4];float score[4];
    for(uint k=0u;k<4u;++k){extrema[k]=.5f*(centerA+contactOffsetA+centerB+contactOffsetB);extGap[k]=bestDistance;score[k]=-1.e30f;}
    for(uint side=0u;side<2u;++side)for(uint sample=0u;sample<PB_COIN_PAIR_SAMPLE_COUNT;++sample){
     const float3 cp=side==0u?centerA+pb_coin_pair_sample_offset(sample,config.scales.y,config.scales.z,float4(coinA.orientation)):
      centerB+pb_coin_pair_sample_offset(sample,config.scales.y,config.scales.z,float4(coinB.orientation));
     float3 sn;const float gap=side==0u?pb_cylinder_sdf(cp-centerB,config.scales.y,config.scales.z,float4(coinB.orientation),sn):
      pb_cylinder_sdf(cp-centerA,config.scales.y,config.scales.z,float4(coinA.orientation),sn);
     const float3 oriented=side==0u?sn:-sn;
     if(gap>bestDistance+2.e-5f||gap<bestDistance-2.e-5f||dot(oriented,n)<.98f)continue;
     const float3 point=cp-.5f*gap*sn;
     const float values[4]={dot(point,t1),-dot(point,t1),dot(point,t2),-dot(point,t2)};
     for(uint k=0u;k<4u;++k)if(values[k]>score[k]){score[k]=values[k];extrema[k]=point;extGap[k]=gap;}
    }
    count=0u;
    for(uint k=0u;k<4u;++k){bool duplicate=false;for(uint j=0u;j<count;++j)if(length(points[j]-extrema[k])<1.e-6f)duplicate=true;
     if(!duplicate){points[count]=extrema[k];gaps[count]=extGap[k];++count;}}
    PlasticBagPairVelocityGeometryGPU cached{};cached.normal=float4(n,bestDistance);
    cached.tangent1=float4(t1,0);cached.tangent2=float4(t2,0);
    for(uint j=0u;j<count;++j)cached.pointsAndGap[j]=float4(points[j],gaps[j]);
    cached.validAndCount=mr_uint4(1u,count,0u,0u);geometry[pairIndex]=cached;
    }
    const float support=bestDistance<=3.5e-5f?max(0.0f,dot(pairLoads[pairIndex].normalImpulse.xyz,n)):0.0f;
    const float sharedSupport=support/float(count);
    const float invMassA=coinA.positionAndInverseMass.w,invMassB=coinB.positionAndInverseMass.w;
    for(uint j=0u;j<count;++j){
     const float3 ra=points[j]-centerA,rb=points[j]-centerB;
     const float3 an=cross(ra,n),bn=cross(rb,n);
     const float3 ian=pb_apply_inverse_inertia(coinA,an,config.scales.y,config.scales.z),ibn=pb_apply_inverse_inertia(coinB,bn,config.scales.y,config.scales.z);
     const float kn=invMassA+invMassB+dot(an,ian)+dot(bn,ibn);
     float3 relative=coinA.velocityAndMass.xyz+cross(coinA.angularVelocity.xyz,ra)-coinB.velocityAndMass.xyz-cross(coinB.angularVelocity.xyz,rb);
     const float incoming=dot(coinA.predictedVelocity.xyz+cross(coinA.predictedAngularVelocity.xyz,ra)-coinB.predictedVelocity.xyz-cross(coinB.predictedAngularVelocity.xyz,rb),n);
     float target=-max(0.0f,gaps[j])/config.scales.x;
     // Zero explicit rigid restitution; retain approved bag/contact solver.
     auto state=pairLoads[pairIndex].lambda[j];
     const float nextN=max(-sharedSupport,state.w+(target-dot(relative,n))/max(kn,1.e-12f));
     const float dn=nextN-state.w;state.w=nextN;
     coinA.velocityAndMass.xyz+=invMassA*dn*n;coinB.velocityAndMass.xyz-=invMassB*dn*n;
     coinA.angularVelocity.xyz+=dn*ian;coinB.angularVelocity.xyz-=dn*ibn;
     relative=coinA.velocityAndMass.xyz+cross(coinA.angularVelocity.xyz,ra)-coinB.velocityAndMass.xyz-cross(coinB.angularVelocity.xyz,rb);
     const float3 ax=cross(ra,t1),az=cross(ra,t2),bx=cross(rb,t1),bz=cross(rb,t2);
     const float3 iax=pb_apply_inverse_inertia(coinA,ax,config.scales.y,config.scales.z),iaz=pb_apply_inverse_inertia(coinA,az,config.scales.y,config.scales.z);
     const float3 ibx=pb_apply_inverse_inertia(coinB,bx,config.scales.y,config.scales.z),ibz=pb_apply_inverse_inertia(coinB,bz,config.scales.y,config.scales.z);
     const float kxx=invMassA+invMassB+dot(ax,iax)+dot(bx,ibx),kzz=invMassA+invMassB+dot(az,iaz)+dot(bz,ibz),kxz=dot(ax,iaz)+dot(bx,ibz);
     const float det=max(kxx*kzz-kxz*kxz,1.e-20f);
     const float2 velocity=float2(dot(relative,t1),dot(relative,t2));
     const float2 oldT=float2(dot(state.xyz,t1),dot(state.xyz,t2));
     const float budget=max(0.0f,PLASTIC_BAG_REEDED_EDGE_FRICTION*(sharedSupport+nextN));
     const float2 nextT=pb_friction_disk(kxx,kzz,kxz,oldT,velocity,budget);
     const float2 dt=nextT-oldT;const float3 impulse=dt.x*t1+dt.y*t2;
     coinA.velocityAndMass.xyz+=invMassA*impulse;coinB.velocityAndMass.xyz-=invMassB*impulse;
     coinA.angularVelocity.xyz+=dt.x*iax+dt.y*iaz;coinB.angularVelocity.xyz-=dt.x*ibx+dt.y*ibz;
     state.xyz=nextT.x*t1+nextT.y*t2;pairLoads[pairIndex].lambda[j]=state;
    }
}

kernel void pb_coin_pair_velocity_solve(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device PlasticBagPairLoadGPU* pairLoads [[buffer(2)]],
    device mr_float4* diagnostics [[buffer(3)]],
    device PlasticBagPairVelocityGeometryGPU* geometry [[buffer(4)]],
    uint coinIndex [[thread_position_in_threadgroup]]) {
    if(config.counts.w==0u)return;
    constexpr uint coinCount=PLASTIC_BAG_COIN_COUNT;
    constexpr uint ringCount=coinCount-1u;
    threadgroup PlasticBagCoinGPU localCoins[coinCount];
    if(coinIndex>=coinCount)return;
    localCoins[coinIndex]=coins[coinIndex];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint round=0u;round<ringCount;++round){
        uint partner=0u;
        if(coinIndex==ringCount)partner=round;
        else if(coinIndex==round)partner=ringCount;
        else partner=(2u*round+ringCount-coinIndex)%ringCount;
        if(coinIndex<partner){
            const uint lo=coinIndex;
            const uint hi=partner;
            const uint pairIndex=lo*(2u*coinCount-lo-1u)/2u+(hi-lo-1u);
            pb_resolve_coin_pair_velocity(localCoins[coinIndex],localCoins[partner],
                config,pairIndex,pairLoads,diagnostics,geometry);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    coins[coinIndex]=localCoins[coinIndex];
}

kernel void pb_coin_pair_velocity_apply(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagCoinPairScratchGPU* scratch [[buffer(2)]],
    device const PlasticBagCoinPairRefGPU* refs [[buffer(3)]],
    device const PlasticBagNodeSpanGPU* spans [[buffer(4)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if(coinIndex>=config.counts.w)return;
    PlasticBagCoinGPU coin=coins[coinIndex];
    const uint begin=spans[coinIndex].range.x;
    const uint count=spans[coinIndex].range.y;
    float3 velocityChange=float3(0.0f);
    float3 angularVelocityChange=float3(0.0f);
    for(uint r=0u;r<count;++r){
        const PlasticBagCoinPairRefGPU ref=refs[begin+r];
        const PlasticBagCoinPairScratchGPU pair=scratch[ref.identity.x];
        velocityChange+=ref.identity.y==0u ? pair.correctionA.xyz : pair.correctionB.xyz;
        angularVelocityChange+=ref.identity.y==0u ?
            pair.angularVelocityChangeA.xyz : pair.angularVelocityChangeB.xyz;
    }
    coin.velocityAndMass.xyz+=velocityChange;
    coin.angularVelocity=mr_float4(pb_limit_angular_velocity(
        coin.angularVelocity.xyz+angularVelocityChange,config.geometry.w),0.0f);
    coins[coinIndex]=coin;
}

kernel void pb_finalize(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagCoinGPU* coins [[buffer(2)]],
    device const PlasticBagContactGPU* contacts [[buffer(3)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(4)]],
    device PlasticBagContactGPU* velocityImpulses [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const float inverseDt=1.0f/config.scales.x;
    if(index<config.counts.x){
        PlasticBagNodeGPU node=nodes[index];
        if(node.positionAndInverseMass.w>0.0f){
            node.velocityAndMass.xyz+=node.contactPositionCorrection.xyz*inverseDt;
            const float shellFloor=config.geometry.y+config.geometry.x;
            if(node.positionAndInverseMass.y<=shellFloor+1.0e-6f&&
               node.velocityAndMass.y<0.0f)node.velocityAndMass.y=0.0f;
            node.velocityAndMass.xyz*=max(0.0f,1.0f-config.solver.x*config.scales.x);
        }else node.velocityAndMass.xyz=float3(0.0f);
        nodes[index]=node;
    }
    if(index<config.counts.w){
        PlasticBagCoinGPU coin=coins[index];
        // Consistent XPBD pose-to-velocity update for both sides of contact.
        // Retaining only the coin's pre-projection velocity breaks momentum
        // exchange when a heavy coin deforms a very light membrane.
        float3 velocity=coin.predictedVelocity.xyz+coin.predictedPosition.xyz*inverseDt;
        float4 delta=float4(coin.predictedOrientation);
        if(delta.w<0.0f)delta=-delta;
        const float sineHalf=length(delta.xyz);
        float3 angularVelocity=coin.predictedAngularVelocity.xyz+(sineHalf>1.0e-8f
            ? delta.xyz*(2.0f*atan2(sineHalf,delta.w)*inverseDt/sineHalf)
            : 2.0f*delta.xyz*inverseDt);
        // Film contact remains active for coins supported by the table.
        // Its ordered manifold follows reconstruction; table contact is solved
        // separately. Do not discard plastic friction on floorContact.
        velocityImpulses[index]=PlasticBagContactGPU{};
        coin.velocityAndMass.xyz=velocity;
        coin.angularVelocity=mr_float4(pb_limit_angular_velocity(angularVelocity,
            config.geometry.w),0.0f);
        coins[index]=coin;
    }
}

kernel void pb_contact_apply_velocity_nodes(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagTriangleGPU* triangles [[buffer(2)]],
    device const PlasticBagContactGPU* velocityImpulses [[buffer(3)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if(nodeIndex>=config.counts.x)return;
    PlasticBagNodeGPU node=nodes[nodeIndex];
    if(!(node.positionAndInverseMass.w>0.0f))return;
    float3 velocityChange=float3(0.0f);
    for(uint coinIndex=0u;coinIndex<config.counts.w;++coinIndex){
        const PlasticBagContactGPU impulse=velocityImpulses[coinIndex];
        if(impulse.normalAndActive.w==0.0f||impulse.identity.x>=config.counts.z)continue;
        const PlasticBagTriangleGPU triangle=triangles[impulse.identity.x];
        if(nodeIndex==triangle.nodes.x)velocityChange+=impulse.correctionA.xyz;
        else if(nodeIndex==triangle.nodes.y)velocityChange+=impulse.correctionB.xyz;
        else if(nodeIndex==triangle.nodes.z)velocityChange+=impulse.correctionC.xyz;
    }
    node.velocityAndMass.xyz+=velocityChange;
    nodes[nodeIndex]=node;
}

kernel void pb_project_ground(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagCoinGPU* coins [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const float shellFloor=config.geometry.y+config.geometry.x;
    if(index<config.counts.x){
        PlasticBagNodeGPU node=nodes[index];
        pb_project_film_floor(node,shellFloor);
        nodes[index]=node;
    }
    if(index<config.counts.w){
        PlasticBagCoinGPU coin=coins[index];
        pb_table_project(coin,config);
        coins[index]=coin;
    }
}

// The side welds and front/back seam are closed boundaries. The triangle
// contact patch handles ordinary impacts; this narrow projection prevents a
// fast, thin coin edge from tunnelling through the zero-thickness seam itself.
kernel void pb_project_pouch_seams(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device const PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagCoinGPU* coins [[buffer(2)]],
    device const uint4* panelPairs [[buffer(3)]],
    device PlasticBagSeamContactGPU* seamContacts [[buffer(4)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if(coinIndex>=config.counts.w)return;
    PlasticBagCoinGPU coin=coins[coinIndex];
    PlasticBagSeamContactGPU seamContact{};
    seamContact.nodesAndActive=mr_uint4(0xffffffffu,0xffffffffu,0u,0u);
    float3 position=pb_project_pouch_seam_position(pb_xyz(coin.positionAndInverseMass),
        coin,config,nodes,panelPairs,true);
    const PlasticBagFrameGPU frame=pb_bag_frame(nodes,panelPairs);
    pb_constrain_coin_to_panel_gap(coin,position,config,nodes,panelPairs,frame);
    position=pb_project_pouch_seam_position(position,coin,config,nodes,panelPairs,true);
    const float3 diskAxis=pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f));
    const float upProjection=dot(diskAxis,frame.upAxis);
    const float extentUp=config.scales.y*sqrt(max(0.0f,1.0f-upProjection*upProjection))+
        config.scales.z*abs(upProjection);
    const float3 local=position-frame.origin;
    const float x=dot(local,frame.widthAxis),y=dot(local,frame.upAxis),z=dot(local,frame.depthAxis);
    const float extentX=config.scales.y*sqrt(max(0.0f,1.0f-dot(diskAxis,frame.widthAxis)*
        dot(diskAxis,frame.widthAxis)))+config.scales.z*abs(dot(diskAxis,frame.widthAxis));
    const float extentDepth=config.scales.y*sqrt(max(0.0f,1.0f-dot(diskAxis,frame.depthAxis)*
        dot(diskAxis,frame.depthAxis)))+config.scales.z*abs(dot(diskAxis,frame.depthAxis));
    const float u=clamp(x/frame.width+0.5f,0.0f,1.0f);
    const float v=clamp(y/frame.height,0.0f,1.0f);
    const float zFront=dot(pb_panel_grid_point(nodes,panelPairs,u,v,false)-
        frame.origin,frame.depthAxis);
    const float zBack=dot(pb_panel_grid_point(nodes,panelPairs,u,v,true)-
        frame.origin,frame.depthAxis);
    const bool insideDepth=z+extentDepth>=min(zFront,zBack)-config.geometry.x&&
        z-extentDepth<=max(zFront,zBack)+config.geometry.x;
    const bool insideWidth=abs(x)<=0.5f*frame.width+extentX;
    const bool belowMouth=y<=frame.height+extentUp;
    const float penetration=extentUp+config.geometry.x-y;
    if(insideDepth&&insideWidth&&belowMouth&&penetration>1.0e-6f){
        constexpr uint width=25u;
        const float gx=u*float(width-1u);
        const uint x0=min(uint(gx),width-2u);
        const float tx=gx-float(x0);
        const uint nodeA=panelPairs[x0].x;
        const uint nodeB=panelPairs[x0+1u].x;
        const float baryA=1.0f-tx,baryB=tx;
        const float floor=config.geometry.y+config.geometry.x;
        const bool downwardReaction=frame.upAxis.y>1.0e-4f;
        const float wa=nodes[nodeA].positionAndInverseMass.w>0.0f&&
            !(downwardReaction&&nodes[nodeA].positionAndInverseMass.y<=floor+1.0e-6f)
            ? nodes[nodeA].positionAndInverseMass.w*1.0e-4f:0.0f;
        const float wb=nodes[nodeB].positionAndInverseMass.w>0.0f&&
            !(downwardReaction&&nodes[nodeB].positionAndInverseMass.y<=floor+1.0e-6f)
            ? nodes[nodeB].positionAndInverseMass.w*1.0e-4f:0.0f;
        const float coinWeight=coin.positionAndInverseMass.w;
        const float denominator=coinWeight+wa*baryA*baryA+wb*baryB*baryB;
        const float lambda=penetration/max(denominator,1.0e-12f)*config.solver.z;
        const float3 correctionCoin=coinWeight*lambda*frame.upAxis;
        seamContact.nodesAndActive=mr_uint4(nodeA,nodeB,0u,1u);
        seamContact.barycentricAndDepth=pb_f4(float3(baryA,baryB,0.0f),penetration);
        seamContact.correctionA=pb_f4(-wa*baryA*lambda*frame.upAxis);
        seamContact.correctionB=pb_f4(-wb*baryB*lambda*frame.upAxis);
        seamContact.correctionCoin=pb_f4(correctionCoin);
        position+=correctionCoin;
    }
    coin.positionAndInverseMass=pb_f4(position,coin.positionAndInverseMass.w);
    coins[coinIndex]=coin;
    seamContacts[coinIndex]=seamContact;
}

kernel void pb_apply_seam_reactions(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagSeamContactGPU* seamContacts [[buffer(2)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if(nodeIndex>=config.counts.x)return;
    PlasticBagNodeGPU node=nodes[nodeIndex];
    if(node.positionAndInverseMass.w>0.0f){
        float3 correction=float3(0.0f);
        for(uint coinIndex=0u;coinIndex<config.counts.w;++coinIndex){
            const PlasticBagSeamContactGPU contact=seamContacts[coinIndex];
            if(contact.nodesAndActive.w==0u)continue;
            if(contact.nodesAndActive.x==nodeIndex)correction+=contact.correctionA.xyz;
            if(contact.nodesAndActive.y==nodeIndex)correction+=contact.correctionB.xyz;
        }
        node.positionAndInverseMass.xyz+=correction;
        const float ground=config.geometry.y+config.geometry.x;
        node.positionAndInverseMass.y=max(node.positionAndInverseMass.y,ground);
        nodes[nodeIndex]=node;
    }
}

inline float pb_coin_directional_extent(
    const PlasticBagCoinGPU coin,const float3 direction,
    const float radius,const float halfThickness) {
    const float3 axis=normalize(pb_rotate(coin.orientation,float3(0.0f,1.0f,0.0f)));
    const float axial=dot(axis,direction);
    return radius*length(cross(axis,direction))+
        halfThickness*abs(axial);
}

// Collision against the actual welded side curves. The contact normal follows
// the full panel centerline and is transported from the previous physical
// substep. A single folded column can reverse its apparent interior direction;
// the temporal frame prevents that inversion from ejecting coins.
kernel void pb_project_pouch_side_seams(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device PlasticBagCoinGPU* coins [[buffer(2)]],
    device const uint4* panelPairs [[buffer(3)]],
    device PlasticBagSeamContactGPU* seamContacts [[buffer(4)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if(coinIndex>=config.counts.w)return;
    constexpr uint width=25u;
    constexpr uint height=23u;
    const float3 center=pb_xyz(coins[coinIndex].positionAndInverseMass);
    PlasticBagSeamContactGPU contact{};
    contact.nodesAndActive=mr_uint4(0xffffffffu,0xffffffffu,0u,0u);
    float bestPenetration=0.0f;
    float bestT=0.0f;
    uint bestA=0xffffffffu,bestB=0xffffffffu;
    float3 bestNormal=float3(0.0f);
    for(uint side=0u;side<2u;++side){
        const uint edgeColumn=side==0u?0u:width-1u;
        constexpr uint centerColumn=width/2u;
        for(uint row=0u;row+1u<height;++row){
            const uint pairA=row*width+edgeColumn;
            const uint pairB=(row+1u)*width+edgeColumn;
            const uint nodeA=panelPairs[pairA].x;
            const uint nodeB=panelPairs[pairB].x;
            const float3 a=pb_xyz(nodes[nodeA].positionAndInverseMass);
            const float3 b=pb_xyz(nodes[nodeB].positionAndInverseMass);
            const float3 segment=b-a;
            const float lengthSquared=dot(segment,segment);
            if(lengthSquared<1.0e-12f)continue;
            const float segmentLength=sqrt(lengthSquared);
            const float3 tangent=segment/segmentLength;
            const float along=dot(center-a,tangent);
            const float t=clamp(along/segmentLength,0.0f,1.0f);
            const float3 q=a+t*segment;
            const uint centerPairA=row*width+centerColumn;
            const uint centerPairB=(row+1u)*width+centerColumn;
            const uint centerFrontA=panelPairs[centerPairA].x;
            const uint centerBackA=panelPairs[centerPairA].y;
            const uint centerFrontB=panelPairs[centerPairB].x;
            const uint centerBackB=panelPairs[centerPairB].y;
            const float3 centerA=0.5f*(pb_xyz(nodes[centerFrontA].positionAndInverseMass)+
                pb_xyz(nodes[centerBackA].positionAndInverseMass));
            const float3 centerB=0.5f*(pb_xyz(nodes[centerFrontB].positionAndInverseMass)+
                pb_xyz(nodes[centerBackB].positionAndInverseMass));
            float3 inward=(1.0f-t)*centerA+t*centerB-q;
            inward-=tangent*dot(inward,tangent);
            const float inwardLength=length(inward);
            if(inwardLength<1.0e-6f)continue;
            inward/=inwardLength;

            // Follow the prior substep, not the held pose: once released the
            // pouch can tumble far from that kinematic frame. Temporal
            // transport keeps the material-side sign continuous as the bag
            // rotates or folds. The transformed rest direction is only a
            // fallback when the previous interior vector is degenerate.
            const float3 previousA=pb_xyz(nodes[nodeA].previousAndFixed);
            const float3 previousB=pb_xyz(nodes[nodeB].previousAndFixed);
            const float3 previousCenterA=0.5f*(
                pb_xyz(nodes[centerFrontA].previousAndFixed)+
                pb_xyz(nodes[centerBackA].previousAndFixed));
            const float3 previousCenterB=0.5f*(
                pb_xyz(nodes[centerFrontB].previousAndFixed)+
                pb_xyz(nodes[centerBackB].previousAndFixed));
            const float3 previousQ=previousA+t*(previousB-previousA);
            const float3 previousCenter=(1.0f-t)*previousCenterA+
                t*previousCenterB;
            float3 transportedInward=previousCenter-previousQ;
            transportedInward-=tangent*dot(transportedInward,tangent);
            const float transportedLength=length(transportedInward);
            bool useRestOrientation=transportedLength<1.0e-6f;
            if(!useRestOrientation){
                transportedInward/=transportedLength;
                const float temporalAlignment=dot(inward,transportedInward);
                if(temporalAlignment<0.0f)inward=-inward;
                useRestOrientation=abs(temporalAlignment)<1.0e-4f;
            }
            if(useRestOrientation){
                const float3 restA=pb_xyz(nodes[nodeA].restPosition);
                const float3 restB=pb_xyz(nodes[nodeB].restPosition);
                const float3 restCenterA=0.5f*(
                    pb_xyz(nodes[centerFrontA].restPosition)+
                    pb_xyz(nodes[centerBackA].restPosition));
                const float3 restCenterB=0.5f*(
                    pb_xyz(nodes[centerFrontB].restPosition)+
                    pb_xyz(nodes[centerBackB].restPosition));
                const float3 restQ=restA+t*(restB-restA);
                const float3 restCenter=(1.0f-t)*restCenterA+t*restCenterB;
                float3 referenceInward=restCenter-restQ;
                referenceInward-=tangent*dot(referenceInward,tangent);
                const float referenceLength=length(referenceInward);
                if(referenceLength<1.0e-6f)continue;
                referenceInward/=referenceLength;
                if(dot(inward,referenceInward)<0.0f)inward=-inward;
            }
            const float tangentExtent=pb_coin_directional_extent(coins[coinIndex],
                tangent,config.scales.y,config.scales.z);
            if(along < -tangentExtent)continue;
            if(row+2u==height&&along>segmentLength+tangentExtent)continue;
            const float support=pb_coin_directional_extent(coins[coinIndex],
                inward,config.scales.y,config.scales.z);
            const float required=support+2.0f*config.geometry.x+2.0e-5f;
            const float penetration=required-dot(center-q,inward);
            if(penetration>bestPenetration){
                bestPenetration=penetration;
                bestT=t;
                bestA=nodeA;
                bestB=nodeB;
                bestNormal=inward;
            }
        }
    }
    if(bestPenetration>1.0e-6f&&bestA<config.counts.x&&bestB<config.counts.x){
        const float baryA=1.0f-bestT,baryB=bestT;
        const float floor=config.geometry.y+config.geometry.x;
        const float wa=pb_contact_node_inverse_mass(nodes[bestA],bestNormal,floor);
        const float wb=pb_contact_node_inverse_mass(nodes[bestB],bestNormal,floor);
        const float coinWeight=coins[coinIndex].positionAndInverseMass.w;
        const float denominator=max(coinWeight+wa*baryA*baryA+
            wb*baryB*baryB,1.0e-12f);
        const float projectedDepth=min(bestPenetration,0.01f);
        const float lambda=projectedDepth/denominator*config.solver.z;
        const float3 correctionCoin=coinWeight*lambda*bestNormal;
        contact.nodesAndActive=mr_uint4(bestA,bestB,0u,1u);
        contact.barycentricAndDepth=pb_f4(float3(baryA,baryB,0.0f),bestPenetration);
        contact.correctionA=pb_f4(-wa*baryA*lambda*bestNormal);
        contact.correctionB=pb_f4(-wb*baryB*lambda*bestNormal);
        contact.correctionCoin=pb_f4(correctionCoin);
        contact.normalAndPadding=pb_f4(bestNormal);
        coins[coinIndex].positionAndInverseMass.xyz+=correctionCoin;
    }
    seamContacts[coinIndex]=contact;
}

// Apply split position corrections to the seam and move its previous position
// by the same amount. The following velocity solve adds the physical impulse,
// keeping positional stabilization out of the impact velocity.
kernel void pb_apply_side_seam_position_reactions(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagSeamContactGPU* seamContacts [[buffer(2)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if(nodeIndex>=config.counts.x)return;
    PlasticBagNodeGPU node=nodes[nodeIndex];
    if(node.positionAndInverseMass.w>0.0f){
        float3 correction=float3(0.0f);
        for(uint coinIndex=0u;coinIndex<config.counts.w;++coinIndex){
            const PlasticBagSeamContactGPU contact=seamContacts[coinIndex];
            if(contact.nodesAndActive.w==0u)continue;
            if(contact.nodesAndActive.x==nodeIndex)correction+=contact.correctionA.xyz;
            if(contact.nodesAndActive.y==nodeIndex)correction+=contact.correctionB.xyz;
        }
        const float3 oldPosition=node.positionAndInverseMass.xyz;
        float3 newPosition=oldPosition+correction;
        newPosition.y=max(newPosition.y,config.geometry.y+config.geometry.x);
        const float3 applied=newPosition-oldPosition;
        node.positionAndInverseMass.xyz=newPosition;
        node.previousAndFixed.xyz+=applied;
        nodes[nodeIndex]=node;
    }
}

// Resolve side-seam impact with the same rigid-body normal/friction impulse
// used for film contacts, including coin angular response and equal/opposite
// seam-node velocity changes.
kernel void pb_side_seam_velocity_solve(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagNodeGPU* nodes [[buffer(2)]],
    device PlasticBagSeamContactGPU* seamContacts [[buffer(3)]],
    uint coinIndex [[thread_position_in_grid]]) {
    if(coinIndex>=config.counts.w)return;
    PlasticBagSeamContactGPU contact=seamContacts[coinIndex];
    contact.correctionA=mr_float4(0.0f);
    contact.correctionB=mr_float4(0.0f);
    if(contact.nodesAndActive.w==0u){
        seamContacts[coinIndex]=contact;
        return;
    }
    PlasticBagCoinGPU coin=coins[coinIndex];
    const uint nodeAIndex=contact.nodesAndActive.x;
    const uint nodeBIndex=contact.nodesAndActive.y;
    const float baryA=contact.barycentricAndDepth.x;
    const float baryB=contact.barycentricAndDepth.y;
    const float3 normal=normalize(contact.normalAndPadding.xyz);
    const float3 arm=pb_coin_support_offset(coin,-normal,
        config.scales.y,config.scales.z);
    const float3 shellVelocity=baryA*nodes[nodeAIndex].velocityAndMass.xyz+
        baryB*nodes[nodeBIndex].velocityAndMass.xyz;
    const float3 relative=coin.velocityAndMass.xyz+
        cross(coin.angularVelocity.xyz,arm)-shellVelocity;
    const float normalSpeed=dot(relative,normal);
    const float floor=config.geometry.y+config.geometry.x;
    const float waN=pb_contact_node_inverse_mass(nodes[nodeAIndex],normal,floor);
    const float wbN=pb_contact_node_inverse_mass(nodes[nodeBIndex],normal,floor);
    const float shellWeightN=waN*baryA*baryA+wbN*baryB*baryB;
    const float3 normalArm=cross(arm,normal);
    const float rotationalWeightN=dot(normalArm,
        pb_apply_inverse_inertia(coin,normalArm,config.scales.y,config.scales.z));
    const float normalDenominator=max(coin.positionAndInverseMass.w+
        shellWeightN+rotationalWeightN,1.0e-12f);
    // The thin welded film is strongly damped; a super-elastic 1.03 factor
    // injected energy at each seam impact and is not a plausible LDPE proxy.
    const float normalMagnitude=normalSpeed<0.0f
        ? -(1.0f+PB_SEAM_RESTITUTION)*normalSpeed/normalDenominator:0.0f;
    const float3 normalImpulse=normalMagnitude*normal;
    const float3 angularAfterNormal=coin.angularVelocity.xyz+
        pb_apply_inverse_inertia(coin,cross(arm,normalImpulse),
            config.scales.y,config.scales.z);
    const float3 relativeAfterNormal=relative+
        (coin.positionAndInverseMass.w+shellWeightN)*normalImpulse+
        cross(angularAfterNormal-coin.angularVelocity.xyz,arm);
    const float3 tangent=relativeAfterNormal-
        dot(relativeAfterNormal,normal)*normal;
    const float tangentSpeed=length(tangent);
    float3 tangentImpulse=float3(0.0f);
    float waT=0.0f,wbT=0.0f;
    if(tangentSpeed>1.0e-8f&&normalMagnitude>0.0f){
        const float3 tangentDirection=tangent/tangentSpeed;
        waT=pb_contact_node_inverse_mass(nodes[nodeAIndex],tangentDirection,floor);
        wbT=pb_contact_node_inverse_mass(nodes[nodeBIndex],tangentDirection,floor);
        const float shellWeightT=waT*baryA*baryA+wbT*baryB*baryB;
        const float3 tangentArm=cross(arm,tangentDirection);
        const float rotationalWeightT=dot(tangentArm,
            pb_apply_inverse_inertia(coin,tangentArm,config.scales.y,config.scales.z));
        const float tangentDenominator=max(coin.positionAndInverseMass.w+
            shellWeightT+rotationalWeightT,1.0e-12f);
        const float magnitude=min(tangentSpeed/tangentDenominator,
            max(0.0f,config.solver.w)*normalMagnitude);
        tangentImpulse=-magnitude*tangentDirection;
    }
    const float3 impulse=normalImpulse+tangentImpulse;
    coin.velocityAndMass.xyz+=coin.positionAndInverseMass.w*impulse;
    coin.angularVelocity=mr_float4(pb_limit_angular_velocity(
        coin.angularVelocity.xyz+pb_apply_inverse_inertia(coin,
            cross(arm,impulse),config.scales.y,config.scales.z),
        config.geometry.w),0.0f);
    contact.correctionA=pb_f4(-baryA*(waN*normalImpulse+waT*tangentImpulse));
    contact.correctionB=pb_f4(-baryB*(wbN*normalImpulse+wbT*tangentImpulse));
    contact.correctionCoin=pb_f4(impulse);
    seamContacts[coinIndex]=contact;
    coins[coinIndex]=coin;
}

kernel void pb_apply_side_seam_velocity_reactions(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagNodeGPU* nodes [[buffer(1)]],
    device const PlasticBagSeamContactGPU* seamContacts [[buffer(2)]],
    uint nodeIndex [[thread_position_in_grid]]) {
    if(nodeIndex>=config.counts.x)return;
    PlasticBagNodeGPU node=nodes[nodeIndex];
    if(!(node.positionAndInverseMass.w>0.0f))return;
    float3 velocityChange=float3(0.0f);
    for(uint coinIndex=0u;coinIndex<config.counts.w;++coinIndex){
        const PlasticBagSeamContactGPU contact=seamContacts[coinIndex];
        if(contact.nodesAndActive.w==0u)continue;
        if(contact.nodesAndActive.x==nodeIndex)velocityChange+=contact.correctionA.xyz;
        if(contact.nodesAndActive.y==nodeIndex)velocityChange+=contact.correctionB.xyz;
    }
    node.velocityAndMass.xyz+=velocityChange;
    nodes[nodeIndex]=node;
}

// Prevent the two independently simulated envelope panels from crossing.
// Corresponding front/back nodes are paired by their shared sheet grid. The
// common welded side and bottom nodes are identical and need no correction.
kernel void pb_project_panel_order(
    device PlasticBagNodeGPU* nodes [[buffer(0)]],
    device const uint4* panelPairs [[buffer(1)]],
    uint pairIndex [[thread_position_in_grid]]) {
    const uint4 pair=panelPairs[pairIndex];
    if(pair.x==pair.y)return;
    PlasticBagNodeGPU front=nodes[pair.x];
    PlasticBagNodeGPU back=nodes[pair.y];
    const float gap=back.positionAndInverseMass.z-front.positionAndInverseMass.z;
    constexpr float minimumGap=1.0e-4f;
    if(gap>=minimumGap)return;
    const float wf=front.positionAndInverseMass.w;
    const float wb=back.positionAndInverseMass.w;
    const float weight=wf+wb;
    if(weight<=0.0f)return;
    const float correction=minimumGap-gap;
    front.positionAndInverseMass.z-=correction*(wf/weight);
    back.positionAndInverseMass.z+=correction*(wb/weight);
    nodes[pair.x]=front;
    nodes[pair.y]=back;
}

// Read-only substep energy instrumentation. Spring energy uses the exact
// compliance of the active membrane constraints; no energy is modified.
kernel void pb_phase_energy(
 constant PlasticBagConfigGPU& c [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const PlasticBagCoinGPU* coins [[buffer(2)]],
 device const PlasticBagEdgeGPU* edges [[buffer(3)]],
 device mr_float4* values [[buffer(4)]],
 constant uint2& slotAndVelocity [[buffer(5)]],
 uint i [[thread_position_in_grid]]) {
 float4 v(0.0f);
 if(i<c.counts.x){
  const PlasticBagNodeGPU n=nodes[i];
  float3 vel=slotAndVelocity.y ? n.velocityAndMass.xyz :
   (n.positionAndInverseMass.xyz-n.previousAndFixed.xyz)/c.scales.x;
  v.x+=0.5f*n.velocityAndMass.w*dot(vel,vel);
  v.y+=n.velocityAndMass.w*(-c.gravityAndBending.y)*(n.positionAndInverseMass.y-c.geometry.y);
 }
 if(i<c.counts.w){
  const PlasticBagCoinGPU coin=coins[i];
  float3 vel=coin.velocityAndMass.xyz;
  float3 omega=coin.angularVelocity.xyz;
  if(!slotAndVelocity.y){
   vel=(coin.positionAndInverseMass.xyz-coin.previousPosition.xyz)/c.scales.x;
   float4 dq=pb_quat_multiply(float4(coin.orientation),pb_quat_conjugate(float4(coin.previousOrientation)));
   if(dq.w<0.0f)dq=-dq;
   float ss=length(dq.xyz);
   omega=ss>1.e-8f?dq.xyz*(2.0f*atan2(ss,dq.w)/(c.scales.x*ss)):2.0f*dq.xyz/c.scales.x;
  }
  float3 local=pb_rotate(pb_quat_conjugate(float4(coin.orientation)),omega);
  float mass=coin.velocityAndMass.w;
  float axial=0.5f*mass*c.scales.y*c.scales.y;
  float radial=mass*(0.25f*c.scales.y*c.scales.y+c.scales.z*c.scales.z/3.0f);
  v.x+=0.5f*mass*dot(vel,vel)+0.5f*(radial*(local.x*local.x+local.z*local.z)+axial*local.y*local.y);
  v.y+=mass*(-c.gravityAndBending.y)*(coin.positionAndInverseMass.y-c.geometry.y);
 }
 if(i<c.counts.y){
  PlasticBagEdgeGPU e=edges[i];
  float len=length(nodes[e.nodes.x].positionAndInverseMass.xyz-nodes[e.nodes.y].positionAndInverseMass.xyz);
  float dl=len-e.material.x;
  v.z=0.5f*dl*dl/max(e.material.y,1.e-20f);
  v.w=len/e.material.x;
 }
 values[slotAndVelocity.x*max(c.counts.x,c.counts.y)+i]=mr_float4(v);
}
// Contact patch impulses. Tangential impulses are accumulated and projected
// onto the Coulomb disk. The position solve's normal impulse supplies the
// friction budget even when the reconstructed normal velocity is already zero.
kernel void pb_table_velocity(
 constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device PlasticBagCoinGPU* coins [[buffer(2)]],
 constant uint& firstCall [[buffer(3)]],device mr_float4* floorLoads [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i<c.counts.x){
  PlasticBagNodeGPU node=nodes[i];
  if(node.positionAndInverseMass.w>0.0f&&node.positionAndInverseMass.y<=c.geometry.y+c.geometry.x+1.e-6f){
   const float incoming=node.velocityAndMass.y;
   node.velocityAndMass.y=max(0.0f,incoming);
   if(firstCall)floorLoads[i].w=node.velocityAndMass.w*max(0.0f,-incoming);
   else floorLoads[i].w+=node.velocityAndMass.w*max(0.0f,-incoming);
   const float2 oldT=floorLoads[i].yz;
   float2 nextT=oldT-node.velocityAndMass.w*node.velocityAndMass.xz;
   const float budget=c.solver.w*max(0.0f,floorLoads[i].x+floorLoads[i].w);
   const float mag=length(nextT);if(mag>budget)nextT*=budget/max(mag,1.e-20f);
   node.velocityAndMass.xz+=node.positionAndInverseMass.w*(nextT-oldT);
   floorLoads[i].yz=nextT;nodes[i]=node;
  }
 }
 if(i>=c.counts.w)return;
 PlasticBagCoinGPU coin=coins[i];
 const float radius=c.scales.y,ht=c.scales.z;
 if(coin.positionAndInverseMass.y-pb_coin_vertical_extent(coin,radius,ht)>c.geometry.y+3.e-5f)return;
 float3 arms[20];uint count=0u;
 for(uint face=0u;face<2u;++face){
  const float y=face==0u?-ht:ht;
  for(uint j=0u;j<8u;++j){
   const float a=float(j)*0.78539816339f;
   const float3 arm=pb_rotate(float4(coin.orientation),float3(radius*cos(a),y,radius*sin(a)));
   if(coin.positionAndInverseMass.y+arm.y<=c.geometry.y+3.e-5f)arms[count++]=arm;
  }
  const float3 center=pb_rotate(float4(coin.orientation),float3(0,y,0));
  if(coin.positionAndInverseMass.y+center.y<=c.geometry.y+3.e-5f)arms[count++]=center;
 }
 const float3 support=pb_coin_support_offset(coin,float3(0,-1,0),radius,ht);
 arms[count++]=support;
 const float massInverse=coin.positionAndInverseMass.w;
 const float positionImpulse=firstCall?max(0.0f,
  (coin.velocityAndMass.y-coin.predictedVelocity.y)/massInverse):0.0f;
 const float sharedSupportImpulse=positionImpulse/float(count);
 float lambdaN[20]={0.0f};float2 lambdaT[20];
 for(uint j=0u;j<count;++j)lambdaT[j]=float2(0.0f);
 for(uint sweep=0u;sweep<8u;++sweep){
  for(uint j=0u;j<count;++j){
   const float3 r=arms[j];
   float3 v=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,r);
   const float3 rn=cross(r,float3(0,1,0));
   const float3 invRn=pb_apply_inverse_inertia(coin,rn,radius,ht);
   const float kn=massInverse+dot(rn,invRn);
   const float incoming=(coin.predictedVelocity.xyz+cross(coin.predictedAngularVelocity.xyz,r)).y;
   const float target=0.0f; // Zero explicit rigid restitution.
   const float newN=max(-sharedSupportImpulse,lambdaN[j]+(target-v.y)/kn);
   const float dn=newN-lambdaN[j];lambdaN[j]=newN;
   coin.velocityAndMass.y+=massInverse*dn;
   coin.angularVelocity.xyz+=dn*invRn;
   v=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,r);
   const float3 rx=cross(r,float3(1,0,0)),rz=cross(r,float3(0,0,1));
   const float3 ix=pb_apply_inverse_inertia(coin,rx,radius,ht),iz=pb_apply_inverse_inertia(coin,rz,radius,ht);
   const float kxx=massInverse+dot(rx,ix),kzz=massInverse+dot(rz,iz),kxz=dot(rx,iz);
   const float invDet=1.0f/max(kxx*kzz-kxz*kxz,1.e-20f);
   const float2 dt=-invDet*float2(kzz*v.x-kxz*v.z,kxx*v.z-kxz*v.x);
   const float budget=max(0.0f,c.solver.w*(lambdaN[j]+sharedSupportImpulse));
   const float2 next=pb_friction_disk(kxx,kzz,kxz,lambdaT[j],v.xz,budget);
   const float2 delta=next-lambdaT[j];lambdaT[j]=next;
   coin.velocityAndMass.xyz+=massInverse*float3(delta.x,0,delta.y);
   coin.angularVelocity.xyz+=delta.x*ix+delta.y*iz;
  }
 }
 coins[i]=coin;
}

// Static graph coloring guarantees disjoint node writes within each color.
// A single owner threadgroup keeps every sweep on GPU and synchronizes between
// colors; no float atomics and no per-constraint host command submission.
kernel void pb_edge_colored_solve(
 constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device PlasticBagEdgeGPU* edges [[buffer(2)]],
 device const uint* order [[buffer(3)]],
 device const mr_uint4* spans [[buffer(4)]],
 uint lane [[thread_index_in_threadgroup]]) {
 const uint colors=spans[0].w;
 for(uint sweep=0u;sweep<c.iterations.x;++sweep){
  for(uint color=0u;color<colors;++color){
   const uint begin=spans[color].x,end=begin+spans[color].y;
   for(uint j=begin+lane;j<end;j+=128u){
    const uint ei=order[j];PlasticBagEdgeGPU edge=edges[ei];
    const uint a=edge.nodes.x,b=edge.nodes.y;
    const float wa=nodes[a].positionAndInverseMass.w,wb=nodes[b].positionAndInverseMass.w;
    const float3 d=pb_node_difference(nodes[b],nodes[a]);
    const float len=length(d),alpha=edge.material.y/(c.scales.x*c.scales.x);
    if(len>1.e-9f){
     const float3 n=d/len;
     const float gamma=edge.material.w/c.scales.x;
     const float previousLength=length(nodes[b].previousAndFixed.xyz-nodes[a].previousAndFixed.xyz);
     const float3 stepDelta=c.scales.x*(nodes[b].velocityAndMass.xyz-nodes[a].velocityAndMass.xyz)+nodes[b].contactPositionCorrection.xyz-nodes[a].contactPositionCorrection.xyz;
     const float rate=dot(n,stepDelta);
     // Implicit Kelvin-Voigt strain damping acts only on extension rate.
     // Equal/opposite central corrections preserve rigid translation/rotation.
     const float dl=(-(len-edge.material.x)-alpha*edge.material.z-gamma*rate)/
          ((1.0f+gamma)*(wa+wb)+alpha);
     pb_node_delta(nodes[a],-wa*dl*n);
     pb_node_delta(nodes[b],wb*dl*n);
     edge.material.z+=dl;edges[ei]=edge;
    }
   }
   threadgroup_barrier(mem_flags::mem_device);
  }
  for(uint i=lane;i<c.counts.x;i+=128u){
   auto node=nodes[i];pb_project_film_floor(node,c.geometry.y+c.geometry.x);nodes[i]=node;
  }
  threadgroup_barrier(mem_flags::mem_device);
 }
}

// One owner updates all contacting bodies in place. A film vertex shared by
// two coins sees the preceding impulse, avoiding additive Jacobi impulses.
kernel void pb_film_load_reset(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagContactGPU* loads [[buffer(1)]],device mr_float4* floorLoads [[buffer(2)]],
 device PlasticBagContactGPU* fullLoads [[buffer(3)]],device PlasticBagPairLoadGPU* pairLoads [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i<c.counts.w*(c.counts.w-1u)/2u)pairLoads[i]=PlasticBagPairLoadGPU{};
 if(i<c.counts.x)floorLoads[i]=mr_float4(0.0f);
 if(i<c.counts.w*c.counts.z)fullLoads[i]=PlasticBagContactGPU{};
 if(i<c.counts.w*uint(c.scales.w))loads[i]=PlasticBagContactGPU{};
}
inline float3 pb_film_inverse_mass(const PlasticBagNodeGPU n,const float floor,
 const float3 reaction){
 float3 w=float3(n.positionAndInverseMass.w);
 if(n.positionAndInverseMass.y<=floor+1.e-7f&&reaction.y<0.0f&&abs(n.velocityAndMass.y)<1.e-8f)w.y=0.0f;
 return w;
}
inline float pb_film_contact_ke(const PlasticBagCoinGPU c,
 const PlasticBagNodeGPU a,const PlasticBagNodeGPU b,const PlasticBagNodeGPU d,
 const float R,const float h){
 const float mass=c.velocityAndMass.w,axial=.5f*mass*R*R,radial=mass*(.25f*R*R+h*h/3.0f);
 const float3 axis=pb_rotate(c.orientation,float3(0,1,0));const float spin=dot(c.angularVelocity.xyz,axis);
 float E=.5f*mass*dot(c.velocityAndMass.xyz,c.velocityAndMass.xyz)+
 .5f*(radial*dot(c.angularVelocity.xyz,c.angularVelocity.xyz)+(axial-radial)*spin*spin);
 E+=.5f*a.velocityAndMass.w*dot(a.velocityAndMass.xyz,a.velocityAndMass.xyz);
 E+=.5f*b.velocityAndMass.w*dot(b.velocityAndMass.xyz,b.velocityAndMass.xyz);
 E+=.5f*d.velocityAndMass.w*dot(d.velocityAndMass.xyz,d.velocityAndMass.xyz);
 return E;
}
kernel void pb_film_load_velocity(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],device PlasticBagCoinGPU* coins [[buffer(2)]],
 device const PlasticBagTriangleGPU* triangles [[buffer(3)]],
 device PlasticBagContactGPU* loads [[buffer(4)]],device mr_float4* diagnostics [[buffer(5)]],device mr_float4* floorLoads [[buffer(6)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane!=0u)return;
 const uint samples=uint(c.scales.w);const float floor=c.geometry.y+c.geometry.x;
 for(uint ci=0u;ci<c.counts.w;++ci){
  PlasticBagCoinGPU coin=coins[ci];
  for(uint si=0u;si<samples;++si){
   const uint li=ci*samples+si;PlasticBagContactGPU load=loads[li];
   if(load.normalAndActive.w==0.0f||load.barycentricAndDepth.w<=0.0f)continue;
   const auto tri=triangles[load.identity.x];const uint3 ids=tri.nodes.xyz;
   const float3 n=-load.normalAndActive.xyz;
   const float3 arm=load.identity.z!=0u?pb_coin_support_offset(coin,-n,c.scales.y,c.scales.z):
       pb_coin_sample_offset(si,c.scales.y,c.scales.z,float4(coin.orientation));
   float3 bary;
   const float3 point=pb_closest_triangle_point(arm,
     pb_node_coin_difference(nodes[ids.x],coin),pb_node_coin_difference(nodes[ids.y],coin),
     pb_node_coin_difference(nodes[ids.z],coin),bary);
   float3 sdfNormal;
   const float gap=pb_cylinder_sdf(point,c.scales.y,c.scales.z,
       float4(coin.orientation),sdfNormal);
   // A saved support impulse grants friction only while contact is still local.
   if(gap>c.geometry.x+2.e-5f)continue;
   const float beforeE=pb_film_contact_ke(coin,nodes[ids.x],nodes[ids.y],nodes[ids.z],c.scales.y,c.scales.z);
   const float3 wa=pb_film_inverse_mass(nodes[ids.x],floor,-n);
   const float3 wb=pb_film_inverse_mass(nodes[ids.y],floor,-n);
   const float3 wc=pb_film_inverse_mass(nodes[ids.z],floor,-n);
   const float3 shellMass=bary.x*bary.x*wa+bary.y*bary.y*wb+bary.z*bary.z*wc;
   float3 sv=bary.x*nodes[ids.x].velocityAndMass.xyz+bary.y*nodes[ids.y].velocityAndMass.xyz+bary.z*nodes[ids.z].velocityAndMass.xyz;
   float3 relative=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,arm)-sv;
   const float3 rn=cross(arm,n);
   const float kn=coin.positionAndInverseMass.w+dot(n,shellMass*n)+
       dot(rn,pb_apply_inverse_inertia(coin,rn,c.scales.y,c.scales.z));
   const float oldN=load.correctionB.w;
   // The pose update already applied J_position. Velocity correction may
   // unload it, while total normal impulse remains nonnegative.
   const float nextN=max(-load.barycentricAndDepth.w,oldN-dot(relative,n)/max(kn,1.e-12f));
   const float3 normalImpulse=(nextN-oldN)*n;
   load.correctionB.w=nextN;
   coin.velocityAndMass.xyz+=coin.positionAndInverseMass.w*normalImpulse;
   coin.angularVelocity.xyz+=pb_apply_inverse_inertia(coin,cross(arm,normalImpulse),c.scales.y,c.scales.z);
   nodes[ids.x].velocityAndMass.xyz-=bary.x*wa*normalImpulse;
   nodes[ids.y].velocityAndMass.xyz-=bary.y*wb*normalImpulse;
   nodes[ids.z].velocityAndMass.xyz-=bary.z*wc*normalImpulse;
   const float3 filmNormalReaction=-normalImpulse;
   if(wa.y==0.0f)floorLoads[ids.x].x+=-bary.x*filmNormalReaction.y;
   if(wb.y==0.0f)floorLoads[ids.y].x+=-bary.y*filmNormalReaction.y;
   if(wc.y==0.0f)floorLoads[ids.z].x+=-bary.z*filmNormalReaction.y;
   sv=bary.x*nodes[ids.x].velocityAndMass.xyz+bary.y*nodes[ids.y].velocityAndMass.xyz+bary.z*nodes[ids.z].velocityAndMass.xyz;
   relative=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,arm)-sv;
   const float3 axis=abs(n.y)<0.9f?float3(0,1,0):float3(1,0,0);
   const float3 t1=normalize(cross(n,axis)),t2=cross(n,t1);
   const float3 r1=cross(arm,t1),r2=cross(arm,t2);
   const float3 ir1=pb_apply_inverse_inertia(coin,r1,c.scales.y,c.scales.z);
   const float3 ir2=pb_apply_inverse_inertia(coin,r2,c.scales.y,c.scales.z);
   const float k11=coin.positionAndInverseMass.w+dot(t1,shellMass*t1)+dot(r1,ir1);
   const float k22=coin.positionAndInverseMass.w+dot(t2,shellMass*t2)+dot(r2,ir2);
   const float k12=dot(t1,shellMass*t2)+dot(r1,ir2);
   const float2 vt=float2(dot(relative,t1),dot(relative,t2));
   const float det=max(k11*k22-k12*k12,1.e-20f);
   const float2 delta=-float2(k22*vt.x-k12*vt.y,k11*vt.y-k12*vt.x)/det;
   const float3 oldT=load.correctionA.xyz;
   const float limit=max(0.0f,c.solver.w*(load.barycentricAndDepth.w+nextN));
   const float2 nextT=pb_friction_disk(k11,k22,k12,float2(dot(oldT,t1),dot(oldT,t2)),vt,limit);
   const float3 newT=nextT.x*t1+nextT.y*t2;
   const float3 impulse=newT-oldT;
   coin.velocityAndMass.xyz+=coin.positionAndInverseMass.w*impulse;
   coin.angularVelocity.xyz+=pb_apply_inverse_inertia(coin,cross(arm,impulse),c.scales.y,c.scales.z);
   nodes[ids.x].velocityAndMass.xyz-=bary.x*wa*impulse;
   nodes[ids.y].velocityAndMass.xyz-=bary.y*wb*impulse;
   nodes[ids.z].velocityAndMass.xyz-=bary.z*wc*impulse;
   const float deltaE=pb_film_contact_ke(coin,nodes[ids.x],nodes[ids.y],nodes[ids.z],c.scales.y,c.scales.z)-beforeE;
   diagnostics[0].x+=1.0f;diagnostics[0].y+=length(normalImpulse);
   diagnostics[0].z+=length(impulse);diagnostics[0].w+=deltaE;
   diagnostics[1].x=max(diagnostics[1].x,max(0.0f,deltaE));
   diagnostics[1].y=max(diagnostics[1].y,load.barycentricAndDepth.w);
   if(coin.positionAndInverseMass.y<=c.geometry.y+pb_coin_vertical_extent(coin,c.scales.y,c.scales.z)+1.e-5f)
       diagnostics[1].z+=1.0f;
   load.correctionA=pb_f4(newT);
   loads[li]=load;
  }
  coins[ci]=coin;
 }
}

// Sparse ordered contact: shared film vertices see each preceding coin's
// correction. This replaces the dense node x contact gather and cross-coin
// Jacobi sum, both of which were unnecessary and destabilizing.
kernel void pb_contact_ordered_solve(
 constant PlasticBagConfigGPU& config [[buffer(0)]],device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const PlasticBagTriangleGPU* triangles [[buffer(2)]],device PlasticBagCoinGPU* coins [[buffer(3)]],
 device PlasticBagContactGPU* contacts [[buffer(4)]],device PlasticBagContactGPU* loads [[buffer(5)]],
 device mr_float4* floorLoads [[buffer(6)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane!=0u)return;
 const uint samples=uint(config.scales.w);
 for(uint ci=0u;ci<config.counts.w;++ci){
  pb_contact_reduce_one(config,contacts,nodes,triangles,coins,loads,floorLoads,ci);
  float3 translation=float3(0),rotation=float3(0);
  for(uint si=0u;si<samples;++si){
   const auto contact=contacts[ci*samples+si];if(contact.normalAndActive.w==0.0f)continue;
   const auto t=triangles[contact.identity.x];
   nodes[t.nodes.x].positionAndInverseMass.xyz+=contact.correctionA.xyz;
   nodes[t.nodes.y].positionAndInverseMass.xyz+=contact.correctionB.xyz;
   nodes[t.nodes.z].positionAndInverseMass.xyz+=contact.correctionC.xyz;
   translation+=contact.correctionCoin.xyz;rotation+=contact.angularVelocityChangeCoin.xyz;
  }
  coins[ci].positionAndInverseMass.xyz+=translation;
  coins[ci].orientation=mr_float4(pb_apply_rotation(float4(coins[ci].orientation),rotation));
 }
}

kernel void pb_full_load_velocity(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],device PlasticBagCoinGPU* coins [[buffer(2)]],
 device const PlasticBagTriangleGPU* triangles [[buffer(3)]],
 device PlasticBagContactGPU* loads [[buffer(4)]],device mr_float4* diagnostics [[buffer(5)]],device mr_float4* floorLoads [[buffer(6)]],device const uint* candidates [[buffer(7)]],device const uint* counts [[buffer(8)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane!=0u)return;
 const uint samples=c.counts.z;const float floor=c.geometry.y+c.geometry.x;
 for(uint ci=0u;ci<c.counts.w;++ci){
  PlasticBagCoinGPU coin=coins[ci];
  for(uint slot=0u;slot<counts[ci];++slot){
   const uint si=candidates[ci*uint(c.geometry.z)+slot];const uint li=ci*samples+si;PlasticBagContactGPU load=loads[li];
   if(load.normalAndActive.w==0.0f||load.barycentricAndDepth.w<=0.0f)continue;
   const auto tri=triangles[load.identity.x];const uint3 ids=tri.nodes.xyz;
   const float3 n=-load.normalAndActive.xyz;
   const float3 arm=pb_rotate(float4(coin.orientation),load.angularVelocityChangeCoin.xyz);
   float3 bary;
   const float3 point=pb_closest_triangle_point(arm,
     pb_node_coin_difference(nodes[ids.x],coin),pb_node_coin_difference(nodes[ids.y],coin),
     pb_node_coin_difference(nodes[ids.z],coin),bary);
   float3 sdfNormal;
   const float gap=pb_cylinder_sdf(point,c.scales.y,c.scales.z,
       float4(coin.orientation),sdfNormal);
   // A saved support impulse grants friction only while contact is still local.
   if(gap>c.geometry.x+2.e-5f)continue;
   const float beforeE=pb_film_contact_ke(coin,nodes[ids.x],nodes[ids.y],nodes[ids.z],c.scales.y,c.scales.z);
   const float3 wa=pb_film_inverse_mass(nodes[ids.x],floor,-n);
   const float3 wb=pb_film_inverse_mass(nodes[ids.y],floor,-n);
   const float3 wc=pb_film_inverse_mass(nodes[ids.z],floor,-n);
   const float3 shellMass=bary.x*bary.x*wa+bary.y*bary.y*wb+bary.z*bary.z*wc;
   float3 sv=bary.x*nodes[ids.x].velocityAndMass.xyz+bary.y*nodes[ids.y].velocityAndMass.xyz+bary.z*nodes[ids.z].velocityAndMass.xyz;
   float3 relative=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,arm)-sv;
   const float3 rn=cross(arm,n);
   const float kn=coin.positionAndInverseMass.w+dot(n,shellMass*n)+
       dot(rn,pb_apply_inverse_inertia(coin,rn,c.scales.y,c.scales.z));
   const float oldN=load.correctionB.w;
   // The pose update already applied J_position. Velocity correction may
   // unload it, while total normal impulse remains nonnegative.
   const float nextN=max(-load.barycentricAndDepth.w,oldN-dot(relative,n)/max(kn,1.e-12f));
   const float3 normalImpulse=(nextN-oldN)*n;
   load.correctionB.w=nextN;
   coin.velocityAndMass.xyz+=coin.positionAndInverseMass.w*normalImpulse;
   coin.angularVelocity.xyz+=pb_apply_inverse_inertia(coin,cross(arm,normalImpulse),c.scales.y,c.scales.z);
   nodes[ids.x].velocityAndMass.xyz-=bary.x*wa*normalImpulse;
   nodes[ids.y].velocityAndMass.xyz-=bary.y*wb*normalImpulse;
   nodes[ids.z].velocityAndMass.xyz-=bary.z*wc*normalImpulse;
   const float3 filmNormalReaction=-normalImpulse;
   if(wa.y==0.0f)floorLoads[ids.x].x+=-bary.x*filmNormalReaction.y;
   if(wb.y==0.0f)floorLoads[ids.y].x+=-bary.y*filmNormalReaction.y;
   if(wc.y==0.0f)floorLoads[ids.z].x+=-bary.z*filmNormalReaction.y;
   sv=bary.x*nodes[ids.x].velocityAndMass.xyz+bary.y*nodes[ids.y].velocityAndMass.xyz+bary.z*nodes[ids.z].velocityAndMass.xyz;
   relative=coin.velocityAndMass.xyz+cross(coin.angularVelocity.xyz,arm)-sv;
   const float3 axis=abs(n.y)<0.9f?float3(0,1,0):float3(1,0,0);
   const float3 t1=normalize(cross(n,axis)),t2=cross(n,t1);
   const float3 r1=cross(arm,t1),r2=cross(arm,t2);
   const float3 ir1=pb_apply_inverse_inertia(coin,r1,c.scales.y,c.scales.z);
   const float3 ir2=pb_apply_inverse_inertia(coin,r2,c.scales.y,c.scales.z);
   const float k11=coin.positionAndInverseMass.w+dot(t1,shellMass*t1)+dot(r1,ir1);
   const float k22=coin.positionAndInverseMass.w+dot(t2,shellMass*t2)+dot(r2,ir2);
   const float k12=dot(t1,shellMass*t2)+dot(r1,ir2);
   const float2 vt=float2(dot(relative,t1),dot(relative,t2));
   const float det=max(k11*k22-k12*k12,1.e-20f);
   const float2 delta=-float2(k22*vt.x-k12*vt.y,k11*vt.y-k12*vt.x)/det;
   const float3 oldT=load.correctionA.xyz;
   const float limit=max(0.0f,c.solver.w*(load.barycentricAndDepth.w+nextN));
   const float2 nextT=pb_friction_disk(k11,k22,k12,float2(dot(oldT,t1),dot(oldT,t2)),vt,limit);
   const float3 newT=nextT.x*t1+nextT.y*t2;
   const float3 impulse=newT-oldT;
   coin.velocityAndMass.xyz+=coin.positionAndInverseMass.w*impulse;
   coin.angularVelocity.xyz+=pb_apply_inverse_inertia(coin,cross(arm,impulse),c.scales.y,c.scales.z);
   nodes[ids.x].velocityAndMass.xyz-=bary.x*wa*impulse;
   nodes[ids.y].velocityAndMass.xyz-=bary.y*wb*impulse;
   nodes[ids.z].velocityAndMass.xyz-=bary.z*wc*impulse;
   const float deltaE=pb_film_contact_ke(coin,nodes[ids.x],nodes[ids.y],nodes[ids.z],c.scales.y,c.scales.z)-beforeE;
   diagnostics[0].x+=1.0f;diagnostics[0].y+=length(normalImpulse);
   diagnostics[0].z+=length(impulse);diagnostics[0].w+=deltaE;
   diagnostics[1].x=max(diagnostics[1].x,max(0.0f,deltaE));
   diagnostics[1].y=max(diagnostics[1].y,load.barycentricAndDepth.w);
   if(coin.positionAndInverseMass.y<=c.geometry.y+pb_coin_vertical_extent(coin,c.scales.y,c.scales.z)+1.e-5f)
       diagnostics[1].z+=1.0f;
   load.correctionA=pb_f4(newT);
   loads[li]=load;
  }
  coins[ci]=coin;
 }
}


// Exact finite triangle / cylinder intersection in cylinder coordinates.
// Clip the triangle against both cap planes, then minimize radial distance
// over the resulting polygon, including its interior. Barycentrics retain
// the actual material witness, including triangle interiors and all edges.
inline bool pb_triangle_cylinder(const float3 a,const float3 b,const float3 d,
 const float R,const float h,thread float3& witness){
 float3 p[6],q[6],bp[6],bq[6];
 p[0]=a;p[1]=b;p[2]=d;bp[0]=float3(1,0,0);bp[1]=float3(0,1,0);bp[2]=float3(0,0,1);
 uint count=3u;
 for(uint cap=0u;cap<2u;++cap){
  uint next=0u;const float sign=cap==0u?1.0f:-1.0f;
  for(uint i=0u;i<count;++i){
   const uint j=(i+1u)%count;
   const float di=sign*p[i].y-h,dj=sign*p[j].y-h;
   if(di<=0.0f){q[next]=p[i];bq[next++]=bp[i];}
   if((di<0.0f&&dj>0.0f)||(di>0.0f&&dj<0.0f)){
    const float t=di/(di-dj);q[next]=mix(p[i],p[j],t);bq[next++]=mix(bp[i],bp[j],t);
   }
  }
  count=next;if(count==0u)return false;
  for(uint i=0u;i<count;++i){p[i]=q[i];bp[i]=bq[i];}
 }
 // An axis piercing the triangle interior has zero radial distance.
 const float2 e=b.xz-a.xz,f=d.xz-a.xz;
 const float det=e.x*f.y-e.y*f.x;
 if(abs(det)>1.e-14f){
  const float u=(-a.x*f.y+a.z*f.x)/det,v=(-e.x*a.z+e.y*a.x)/det;
  if(u>=0.0f&&v>=0.0f&&u+v<=1.0f){
   const float yy=a.y+u*(b.y-a.y)+v*(d.y-a.y);
   if(abs(yy)<=h){witness=float3(1.0f-u-v,u,v);return true;}
  }
 }
 float best=INFINITY;float3 bary=float3(1,0,0);
 for(uint i=0u;i<count;++i){
  const uint j=(i+1u)%count;const float2 delta=p[j].xz-p[i].xz;
  const float t=clamp(-dot(p[i].xz,delta)/max(dot(delta,delta),1.e-30f),0.0f,1.0f);
  const float2 radial=p[i].xz+t*delta;const float rr=dot(radial,radial);
  if(rr<best){best=rr;bary=mix(bp[i],bp[j],t);}
 }
 witness=bary;return best<=R*R;
}

struct PBSupport {float3 v,bary,arm;};
inline PBSupport pb_convex_support(const float3 dir,const float3 a,const float3 b,const float3 d,const float R,const float h){
 PBSupport s;s.bary=float3(1,0,0);float3 p=a;
 if(dot(b,dir)>dot(p,dir)){p=b;s.bary=float3(0,1,0);}
 if(dot(d,dir)>dot(p,dir)){p=d;s.bary=float3(0,0,1);}
 const float rr=length(dir.xz);float3 q=float3(0,dir.y>0.0f?h:dir.y<0.0f?-h:0.0f,0);
 if(rr>1.e-20f)q.xz=R*dir.xz/rr;
 s.arm=-q;s.v=p+q;return s;
}
inline float3 pb_convex_simplex(thread PBSupport* simplex,thread uint& count,thread float3& bary,thread float3& arm){
 float best=INFINITY;float4 weights=float4(0);
 for(uint mask=1u;mask<(1u<<count);++mask){
  uint ids[4],n=0u;for(uint i=0u;i<count;++i)if(mask&(1u<<i))ids[n++]=i;
  float4 w=float4(1,0,0,0);const float3 a=simplex[ids[0]].v;
  if(n==2u){const float3 e=simplex[ids[1]].v-a;const float den=dot(e,e);if(den<1.e-25f)continue;w.y=-dot(a,e)/den;w.x=1.0f-w.y;}
  if(n==3u){
   const float3 e=simplex[ids[1]].v-a,f=simplex[ids[2]].v-a;
   // Cross-product barycentrics avoid subtracting nearly equal Gram terms
   // for the narrow simplices produced by grazing thin-cylinder contacts.
   const float3 area=cross(e,f);const float den=dot(area,area);
   if(den<1.e-32f)continue;
   w.y=dot(cross(-a,f),area)/den;w.z=dot(cross(e,-a),area)/den;w.x=1.0f-w.y-w.z;
  }
  if(n==4u){
   const float3 e=simplex[ids[1]].v-a,f=simplex[ids[2]].v-a,g=simplex[ids[3]].v-a;
   const float den=dot(e,cross(f,g));if(abs(den)<1.e-25f)continue;
   w.y=dot(-a,cross(f,g))/den;w.z=dot(e,cross(-a,g))/den;w.w=dot(e,cross(f,-a))/den;w.x=1.0f-w.y-w.z-w.w;
  }
  bool valid=true;for(uint j=0u;j<n;++j)valid=valid&&(w[j]>=0.0f);if(!valid)continue;
  float3 p=float3(0);for(uint j=0u;j<n;++j)p+=w[j]*simplex[ids[j]].v;
  const float dd=dot(p,p);if(dd<best){best=dd;weights=float4(0);for(uint j=0u;j<n;++j)weights[ids[j]]=w[j];}
 }
 PBSupport next[4];uint nn=0u;float3 p=float3(0);bary=float3(0);arm=float3(0);
 for(uint i=0u;i<count;++i)if(weights[i]>0.0f){p+=weights[i]*simplex[i].v;bary+=weights[i]*simplex[i].bary;arm+=weights[i]*simplex[i].arm;next[nn++]=simplex[i];}
 for(uint i=0u;i<nn;++i)simplex[i]=next[i];count=nn;return p;
}
// GJK distance with explicit upper/lower bounds and material witnesses. The
// lower bound is a supporting plane, and remains conservative on early exit.
inline float pb_convex_distance(const float3 a,const float3 b,const float3 d,const float R,const float h,
 thread float3& bary,thread float3& arm,thread float3& normal,thread float& upper){
 // Try a closest triangle point and its exact cylinder projection first.
 // Accept it only when an independent support-plane LOWER bound closes the
 // distance interval. Flat faces and radial rim planes then solve in one
 // query; oblique feature pairs continue through the general simplex solve.
 const float3 point=pb_closest_triangle_point(float3(0),a,b,d,bary);
 arm=point;arm.y=clamp(arm.y,-h,h);const float radial=length(arm.xz);
 if(radial>R)arm.xz*=R/radial;
 const float3 delta=point-arm;upper=length(delta);
 if(upper<2.e-8f){pb_cylinder_sdf(point,R,h,float4(0,0,0,1),normal);return 0.0f;}
 normal=delta/upper;
 const float support=R*length(normal.xz)+h*abs(normal.y);
 const float certificate=min(min(dot(a,normal),dot(b,normal)),dot(d,normal))-support;
 if(upper-certificate<1.e-7f)return max(0.0f,certificate);
 float3 dir=-(a+b+d);if(length_squared(dir)<1.e-20f)dir=float3(1,0,0);
 PBSupport simplex[4];simplex[0]=pb_convex_support(dir,a,b,d,R,h);uint count=1u;
 float3 closest=simplex[0].v;bary=simplex[0].bary;arm=simplex[0].arm;
 float lower=0.0f;upper=length(closest);
 for(uint iteration=0u;iteration<64u;++iteration){
  upper=length(closest);
  if(upper<2.e-8f){normal=float3(0,1,0);return 0.0f;}
  normal=closest/upper;const PBSupport support=pb_convex_support(-normal,a,b,d,R,h);
  lower=max(lower,dot(support.v,normal));
  if(upper-lower<1.e-7f)return max(0.0f,lower);
  bool duplicate=false;for(uint i=0u;i<count;++i)duplicate=duplicate||length_squared(support.v-simplex[i].v)<1.e-20f;
  if(duplicate||count>=4u)return max(0.0f,lower);
  simplex[count++]=support;closest=pb_convex_simplex(simplex,count,bary,arm);
 }
 return max(0.0f,lower);
}

// Analytic finite-cylinder support, evaluated with a cross-product radial
// direction. The payload uses the existing generic simplex reducer; no samples
// are used to certify separation. Unresolved queries remain possible contact.
inline float3 pb_pair_support_offset(const PlasticBagCoinGPU c,const float3 dir,const float R,const float h){
 const float3 axis=normalize(pb_rotate(float4(c.orientation),float3(0,1,0)));
 const float axial=dot(axis,dir);const float3 radial=cross(axis,cross(dir,axis));
 const float rr=length(radial);
 return (rr>1.e-10f?R*radial/rr:float3(0))+h*(axial>0.0f?1.0f:axial<0.0f?-1.0f:0.0f)*axis;
}
inline PBSupport pb_pair_support(const PlasticBagCoinGPU a,const PlasticBagCoinGPU b,const float3 dir,const float R,const float h){
 PBSupport s;s.bary=pb_pair_support_offset(a,dir,R,h);s.arm=pb_pair_support_offset(b,-dir,R,h);
 s.v=pb_coin_difference(a,b)+s.bary-s.arm;return s;
}
#include "pair_mpr.metalinc"
#include "pair_contact_dual.metalinc"

inline bool pb_pair_certified_separated(const PlasticBagCoinGPU a,const PlasticBagCoinGPU b,const float R,const float h){
 float3 dir=b.positionAndInverseMass.xyz-a.positionAndInverseMass.xyz;
 if(length_squared(dir)<1.e-20f)dir=float3(1,0,0);dir=normalize(dir);
 PBSupport simplex[4];simplex[0]=pb_pair_support(a,b,dir,R,h);uint count=1u;
 float3 closest=simplex[0].v,wa,wb;
 for(uint iteration=0u;iteration<64u;++iteration){
  const float distance=length(closest);if(distance<2.e-8f)return false;
  const float3 n=closest/distance;const PBSupport support=pb_pair_support(a,b,-n,R,h);
  // 50 nm margin covers float support/pose round-off at this scene scale.
  // A positive supporting-plane gap proves separation; a failed/uncertain
  // query never silently removes a potentially intersecting pair.
  if(dot(support.v,n)>5.e-8f)return true;
  bool duplicate=false;for(uint j=0u;j<count;++j)duplicate=duplicate||length_squared(simplex[j].v-support.v)<1.e-20f;
  if(duplicate||count>=4u)return false;
  simplex[count++]=support;closest=pb_convex_simplex(simplex,count,wa,wb);
 }
 return false;
}

kernel void pb_full_contact_seed(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagTriangleGPU* tris [[buffer(2)]],
 device const PlasticBagCoinGPU* coins [[buffer(3)]],device PlasticBagContactGPU* contacts [[buffer(4)]],
 device const uint* candidates [[buffer(5)]],device const uint* counts [[buffer(6)]],uint i [[thread_position_in_grid]]){
 const uint cap=uint(c.geometry.z),ci=i/cap,slot=i%cap;
 if(ci>=c.counts.w||slot>=counts[ci])return;
 const uint ti=candidates[i];const uint3 ids=tris[ti].nodes.xyz;const auto coin=coins[ci];
 const float4 inv=pb_quat_conjugate(float4(coin.orientation));const float3 center=coin.positionAndInverseMass.xyz;
 const float3 a=pb_rotate(inv,pb_node_coin_difference(nodes[ids.x],coin));
 const float3 b=pb_rotate(inv,pb_node_coin_difference(nodes[ids.y],coin));
 const float3 d=pb_rotate(inv,pb_node_coin_difference(nodes[ids.z],coin));
 const float margin=c.geometry.x+2.e-6f;float3 bary;
 PlasticBagContactGPU contact{};
 if(pb_triangle_cylinder(a,b,d,c.scales.y+margin,c.scales.z+margin,bary)){
  float3 best=bary,normal;
  float lower;
  if(pb_triangle_cylinder(a,b,d,c.scales.y,c.scales.z,bary)){
   // Signed interior depth is found by intersecting inset cylinders.
   lower=0.0f;float upper=c.scales.z;best=bary;
   for(uint j=0u;j<11u;++j){const float depth=.5f*(lower+upper);float3 test;
    if(pb_triangle_cylinder(a,b,d,c.scales.y-depth,c.scales.z-depth,test)){lower=depth;best=test;}else upper=depth;}
   const float3 local=a*best.x+b*best.y+d*best.z;
   pb_cylinder_sdf(pb_rotate(float4(coin.orientation),local),c.scales.y,c.scales.z,float4(coin.orientation),normal);
  }else{
   // Outside the metal, the film thickness is a rounded Euclidean offset.
   // A square cap/rim expansion can choose the wrong material witness.
   float3 arm,n;float upper;
   pb_convex_distance(a,b,d,c.scales.y,c.scales.z,best,arm,n,upper);
   lower=-upper;normal=pb_rotate(float4(coin.orientation),n);
   if(upper>margin+1.e-7f){contacts[ci*c.counts.z+ti]=contact;return;}
  }
  contact.normalAndActive=pb_f4(normal,1.0f);
  contact.barycentricAndDepth=pb_f4(best,lower+margin);
  contact.identity=mr_uint4(ti,0u,2u,0u);
 }
 contacts[ci*c.counts.z+ti]=contact;
}

// Sparse ordered exact-primitive supplement. Every intersecting candidate
// triangle receives a reciprocal impulse, rather than competing for one of
// the old 66 deepest cylinder samples. All shared film writes are ordered.
// Solve the normal row with up to three active film/table bounds. A node
// that reaches the table contributes its finite travel to the contact gap;
// its remaining vertical inverse mass is removed from the Schur denominator.
// The returned bound displacement also gives the reciprocal floor impulse.
inline float pb_floor_active_lambda(const float depth,const float3 bary,const float normalY,
 const float3 heights,const float floor,thread float3& inverseY,thread float3& boundDisplacement,float denominator,const float lowerLambda=0.0f){
 boundDisplacement=float3(0.0f);
 float lambda=max(lowerLambda,depth/max(denominator,1.e-12f));
 for(uint pass=0u;pass<3u;++pass){
  bool changed=false;
  for(uint j=0u;j<3u;++j){
   if(inverseY[j]>0.0f&&heights[j]+lambda*bary[j]*inverseY[j]*normalY<floor){
    boundDisplacement[j]=floor-heights[j];
    denominator-=bary[j]*bary[j]*normalY*normalY*inverseY[j];
    inverseY[j]=0.0f;changed=true;
   }
  }
  if(!changed)break;
  lambda=max(lowerLambda,(depth-normalY*dot(bary,boundDisplacement))/max(denominator,1.e-12f));
 }
 return lambda;
}

// Position-level sticking under a Coulomb bound. The same material point is
// evaluated in current and previous rigid poses; all free-body impulses are
// reciprocal. A stuck floor vertex transfers the reaction into floor traction.
inline void pb_static_film_position(
 constant PlasticBagConfigGPU& c,device PlasticBagNodeGPU* nodes,
 thread PlasticBagCoinGPU& coin,thread PlasticBagContactGPU& load,
 const uint3 ids,const float3 arm,device mr_float4* floorLoads){
 if(load.normalAndActive.w==0.0f||(load.barycentricAndDepth.w<=0.0f&&length_squared(load.correctionCoin.xyz)==0.0f))return;
 const float dt=c.scales.x,floor=c.geometry.y+c.geometry.x;
 const float3 bary=load.barycentricAndDepth.xyz;
 const float3 n=-load.normalAndActive.xyz;
 const float3 shellPoint=bary.x*nodes[ids.x].positionAndInverseMass.xyz+bary.y*nodes[ids.y].positionAndInverseMass.xyz+bary.z*nodes[ids.z].positionAndInverseMass.xyz;
 float3 gapNormal;
 const float gap=pb_cylinder_sdf(shellPoint-coin.positionAndInverseMass.xyz,c.scales.y,c.scales.z,float4(coin.orientation),gapNormal);
 if(gap>c.geometry.x+4.e-6f)return;
 const float3 ref=abs(n.y)<.9f?float3(0,1,0):float3(1,0,0);
 const float3 t1=normalize(cross(n,ref)),t2=cross(n,t1);
 const float3 localArm=pb_rotate(pb_quat_conjugate(float4(coin.orientation)),arm);
 const float3 previousArm=pb_rotate(float4(coin.previousOrientation),localArm);
 const float3 rigidMove=coin.positionAndInverseMass.xyz-coin.previousPosition.xyz+arm-previousArm;
 const float3 shellMove=bary.x*(nodes[ids.x].positionAndInverseMass.xyz-nodes[ids.x].previousAndFixed.xyz)+
  bary.y*(nodes[ids.y].positionAndInverseMass.xyz-nodes[ids.y].previousAndFixed.xyz)+
  bary.z*(nodes[ids.z].positionAndInverseMass.xyz-nodes[ids.z].previousAndFixed.xyz);
 const float3 rel=rigidMove-shellMove;
 // A moving fold stays on the approved velocity-level contact path.
 if(length(rel)>0.02f*dt)return;
 const float3 r1=cross(arm,t1),r2=cross(arm,t2);
 const float3 ir1=pb_apply_inverse_inertia(coin,r1,c.scales.y,c.scales.z);
 const float3 ir2=pb_apply_inverse_inertia(coin,r2,c.scales.y,c.scales.z);
 const float3 oldT=load.correctionCoin.xyz;
 const float budget=c.solver.w*load.barycentricAndDepth.w*dt;
 bool stuck[3];float ownN[3];
 for(uint j=0u;j<3u;++j){
  const uint ni=ids[j];const auto nd=nodes[ni];const float w=nd.positionAndInverseMass.w;
  stuck[j]=w>0.0f&&floorLoads[ni].w>.5f&&nd.positionAndInverseMass.y<=floor+1.e-7f;
  ownN[j]=0.0f; // Already included in the accumulated floor multiplier.
 }
 float3 impulse=float3(0),nextT=oldT;float3 w[3];
 for(uint activePass=0u;activePass<4u;++activePass){
  float3 shellMass=float3(0);
  for(uint j=0u;j<3u;++j){w[j]=float3(stuck[j]?0.0f:nodes[ids[j]].positionAndInverseMass.w);shellMass+=bary[j]*bary[j]*w[j];}
  const float k11=coin.positionAndInverseMass.w+dot(t1,shellMass*t1)+dot(r1,ir1);
  const float k22=coin.positionAndInverseMass.w+dot(t2,shellMass*t2)+dot(r2,ir2);
  const float k12=dot(t1,shellMass*t2)+dot(r1,ir2);
  const float determinant=k11*k22-k12*k12;
  if(determinant<=1.e-20f)return;
  const float2 err=float2(dot(rel,t1),dot(rel,t2));
  const float2 tangent=pb_friction_disk(k11,k22,k12,float2(dot(oldT,t1),dot(oldT,t2)),err,budget);
  nextT=tangent.x*t1+tangent.y*t2;
  impulse=nextT-oldT;
  bool released=false;
  for(uint j=0u;j<3u;++j)if(stuck[j]){
   const uint ni=ids[j];const float3 reaction=bary[j]*impulse/dt;
   const float2 trial=floorLoads[ni].yz+reaction.xz;
   const float normal=floorLoads[ni].x+ownN[j]+reaction.y;
   // A unilateral table cannot sustain tensile normal traction.
   if(normal<0.0f||length(trial)>c.solver.w*normal){stuck[j]=false;released=true;}
  }
  if(!released)break;
 }
 pb_coin_translation(coin,coin.positionAndInverseMass.w*impulse);
 pb_coin_rotation(coin,pb_apply_inverse_inertia(coin,cross(arm,impulse),c.scales.y,c.scales.z));
 for(uint j=0u;j<3u;++j){
  const uint ni=ids[j];const float3 reaction=bary[j]*impulse/dt;
  if(stuck[j]){floorLoads[ni].yz+=reaction.xz;floorLoads[ni].x+=reaction.y;nodes[ni].restPosition.w+=reaction.y*dt;}
  else{
   floorLoads[ni].w=0.0f;
   const float3 delta=-bary[j]*w[j]*impulse;
   pb_node_delta(nodes[ni],delta);float3 pos=nodes[ni].positionAndInverseMass.xyz;
   if(pos.y<floor){const float wi=nodes[ni].positionAndInverseMass.w;
    if(wi>0.0f){const float dl=(floor-pos.y)/wi;nodes[ni].restPosition.w+=dl;floorLoads[ni].x+=dl/dt;}pos.y=floor;}
   pb_node_delta(nodes[ni],pos-nodes[ni].positionAndInverseMass.xyz);
  }
 }
 load.correctionCoin=pb_f4(nextT);
 // Finalize will reconstruct this already-applied tangential impulse.
 load.correctionA=pb_f4(nextT/dt);
}

kernel void pb_full_contact_position(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagTriangleGPU* tris [[buffer(2)]],
 device PlasticBagCoinGPU* coins [[buffer(3)]],device PlasticBagContactGPU* contacts [[buffer(4)]],
 device PlasticBagContactGPU* loads [[buffer(5)]],device const uint* candidates [[buffer(6)]],
 device const uint* counts [[buffer(7)]],device mr_float4* floorLoads [[buffer(8)]],device mr_float4* diagnostics [[buffer(9)]],device PlasticBagContactGPU* sampleLoads [[buffer(10)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane)return;
 const uint cap=uint(c.geometry.z);const float floor=c.geometry.y+c.geometry.x;
 for(uint sparseSweep=0u;sparseSweep<1u;++sparseSweep){
 for(uint ci=0u;ci<c.counts.w;++ci){
  auto coin=coins[ci];
  for(uint slot=0u;slot<counts[ci];++slot){
   const uint ti=candidates[ci*cap+slot],li=ci*c.counts.z+ti;
   auto ct=contacts[li];const auto prior=loads[li];
   // An accumulated contact multiplier must unload when another constraint
   // separates the surfaces. Keep its row until its multiplier reaches zero.
   if(ct.normalAndActive.w==0.0f){
    if(prior.normalAndActive.w==0.0f||prior.barycentricAndDepth.w<=0.0f)continue;
    ct=prior;
   }
   const uint3 ids=tris[ti].nodes.xyz;const float3 bary=ct.barycentricAndDepth.xyz;
   const float3 point=bary.x*pb_node_coin_difference(nodes[ids.x],coin)+bary.y*pb_node_coin_difference(nodes[ids.y],coin)+bary.z*pb_node_coin_difference(nodes[ids.z],coin);
   float3 normal,arm;float depth;
   if(ct.identity.z==3u){
    normal=ct.normalAndActive.xyz;arm=pb_rotate(float4(coin.orientation),ct.angularVelocityChangeCoin.xyz);
    depth=c.geometry.x+2.e-6f-dot(point-arm,normal);
   }else{
    const float sdf=pb_cylinder_sdf(point,c.scales.y,c.scales.z,float4(coin.orientation),normal);
    depth=c.geometry.x+2.e-6f-sdf;arm=point-sdf*normal;
   }
   const bool same=prior.normalAndActive.w>0.0f&&dot(prior.normalAndActive.xyz,normal)>.95f;
   const float oldLambda=same?max(0.0f,prior.barycentricAndDepth.w*c.scales.x):0.0f;
   if(depth<=0.0f&&oldLambda<=0.0f)continue;
   diagnostics[2].x+=1.0f;
   if(ct.identity.z==3u){diagnostics[2].y+=1.0f;diagnostics[2].z+=ct.identity.w?1.0f:0.0f;diagnostics[2].w=max(diagnostics[2].w,depth);}
   // Start with the true masses. The active-set solve retains each node's
   // finite travel to the plane instead of removing mobility one micron early.
   float3 wa=float3(nodes[ids.x].positionAndInverseMass.w),wb=float3(nodes[ids.y].positionAndInverseMass.w),wc=float3(nodes[ids.z].positionAndInverseMass.w);
   const float3 wm=bary.x*bary.x*wa+bary.y*bary.y*wb+bary.z*bary.z*wc;
   const float3 rn=cross(arm,normal),ir=pb_apply_inverse_inertia(coin,rn,c.scales.y,c.scales.z);
   const float denom=coin.positionAndInverseMass.w+dot(normal,wm*normal)+dot(rn,ir);
   const float3 heights=float3((nodes[ids.x].positionAndInverseMass.y-floor)-nodes[ids.x].positionRemainder.y,(nodes[ids.y].positionAndInverseMass.y-floor)-nodes[ids.y].positionRemainder.y,(nodes[ids.z].positionAndInverseMass.y-floor)-nodes[ids.z].positionRemainder.y);
   const float3 originalInverse=float3(nodes[ids.x].positionAndInverseMass.w,nodes[ids.y].positionAndInverseMass.w,nodes[ids.z].positionAndInverseMass.w);
   float3 inverseY=float3(wa.y,wb.y,wc.y),boundDisplacement;
   const float lambda=pb_floor_active_lambda(depth,bary,normal.y,heights,0.0f,inverseY,boundDisplacement,denom,-oldLambda);
   wa.y=inverseY.x;wb.y=inverseY.y;wc.y=inverseY.z;
   pb_node_delta(nodes[ids.x],bary.x*wa*lambda*normal+float3(0,boundDisplacement.x,0));
   pb_node_delta(nodes[ids.y],bary.y*wb*lambda*normal+float3(0,boundDisplacement.y,0));
   pb_node_delta(nodes[ids.z],bary.z*wc*lambda*normal+float3(0,boundDisplacement.z,0));
   for(uint j=0u;j<3u;++j){
    if(inverseY[j]==0.0f&&originalInverse[j]>0.0f){
     const float reaction=boundDisplacement[j]/originalInverse[j]-bary[j]*normal.y*lambda;
     nodes[ids[j]].restPosition.w+=max(0.0f,reaction);
     floorLoads[ids[j]].x+=max(0.0f,reaction)/c.scales.x;
    }
   }
   const float3 localArm=pb_rotate(pb_quat_conjugate(float4(coin.orientation)),arm);
   pb_coin_translation(coin,-coin.positionAndInverseMass.w*lambda*normal);
   pb_coin_rotation(coin,-lambda*ir);
   auto load=prior;
   const float budget=oldLambda/c.scales.x;
   const float3 savedStatic=same?load.correctionCoin.xyz:float3(0);
   load=PlasticBagContactGPU{};load.normalAndActive=pb_f4(normal,1.0f);load.identity=ct.identity;
   load.barycentricAndDepth=pb_f4(bary,max(0.0f,budget+lambda/c.scales.x));
   load.correctionCoin=pb_f4(savedStatic);load.correctionA=pb_f4(savedStatic/c.scales.x);
   load.angularVelocityChangeCoin=pb_f4(localArm);loads[li]=load;
  }
  coins[ci]=coin;
 }
  }
}

// Independent floor nodes run in parallel between the unchanged ordered
// normal-contact and static-film sweeps. No physical equation or row order changes.
kernel void pb_full_static_floor(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagCoinGPU* coins [[buffer(2)]],
 device mr_float4* floorLoads [[buffer(3)]],uint ni [[thread_position_in_grid]]){
 const float floor=c.geometry.y+c.geometry.x;
 // Static contact is added only in the quasi-static branch. Use both rigid
 // translation and surface angular speed to keep the approved impact branch.
 bool quietNetwork=true;
 for(uint ci=0u;ci<c.counts.w;++ci){
  if(length(coins[ci].velocityAndMass.xyz)>0.02f ||
     c.scales.y*length(coins[ci].angularVelocity.xyz)>0.02f){quietNetwork=false;break;}
 }
 if(!quietNetwork)return;
 // Static floor traction acts on displacement before velocity reconstruction.
 // The cone includes transmitted coin load and the node's own floor reaction.
 // No clock, sleeping state, speed clamp, or body freeze is used.
 if(ni>=c.counts.x)return;
  auto node=nodes[ni];const float w=node.positionAndInverseMass.w;
  floorLoads[ni].w=0.0f;
  if(w<=0.0f||node.positionAndInverseMass.y>floor+1.e-7f||
     node.previousAndFixed.y>floor+1.e-6f)return;
  const float dt=c.scales.x;
  const float ownDepth=max(0.0f,floor-node.previousAndFixed.y-node.contactPositionCorrection.w*dt);
  const float normalImpulse=max(0.0f,node.restPosition.w/dt);
  floorLoads[ni].x=normalImpulse;
  const float2 oldT=floorLoads[ni].yz;
  const float2 displacement=node.positionAndInverseMass.xz-node.previousAndFixed.xz;
  // Velocity-level sliding/impact response remains the approved path. The
  // position-level static branch resolves only quiet accumulated drift.
  if(length(displacement)>0.02f*dt || abs(node.contactPositionCorrection.w)>0.02f)return;
  float2 required=oldT-displacement/(w*dt);
  const float budget=c.solver.w*normalImpulse;
  const float magnitude=length(required);const bool sticks=magnitude<=budget;
  if(magnitude>budget)required*=budget/max(magnitude,1.e-30f);
  {const float2 delta=required-oldT;
   pb_node_delta(node,w*dt*float3(delta.x,0,delta.y));
   floorLoads[ni].yz=required;floorLoads[ni].w=sticks?1.0f:0.0f;
   nodes[ni]=node;
  }
 
}

kernel void pb_full_static_contacts(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagTriangleGPU* tris [[buffer(2)]],
 device PlasticBagCoinGPU* coins [[buffer(3)]],device PlasticBagContactGPU* loads [[buffer(4)]],
 device const uint* candidates [[buffer(5)]],device const uint* counts [[buffer(6)]],
 device mr_float4* floorLoads [[buffer(7)]],device PlasticBagContactGPU* sampleLoads [[buffer(8)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane)return;
 const uint cap=uint(c.geometry.z);const float floor=c.geometry.y+c.geometry.x;
 // Static contact is added only in the quasi-static branch. Use both rigid
 // translation and surface angular speed to keep the approved impact branch.
 bool quietNetwork=true;
 for(uint ci=0u;ci<c.counts.w;++ci){
  if(length(coins[ci].velocityAndMass.xyz)>0.02f ||
     c.scales.y*length(coins[ci].angularVelocity.xyz)>0.02f){quietNetwork=false;break;}
 }
 if(!quietNetwork)return;
 // Static floor traction acts on displacement before velocity reconstruction.
 // The cone includes transmitted coin load and the node's own floor reaction.
 // No clock, sleeping state, speed clamp, or body freeze is used.

 // Couple static floor support to both existing contact coverage paths.
 for(uint ci=0u;ci<c.counts.w;++ci){
  auto coin=coins[ci];
  const uint samples=uint(c.scales.w);
  for(uint si=0u;si<samples;++si){
   const uint li=ci*samples+si;auto load=sampleLoads[li];
   if(load.normalAndActive.w==0.0f||load.barycentricAndDepth.w<=0.0f)continue;
   const auto tri=tris[load.identity.x];const float3 n=-load.normalAndActive.xyz;
   const float3 arm=load.identity.z!=0u?pb_coin_support_offset(coin,-n,c.scales.y,c.scales.z):
    pb_coin_sample_offset(si,c.scales.y,c.scales.z,float4(coin.orientation));
   pb_static_film_position(c,nodes,coin,load,tri.nodes.xyz,arm,floorLoads);
   sampleLoads[li]=load;
  }
  for(uint slot=0u;slot<counts[ci];++slot){
   const uint ti=candidates[ci*cap+slot],li=ci*c.counts.z+ti;auto load=loads[li];
   if(load.normalAndActive.w==0.0f||load.barycentricAndDepth.w<=0.0f)continue;
   const float3 arm=pb_rotate(float4(coin.orientation),load.angularVelocityChangeCoin.xyz);
   pb_static_film_position(c,nodes,coin,load,tris[ti].nodes.xyz,arm,floorLoads);
   loads[li]=load;
  }
  coins[ci]=coin;
 }

}

inline float4 pb_interpolate_quaternion(float4 a,float4 b,float t){
 if(dot(a,b)<0.0f)b=-b;
 const float angle=acos(clamp(dot(a,b),-1.0f,1.0f));
 return angle>1.e-4f?pb_quat_normalize((sin((1.0f-t)*angle)*a+sin(t*angle)*b)/sin(angle)):
  pb_quat_normalize(mix(a,b,t));
}
inline float3 pb_project_cylinder(const float3 p,const float R,const float h){
 float3 q=p;q.y=clamp(q.y,-h,h);const float radial=length(p.xz);
 if(radial>R)q.xz*=R/radial;return q;
}
// Alternating convex projections supply witnesses. The support-plane gap is
// a LOWER bound on distance regardless of convergence; advancement uses only
// this bound, never the (possibly too large) witness distance.
inline float pb_triangle_cylinder_gap(const float3 a,const float3 b,const float3 d,
 const float R,const float h,thread float3& bary,thread float3& cylinderPoint,thread float3& normal){
 float upper;return pb_convex_distance(a,b,d,R,h,bary,cylinderPoint,normal,upper);
}
// Swept triangle / rotating finite cylinder conservative advancement. Vertex
// paths and the center are linear; cylinder rotation follows quaternion slerp.
// The motion bound includes every vertex and the cylinder's angular sweep.
// Unresolved advancement is retained as conservative possible contact, never
// silently declared separated. Response uses the material point and boundary
// point at contact, transported with the rigid body to the final pose.
kernel void pb_full_ccd_seed(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagTriangleGPU* tris [[buffer(2)]],
 device const PlasticBagCoinGPU* coins [[buffer(3)]],device PlasticBagContactGPU* contacts [[buffer(4)]],
 device const uint* candidates [[buffer(5)]],device const uint* counts [[buffer(6)]],uint i [[thread_position_in_grid]]){
 const uint cap=uint(c.geometry.z),ci=i/cap,slot=i%cap;
 if(ci>=c.counts.w||slot>=counts[ci])return;
 const uint ti=candidates[i],li=ci*c.counts.z+ti;if(contacts[li].normalAndActive.w>0.0f)return;
 const uint3 ids=tris[ti].nodes.xyz;const auto coin=coins[ci];
 const float3 ac=nodes[ids.x].positionAndInverseMass.xyz,bc=nodes[ids.y].positionAndInverseMass.xyz,dc=nodes[ids.z].positionAndInverseMass.xyz;
 const float3 ap=nodes[ids.x].previousAndFixed.xyz,bp=nodes[ids.y].previousAndFixed.xyz,dp=nodes[ids.z].previousAndFixed.xyz;
 const float3 center0=coin.previousPosition.xyz,center1=coin.positionAndInverseMass.xyz,dcen=center1-center0;
 const float4 q0=pb_quat_normalize(float4(coin.previousOrientation)),q1=pb_quat_normalize(float4(coin.orientation));
 const float angle=2.0f*acos(clamp(abs(dot(q0,q1)),0.0f,1.0f));
 const float bound=max(max(length(ac-ap-dcen),length(bc-bp-dcen)),length(dc-dp-dcen))+
  length(float2(c.scales.y,c.scales.z))*angle;
 if(bound<1.e-8f)return;
 float t=0.0f;float3 bary=float3(1,0,0),arm=float3(0),normal=float3(0,1,0);bool possible=false,uncertain=false;
 const float margin=c.geometry.x+2.e-6f;
 for(uint iteration=0u;iteration<32u;++iteration){
  const float4 q=pb_interpolate_quaternion(q0,q1,t),inv=pb_quat_conjugate(q);
  const float3 center=mix(center0,center1,t);
  const float3 a=pb_rotate(inv,mix(ap,ac,t)-center),b=pb_rotate(inv,mix(bp,bc,t)-center),d=pb_rotate(inv,mix(dp,dc,t)-center);
  const float gap=pb_triangle_cylinder_gap(a,b,d,c.scales.y,c.scales.z,bary,arm,normal);
  if(gap<=margin+1.e-7f){possible=true;uncertain=length(a*bary.x+b*bary.y+d*bary.z-arm)>margin+1.e-6f;break;}
  const float advance=.9f*(gap-margin)/bound;
  if(t+advance>1.0f)return;
  t+=advance;
  if(iteration==31u){possible=true;uncertain=true;}
 }
 if(!possible)return;
 const float3 n=pb_rotate(q1,normal),r=pb_rotate(q1,arm);
 const float3 point=bary.x*ac+bary.y*bc+bary.z*dc;
 const float depth=margin-dot(point-center1-r,n);
 if(depth<=0.0f)return;
 PlasticBagContactGPU ct{};ct.normalAndActive=pb_f4(n,1.0f);ct.barycentricAndDepth=pb_f4(bary,depth);
 ct.angularVelocityChangeCoin=pb_f4(arm);ct.identity=mr_uint4(ti,0u,3u,uncertain?1u:0u);contacts[li]=ct;
}

// Position-controlled Panda with finite compliant pads. No selected vertex IDs,
// fixed nodes, attachments, sleeping, or kinematic motion of the bag/coins.
inline FrankaGripPoseGPU fg_pose(device const FrankaGripPoseGPU* path,float time,uint lastStep){
 float index=clamp(time,0.0f,float(lastStep));uint i=min(uint(index),lastStep-1u);float a=index-float(i);
 auto p=path[i];p.centerAndHalfGap=mix(path[i].centerAndHalfGap,path[i+1].centerAndHalfGap,a);
 p.normal.xyz=normalize(mix(path[i].normal.xyz,path[i+1].normal.xyz,a));
 p.tangentX.xyz=mix(path[i].tangentX.xyz,path[i+1].tangentX.xyz,a);
 p.tangentX.xyz=normalize(p.tangentX.xyz-p.normal.xyz*dot(p.normal.xyz,p.tangentX.xyz));
 p.tangentZ.xyz=normalize(cross(p.tangentX.xyz,p.normal.xyz));return p;
}
kernel void fg_begin(constant PlasticBagConfigGPU& c [[buffer(0)]],device const FrankaGripPoseGPU* path [[buffer(1)]],
 device FrankaGripRuntimeGPU* runtime [[buffer(2)]],device FrankaGripContactGPU* contacts [[buffer(3)]],
 constant float2& times [[buffer(4)]],constant uint& lastStep [[buffer(5)]],uint i [[thread_position_in_grid]]){
 if(i<c.counts.x){const bool wasContact=contacts[i].normalMultipliers.x+contacts[i].normalMultipliers.y>0.0f;contacts[i]=FrankaGripContactGPU{};contacts[i].normalMultipliers.w=wasContact?1.0f:0.0f;}
 if(i==0u){runtime->previous=fg_pose(path,times.x,lastStep);runtime->current=fg_pose(path,times.y,lastStep);}
}
kernel void fg_position(constant PlasticBagConfigGPU& c [[buffer(0)]],device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const FrankaGripRuntimeGPU* runtime [[buffer(2)]],device FrankaGripContactGPU* contacts [[buffer(3)]],uint i [[thread_position_in_grid]]){
 if(i>=c.counts.x)return;auto node=nodes[i];float w=node.positionAndInverseMass.w;if(w<=0)return;
 auto ct=contacts[i];const auto p=runtime->current;const float3 n=p.normal.xyz,tx=p.tangentX.xyz,tz=p.tangentZ.xyz;
 const float3 offset=(node.positionAndInverseMass.xyz-p.centerAndHalfGap.xyz)-node.positionRemainder.xyz;
 const float s=dot(offset,n),h=p.centerAndHalfGap.w,r=c.geometry.x;
 // Finite 17 x 17 mm rubber pads, 3 mm thick. Leaving their footprint releases contact.
 const float previousSide=dot(node.previousAndFixed.xyz-runtime->previous.centerAndHalfGap.xyz,runtime->previous.normal.xyz);
 // These are the inward-facing pad patches. A point entering from the back
 // cannot be projected through the pad onto its inner face.
 const bool fromGap=abs(previousSide)<=runtime->previous.centerAndHalfGap.w+r+2.e-6f;
 if(abs(dot(offset,tx))>.0085f||abs(dot(offset,tz))>.0085f||abs(s)>h+.003f+r||(!fromGap&&ct.normalMultipliers.w<=0.0f&&ct.normalMultipliers.x+ct.normalMultipliers.y<=0.0f)){
  const float4 work=ct.incrementalWork;ct=FrankaGripContactGPU{};ct.incrementalWork=work;contacts[i]=ct;return;
 }
 const float dt=c.scales.x,alpha=1.0f/(200000.0f*dt*dt),A=w+alpha;
 const float2 oldN=ct.normalMultipliers.xy,C=float2(s+h-r,h-s-r);
 const float2 rhs=-C-alpha*oldN;
 const float determinant=alpha*(2.0f*w+alpha);
 float2 nextN=oldN+float2(A*rhs.x+w*rhs.y,w*rhs.x+A*rhs.y)/determinant;
 // Solve the two-plane LCP by enumerating its active sets. Both entries
 // of the unconstrained solution may be negative even when one plane is
 // penetrating; clamping one entry first loses that valid one-sided contact.
 if(nextN.x<0.0f||nextN.y<0.0f){
  const float left=max(0.0f,(w*(oldN.x-oldN.y)-C.x)/A);
  const float inactiveRight=C.y-w*(left-oldN.x+oldN.y);
  if(inactiveRight>=0.0f)nextN=float2(left,0.0f);
  else nextN=float2(0.0f,max(0.0f,(w*(oldN.y-oldN.x)-C.y)/A));
 }
 // 20 N per pad/node transient bound; preload comes from compressed pad springs.
 nextN=clamp(nextN,0.0f,20.0f*dt*dt);
 pb_node_delta(node,w*((nextN.x-oldN.x)-(nextN.y-oldN.y))*n);
 const float3 padMove=p.centerAndHalfGap.xyz-runtime->previous.centerAndHalfGap.xyz;
 const float3 relative=dt*node.velocityAndMass.xyz+node.contactPositionCorrection.xyz-padMove;
 const float3 tangent=relative-dot(relative,n)*n;
 const float3 oldT=ct.tangentMultiplier.xyz;
 float3 nextT=oldT-tangent/w;nextT-=dot(nextT,n)*n;
 const float bound=.8f*(nextN.x+nextN.y),magnitude=length(nextT);
 if(magnitude>bound)nextT*=bound/max(magnitude,1.e-30f);
 pb_node_delta(node,w*(nextT-oldT));
 pb_project_film_floor(node,c.geometry.y+c.geometry.x);
 const float2 deltaN=nextN-oldN;const float dh=p.centerAndHalfGap.w-runtime->previous.centerAndHalfGap.w;
 const float W=(((deltaN.x-deltaN.y)*dot(n,padMove))-(deltaN.x+deltaN.y)*dh+dot(nextT-oldT,padMove))/(dt*dt);
 ct.incrementalWork.x+=W;
 ct.normalMultipliers.xy=nextN;ct.tangentMultiplier=pb_f4(nextT);contacts[i]=ct;nodes[i]=node;
}
kernel void fg_velocity(constant PlasticBagConfigGPU& c [[buffer(0)]],device PlasticBagNodeGPU* nodes [[buffer(1)]],
 device const FrankaGripRuntimeGPU* runtime [[buffer(2)]],device FrankaGripContactGPU* contacts [[buffer(3)]],uint i [[thread_position_in_grid]]){
 if(i>=c.counts.x)return;auto ct=contacts[i];if(ct.normalMultipliers.x+ct.normalMultipliers.y<=0)return;
 auto nd=nodes[i];const float w=nd.positionAndInverseMass.w,dt=c.scales.x;const float3 n=runtime->current.normal.xyz;
 const float3 pv=(runtime->current.centerAndHalfGap.xyz-runtime->previous.centerAndHalfGap.xyz)/dt;
 float3 relative=nd.velocityAndMass.xyz-pv;
 const float normalJ=clamp(-dot(relative,n)/w,-ct.normalMultipliers.y/dt,ct.normalMultipliers.x/dt);
 nd.velocityAndMass.xyz+=w*normalJ*n;ct.normalMultipliers.z=normalJ*dt;
 relative=nd.velocityAndMass.xyz-pv;const float3 old=ct.tangentMultiplier.xyz/dt;
 float3 next=old-(relative-dot(relative,n)*n)/w;const float bound=.8f*(ct.normalMultipliers.x+ct.normalMultipliers.y)/dt;
 if(length(next)>bound)next*=bound/max(length(next),1.e-30f);
 nd.velocityAndMass.xyz+=w*(next-old);
 const float3 dc=runtime->current.centerAndHalfGap.xyz-runtime->previous.centerAndHalfGap.xyz;
 const float dh=runtime->current.centerAndHalfGap.w-runtime->previous.centerAndHalfGap.w;
 ct.incrementalWork.x+=(dot(normalJ*n+next-old,dc)-abs(normalJ)*dh)/dt;
 ct.tangentMultiplier=pb_f4(next*dt);contacts[i]=ct;nodes[i]=nd;
}
kernel void fg_work(constant PlasticBagConfigGPU& c [[buffer(0)]],device const FrankaGripRuntimeGPU* runtime [[buffer(1)]],
 device const FrankaGripContactGPU* contacts [[buffer(2)]],device mr_float4* metric [[buffer(3)]],uint lane [[thread_index_in_threadgroup]]){
 if(lane)return;const float dt=c.scales.x;const auto p=runtime->current;const auto old=runtime->previous;
 const float3 dc=p.centerAndHalfGap.xyz-old.centerAndHalfGap.xyz;const float dh=p.centerAndHalfGap.w-old.centerAndHalfGap.w;
 float signedWork=0,positiveWork=0,left=0,right=0;uint count=0;
 for(uint i=0;i<c.counts.x;++i){auto ct=contacts[i];
  // Accumulated increments survive feature exit; do not erase work merely
  // because a point is no longer touching the pad at the end of the solve.
  const float W=ct.incrementalWork.x;signedWork+=W;positiveWork+=max(0.0f,W);
  float2 N=ct.normalMultipliers.xy/(dt*dt);if(N.x+N.y<=0)continue;
  ++count;left+=N.x;right+=N.y;
 }
 metric[0].x+=signedWork;metric[0].y+=positiveWork;metric[0].z=max(metric[0].z,float(count));metric[0].w=max(metric[0].w,max(left,right));
 metric[1]=float4(float(count),left,right,0);
}

// Diagnostic copies only. Never changes the simulated state.
kernel void pb_debug_snapshot(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],device const PlasticBagCoinGPU* coins [[buffer(2)]],
 device PlasticBagNodeGPU* outNodes [[buffer(3)]],device PlasticBagCoinGPU* outCoins [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i<c.counts.x)outNodes[i]=nodes[i];if(i<c.counts.w)outCoins[i]=coins[i];
}

// One round-robin sweep: no body participates in two simultaneous pairs.
kernel void pb_coin_pair_ordered_sweep(
    constant PlasticBagConfigGPU& config [[buffer(0)]],
    device PlasticBagCoinGPU* coins [[buffer(1)]],
    device const PlasticBagNodeGPU* nodes [[buffer(2)]],
    device const uint4* panelPairs [[buffer(3)]],
    device PlasticBagPairLoadGPU* pairLoads [[buffer(4)]],
    device mr_float4* diagnostics [[buffer(5)]],
    uint coinIndex [[thread_position_in_threadgroup]]) {
    if(config.counts.w==0u)return;
    constexpr uint coinCount=PLASTIC_BAG_COIN_COUNT;
    constexpr uint ringCount=coinCount-1u;
    threadgroup PlasticBagCoinGPU localCoins[coinCount];
    threadgroup float4 corrections[coinCount];
    threadgroup float4 rotations[coinCount];
    if(coinIndex>=coinCount)return;
    localCoins[coinIndex]=coins[coinIndex];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint orderedSweepCount=1u;
    for(uint sweep=0u;sweep<orderedSweepCount;++sweep){

        for(uint round=0u;round<ringCount;++round){
            uint partner=0u;
            if(coinIndex==ringCount)partner=round;
            else if(coinIndex==round)partner=ringCount;
            else partner=(2u*round+ringCount-coinIndex)%ringCount;
            corrections[coinIndex]=float4(0.0f);rotations[coinIndex]=float4(0.0f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(coinIndex<partner){
                const PlasticBagCoinGPU coinA=localCoins[coinIndex];
                const PlasticBagCoinGPU coinB=localCoins[partner];
                const float3 centerA=pb_xyz(coinA.positionAndInverseMass);
                const float3 centerB=pb_xyz(coinB.positionAndInverseMass);
                const float3 centerDelta=centerA-centerB;
                const float radius=config.scales.y,halfThickness=config.scales.z;
                const float pairBound=2.0f*sqrt(radius*radius+halfThickness*halfThickness);
                float3 correctionA=float3(0.0f),correctionB=float3(0.0f),rotationA=float3(0),rotationB=float3(0);
                const uint pi=coinIndex*(2u*coinCount-coinIndex-1u)/2u+(partner-coinIndex-1u);
                auto load=pairLoads[pi];
                if(dot(centerDelta,centerDelta)<=pairBound*pairBound||load.normalImpulse.w>0.0f){
                    const int status=pb_pair_position_dual(coinA,coinB,radius,halfThickness,config.scales.x,
                        load,correctionA,correctionB,rotationA,rotationB);
                    pairLoads[pi]=load;
                    if(status==0&&atomic_fetch_or_explicit(((device atomic_uint*)diagnostics)+7,1u,memory_order_relaxed)==0u){
                        device PlasticBagCoinGPU* debug=(device PlasticBagCoinGPU*)(diagnostics+3);debug[0]=coinA;debug[1]=coinB;
                    }
                }
                corrections[coinIndex]=float4(correctionA,0.0f);
                corrections[partner]=float4(correctionB,0.0f);
                rotations[coinIndex]=float4(rotationA,0);rotations[partner]=float4(rotationB,0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pb_coin_translation(localCoins[coinIndex],corrections[coinIndex].xyz);
            pb_coin_rotation(localCoins[coinIndex],rotations[coinIndex].xyz);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    }
    coins[coinIndex]=localCoins[coinIndex];
}


kernel void pb_sync_floor_support(constant PlasticBagConfigGPU& c [[buffer(0)]],
 device const PlasticBagNodeGPU* nodes [[buffer(1)]],device mr_float4* floorLoads [[buffer(2)]],uint i [[thread_position_in_grid]]){
 if(i<c.counts.x)floorLoads[i].x=max(0.0f,nodes[i].restPosition.w/c.scales.x);
}
