#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread/qos.h>

#include "plastic_bag_gpu.h"
#include "franka_grip_gpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef PLASTIC_BAG_METALLIB
#error "PLASTIC_BAG_METALLIB must name the compiled scene Metal library"
#endif

namespace {

NSMutableArray<id<MTLBuffer>>* checkpointBuffers;
bool legacyPoseCheckpoint=false;std::uint32_t resumeSerializationStep=0;
struct CheckpointHeader {
    std::uint64_t magic=0x4e554d4942473031ull,step=0,count=0;
    double initialEnergy=0,positiveWork=0,signedWork=0;
    std::uint64_t contacts=0;
};
void checkpoint(const std::filesystem::path& file,CheckpointHeader& h,bool read) {
    std::fstream io(file,read?std::ios::binary|std::ios::in:std::ios::binary|std::ios::out|std::ios::trunc);
    if(!io)throw std::runtime_error("checkpoint open failed: "+file.string());
    auto bytes=[&](void* data,std::size_t size){if(read)io.read((char*)data,size);else io.write((char*)data,size);if(!io)throw std::runtime_error("checkpoint IO failed");};
    if(!read)h.count=checkpointBuffers.count;
    bytes(&h,sizeof(h));
    if(h.magic!=0x4e554d4942473031ull||h.count!=checkpointBuffers.count)throw std::runtime_error("checkpoint ABI mismatch");
    for(id<MTLBuffer> b in checkpointBuffers){
        std::uint64_t size=b.length;bytes(&size,sizeof(size));
        // ABI 12 -> 13 adds only solver-accuracy controls to the constant
        // buffer. All physical state buffers retain their exact byte layout.
        if(read&&size==80u&&b.length==96u&&[b.label isEqualToString:@"scene constants"]){bytes(b.contents,size);continue;}
        if(read&&size==48u&&b.length==368u&&[b.label isEqualToString:@"film velocity work and load diagnostics"]){bytes(b.contents,size);continue;}
        if(read&&[b.label isEqualToString:@"LDPE shell nodes"]&&size/80u==b.length/sizeof(PlasticBagNodeGPU)&&size%80u==0u&&size!=b.length){
            std::vector<char> old(size);bytes(old.data(),size);std::memset(b.contents,0,b.length);
            for(std::size_t i=0;i<size/80u;++i)std::memcpy((char*)b.contents+i*sizeof(PlasticBagNodeGPU),old.data()+i*80u,80u);
            legacyPoseCheckpoint=true;continue;
        }
        if(read&&[b.label isEqualToString:@"10 rigid metallic coins"]&&size==160u*PLASTIC_BAG_COIN_COUNT&&b.length==sizeof(PlasticBagCoinGPU)*PLASTIC_BAG_COIN_COUNT){
            std::vector<char> old(size);bytes(old.data(),size);std::memset(b.contents,0,b.length);
            for(std::size_t i=0;i<PLASTIC_BAG_COIN_COUNT;++i)std::memcpy((char*)b.contents+i*sizeof(PlasticBagCoinGPU),old.data()+i*160u,160u);continue;
        }
        if(read&&(size==48u||size==368u)&&b.length==400u&&[b.label isEqualToString:@"film velocity work and load diagnostics"]){bytes(b.contents,size);continue;}
        // Future actuator poses may be authored from the actual landed state.
        // Every executed pose through the checkpoint must match byte-for-byte;
        // physical state buffers are restored normally. Unexecuted poses are inputs.
        if(read&&[b.label isEqualToString:@"Panda FK fingertip trajectory"]){
            std::vector<char> oldPath(size);bytes(oldPath.data(),size);
            const auto prefix=(h.step+1u)*sizeof(FrankaGripPoseGPU);
            if(size%sizeof(FrankaGripPoseGPU)!=0u||size<prefix||b.length<prefix||std::memcmp(oldPath.data(),b.contents,prefix)!=0)
                throw std::runtime_error("extended gripper path changes checkpoint prefix");
            continue;
        }
        if(size!=b.length)throw std::runtime_error("checkpoint buffer size mismatch: "+std::string(b.label.UTF8String));
        bytes(b.contents,b.length);
    }
}
std::vector<std::string> robotFrameJSON;
double positiveActuatorWork=0.0,signedActuatorWork=0.0;
unsigned lastGripContacts=0;
id<MTLBuffer> savedGripContactBuffer=nil,savedGripConfigBuffer=nil;

constexpr std::size_t kCoinCount = PLASTIC_BAG_COIN_COUNT;
static_assert(kCoinCount>=2u && kCoinCount%2u==0u,"round-robin contact requires an even coin count");
constexpr double kFilmGauge = 75.0e-6;
constexpr double kFilmDensity = 923.0;
constexpr double kYoungsModulus = 100.0e6;
constexpr double kPoissonRatio = 0.45;
// Exploratory loss parameters, not measured LDPE calibration. No rigid drag.
constexpr double kStrainViscosityTime = 0.0004;
constexpr double kBendViscosityTime = 0.010;
constexpr double kCoinMass = 0.01134;
constexpr double kCoinRadius = 0.03061 * 0.5;
constexpr double kCoinThickness = 0.00215;
constexpr double kGroundHeight = 0.108;
constexpr double kBagWidth = 0.20;
constexpr double kBagHeight = 0.23;
// Maximum initial opening at the held top lips. The closed bottom is a shared
// front/back seam; there is no gusset or separate floor panel.
constexpr double kBagDepth = 0.034;
constexpr double kBagBodyDepth = 0.008;
constexpr double kDropHeight = 0.25;
constexpr double kBagBottomY = kGroundHeight + 0.5*kFilmGauge + kDropHeight;
constexpr double kInitialTilt = 12.0 * std::numbers::pi / 180.0;
constexpr double kCoinContactFriction = 0.30;
constexpr std::uint32_t kContactSamples = 66u;
constexpr double kFrameDt = 1.0 / 240.0;
constexpr std::uint32_t kSubsteps = 16u;
constexpr std::uint32_t kConstraintIterations = 128u;
constexpr std::uint32_t kCoinPairPositionIterations = 24u;
constexpr std::uint32_t kCoinPairCleanupIterations = 8u;
constexpr std::uint32_t kCoinPairVelocityIterations = 64u;
constexpr std::uint32_t kContactIterations = 8u;
constexpr double kGravity = 9.81;

using Vec3 = std::array<double, 3u>;
using Key3 = std::array<std::int64_t, 3u>;

struct Grid {
    std::string name;
    std::size_t nu = 0u;
    std::size_t nv = 0u;
    std::vector<std::uint32_t> nodes;
};

struct EdgeKey {
    std::uint32_t a = 0u;
    std::uint32_t b = 0u;
    friend bool operator<(const EdgeKey& left, const EdgeKey& right) {
        return std::tie(left.a, left.b) < std::tie(right.a, right.b);
    }
};

struct EdgeAccum {
    double stiffness = 0.0;
    double restLength = 0.0;
};

struct alignas(16) AdjacencyEntry {
    std::uint32_t index = 0u;
    std::uint32_t local = 0u;
    std::uint32_t payload = 0u;
    std::uint32_t reserved = 0u;
};
static_assert(sizeof(AdjacencyEntry)==16u);

mr_float4 f4(const Vec3& v, const float w = 0.0f) {
    return {static_cast<float>(v[0]), static_cast<float>(v[1]),
            static_cast<float>(v[2]), w};
}

template<class T> Vec3 physicalPosition(const T& e){
 const auto p=e.positionAndInverseMass,r=e.positionRemainder;
 return {double(p.x)-r.x,double(p.y)-r.y,double(p.z)-r.z};
}
Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
double dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}
double norm(const Vec3& a) { return std::sqrt(dot(a,a)); }
Vec3 add(const Vec3& a,const Vec3& b) {
    return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
}
Vec3 scale(const Vec3& a,const double s) {
    return {a[0]*s,a[1]*s,a[2]*s};
}

using Quat = std::array<double,4u>;
Quat multiply(const Quat& a,const Quat& b) {
    const Vec3 av{a[0],a[1],a[2]},bv{b[0],b[1],b[2]};
    const Vec3 xyz=add(add(scale(bv,a[3]),scale(av,b[3])),cross(av,bv));
    return {xyz[0],xyz[1],xyz[2],a[3]*b[3]-dot(av,bv)};
}
Quat axisAngle(const Vec3& axis,const double angle) {
    const double n=norm(axis);
    if(n<1.0e-12)return {0.0,0.0,0.0,1.0};
    const double s=std::sin(0.5*angle)/n;
    return {axis[0]*s,axis[1]*s,axis[2]*s,std::cos(0.5*angle)};
}
Vec3 rotate(const Quat& q,const Vec3& v) {
    const Vec3 u{q[0],q[1],q[2]};
    const Vec3 t=scale(cross(u,v),2.0);
    return add(v,add(scale(t,q[3]),cross(u,t)));
}

// Initial pose only: after t=0 every vertex and coin is freely dynamic.
Vec3 initialPoint(const Vec3& p) {
    const Vec3 origin{0.0,kBagBottomY,0.0};
    return add(origin,rotate(axisAngle({1.0,0.0,0.0},kInitialTilt),sub(p,origin)));
}

struct SceneTopology {
    std::vector<Vec3> rest;
    std::vector<double> mass;
    std::vector<std::uint32_t> fixedNodes;
    std::vector<PlasticBagTriangleGPU> triangles;
    std::vector<PlasticBagEdgeGPU> edges;
    std::vector<PlasticBagHingeGPU> hinges;
    std::vector<PlasticBagLapRowGPU> lapRows;
    std::vector<Grid> grids;
    std::vector<std::vector<AdjacencyEntry>> nodeEdges;
    std::vector<std::vector<AdjacencyEntry>> nodeLaps;
    std::map<EdgeKey, EdgeAccum> edgeMap;
    std::map<Key3, std::uint32_t> nodeMap;
    std::size_t skippedDegenerateTriangles = 0u;
};

bool fullTriangleCylinderIntersection(Vec3 a,Vec3 b,Vec3 c,double radius,double halfHeight,Vec3& witness){
 std::vector<Vec3> poly{a,b,c},next;
 for(double sign:{1.,-1.}){
  next.clear();for(size_t i=0;i<poly.size();++i){Vec3 p=poly[i],q=poly[(i+1)%poly.size()];double dp=sign*p[1]-halfHeight,dq=sign*q[1]-halfHeight;
   if(dp<=0)next.push_back(p);
   if((dp<0&&dq>0)||(dp>0&&dq<0))next.push_back(add(p,scale(sub(q,p),dp/(dp-dq))));
  }poly=next;if(poly.empty())return false;
 }
 Vec3 e=sub(b,a),f=sub(c,a);double det=e[0]*f[2]-e[2]*f[0];
 if(std::abs(det)>1e-20){double u=(-a[0]*f[2]+a[2]*f[0])/det,v=(-e[0]*a[2]+e[2]*a[0])/det;
  if(u>=0&&v>=0&&u+v<=1){Vec3 p=add(a,add(scale(e,u),scale(f,v)));if(std::abs(p[1])<=halfHeight){witness=p;return true;}}
 }
 double best=INFINITY;Vec3 point{};
 for(size_t i=0;i<poly.size();++i){Vec3 p=poly[i],d=sub(poly[(i+1)%poly.size()],p);double t=std::clamp(-(p[0]*d[0]+p[2]*d[2])/std::max(d[0]*d[0]+d[2]*d[2],1e-30),0.,1.);Vec3 q=add(p,scale(d,t));double rr=q[0]*q[0]+q[2]*q[2];if(rr<best){best=rr;point=q;}}
 witness=point;return best<=radius*radius;
}

// Read-only convex distance certificate. A support plane gives a lower bound;
// a point on the Minkowski simplex gives an upper bound. No physics changes.
struct GJKDistance {double lower=0,upper=INFINITY;unsigned iterations=0;};
Vec3 cylinderTriangleSupport(Vec3 direction,Vec3 a,Vec3 b,Vec3 c,double R,double h){
 Vec3 v=a;if(dot(b,direction)>dot(v,direction))v=b;if(dot(c,direction)>dot(v,direction))v=c;
 const double rr=std::hypot(direction[0],direction[2]);Vec3 q{0,0,0};
 if(rr>1e-30){q[0]=R*direction[0]/rr;q[2]=R*direction[2]/rr;}
 q[1]=direction[1]>0?h:direction[1]<0?-h:0;return add(v,q);
}
Vec3 closestSimplex(std::array<Vec3,4>& simplex,unsigned& count){
 double best=INFINITY;std::array<double,4> bestWeights{};
 for(unsigned mask=1;mask<(1u<<count);++mask){
  unsigned ids[4],n=0;for(unsigned i=0;i<count;++i)if(mask&(1u<<i))ids[n++]=i;
  double w[4]{1,0,0,0};const Vec3 a=simplex[ids[0]];
  if(n==2){Vec3 e=sub(simplex[ids[1]],a);const double den=dot(e,e);if(den<1e-30)continue;w[1]=-dot(a,e)/den;w[0]=1-w[1];}
  if(n==3){Vec3 e=sub(simplex[ids[1]],a),f=sub(simplex[ids[2]],a);const double ee=dot(e,e),ff=dot(f,f),ef=dot(e,f),den=ee*ff-ef*ef;if(den<1e-40)continue;const double ea=dot(e,a),fa=dot(f,a);w[1]=(ef*fa-ff*ea)/den;w[2]=(ef*ea-ee*fa)/den;w[0]=1-w[1]-w[2];}
  if(n==4){Vec3 e=sub(simplex[ids[1]],a),f=sub(simplex[ids[2]],a),g=sub(simplex[ids[3]],a),minus=scale(a,-1);const double den=dot(e,cross(f,g));if(std::abs(den)<1e-30)continue;w[1]=dot(minus,cross(f,g))/den;w[2]=dot(e,cross(minus,g))/den;w[3]=dot(e,cross(f,minus))/den;w[0]=1-w[1]-w[2]-w[3];}
  bool valid=true;for(unsigned j=0;j<n;++j)valid&=w[j]>=0.;if(!valid)continue;
  Vec3 p{0,0,0};for(unsigned j=0;j<n;++j)p=add(p,scale(simplex[ids[j]],w[j]));const double d=dot(p,p);
  if(d<best){best=d;bestWeights.fill(0);for(unsigned j=0;j<n;++j)bestWeights[ids[j]]=w[j];}
 }
 std::array<Vec3,4> next{};unsigned nn=0;Vec3 p{0,0,0};
 for(unsigned i=0;i<count;++i)if(bestWeights[i]>0){p=add(p,scale(simplex[i],bestWeights[i]));next[nn++]=simplex[i];}
 simplex=next;count=nn;return p;
}
GJKDistance triangleCylinderDistance(Vec3 a,Vec3 b,Vec3 c,double R,double h,double tolerance=1e-10){
 GJKDistance result;Vec3 dir=scale(add(a,add(b,c)),-1);if(norm(dir)<1e-20)dir={1,0,0};
 std::array<Vec3,4> simplex{};simplex[0]=cylinderTriangleSupport(dir,a,b,c,R,h);unsigned count=1;Vec3 closest=simplex[0];
 for(unsigned it=0;it<128u;++it){
  result.iterations=it+1;result.upper=std::min(result.upper,norm(closest));
  if(result.upper<1e-13){result.lower=0;result.upper=0;return result;}
  const Vec3 normal=scale(closest,1./norm(closest));const Vec3 support=cylinderTriangleSupport(scale(normal,-1),a,b,c,R,h);
  result.lower=std::max(result.lower,dot(support,normal));
  if(result.upper-result.lower<=tolerance)return result;
  bool duplicate=false;for(unsigned i=0;i<count;++i)duplicate|=norm(sub(support,simplex[i]))<1e-15;
  if(duplicate||count>=4u)return result;
  simplex[count++]=support;closest=closestSimplex(simplex,count);
 }
 return result;
}

struct FilmAccuracy {std::size_t bad=0,uncertain=0;double maxDeficitBound=0;};
// Read-only FP64 accuracy observer. It controls time resolution only; all
// positions, impulses and velocities remain owned by native Metal kernels.
FilmAccuracy filmAccuracy(const std::vector<PlasticBagNodeGPU>& nodes,
 const std::vector<PlasticBagCoinGPU>& coins,const SceneTopology& scene){
 FilmAccuracy result;constexpr double clearance=.5*kFilmGauge-2.e-6;
 const auto v3=[](const mr_float4& v)->Vec3{return {v.x,v.y,v.z};};

 for(const auto& coin:coins){const Vec3 center=physicalPosition(coin);const Quat inv{-coin.orientation.x,-coin.orientation.y,-coin.orientation.z,coin.orientation.w};
  for(const auto& triangle:scene.triangles){
   Vec3 a=physicalPosition(nodes[triangle.nodes.x]),b=physicalPosition(nodes[triangle.nodes.y]),c=physicalPosition(nodes[triangle.nodes.z]);
   const Vec3 tc=scale(add(a,add(b,c)),1./3.);const double tr=std::max({norm(sub(a,tc)),norm(sub(b,tc)),norm(sub(c,tc))});
   if(norm(sub(tc,center))>std::hypot(kCoinRadius,.5*kCoinThickness)+tr+.5*kFilmGauge)continue;
   a=rotate(inv,sub(a,center));b=rotate(inv,sub(b,center));c=rotate(inv,sub(c,center));Vec3 witness;
   // Expanded product cylinder bounds the true rounded thickness offset.
   if(!fullTriangleCylinderIntersection(a,b,c,kCoinRadius+clearance,.5*kCoinThickness+clearance,witness))continue;
   const auto distance=triangleCylinderDistance(a,b,c,kCoinRadius,.5*kCoinThickness,1.e-9);
   const double lower=distance.lower,upper=distance.upper;
   const bool uncertain=lower<clearance&&upper>=clearance;
   const bool bad=lower<clearance;
   if(uncertain)++result.uncertain;
   if(bad){++result.bad;result.maxDeficitBound=std::max(result.maxDeficitBound,clearance-lower);}
  }
 }
 return result;
}
struct EnergyState {
    double kinetic=0,gravity=0,membrane=0,bending=0;
    double total() const{return kinetic+gravity+membrane+bending;}
};
EnergyState sceneEnergy(const std::vector<PlasticBagNodeGPU>& nodes,
 const std::vector<PlasticBagCoinGPU>& coins,const SceneTopology& scene,const double rigidity){
    EnergyState e;
    const auto v3=[](const mr_float4& v)->Vec3{return {v.x,v.y,v.z};};
    for(const auto& n:nodes){const auto v=v3(n.velocityAndMass);e.kinetic+=0.5*n.velocityAndMass.w*dot(v,v);e.gravity+=n.velocityAndMass.w*kGravity*(physicalPosition(n)[1]-kGroundHeight);}
    for(const auto& c:coins){
        const Vec3 v=v3(c.velocityAndMass);const Quat qc{-c.orientation.x,-c.orientation.y,-c.orientation.z,c.orientation.w};
        const Vec3 w=rotate(qc,v3(c.angularVelocity));
        const double mass=c.velocityAndMass.w;
        const double axial=0.5*mass*kCoinRadius*kCoinRadius,radial=mass*(0.25*kCoinRadius*kCoinRadius+kCoinThickness*kCoinThickness/12.0);
        e.kinetic+=0.5*mass*dot(v,v)+0.5*(radial*(w[0]*w[0]+w[2]*w[2])+axial*w[1]*w[1]);
        e.gravity+=mass*kGravity*(physicalPosition(c)[1]-kGroundHeight);
    }
    for(const auto& edge:scene.edges){const double dl=norm(sub(physicalPosition(nodes[edge.nodes.x]),physicalPosition(nodes[edge.nodes.y])))-edge.material.x;e.membrane+=0.5*dl*dl/edge.material.y;}
    for(const auto& h:scene.hinges){
        const auto p0=physicalPosition(nodes[h.nodes.x]),p1=physicalPosition(nodes[h.nodes.y]);
        const auto p2=physicalPosition(nodes[h.nodes.z]),p3=physicalPosition(nodes[h.nodes.w]);
        const auto edge=sub(p1,p0),n1=cross(edge,sub(p2,p0)),n2=cross(sub(p3,p0),edge);
        const double den=norm(n1)*norm(n2);
        if(den<=1.e-24)continue;
        const double theta=std::atan2(dot(cross(n1,n2),scale(edge,1.0/norm(edge)))/den,dot(n1,n2)/den);
        const double d=std::remainder(theta-h.material.x,2.0*std::numbers::pi);
        e.bending+=0.5*d*d/h.material.y;
    }
    (void)rigidity;
    return e;
}

std::uint32_t addNode(SceneTopology& scene, const Vec3& p) {
    constexpr double keyScale = 1.0e9;
    const Key3 key{
        std::llround(p[0]*keyScale), std::llround(p[1]*keyScale),
        std::llround(p[2]*keyScale)};
    if (const auto found = scene.nodeMap.find(key); found != scene.nodeMap.end()) {
        return found->second;
    }
    const auto index = static_cast<std::uint32_t>(scene.rest.size());
    scene.nodeMap.emplace(key, index);
    scene.rest.push_back(p);
    scene.mass.push_back(0.0);
    return index;
}

void addTriangle(SceneTopology& scene, const std::uint32_t a,
                 const std::uint32_t b, const std::uint32_t c,
                 const std::uint32_t surfaceId) {
    const Vec3 ab = sub(scene.rest[b], scene.rest[a]);
    const Vec3 ac = sub(scene.rest[c], scene.rest[a]);
    const double area = 0.5 * norm(cross(ab, ac));
    if (!(area > 1.0e-12)) {
        ++scene.skippedDegenerateTriangles;
        return;
    }
    const double nodalMass = kFilmDensity * kFilmGauge * area / 3.0;
    scene.mass[a] += nodalMass;
    scene.mass[b] += nodalMass;
    scene.mass[c] += nodalMass;
    const Vec3 center{(scene.rest[a][0]+scene.rest[b][0]+scene.rest[c][0])/3.0,
        (scene.rest[a][1]+scene.rest[b][1]+scene.rest[c][1])/3.0,
        (scene.rest[a][2]+scene.rest[b][2]+scene.rest[c][2])/3.0};
    const double radius=std::max({norm(sub(scene.rest[a],center)),
        norm(sub(scene.rest[b],center)),norm(sub(scene.rest[c],center))});
    PlasticBagTriangleGPU triangle{};
    triangle.nodes={a,b,c,surfaceId};
    triangle.centerAndRadius=f4(center,static_cast<float>(radius));
    triangle.previousCenterAndRadius=triangle.centerAndRadius;
    scene.triangles.push_back(triangle);
}

void addSpring(SceneTopology& scene, const std::uint32_t a,
               const std::uint32_t b, const double stiffness) {
    if (a == b || !(stiffness > 0.0)) return;
    const EdgeKey key{std::min(a,b), std::max(a,b)};
    auto& edge = scene.edgeMap[key];
    const double length = norm(sub(scene.rest[b], scene.rest[a]));
    if (!(length > 1.0e-8)) throw std::runtime_error("zero-length membrane edge");
    if (edge.stiffness == 0.0) edge.restLength = length;
    else if (std::abs(edge.restLength-length) > 1.0e-7)
        throw std::runtime_error("welded shell edge has inconsistent rest length");
    edge.stiffness += stiffness;
}

std::size_t gridIndex(const Grid& grid, const std::size_t u,
                      const std::size_t v) {
    return v*grid.nu+u;
}

void addGridSprings(SceneTopology& scene, const Grid& grid,
                    const double du, const double dv) {
    const double extensional = kYoungsModulus*kFilmGauge;
    const double shear = extensional/(2.0*(1.0+kPoissonRatio));
    for (std::size_t v=0u; v<grid.nv; ++v) {
        for (std::size_t u=0u; u+1u<grid.nu; ++u) {
            const auto a=grid.nodes[gridIndex(grid,u,v)];
            const auto b=grid.nodes[gridIndex(grid,u+1u,v)];
            addSpring(scene,a,b,extensional*dv/du);
        }
    }
    for (std::size_t v=0u; v+1u<grid.nv; ++v) {
        for (std::size_t u=0u; u<grid.nu; ++u) {
            const auto a=grid.nodes[gridIndex(grid,u,v)];
            const auto b=grid.nodes[gridIndex(grid,u,v+1u)];
            addSpring(scene,a,b,extensional*du/dv);
        }
    }
    const double diagonalStiffness = shear*(2.0*du*dv)/(du*du+dv*dv);
    for (std::size_t v=0u; v+1u<grid.nv; ++v) {
        for (std::size_t u=0u; u+1u<grid.nu; ++u) {
            const auto a=grid.nodes[gridIndex(grid,u,v)];
            const auto b=grid.nodes[gridIndex(grid,u+1u,v)];
            const auto c=grid.nodes[gridIndex(grid,u,v+1u)];
            const auto d=grid.nodes[gridIndex(grid,u+1u,v+1u)];
            addSpring(scene,a,d,diagonalStiffness);
            addSpring(scene,b,c,diagonalStiffness);
        }
    }
}

template<class Fn>
Grid addGrid(SceneTopology& scene, const std::string& name,
             const std::size_t nu, const std::size_t nv,
             const double du, const double dv, const Vec3& normal,
             Fn position) {
    Grid grid{name,nu,nv,{}};
    grid.nodes.resize(nu*nv);
    const std::uint32_t surfaceId=name=="front" ? 0u :
        name=="back" ? 1u : 2u;
    for (std::size_t v=0u; v<nv; ++v) {
        for (std::size_t u=0u; u<nu; ++u) {
            grid.nodes[gridIndex(grid,u,v)]=addNode(scene,position(u,v));
        }
    }
    scene.nodeEdges.resize(scene.rest.size());
    scene.nodeLaps.resize(scene.rest.size());
    for (std::size_t v=0u; v+1u<nv; ++v) {
        for (std::size_t u=0u; u+1u<nu; ++u) {
            const auto a=grid.nodes[gridIndex(grid,u,v)];
            const auto b=grid.nodes[gridIndex(grid,u+1u,v)];
            const auto c=grid.nodes[gridIndex(grid,u,v+1u)];
            const auto d=grid.nodes[gridIndex(grid,u+1u,v+1u)];
            addTriangle(scene,a,b,d,surfaceId);
            addTriangle(scene,a,d,c,surfaceId);
        }
    }
    addGridSprings(scene,grid,du,dv);
    (void)normal; // dihedral bending is assembled from triangle adjacency below.
    scene.grids.push_back(grid);
    return grid;
}

template<class T>
std::pair<std::vector<PlasticBagNodeSpanGPU>,std::vector<T>> packAdjacency(
    const std::vector<std::vector<T>>& adjacency) {
    std::vector<PlasticBagNodeSpanGPU> spans(adjacency.size());
    std::vector<T> packed;
    std::size_t total=0u;
    for (const auto& list:adjacency) total+=list.size();
    packed.reserve(total);
    for (std::size_t i=0u;i<adjacency.size();++i) {
        spans[i].range={static_cast<std::uint32_t>(packed.size()),
            static_cast<std::uint32_t>(adjacency[i].size()),0u,0u};
        packed.insert(packed.end(),adjacency[i].begin(),adjacency[i].end());
    }
    return {std::move(spans),std::move(packed)};
}

SceneTopology makeScene() {
    constexpr std::size_t nx=25u, ny=23u;
    constexpr double width=kBagWidth, height=kBagHeight, depth=kBagDepth;
    constexpr double bottomY=kBagBottomY;
    constexpr double mouthY=bottomY+height;
    constexpr double seamDepth=0.012;
    // The two panels share one bottom edge: a continuous folded sheet.
    const double dx=width/static_cast<double>(nx-1u);
    const double dy=height/static_cast<double>(ny-1u);
    SceneTopology scene;

    const auto sideProfile=[&](const double u) {
        const double edgeDistance=std::min(u,1.0-u)*width;
        const double t=std::clamp(edgeDistance/seamDepth,0.0,1.0);
        return t*t*(3.0-2.0*t);
    };
    for (std::size_t panel=0u;panel<2u;++panel) {
        const double sign=panel==0u ? -1.0 : 1.0;
        const Vec3 normal=rotate(axisAngle({1.0,0.0,0.0},kInitialTilt),{0.0,0.0,sign});
        const std::string name=panel==0u ? "front" : "back";
        addGrid(scene,name,nx,ny,dx,dy,normal,
            [=](const std::size_t i,const std::size_t j) {
                const double u=static_cast<double>(i)/(nx-1u);
                const double v=static_cast<double>(j)/(ny-1u);
                const double x=-0.5*width+u*width;
                const double y=bottomY+v*height;
                // Folded bottom, welded side edges, and freely open mouth.
                // No connector floor or horizontal bottom weld strip.
                const double bottomTransition=std::clamp(v/0.05,0.0,1.0);
                const double smoothBottom=bottomTransition*bottomTransition*
                    (3.0-2.0*bottomTransition);
                const double mouthTransition=std::clamp((v-0.80)/0.20,0.0,1.0);
                const double smoothMouth=mouthTransition*mouthTransition*
                    (3.0-2.0*mouthTransition);
                const double openingDepth=kBagBodyDepth+
                    (depth-kBagBodyDepth)*smoothMouth;
                const double localDepth=openingDepth*smoothBottom;
                const double z=sign*0.5*localDepth*sideProfile(u);
                return initialPoint(Vec3{x,y,z});
            });
    }
    scene.nodeEdges.resize(scene.rest.size());
    scene.nodeLaps.resize(scene.rest.size());
    for (const auto& [key,value]:scene.edgeMap) {
        PlasticBagEdgeGPU edge{};
        edge.nodes={key.a,key.b,0u,0u};
        edge.material={static_cast<float>(value.restLength),
            static_cast<float>(1.0/value.stiffness),0.0f,static_cast<float>(kStrainViscosityTime)};
        const auto edgeIndex=static_cast<std::uint32_t>(scene.edges.size());
        scene.edges.push_back(edge);
        scene.nodeEdges[key.a].push_back({edgeIndex,0u,0u,0u});
        scene.nodeEdges[key.b].push_back({edgeIndex,1u,0u,0u});
    }
    // Interior panel dihedrals. A shared boundary is a freely folding seam,
    // not a fictitious connector triangle. Panels have zero manufactured
    // rest curvature instead of remembering the initially opened pouch.
    struct Half {std::uint32_t opposite,surface;};
    std::map<EdgeKey,Half> halves;
    const double D=kYoungsModulus*std::pow(kFilmGauge,3.0)/(12.0*(1.0-kPoissonRatio*kPoissonRatio));
    for(const auto& t:scene.triangles){
        const std::array<std::uint32_t,3u> ids{t.nodes.x,t.nodes.y,t.nodes.z};
        for(std::size_t i=0;i<3u;++i){
            const auto u=ids[i],v=ids[(i+1u)%3u],o=ids[(i+2u)%3u];
            const EdgeKey key{std::min(u,v),std::max(u,v)};
            const auto it=halves.find(key);
            if(it==halves.end()){halves.emplace(key,Half{o,t.nodes.w});continue;}
            const auto other=it->second; if(other.surface!=t.nodes.w)continue;
            const auto e=sub(scene.rest[key.b],scene.rest[key.a]);
            const double a1=.5*norm(cross(e,sub(scene.rest[other.opposite],scene.rest[key.a])));
            const double a2=.5*norm(cross(sub(scene.rest[o],scene.rest[key.a]),e));
            const double stiffness=1.5*D*dot(e,e)/(a1+a2);
            PlasticBagHingeGPU h{};h.nodes={key.a,key.b,other.opposite,o};
            h.material={0.0f,static_cast<float>(1.0/stiffness),0.0f,static_cast<float>(kBendViscosityTime)};
            scene.hinges.push_back(h);
        }
    }
    // No fixture vertices. Bottom and side seams are welded by shared
    // topology; the mouth has no positional or kinematic constraints.
    if (scene.rest.size()!=scene.mass.size() || scene.triangles.empty() || scene.edges.empty())
        throw std::runtime_error("invalid generated thin-shell topology");
    if (mouthY<=0.0) throw std::runtime_error("invalid bag mouth height");
    return scene;
}

template<class T>
id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device,
                               const std::vector<T>& values,
                               const std::string& label) {
    const std::size_t bytes=std::max<std::size_t>(sizeof(T),values.size()*sizeof(T));
    id<MTLBuffer> buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer==nil) throw std::runtime_error("Metal buffer allocation failed: "+label);
    buffer.label=[NSString stringWithUTF8String:label.c_str()];
    if (!values.empty()) std::memcpy(buffer.contents,values.data(),values.size()*sizeof(T));
    else std::memset(buffer.contents,0,bytes);
    [checkpointBuffers addObject:buffer];
    return buffer;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,id<MTLLibrary> library,
                                         const char* functionName) {
    NSString* name=[NSString stringWithUTF8String:functionName];
    id<MTLFunction> function=[library newFunctionWithName:name];
    if (function==nil) throw std::runtime_error(std::string("Metal function is missing: ")+functionName);
    NSError* error=nil;
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline==nil) {
        throw std::runtime_error(std::string("Metal pipeline failed: ")+functionName+
            " "+(error==nil?"unknown":error.localizedDescription.UTF8String));
    }
    return pipeline;
}

void encodeKernel(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  const std::initializer_list<id<MTLBuffer>>& buffers,
                  const std::size_t count) {
    if (count==0u) return;
    [encoder setComputePipelineState:pipeline];
    NSUInteger index=0u;
    for (id<MTLBuffer> buffer:buffers) [encoder setBuffer:buffer offset:0u atIndex:index++];
    const NSUInteger width=std::min<NSUInteger>(128u,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count,1u,1u)
        threadsPerThreadgroup:MTLSizeMake(std::max<NSUInteger>(width,1u),1u,1u)];
}

struct CoinSeed {
    Vec3 position;
    std::uint32_t stack = 0u;
};

std::array<CoinSeed,kCoinCount> makeCoins() {
    std::array<CoinSeed,kCoinCount> result{};
    // Nonoverlapping three-row pack. All bodies start at rest together.
    constexpr double coinPitch=2.0*kCoinRadius+0.0005;
    constexpr double rowPitch=0.8660254037844386*coinPitch;
    constexpr std::array<std::uint32_t,3u> rowCounts{{4u,3u,3u}};
    std::size_t coin=0u;
    for(std::uint32_t row=0u;row<rowCounts.size();++row){
        const std::uint32_t count=rowCounts[row];
        const double firstX=-0.5*static_cast<double>(count-1u)*coinPitch;
        for(std::uint32_t column=0u;column<count;++column){
            result[coin].stack=row;
            result[coin].position=initialPoint({firstX+column*coinPitch,
                kBagBottomY+kCoinRadius+0.007+(row==2u?rowPitch+coinPitch:row*rowPitch),0.0});
            ++coin;
        }
    }
    if(coin!=result.size())throw std::runtime_error("invalid 10-coin pack");
    return result;
}

void writeGrid(std::ostream& out,const Grid& grid,const PlasticBagNodeGPU* nodes) {
    out<<'"'<<grid.name<<"\":[";
    for(std::size_t i=0u;i<grid.nodes.size();++i){
        if(i)out<<',';
        const auto p=physicalPosition(nodes[grid.nodes[i]]);
        out<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';
    }
    out<<']';
}

Vec3 nodePosition(const std::vector<PlasticBagNodeGPU>& nodes,std::uint32_t index) {
    return physicalPosition(nodes[index]);
}

Vec3 interpolateGridPoint(const Grid& grid,const std::vector<PlasticBagNodeGPU>& nodes,
                          const double u,const double v) {
    const double gx=std::clamp(u,0.0,1.0)*static_cast<double>(grid.nu-1u);
    const double gy=std::clamp(v,0.0,1.0)*static_cast<double>(grid.nv-1u);
    const std::size_t x0=std::min<std::size_t>(static_cast<std::size_t>(gx),grid.nu-2u);
    const std::size_t y0=std::min<std::size_t>(static_cast<std::size_t>(gy),grid.nv-2u);
    const double tx=gx-static_cast<double>(x0),ty=gy-static_cast<double>(y0);
    const Vec3 p00=nodePosition(nodes,grid.nodes[y0*grid.nu+x0]);
    const Vec3 p10=nodePosition(nodes,grid.nodes[y0*grid.nu+x0+1u]);
    const Vec3 p01=nodePosition(nodes,grid.nodes[(y0+1u)*grid.nu+x0]);
    const Vec3 p11=nodePosition(nodes,grid.nodes[(y0+1u)*grid.nu+x0+1u]);
    const Vec3 low=add(scale(p00,1.0-tx),scale(p10,tx));
    const Vec3 high=add(scale(p01,1.0-tx),scale(p11,tx));
    return add(scale(low,1.0-ty),scale(high,ty));
}

std::size_t panelRayCrossings(const Vec3& origin,const Vec3& direction,
                              const SceneTopology& scene,
                              const std::vector<PlasticBagNodeGPU>& nodes) {
    std::vector<double> intersections;
    intersections.reserve(scene.triangles.size()/2u);
    for(const auto& triangle:scene.triangles){
        if(triangle.nodes.w>1u)continue;
        const Vec3 a=nodePosition(nodes,triangle.nodes.x);
        const Vec3 b=nodePosition(nodes,triangle.nodes.y);
        const Vec3 c=nodePosition(nodes,triangle.nodes.z);
        const Vec3 edge1=sub(b,a),edge2=sub(c,a);
        const Vec3 h=cross(direction,edge2);
        const double determinant=dot(edge1,h);
        if(std::abs(determinant)<1.0e-11)continue;
        const double inverse=1.0/determinant;
        const Vec3 fromA=sub(origin,a);
        const double u=dot(fromA,h)*inverse;
        if(u< -1.0e-9||u>1.0+1.0e-9)continue;
        const Vec3 q=cross(fromA,edge1);
        const double v=dot(direction,q)*inverse;
        if(v< -1.0e-9||u+v>1.0+1.0e-9)continue;
        const double t=dot(edge2,q)*inverse;
        if(std::isfinite(t))intersections.push_back(t);
    }
    std::sort(intersections.begin(),intersections.end());
    std::size_t crossings=0u;
    double previous=-std::numeric_limits<double>::infinity();
    for(const double t:intersections){
        if(t>=-2.0e-5)continue;
        if(t-previous>2.0e-5){++crossings;previous=t;}
    }
    return crossings;
}

struct PouchCounts {
    std::size_t inside=0,aboveMouth=0,escaped=0,projectedDepthSpanExceeded=0;
};

PouchCounts classifyCoins(const std::vector<PlasticBagCoinGPU>& coins,
                          const SceneTopology& scene,
                          const std::vector<PlasticBagNodeGPU>& nodes) {
    PouchCounts result;
    const Grid& front=scene.grids[0];
    const Grid& back=scene.grids[1];
    Vec3 bottom{0.0,0.0,0.0},top{0.0,0.0,0.0};
    Vec3 left{0.0,0.0,0.0},right{0.0,0.0,0.0};
    for(std::size_t u=0u;u<front.nu;++u){
        bottom=add(bottom,scale(add(nodePosition(nodes,front.nodes[gridIndex(front,u,0u)]),
            nodePosition(nodes,back.nodes[gridIndex(back,u,0u)])),0.5));
        top=add(top,scale(add(nodePosition(nodes,front.nodes[gridIndex(front,u,front.nv-1u)]),
            nodePosition(nodes,back.nodes[gridIndex(back,u,back.nv-1u)])),0.5));
    }
    bottom=scale(bottom,1.0/static_cast<double>(front.nu));
    top=scale(top,1.0/static_cast<double>(front.nu));
    for(std::size_t v=0u;v<front.nv;++v){
        left=add(left,nodePosition(nodes,front.nodes[gridIndex(front,0u,v)]));
        right=add(right,nodePosition(nodes,front.nodes[gridIndex(front,front.nu-1u,v)]));
    }
    left=scale(left,1.0/static_cast<double>(front.nv));
    right=scale(right,1.0/static_cast<double>(front.nv));
    Vec3 upDelta=sub(top,bottom);
    const double upLength=norm(upDelta);
    Vec3 up=upLength>1.0e-8 ? scale(upDelta,1.0/upLength) : Vec3{0.0,1.0,0.0};
    Vec3 width=sub(right,left);
    width=sub(width,scale(up,dot(width,up)));
    const double widthNorm=norm(width);
    width=widthNorm>1.0e-8 ? scale(width,1.0/widthNorm) : Vec3{1.0,0.0,0.0};
    Vec3 depth=cross(width,up);
    depth=scale(depth,1.0/std::max(norm(depth),1.0e-9));
    const Vec3 frontMid=interpolateGridPoint(front,nodes,0.5,0.5);
    const Vec3 backMid=interpolateGridPoint(back,nodes,0.5,0.5);
    if(dot(sub(backMid,frontMid),depth)<0.0)depth=scale(depth,-1.0);
    const double widthLength=std::max(std::abs(dot(sub(right,left),width)),1.0e-6);
    const double heightLength=std::max(dot(sub(top,bottom),up),1.0e-6);
    const Vec3 frontNormal=depth;
    for(const auto& coin:coins){
        const auto p4=coin.positionAndInverseMass;
        const Vec3 p{p4.x,p4.y,p4.z};
        const Vec3 local=sub(p,bottom);
        const double x=dot(local,width),y=dot(local,up),z=dot(local,depth);
        const Quat q{coin.orientation.x,coin.orientation.y,coin.orientation.z,coin.orientation.w};
        const Vec3 diskAxis=rotate(q,{0.0,1.0,0.0});
        const auto extentAlong=[&](const Vec3& axis){
            const double projection=dot(diskAxis,axis);
            return kCoinRadius*std::sqrt(std::max(0.0,1.0-projection*projection))+
                0.5*kCoinThickness*std::abs(projection);
        };
        const double extentX=extentAlong(width),extentY=extentAlong(up),extentZ=extentAlong(frontNormal);
        if(y>heightLength)++result.aboveMouth;
        if(y>heightLength+extentY||y< -extentY||
           std::abs(x)>0.5*widthLength+extentX){
            ++result.escaped;
            continue;
        }
        const double u=std::clamp(x/widthLength+0.5,0.0,1.0);
        const double v=std::clamp(y/heightLength,0.0,1.0);
        const Vec3 frontPoint=interpolateGridPoint(front,nodes,u,v);
        const Vec3 backPoint=interpolateGridPoint(back,nodes,u,v);
        const double zFront=dot(sub(frontPoint,bottom),depth);
        const double zBack=dot(sub(backPoint,bottom),depth);
        const double interiorDepth=std::min(zFront,zBack);
        const double exteriorDepth=std::max(zFront,zBack);
        const double filmTolerance=0.5*kFilmGauge;
        if(z-extentZ<interiorDepth-filmTolerance||
           z+extentZ>exteriorDepth+filmTolerance)
            ++result.projectedDepthSpanExceeded;
        // This old flat-frame projection is not a valid escape test once the
        // envelope folds. Classify the coin center against the actual current
        // front/back triangle sheets by odd/even ray crossings instead.
        const bool centerInsidePanels=(panelRayCrossings(p,depth,scene,nodes)&1u)!=0u;
        if(centerInsidePanels)++result.inside;
        else ++result.escaped;
    }
    return result;
}

void writeFrame(std::ostream& out,const std::uint32_t step,const SceneTopology& scene,
                const std::vector<PlasticBagNodeGPU>& nodes,
                const std::vector<PlasticBagCoinGPU>& coins,
                const PouchCounts& pouch,bool& first) {
    if(!first)out<<',';
    first=false;
    const double time=static_cast<double>(step)*kFrameDt;
    const char* phase=time<.23 ? "first_drop" : time<=1.0 ? "first_impact_and_settle" : time<2.6 ? "robot_approach" : time<3.2 ? "pinch_corner" : time<6.2 ? "lift" : time<=6.4 ? "hold" : time<6.6 ? "release" : "second_drop_and_settle";
    out<<"{\"step\":"<<step<<",\"time_s\":"<<std::setprecision(legacyPoseCheckpoint&&step==resumeSerializationStep?8:12)
       <<time<<",\"motion_phase\":\""<<phase<<"\",\"gripped\":"
       <<(lastGripContacts>0?"true":"false")<<",\"inside_coins\":"<<pouch.inside
       <<",\"above_mouth_coins\":"<<pouch.aboveMouth<<",\"escaped_coins\":"
       <<pouch.escaped<<",\"projected_depth_span_exceeded_coins\":"
       <<pouch.projectedDepthSpanExceeded<<',';
    out<<"\"grip_contacts\":[";
    bool firstGripPoint=true;
    if(savedGripContactBuffer!=nil&&savedGripConfigBuffer!=nil){
        const auto* ct=static_cast<const FrankaGripContactGPU*>(savedGripContactBuffer.contents);
        const auto* cfg=static_cast<const PlasticBagConfigGPU*>(savedGripConfigBuffer.contents);
        const double dt=cfg->scales.x;
        for(std::size_t i=0;i<nodes.size();++i){
            if(ct[i].normalMultipliers.x+ct[i].normalMultipliers.y<=0)continue;
            if(!firstGripPoint)out<<',';firstGripPoint=false;
            const auto p=physicalPosition(nodes[i]);
            out<<"{\"node\":"<<i<<",\"point\":["<<p[0]<<','<<p[1]<<','<<p[2]
               <<"],\"left_normal_N\":"<<ct[i].normalMultipliers.x/(dt*dt)
               <<",\"right_normal_N\":"<<ct[i].normalMultipliers.y/(dt*dt)<<'}';
        }
    }
    out<<"],";
    for(std::size_t i=0u;i<scene.grids.size();++i){
        if(i)out<<',';
        writeGrid(out,scene.grids[i],nodes.data());
    }
    out<<",\"bottom_seam\":[";
    const Grid& front=scene.grids[0];
    for(std::size_t u=0u;u<front.nu;++u){
        if(u)out<<',';
        const auto p=physicalPosition(nodes[front.nodes[gridIndex(front,u,0u)]]);
        out<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';
    }
    out<<']';
    out<<",\"coins\":[";
    for(std::size_t i=0u;i<coins.size();++i){
        if(i)out<<',';
        const auto p=physicalPosition(coins[i]);
        out<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';
    }
    out<<"],\"coin_orientations\":[";
    for(std::size_t i=0u;i<coins.size();++i){
        if(i)out<<',';
        const auto q=coins[i].orientation;
        out<<'['<<q.x<<','<<q.y<<','<<q.z<<','<<q.w<<']';
    }
    out<<"],\"coin_velocities\":[";
    for(std::size_t i=0u;i<coins.size();++i){if(i)out<<',';const auto v=coins[i].velocityAndMass;out<<'['<<v.x<<','<<v.y<<','<<v.z<<']';}
    out<<"],\"coin_angular_velocities\":[";
    for(std::size_t i=0u;i<coins.size();++i){if(i)out<<',';const auto w=coins[i].angularVelocity;out<<'['<<w.x<<','<<w.y<<','<<w.z<<']';}
    out<<"],\"robot\":"<<robotFrameJSON.at(step)<<",\"grip_contact_nodes\":"<<lastGripContacts<<",\"grip_positive_work_J\":"<<positiveActuatorWork<<"}";
}

int run(int argc,char** argv) {
    std::filesystem::path output,resume;
    std::uint32_t traceFrame=0,checkpointEvery=120,startStep=0;
    std::uint32_t steps=1920u,recordEvery=1u;
    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];
        if(arg=="--output"&&i+1<argc)output=argv[++i];
        else if(arg=="--steps"&&i+1<argc)steps=static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if(arg=="--record-every"&&i+1<argc)recordEvery=static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if(arg=="--resume"&&i+1<argc)resume=argv[++i];
        else if(arg=="--trace-frame"&&i+1<argc)traceFrame=std::stoul(argv[++i]);
        else if(arg=="--checkpoint-every"&&i+1<argc)checkpointEvery=std::stoul(argv[++i]);
        else if(arg=="--help"){
            std::cout<<"usage: metalrobo_plastic_bag_scene --output DIR [--steps N] [--record-every N]\n";
            return 0;
        }else throw std::runtime_error("unknown or incomplete argument: "+arg);
    }
    if(output.empty()||steps==0u||recordEvery==0u)throw std::runtime_error("--output, positive --steps, and --record-every are required");
    std::filesystem::create_directories(output);

    @autoreleasepool {
        checkpointBuffers=[NSMutableArray array];
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(device==nil)throw std::runtime_error("no Metal device available");
        NSError* error=nil;
        id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@PLASTIC_BAG_METALLIB] error:&error];
        if(library==nil)throw std::runtime_error(std::string("cannot load PlasticBag.metallib: ")+
            (error==nil?"unknown":error.localizedDescription.UTF8String));
        id<MTLCommandQueue> queue=[device newCommandQueue];
        if(queue==nil)throw std::runtime_error("Metal command queue allocation failed");

        const std::array<const char*,15u> functionNames{
            "pb_hinge_colored_solve","pb_integrate_nodes","pb_integrate_coins",
            "pb_reset_edge_lambdas","pb_edge_solve","pb_edge_apply",
            "pb_update_triangle_bounds","pb_contact_solve","pb_contact_reduce","pb_contact_apply_nodes",
            "pb_contact_apply_coins","pb_coin_pair_solve","pb_coin_pair_apply",
            "pb_build_coin_candidates","pb_project_panel_order"};
        std::array<id<MTLComputePipelineState>,37u> pipelines{};
        for(std::size_t i=0u;i<functionNames.size();++i)
            pipelines[i]=makePipeline(device,library,functionNames[i]);
        pipelines[15]=makePipeline(device,library,"pb_finalize");
        pipelines[16]=makePipeline(device,library,"pb_project_ground");
        pipelines[18]=makePipeline(device,library,"pb_coin_pair_velocity_solve");
        pipelines[19]=makePipeline(device,library,"pb_coin_pair_velocity_apply");
        pipelines[21]=makePipeline(device,library,"pb_coin_pair_ordered_cleanup");
        pipelines[22]=makePipeline(device,library,"pb_contact_apply_velocity_nodes");
        pipelines[23]=makePipeline(device,library,"pb_shell_self_contact_solve");
        pipelines[24]=makePipeline(device,library,"pb_shell_self_contact_apply");
        pipelines[17]=makePipeline(device,library,"pb_project_pouch_side_seams");
        pipelines[20]=makePipeline(device,library,"pb_apply_side_seam_position_reactions");
        pipelines[25]=makePipeline(device,library,"pb_side_seam_velocity_solve");
        pipelines[26]=makePipeline(device,library,"pb_apply_side_seam_velocity_reactions");

        pipelines[27]=makePipeline(device,library,"pb_phase_energy");

        pipelines[28]=makePipeline(device,library,"pb_table_velocity");
        pipelines[29]=makePipeline(device,library,"pb_edge_colored_solve");
        pipelines[30]=makePipeline(device,library,"pb_film_load_reset");
        pipelines[31]=makePipeline(device,library,"pb_film_load_velocity");
        pipelines[32]=makePipeline(device,library,"pb_contact_ordered_solve");
        pipelines[33]=makePipeline(device,library,"pb_full_contact_seed");
        pipelines[34]=makePipeline(device,library,"pb_full_contact_position");
        pipelines[35]=makePipeline(device,library,"pb_full_load_velocity");
        pipelines[36]=makePipeline(device,library,"pb_full_ccd_seed");
        auto pairSweepPipeline=makePipeline(device,library,"pb_coin_pair_ordered_sweep");
        auto floorSupportPipeline=makePipeline(device,library,"pb_sync_floor_support");
        SceneTopology scene=makeScene();
        const std::size_t contactCandidateCapacity=scene.triangles.size();
        std::vector<mr_uint4> panelPairs;
        const Grid& frontPanel=scene.grids[0];
        const Grid& backPanel=scene.grids[1];
        panelPairs.reserve(frontPanel.nodes.size());
        for(std::size_t i=0u;i<frontPanel.nodes.size();++i){
            const auto frontNode=frontPanel.nodes[i];
            const auto backNode=backPanel.nodes[i];
            panelPairs.push_back({frontNode,backNode,0u,0u});
        }
        auto [edgeSpans,nodeEdgeRefs]=packAdjacency(scene.nodeEdges);
        auto [lapSpans,nodeLapRefs]=packAdjacency(scene.nodeLaps);
        std::vector<PlasticBagEdgeGPU> edges=scene.edges;
        std::vector<PlasticBagNodeGPU> nodes(scene.rest.size());
        for(std::size_t i=0u;i<nodes.size();++i){
            const bool fixed=std::binary_search(scene.fixedNodes.begin(),scene.fixedNodes.end(),static_cast<std::uint32_t>(i));
            const double mass=std::max(scene.mass[i],1.0e-12);
            nodes[i].positionAndInverseMass=f4(scene.rest[i],fixed?0.0f:static_cast<float>(1.0/mass));
            nodes[i].velocityAndMass={0.0f,0.0f,0.0f,static_cast<float>(mass)};
            nodes[i].previousAndFixed=f4(scene.rest[i],fixed?1.0f:0.0f);
            nodes[i].restPosition=f4(scene.rest[i]);
        }

        std::vector<PlasticBagCoinGPU> coins(kCoinCount);
        const auto seeds=makeCoins();
        for(std::size_t i=0u;i<kCoinCount;++i){
            coins[i].positionAndInverseMass=f4(seeds[i].position,static_cast<float>(1.0/kCoinMass));
            coins[i].velocityAndMass={0.0f,0.0f,0.0f,static_cast<float>(kCoinMass)};
            coins[i].previousPosition=f4(seeds[i].position);
            constexpr float inverseSqrtTwo=0.70710678118654752440f;
            const Quat orientation=multiply(axisAngle({1.0,0.0,0.0},kInitialTilt),
                Quat{inverseSqrtTwo,0.0,0.0,inverseSqrtTwo});
            coins[i].orientation={static_cast<float>(orientation[0]),static_cast<float>(orientation[1]),
                static_cast<float>(orientation[2]),static_cast<float>(orientation[3])};
            coins[i].previousOrientation=coins[i].orientation;
            // A stationary loaded pouch has no authored pre-impact spin.
            // Pick-up and impact acquire angular velocity from contact impulses.
            coins[i].angularVelocity={0.0f,0.0f,0.0f,0.0f};
        }
        NSData* initialData=[NSData dataWithContentsOfFile:@"initial_state.json"];
        NSDictionary* initial=[NSJSONSerialization JSONObjectWithData:initialData options:0 error:nullptr];
        if(initial==nil)throw std::runtime_error("Missing settled initial state");
        auto readV=[](NSArray* a){return Vec3{[a[0] doubleValue],[a[1] doubleValue],[a[2] doubleValue]};};
        for(const auto& grid:scene.grids){NSArray* values=initial[[NSString stringWithUTF8String:grid.name.c_str()]];
            if([values count]!=grid.nodes.size())throw std::runtime_error("Initial film topology mismatch");
            for(size_t j=0;j<grid.nodes.size();++j){auto& n=nodes[grid.nodes[j]];auto v=readV(values[j]);n.positionAndInverseMass=f4(v,n.positionAndInverseMass.w);n.previousAndFixed=f4(v);n.velocityAndMass.x=n.velocityAndMass.y=n.velocityAndMass.z=0;}}
        for(size_t i=0;i<coins.size();++i){auto& coin=coins[i];auto v=readV(initial[@"coins"][i]);coin.positionAndInverseMass=f4(v,coin.positionAndInverseMass.w);coin.previousPosition=f4(v);
            coin.velocityAndMass=f4(readV(initial[@"coin_velocities"][i]),coin.velocityAndMass.w);coin.angularVelocity=f4(readV(initial[@"coin_angular_velocities"][i]));NSArray* q=initial[@"coin_orientations"][i];coin.orientation={[q[0] floatValue],[q[1] floatValue],[q[2] floatValue],[q[3] floatValue]};coin.previousOrientation=coin.orientation;}
        NSDictionary* robot=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@"robot.json"] options:0 error:nullptr];
        NSArray* robotFrames=robot[@"frames"];if([robotFrames count]<2u||[robotFrames count]<=steps)throw std::runtime_error("Robot trajectory does not cover requested steps");
        for(NSDictionary* f in robotFrames){NSData* bytes=[NSJSONSerialization dataWithJSONObject:f options:0 error:nullptr];robotFrameJSON.emplace_back(static_cast<const char*>(bytes.bytes),bytes.length);}
        std::vector<FrankaGripPoseGPU> gripPath;std::ifstream gripInput("grip.csv");std::string line;std::getline(gripInput,line);
        while(std::getline(gripInput,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);double t,x,y,z,nx,ny,nz,h,tx,ty,tz;if(!(in>>t>>x>>y>>z>>nx>>ny>>nz>>h>>tx>>ty>>tz))throw std::runtime_error("Invalid grip path row");
            FrankaGripPoseGPU g{};g.centerAndHalfGap=f4({x,y,z},h);g.normal=f4({nx,ny,nz});g.tangentX=f4({tx,ty,tz});g.tangentZ=f4(cross({tx,ty,tz},{nx,ny,nz}));gripPath.push_back(g);}
        if(gripPath.size()!=robotFrameJSON.size())throw std::runtime_error("Robot/grip trajectory lengths differ");
        const std::uint32_t gripLastStep=std::uint32_t(gripPath.size()-1u);
        auto gripPathBuffer=makeSharedBuffer(device,gripPath,"Panda FK fingertip trajectory");
        std::vector<FrankaGripRuntimeGPU> gripRuntime(1);std::vector<FrankaGripContactGPU> gripContacts(nodes.size());std::vector<mr_float4> gripMetric(2);
        auto gripRuntimeBuffer=makeSharedBuffer(device,gripRuntime,"Panda prescribed pad poses");
        auto gripContactBuffer=makeSharedBuffer(device,gripContacts,"finite compliant pad contact");
        auto gripMetricBuffer=makeSharedBuffer(device,gripMetric,"external actuator work");
        savedGripContactBuffer=gripContactBuffer;
        const auto staticFloorPipeline=makePipeline(device,library,"pb_full_static_floor"),staticContactsPipeline=makePipeline(device,library,"pb_full_static_contacts");
        const auto gripBeginPipeline=makePipeline(device,library,"fg_begin"),gripPositionPipeline=makePipeline(device,library,"fg_position"),gripVelocityPipeline=makePipeline(device,library,"fg_velocity"),gripWorkPipeline=makePipeline(device,library,"fg_work");
        std::ofstream gripHistory(output/"grip_history.csv");gripHistory<<"step,time_s,contact_nodes,max_contact_nodes,peak_pad_normal_force_N,signed_work_J,positive_work_J\n";
        std::vector<PlasticBagCoinPairGPU> coinPairs;
        std::vector<std::vector<AdjacencyEntry>> coinPairAdj(kCoinCount);
        for(std::uint32_t a=0u;a<kCoinCount;++a){
            for(std::uint32_t b=a+1u;b<kCoinCount;++b){
                const auto pairIndex=static_cast<std::uint32_t>(coinPairs.size());
                coinPairs.push_back({.coins={a,b,0u,0u}});
                coinPairAdj[a].push_back({pairIndex,0u,0u,0u});
                coinPairAdj[b].push_back({pairIndex,1u,0u,0u});
            }
        }
        std::vector<PlasticBagCoinPairRefGPU> packedCoinPairRefs;
        std::vector<PlasticBagNodeSpanGPU> coinPairSpans(kCoinCount);
        for(std::size_t i=0u;i<kCoinCount;++i){
            coinPairSpans[i].range={static_cast<std::uint32_t>(packedCoinPairRefs.size()),
                static_cast<std::uint32_t>(coinPairAdj[i].size()),0u,0u};
            for(const auto& ref:coinPairAdj[i])packedCoinPairRefs.push_back({.identity={ref.index,ref.local,0u,0u}});
        }
        std::vector<std::uint32_t> candidateTriangles(kCoinCount*contactCandidateCapacity);
        std::vector<std::uint32_t> candidateCounts(kCoinCount,0u);
        std::vector<std::uint32_t> candidateOverflow(kCoinCount,0u);
        std::vector<PlasticBagContactGPU> contacts(kCoinCount*kContactSamples);
        std::vector<PlasticBagContactGPU> filmLoads(kCoinCount*kContactSamples);
        std::vector<PlasticBagContactGPU> fullContacts(kCoinCount*contactCandidateCapacity),fullLoads(fullContacts.size());
        std::vector<mr_float4> floorLoads(nodes.size());
        std::vector<PlasticBagContactGPU> contactVelocityImpulses(kCoinCount);
        std::vector<PlasticBagCoinPairScratchGPU> pairScratch(coinPairs.size());
        std::vector<PlasticBagEdgeScratchGPU> edgeScratch(edges.size());
        std::vector<PlasticBagContactGPU> selfContactCorrections(nodes.size());
        std::vector<PlasticBagSeamContactGPU> sideSeamContacts(kCoinCount);

        const double bendingRigidity=kYoungsModulus*std::pow(kFilmGauge,3.0)/
            (12.0*(1.0-kPoissonRatio*kPoissonRatio));
        PlasticBagConfigGPU config{};
        config.counts={static_cast<std::uint32_t>(nodes.size()),static_cast<std::uint32_t>(edges.size()),
            static_cast<std::uint32_t>(scene.triangles.size()),static_cast<std::uint32_t>(kCoinCount)};
        config.scales={static_cast<float>(kFrameDt/kSubsteps),static_cast<float>(kCoinRadius),
            static_cast<float>(0.5*kCoinThickness),static_cast<float>(kContactSamples)};
        config.gravityAndBending={0.0f,-static_cast<float>(kGravity),0.0f,static_cast<float>(bendingRigidity)};
        config.solver={0.18f,0.32f,1.0f,static_cast<float>(kCoinContactFriction)};
        config.geometry={static_cast<float>(0.5*kFilmGauge),static_cast<float>(kGroundHeight),
            static_cast<float>(contactCandidateCapacity),1.0e6f};
        config.iterations={16u,4u,1u,0u};

        std::vector<std::set<std::uint32_t>> usedColors(nodes.size());
        std::vector<std::vector<std::uint32_t>> colors;
        for(std::uint32_t ei=0u;ei<edges.size();++ei){
            const auto edge=edges[ei];std::uint32_t color=0u;
            while(usedColors[edge.nodes.x].contains(color)||usedColors[edge.nodes.y].contains(color))++color;
            if(color>=colors.size())colors.resize(color+1u);
            colors[color].push_back(ei);usedColors[edge.nodes.x].insert(color);usedColors[edge.nodes.y].insert(color);
        }
        std::vector<std::uint32_t> coloredOrder;
        std::vector<mr_uint4> coloredSpans;
        for(const auto& color:colors){coloredSpans.push_back({static_cast<std::uint32_t>(coloredOrder.size()),static_cast<std::uint32_t>(color.size()),0u,static_cast<std::uint32_t>(colors.size())});coloredOrder.insert(coloredOrder.end(),color.begin(),color.end());}
        auto coloredOrderBuffer=makeSharedBuffer(device,coloredOrder,"disjoint edge color order");
        auto coloredSpanBuffer=makeSharedBuffer(device,coloredSpans,"edge color spans");
        std::cout<<"membrane_edge_colors="<<colors.size()<<std::endl;
        std::vector<std::set<std::uint32_t>> usedHingeColors(nodes.size());
        std::vector<std::vector<std::uint32_t>> hingeColors;
        for(std::uint32_t hi=0;hi<scene.hinges.size();++hi){
            const auto ids=scene.hinges[hi].nodes;std::uint32_t color=0;
            while(usedHingeColors[ids.x].contains(color)||usedHingeColors[ids.y].contains(color)||
                  usedHingeColors[ids.z].contains(color)||usedHingeColors[ids.w].contains(color))++color;
            if(color>=hingeColors.size())hingeColors.resize(color+1u);
            hingeColors[color].push_back(hi);
            for(const auto ni:{ids.x,ids.y,ids.z,ids.w})usedHingeColors[ni].insert(color);
        }
        std::vector<std::uint32_t> hingeOrder;std::vector<mr_uint4> hingeSpans;
        for(const auto& color:hingeColors){hingeSpans.push_back({static_cast<std::uint32_t>(hingeOrder.size()),static_cast<std::uint32_t>(color.size()),0u,static_cast<std::uint32_t>(hingeColors.size())});hingeOrder.insert(hingeOrder.end(),color.begin(),color.end());}
        auto hingeOrderBuffer=makeSharedBuffer(device,hingeOrder,"disjoint hinge colors");
        auto hingeSpanBuffer=makeSharedBuffer(device,hingeSpans,"hinge color spans");
        auto hingeBuffer=makeSharedBuffer(device,scene.hinges,"objective panel bending");
        std::cout<<"bending_hinges="<<scene.hinges.size()<<" colors="<<hingeColors.size()<<std::endl;
        auto fullContactBuffer=makeSharedBuffer(device,fullContacts,"full triangle cylinder witnesses");
        auto fullLoadBuffer=makeSharedBuffer(device,fullLoads,"full triangle cylinder support impulses");
        auto floorLoadBuffer=makeSharedBuffer(device,floorLoads,"transmitted floor support impulses");
        auto filmLoadBuffer=makeSharedBuffer(device,filmLoads,"accumulated film support impulses");
        std::vector<mr_float4> filmDiagnostics(25u);
        auto filmDiagnosticsBuffer=makeSharedBuffer(device,filmDiagnostics,"film velocity work and load diagnostics");
        auto nodeBuffer=makeSharedBuffer(device,nodes,"LDPE shell nodes");
        auto panelPairBuffer=makeSharedBuffer(device,panelPairs,"envelope panel non-crossing pairs");
        auto edgeBuffer=makeSharedBuffer(device,edges,"LDPE membrane constraints");
        auto triangleBuffer=makeSharedBuffer(device,scene.triangles,"LDPE shell triangles");
        auto edgeRefBuffer=makeSharedBuffer(device,nodeEdgeRefs,"node-to-edge adjacency");
        auto edgeSpanBuffer=makeSharedBuffer(device,edgeSpans,"node edge spans");
        auto coinBuffer=makeSharedBuffer(device,coins,"10 rigid metallic coins");
        auto candidateTriangleBuffer=makeSharedBuffer(device,candidateTriangles,"bounded coin-film triangle candidates");
        auto candidateCountBuffer=makeSharedBuffer(device,candidateCounts,"coin-film candidate counts");
        auto candidateOverflowBuffer=makeSharedBuffer(device,candidateOverflow,"coin-film candidate overflow flags");
        auto contactBuffer=makeSharedBuffer(device,contacts,"selected coin-to-film contacts");
        auto contactVelocityImpulseBuffer=makeSharedBuffer(device,contactVelocityImpulses,
            "coin-film normal and friction impulses");
        auto pairBuffer=makeSharedBuffer(device,coinPairs,"coin collision pairs");
        auto pairScratchBuffer=makeSharedBuffer(device,pairScratch,"coin pair corrections");
        std::vector<PlasticBagPairLoadGPU> pairLoads(coinPairs.size());
        auto pairLoadBuffer=makeSharedBuffer(device,pairLoads,"coin pair support and friction impulses");
        auto selfContactCorrectionBuffer=makeSharedBuffer(device,selfContactCorrections,
            "thin-shell self-contact corrections");
        auto sideSeamContactBuffer=makeSharedBuffer(device,sideSeamContacts,
            "coin-to-welded-side-seam contacts");
        auto pairRefBuffer=makeSharedBuffer(device,packedCoinPairRefs,"coin pair adjacency");
        auto pairSpanBuffer=makeSharedBuffer(device,coinPairSpans,"coin pair spans");
        auto configBuffer=makeSharedBuffer(device,std::vector<PlasticBagConfigGPU>{config},"scene constants");
        savedGripConfigBuffer=configBuffer;

        CheckpointHeader saved;
        if(!resume.empty()){
            checkpoint(resume,saved,true);startStep=std::uint32_t(saved.step);resumeSerializationStep=startStep;
            positiveActuatorWork=saved.positiveWork;signedActuatorWork=saved.signedWork;lastGripContacts=unsigned(saved.contacts);
            std::memcpy(nodes.data(),nodeBuffer.contents,nodes.size()*sizeof(nodes[0]));
            std::memcpy(coins.data(),coinBuffer.contents,coins.size()*sizeof(coins[0]));
        }
        auto padEnergy=[&](){
            const auto& pose=static_cast<const FrankaGripRuntimeGPU*>(gripRuntimeBuffer.contents)->current;
            const auto* ct=static_cast<const FrankaGripContactGPU*>(gripContactBuffer.contents);
            double energy=0;
            for(std::size_t i=0;i<nodes.size();++i){
                if(ct[i].normalMultipliers.x+ct[i].normalMultipliers.y<=0)continue;
                const auto x=physicalPosition(nodes[i]);
                const double side=(x[0]-pose.centerAndHalfGap.x)*pose.normal.x+(x[1]-pose.centerAndHalfGap.y)*pose.normal.y+(x[2]-pose.centerAndHalfGap.z)*pose.normal.z;
                for(double delta:{.5*kFilmGauge-side-pose.centerAndHalfGap.w,.5*kFilmGauge+side-pose.centerAndHalfGap.w}){
                    delta=std::max(0.0,delta);
                    // Integral of the native capped linear normal spring:
                    // F=min(k*compression,20 N), with k=200000 N/m.
                    constexpr double stiffness=200000.0,cap=20.0,transition=cap/stiffness;
                    energy+=delta<=transition?.5*stiffness*delta*delta:cap*(delta-.5*transition);
                }
            }
            return energy;
        };
        double acceptedPadEnergy=padEnergy();
        std::ofstream balanceHistory(output/"energy_balance.csv");
        balanceHistory<<"step,time_s,pad_elastic_J,system_energy_J,signed_actuator_work_J,frame_energy_balance_J,local_tolerance_J,local_limit_J,global_limit_J\n";
        const std::size_t snapshotSlots=256u*8u;
        id<MTLBuffer> traceNodes=traceFrame?[device newBufferWithLength:snapshotSlots*nodeBuffer.length options:MTLResourceStorageModeShared]:nil;
        id<MTLBuffer> traceCoins=traceFrame?[device newBufferWithLength:snapshotSlots*coinBuffer.length options:MTLResourceStorageModeShared]:nil;
        auto tracePipeline=makePipeline(device,library,"pb_debug_snapshot");
        auto pairVelocityResetPipeline=makePipeline(device,library,"pb_pair_velocity_reset");
        // Purely derived geometry: invalidate at each substep velocity phase.
        // Do not add it to checkpointBuffers; it contains no physical state.
        id<MTLBuffer> pairVelocityGeometryBuffer=[device newBufferWithLength:
            sizeof(PlasticBagPairVelocityGeometryGPU)*kCoinCount*(kCoinCount-1u)/2u
            options:MTLResourceStorageModePrivate];
        if(pairVelocityGeometryBuffer==nil)throw std::runtime_error("pair geometry cache allocation failed");
        std::ofstream stageHistory(output/"stage_energy.csv");
        stageHistory<<"step,substeps,substep,stage,kinetic_J,gravity_J,membrane_J,bending_J,total_J,max_stretch,edge_a,edge_b,max_edge_energy_J\n";
        std::ofstream trajectory(output/"trajectory.json",std::ios::binary|std::ios::trunc);
        if(!trajectory)throw std::runtime_error("cannot create trajectory.json");
        trajectory<<"{\"schema\":\"numi.plastic-bag-metal-trajectory.v5\",\"fps\":240,\"frames\":[";
        bool firstFrame=true;
        std::uint64_t acceptedSteps=0u;
        double gpuMilliseconds=0.0;
        std::size_t maxContacts=0u;
        std::size_t maxContactCandidates=0u;
        std::size_t maxSelfContactVertices=0u;
        std::size_t maxSideSeamContacts=0u;
        double maxSideSeamPenetration=0.0;
        double maxSideSeamImpulseNs=0.0;
        double windowMaxSideSeamPenetration=0.0;
        double windowMaxSideSeamImpulseNs=0.0;
        double windowMaxCoinLinearSpeed=0.0;
        double windowMaxCoinAngularSpeed=0.0;
        std::size_t maxEscapedCoins=0u;
        std::size_t maxProjectedDepthSpanExceeded=0u;
        std::size_t escapedCoinsFinal=0u;
        std::size_t minInsideAfterDrop=kCoinCount;
        double maxStretch=1.0;
        double maxCoinLinearSpeed=0.0;
        double maxCoinAngularSpeed=0.0;
        double maxShellGroundPenetration=0.0;
        double maxCoinGroundPenetration=0.0;
        EnergyState acceptedEnergy=sceneEnergy(nodes,coins,scene,bendingRigidity);
        const double initialEnergy=resume.empty()?acceptedEnergy.total():saved.initialEnergy;
        double peakAcceptedEnergy=initialEnergy;
        std::uint32_t maximumSubsteps=kSubsteps;
        std::uint64_t energyRejectedFrames=0u;
        std::ofstream filmHistory(output/"film_load_history.csv");
        filmHistory<<"step,time_s,witness_solves,normal_delta_impulse_Ns,tangent_delta_impulse_Ns,velocity_solver_delta_kinetic_J,max_positive_witness_delta_kinetic_J,max_position_normal_impulse_Ns,film_solves_for_coins_on_table\n";
        std::ofstream coverageHistory(output/"coverage_history.csv");
        coverageHistory<<"step,time_s,full_triangle_position_solves,ccd_constraint_solves,ccd_conservative_uncertain_solves,max_ccd_constraint_depth_m\n";
        std::ofstream contactRetries(output/"contact_accuracy_retries.csv");
        contactRetries<<"step,substeps,bad_triangle_coin_pairs,uncertain_distance_pairs,max_clearance_deficit_bound_m,film_table_penetration_m\n";
        std::uint64_t contactRejectedFrames=0;
        std::ofstream energyHistory(output/"energy_history.csv");
        energyHistory<<"step,time_s,substeps,attempts,kinetic_J,gravity_J,membrane_J,bending_J,total_J\n";
        energyHistory<<"0,0,0,0,"<<std::setprecision(14)<<acceptedEnergy.kinetic<<','<<acceptedEnergy.gravity<<','<<acceptedEnergy.membrane<<','<<acceptedEnergy.bending<<','<<initialEnergy<<'\n';
        std::ofstream retries(output/"energy_retries.csv");
        retries<<"step,substeps,energy_before_J,energy_trial_J,limit_J,kinetic_J,gravity_J,membrane_J,bending_J\n";
        const auto start=std::chrono::steady_clock::now();
        const std::size_t contactSampleCount=contacts.size();


        writeFrame(trajectory,startStep,scene,nodes,coins,classifyCoins(coins,scene,nodes),firstFrame);trajectory.flush();

        const auto* restoredConfig=static_cast<const PlasticBagConfigGPU*>(configBuffer.contents);
        const std::uint32_t restoredSubsteps=resume.empty()?kSubsteps:std::clamp(std::uint32_t(std::lround(kFrameDt/restoredConfig->scales.x)),kSubsteps,256u);
        std::uint32_t preferredSubsteps=16u,preferredRefinement=4u;
        std::uint32_t cleanIntervals=0u;
        std::ofstream accuracyHistory(output/"solver_accuracy.csv");
        accuracyHistory<<"step,substeps,material_refinement,attempt,energy_bad,film_bad,balance_error_J\n";
        for(std::uint32_t step=startStep+1u;step<=steps;++step){
            // Retry the same physical output interval from its saved state.
            // Energy is measured, never rescaled or removed to pass a gate.
            const auto previousNodes=nodes;
            const auto previousCoins=coins;
            std::vector<char> previousGrip(gripContactBuffer.length);
            std::memcpy(previousGrip.data(),gripContactBuffer.contents,previousGrip.size());
            std::uint32_t activeSubsteps=preferredSubsteps,activeRefinement=preferredRefinement,attempts=0u;
            // Nonsmooth contacts and finite-precision constraint velocities can
            // make the defect nonmonotone in dt. Search bounded integration and
            // coupled-solve budgets from the identical accepted state. Every
            // candidate must satisfy the same work-energy and geometry gates.
            std::vector<std::pair<std::uint32_t,std::uint32_t>> retryBudgets;
            auto addBudget=[&](std::uint32_t n,std::uint32_t r){
                const std::pair<std::uint32_t,std::uint32_t> b{n,r};
                if(std::find(retryBudgets.begin(),retryBudgets.end(),b)==retryBudgets.end())retryBudgets.push_back(b);
            };
            addBudget(activeSubsteps,activeRefinement);
            for(auto r:{8u,16u,32u,64u})
                for(auto n:{16u,20u,12u,24u,28u,32u,8u,48u,64u,128u,256u})addBudget(n,r);
            for(;;){
            ++attempts;
            std::memset(gripMetricBuffer.contents,0,2u*sizeof(mr_float4));
            std::memset(filmDiagnosticsBuffer.contents,0,3u*sizeof(mr_float4));
            config.scales.x=static_cast<float>(kFrameDt/activeSubsteps);
            config.iterations={4u,1u,1u,0u};
            std::memcpy(configBuffer.contents,&config,sizeof(config));
            std::memcpy(coinBuffer.contents,coins.data(),coins.size()*sizeof(coins[0]));
            std::memcpy(nodeBuffer.contents,nodes.data(),nodes.size()*sizeof(nodes[0]));
            id<MTLCommandBuffer> command=[queue commandBuffer];
            if(command==nil)throw std::runtime_error("Metal command buffer allocation failed");
            command.label=[NSString stringWithFormat:@"plastic bag frame %u",step];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(encoder==nil)throw std::runtime_error("Metal compute encoder allocation failed");
            encoder.label=@"LDPE shell and coin scene";
            std::vector<std::array<std::uint32_t,2>> traceLabels;
            auto sampleEnergy=[&](std::uint32_t sub,std::uint32_t stage,bool){
                if(step!=traceFrame)return;
                const std::uint32_t offset=traceLabels.size();
                if(offset>=snapshotSlots)throw std::runtime_error("stage snapshot capacity exceeded");
                traceLabels.push_back({sub,stage});
                [encoder setComputePipelineState:tracePipeline];
                [encoder setBuffer:configBuffer offset:0 atIndex:0];
                [encoder setBuffer:nodeBuffer offset:0 atIndex:1];[encoder setBuffer:coinBuffer offset:0 atIndex:2];
                [encoder setBuffer:traceNodes offset:offset*nodeBuffer.length atIndex:3];
                [encoder setBuffer:traceCoins offset:offset*coinBuffer.length atIndex:4];
                [encoder dispatchThreads:MTLSizeMake(nodes.size(),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto gripPosition=[&](){encodeKernel(encoder,gripPositionPipeline,{configBuffer,nodeBuffer,gripRuntimeBuffer,gripContactBuffer},nodes.size());[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];};
            auto pairPosition=[&](){
                [encoder setComputePipelineState:pairSweepPipeline];
                std::array<id<MTLBuffer>,6u> bs{configBuffer,coinBuffer,nodeBuffer,panelPairBuffer,pairLoadBuffer,filmDiagnosticsBuffer};
                for(std::size_t j=0;j<bs.size();++j)[encoder setBuffer:bs[j] offset:0 atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(kCoinCount,1,1)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto filmPosition=[&](){
                [encoder setComputePipelineState:pipelines[32]];
                std::array<id<MTLBuffer>,7u> buffers{configBuffer,nodeBuffer,triangleBuffer,coinBuffer,contactBuffer,filmLoadBuffer,floorLoadBuffer};
                for(std::size_t j=0;j<buffers.size();++j)[encoder setBuffer:buffers[j] offset:0u atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto fullPosition=[&](bool ccd=false){
                encodeKernel(encoder,pipelines[33],{configBuffer,nodeBuffer,triangleBuffer,coinBuffer,fullContactBuffer,candidateTriangleBuffer,candidateCountBuffer},fullContacts.size());
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if(ccd){
                    encodeKernel(encoder,pipelines[36],{configBuffer,nodeBuffer,triangleBuffer,coinBuffer,fullContactBuffer,candidateTriangleBuffer,candidateCountBuffer},fullContacts.size());
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [encoder setComputePipelineState:pipelines[34]];
                std::array<id<MTLBuffer>,11u> buffers{configBuffer,nodeBuffer,triangleBuffer,coinBuffer,fullContactBuffer,fullLoadBuffer,candidateTriangleBuffer,candidateCountBuffer,floorLoadBuffer,filmDiagnosticsBuffer,filmLoadBuffer};
                for(std::size_t j=0;j<buffers.size();++j)[encoder setBuffer:buffers[j] offset:0u atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,staticFloorPipeline,{configBuffer,nodeBuffer,coinBuffer,floorLoadBuffer},nodes.size());
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder setComputePipelineState:staticContactsPipeline];
                std::array<id<MTLBuffer>,9u> staticBuffers{configBuffer,nodeBuffer,triangleBuffer,coinBuffer,fullLoadBuffer,candidateTriangleBuffer,candidateCountBuffer,floorLoadBuffer,filmLoadBuffer};
                for(std::size_t j=0;j<staticBuffers.size();++j)[encoder setBuffer:staticBuffers[j] offset:0u atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto filmVelocity=[&](){
                [encoder setComputePipelineState:pipelines[31]];
                std::array<id<MTLBuffer>,7u> buffers{configBuffer,nodeBuffer,coinBuffer,triangleBuffer,filmLoadBuffer,filmDiagnosticsBuffer,floorLoadBuffer};
                for(std::size_t j=0;j<buffers.size();++j)[encoder setBuffer:buffers[j] offset:0u atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto fullVelocity=[&](){
                [encoder setComputePipelineState:pipelines[35]];
                std::array<id<MTLBuffer>,9u> buffers{configBuffer,nodeBuffer,coinBuffer,triangleBuffer,fullLoadBuffer,filmDiagnosticsBuffer,floorLoadBuffer,candidateTriangleBuffer,candidateCountBuffer};
                for(std::size_t j=0;j<buffers.size();++j)[encoder setBuffer:buffers[j] offset:0u atIndex:j];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto solveHinges=[&](bool reset){
                [encoder setComputePipelineState:pipelines[0]];
                std::array<id<MTLBuffer>,5u> buffers{configBuffer,nodeBuffer,hingeBuffer,hingeOrderBuffer,hingeSpanBuffer};
                for(std::size_t j=0;j<buffers.size();++j)[encoder setBuffer:buffers[j] offset:0u atIndex:j];
                const std::uint32_t flag=reset?1u:0u;
                [encoder setBytes:&flag length:sizeof(flag) atIndex:5u];
                [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(128u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            auto tableVelocity=[&](bool first){
                [encoder setComputePipelineState:pipelines[28]];
                [encoder setBuffer:configBuffer offset:0u atIndex:0u];
                [encoder setBuffer:nodeBuffer offset:0u atIndex:1u];
                [encoder setBuffer:coinBuffer offset:0u atIndex:2u];
                const std::uint32_t flag=first?1u:0u;
                [encoder setBytes:&flag length:sizeof(flag) atIndex:3u];
                [encoder setBuffer:floorLoadBuffer offset:0u atIndex:4u];
                [encoder dispatchThreads:MTLSizeMake(std::max(nodes.size(),coins.size()),1u,1u) threadsPerThreadgroup:MTLSizeMake(128u,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
            for(std::uint32_t substep=0u;substep<activeSubsteps;++substep){
                const std::array<float,2> gripTimes{float(step-1u)+float(substep)/activeSubsteps,float(step-1u)+float(substep+1u)/activeSubsteps};
                [encoder setComputePipelineState:gripBeginPipeline];[encoder setBuffer:configBuffer offset:0 atIndex:0];[encoder setBuffer:gripPathBuffer offset:0 atIndex:1];[encoder setBuffer:gripRuntimeBuffer offset:0 atIndex:2];[encoder setBuffer:gripContactBuffer offset:0 atIndex:3];[encoder setBytes:gripTimes.data() length:sizeof(gripTimes) atIndex:4];[encoder setBytes:&gripLastStep length:sizeof(gripLastStep) atIndex:5];[encoder dispatchThreads:MTLSizeMake(nodes.size(),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                sampleEnergy(substep,0u,true);
                encodeKernel(encoder,pipelines[30],{configBuffer,filmLoadBuffer,floorLoadBuffer,fullLoadBuffer,pairLoadBuffer},fullLoads.size());
                solveHinges(true);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,pipelines[1],{configBuffer,nodeBuffer},nodes.size());
                encodeKernel(encoder,pipelines[2],{configBuffer,coinBuffer},kCoinCount);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,pipelines[3],{configBuffer,edgeBuffer},edges.size());
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                sampleEnergy(substep,1u,false);
                const std::uint32_t couplingRounds=activeRefinement*std::max({kCoinPairPositionIterations,kContactIterations,1u});
                const std::uint32_t filmContactsInRounds=kContactIterations>0u
                    ? kContactIterations-1u : 0u;
                for(std::uint32_t round=0u;round<couplingRounds;++round){
                    solveHinges(false);
                    [encoder setComputePipelineState:pipelines[29]];
                    std::array<id<MTLBuffer>,5u> membraneBuffers{configBuffer,nodeBuffer,edgeBuffer,coloredOrderBuffer,coloredSpanBuffer};
                    for(std::size_t j=0u;j<membraneBuffers.size();++j)[encoder setBuffer:membraneBuffers[j] offset:0u atIndex:j];
                    [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(128u,1u,1u)];
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    gripPosition();
                    if(round==0u)sampleEnergy(substep,2u,false);
                    pairPosition();
                    const bool runFilmContact=true;
                    if(runFilmContact){
                        encodeKernel(encoder,pipelines[6],{configBuffer,nodeBuffer,triangleBuffer},scene.triangles.size());
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        encodeKernel(encoder,pipelines[23],{configBuffer,nodeBuffer,triangleBuffer,
                            selfContactCorrectionBuffer,edgeRefBuffer,edgeSpanBuffer,edgeBuffer},nodes.size());
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        encodeKernel(encoder,pipelines[24],{configBuffer,nodeBuffer,triangleBuffer,selfContactCorrectionBuffer},1u);
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        encodeKernel(encoder,pipelines[6],{configBuffer,nodeBuffer,triangleBuffer},scene.triangles.size());
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        encodeKernel(encoder,pipelines[13],{configBuffer,triangleBuffer,nodeBuffer,coinBuffer,
                            candidateTriangleBuffer,candidateCountBuffer,candidateOverflowBuffer},kCoinCount);
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        // Full triangle contact owns coverage; the unused sampled seed is omitted.
                        /* Full triangle manifold below owns physical film contact. */
                fullPosition();gripPosition();
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                }
                // All normal contacts remain coupled to material sweeps;
                // there are no uncoupled contact-only cleanup batches.
                // Revisit all exact triangle contacts after pair/table
                // cleanup. Neighboring material reactions can otherwise leave
                // one newly intersecting patch after the final witness pass.
                for(std::uint32_t closure=0u;closure<32u*(activeRefinement/4u);++closure){
                    solveHinges(false);
                    [encoder setComputePipelineState:pipelines[29]];
                    std::array<id<MTLBuffer>,5u> closureMaterial{configBuffer,nodeBuffer,edgeBuffer,coloredOrderBuffer,coloredSpanBuffer};
                    for(std::size_t j=0;j<closureMaterial.size();++j)[encoder setBuffer:closureMaterial[j] offset:0u atIndex:j];
                    [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(128u,1u,1u)];
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    pairPosition();
                    encodeKernel(encoder,pipelines[16],{configBuffer,nodeBuffer,coinBuffer},std::max(nodes.size(),coins.size()));
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    encodeKernel(encoder,pipelines[6],{configBuffer,nodeBuffer,triangleBuffer},scene.triangles.size());
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    encodeKernel(encoder,pipelines[13],{configBuffer,triangleBuffer,nodeBuffer,coinBuffer,candidateTriangleBuffer,candidateCountBuffer,candidateOverflowBuffer},kCoinCount);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    fullPosition(closure==0u);gripPosition();
                }
                sampleEnergy(substep,3u,false);
                encodeKernel(encoder,pipelines[15],{configBuffer,nodeBuffer,coinBuffer,contactBuffer,
                    triangleBuffer,contactVelocityImpulseBuffer},
                    std::max(nodes.size(),coins.size()));
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,pipelines[22],{configBuffer,nodeBuffer,triangleBuffer,
                    contactVelocityImpulseBuffer},nodes.size());
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,floorSupportPipeline,{configBuffer,nodeBuffer,floorLoadBuffer},nodes.size());
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,pairVelocityResetPipeline,{configBuffer,pairLoadBuffer,pairVelocityGeometryBuffer},kCoinCount*(kCoinCount-1u)/2u);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                /* Full triangle velocity manifold owns film response. */
                fullVelocity();
                tableVelocity(true);
                for(std::uint32_t velocityRound=0u;velocityRound<kCoinPairVelocityIterations;++velocityRound){
                    [encoder setComputePipelineState:pipelines[18]];
                    [encoder setBuffer:pairLoadBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:filmDiagnosticsBuffer offset:0u atIndex:3u];
                    [encoder setBuffer:pairVelocityGeometryBuffer offset:0u atIndex:4u];
                    [encoder setBuffer:configBuffer offset:0u atIndex:0u];
                    [encoder setBuffer:coinBuffer offset:0u atIndex:1u];
                    [encoder dispatchThreadgroups:MTLSizeMake(1u,1u,1u)
                        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(kCoinCount),1u,1u)];
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    /* Full triangle velocity manifold owns film response. */
                fullVelocity();
                    tableVelocity(false);
                }
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,gripVelocityPipeline,{configBuffer,nodeBuffer,gripRuntimeBuffer,gripContactBuffer},nodes.size());[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encodeKernel(encoder,gripWorkPipeline,{configBuffer,gripRuntimeBuffer,gripContactBuffer,gripMetricBuffer},1u);[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                sampleEnergy(substep,4u,true);
                sampleEnergy(substep,5u,true);
            }
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted){
                throw std::runtime_error(std::string("Metal scene command failed: ")+
                    (command.error==nil?"unknown":command.error.localizedDescription.UTF8String));
            }
            if(command.GPUEndTime>command.GPUStartTime)gpuMilliseconds+=
                (command.GPUEndTime-command.GPUStartTime)*1000.0;
            for(std::size_t ti=0;ti<traceLabels.size();++ti){
                std::vector<PlasticBagNodeGPU> tn(nodes.size());std::vector<PlasticBagCoinGPU> tc(coins.size());
                std::memcpy(tn.data(),(char*)traceNodes.contents+ti*nodeBuffer.length,nodeBuffer.length);
                std::memcpy(tc.data(),(char*)traceCoins.contents+ti*coinBuffer.length,coinBuffer.length);
                auto te=sceneEnergy(tn,tc,scene,bendingRigidity);
                double stretch=1,edgeEnergy=0;unsigned a=0,b=0;
                for(const auto& edge:scene.edges){
                    auto pa=tn[edge.nodes.x].positionAndInverseMass,pb=tn[edge.nodes.y].positionAndInverseMass;
                    double len=std::hypot(double(pa.x-pb.x),double(pa.y-pb.y),double(pa.z-pb.z));
                    stretch=std::max(stretch,len/edge.material.x);
                    double en=.5*(len-edge.material.x)*(len-edge.material.x)/edge.material.y;
                    if(en>edgeEnergy){edgeEnergy=en;a=edge.nodes.x;b=edge.nodes.y;}
                }
                stageHistory<<step<<','<<activeSubsteps<<','<<traceLabels[ti][0]<<','<<traceLabels[ti][1]<<','<<std::setprecision(14)<<te.kinetic<<','<<te.gravity<<','<<te.membrane<<','<<te.bending<<','<<te.total()<<','<<stretch<<','<<a<<','<<b<<','<<edgeEnergy<<'\n';
            }
            stageHistory.flush();
            std::memcpy(nodes.data(),nodeBuffer.contents,nodes.size()*sizeof(nodes[0]));
            std::memcpy(coins.data(),coinBuffer.contents,coins.size()*sizeof(coins[0]));
            const EnergyState trialEnergy=sceneEnergy(nodes,coins,scene,bendingRigidity);
            // Symplectic/discrete elastic energy may oscillate between frames.
            // Enforce the closed-system initial budget; preserve and report
            // local positive drift instead of assuming exact stepwise passivity.
            const auto* gm=static_cast<const mr_float4*>(gripMetricBuffer.contents);
            const double trialPadEnergy=padEnergy();
            const double systemEnergy=trialEnergy.total()+trialPadEnergy;
            const double previousSystemEnergy=acceptedEnergy.total()+acceptedPadEnergy;
            // Local work-energy error control. Detect injection immediately;
            // dissipated energy from old frames is not a reservoir of credit.
            const double localTolerance=2.e-6+1.e-4*std::max(previousSystemEnergy,std::abs(double(gm[0].x)));
            const double globalLimit=initialEnergy+positiveActuatorWork+gm[0].y+1.e-4;
            const double localLimit=previousSystemEnergy+double(gm[0].x)+localTolerance;
            const double limit=std::min(globalLimit,localLimit);
            const double balanceError=systemEnergy-previousSystemEnergy-double(gm[0].x);
            const FilmAccuracy accuracy=filmAccuracy(nodes,coins,scene);
            double filmTablePenetration=0.0;
            for(const auto& node:nodes)filmTablePenetration=std::max(filmTablePenetration,
                kGroundHeight+.5*kFilmGauge-physicalPosition(node)[1]);
            const bool tableBad=filmTablePenetration>1.e-7;

            std::uint32_t unresolvedPairQuery=0u;
            std::memcpy(&unresolvedPairQuery,(char*)filmDiagnosticsBuffer.contents+7u*sizeof(float),sizeof(unresolvedPairQuery));
            const bool pairQueryBad=unresolvedPairQuery!=0u;
            if(pairQueryBad){std::ofstream q(output/("unresolved_pair_"+std::to_string(step)+"_"+std::to_string(activeSubsteps)+"_"+std::to_string(activeRefinement)+".bin"),std::ios::binary);q.write((char*)filmDiagnosticsBuffer.contents+3u*sizeof(mr_float4),2u*sizeof(PlasticBagCoinGPU));}
            const bool energyBad=!std::isfinite(systemEnergy)||systemEnergy>limit;
            accuracyHistory<<step<<','<<activeSubsteps<<','<<activeRefinement<<','<<attempts<<','<<energyBad<<','<<(accuracy.bad>0u||tableBad||pairQueryBad)<<','<<std::setprecision(14)<<balanceError<<'\n';accuracyHistory.flush();
            if(energyBad||accuracy.bad>0u||tableBad||pairQueryBad){
                if(accuracy.bad>0u||tableBad){++contactRejectedFrames;contactRetries<<step<<','<<activeSubsteps<<','<<accuracy.bad<<','<<accuracy.uncertain<<','<<std::setprecision(14)<<accuracy.maxDeficitBound<<','<<filmTablePenetration<<'\n';contactRetries.flush();}
                if(energyBad)++energyRejectedFrames;
                if(energyBad)retries<<step<<','<<activeSubsteps<<','<<std::setprecision(14)<<acceptedEnergy.total()<<','<<systemEnergy<<','<<limit<<','<<trialEnergy.kinetic<<','<<trialEnergy.gravity<<','<<trialEnergy.membrane<<','<<trialEnergy.bending<<'\n';retries.flush();
                if(attempts>=retryBudgets.size()){
                    std::ofstream failedState(output/("rejected_frame_"+std::to_string(step)+".json"));
                    failedState<<"{\"fps\":240,\"frames\":[";bool firstFailed=true;
                    writeFrame(failedState,step,scene,nodes,coins,classifyCoins(coins,scene,nodes),firstFailed);failedState<<"]}";
                    throw std::runtime_error(std::string(energyBad?"energy":"film contact accuracy")+" acceptance failed at frame "+std::to_string(step)+" after all bounded integration/contact budgets");
                }
                nodes=previousNodes;coins=previousCoins;std::memcpy(gripContactBuffer.contents,previousGrip.data(),previousGrip.size());
                // Isolated energy defects can indicate under-converged coupled
                // contact. Refine its iterations before shortening dt, which
                // amplifies constraint-velocity noise near rest. Geometry
                // failures still request smaller integration steps first.
                activeSubsteps=retryBudgets[attempts].first;
                activeRefinement=retryBudgets[attempts].second;
                continue;
            }
            balanceHistory<<step<<','<<step*kFrameDt<<','<<std::setprecision(14)<<trialPadEnergy<<','<<systemEnergy<<','<<gm[0].x<<','<<balanceError<<','<<localTolerance<<','<<localLimit<<','<<globalLimit<<'\n';balanceHistory.flush();
            acceptedPadEnergy=trialPadEnergy;
            cleanIntervals=(attempts==1u&&balanceError< -4.0*localTolerance)?cleanIntervals+1u:0u;
            preferredSubsteps=activeSubsteps;
            // Retain the successful coupled iteration budget across the
            // remaining dense-contact settling intervals.
            preferredRefinement=activeRefinement;
            positiveActuatorWork+=gm[0].y;signedActuatorWork+=gm[0].x;lastGripContacts=unsigned(gm[1].x);
            gripHistory<<step<<','<<step*kFrameDt<<','<<lastGripContacts<<','<<gm[0].z<<','<<gm[0].w<<','<<signedActuatorWork<<','<<positiveActuatorWork<<'\n';gripHistory.flush();
            acceptedEnergy=trialEnergy;peakAcceptedEnergy=std::max(peakAcceptedEnergy,trialEnergy.total());
            maximumSubsteps=std::max(maximumSubsteps,activeSubsteps);
            const auto* fd=static_cast<const mr_float4*>(filmDiagnosticsBuffer.contents);
            coverageHistory<<step<<','<<step*kFrameDt<<','<<std::setprecision(14)<<fd[2].x<<','<<fd[2].y<<','<<fd[2].z<<','<<fd[2].w<<'\n';coverageHistory.flush();
            filmHistory<<step<<','<<step*kFrameDt<<','<<std::setprecision(14)<<fd[0].x<<','<<fd[0].y<<','<<fd[0].z<<','<<fd[0].w<<','<<fd[1].x<<','<<fd[1].y<<','<<fd[1].z<<'\n';filmHistory.flush();
            energyHistory<<step<<','<<step*kFrameDt<<','<<activeSubsteps<<','<<attempts<<','<<std::setprecision(14)<<trialEnergy.kinetic<<','<<trialEnergy.gravity<<','<<trialEnergy.membrane<<','<<trialEnergy.bending<<','<<trialEnergy.total()<<'\n';energyHistory.flush();
            break;
            }
            ++acceptedSteps;
            if(checkpointEvery>0&&(step%checkpointEvery==0||step>=startStep+1u)){
                CheckpointHeader h;h.step=step;h.initialEnergy=initialEnergy;h.positiveWork=positiveActuatorWork;h.signedWork=signedActuatorWork;h.contacts=lastGripContacts;
                checkpoint(output/"latest_checkpoint.bin",h,false);
                if(step%checkpointEvery==0)std::filesystem::copy_file(output/"latest_checkpoint.bin",output/("checkpoint_"+std::to_string(step)+".bin"),std::filesystem::copy_options::overwrite_existing);
            }
            auto* nodeData=static_cast<PlasticBagNodeGPU*>(nodeBuffer.contents);
            auto* coinData=static_cast<PlasticBagCoinGPU*>(coinBuffer.contents);
            auto* selfContactData=static_cast<PlasticBagContactGPU*>(selfContactCorrectionBuffer.contents);
            auto* sideSeamData=static_cast<PlasticBagSeamContactGPU*>(sideSeamContactBuffer.contents);
            std::memcpy(nodes.data(),nodeData,nodes.size()*sizeof(nodes[0]));
            std::memcpy(coins.data(),coinData,coins.size()*sizeof(coins[0]));
            std::size_t activeSelfContactVertices=0u;
            for(std::size_t nodeIndex=0u;nodeIndex<nodes.size();++nodeIndex)
                if(selfContactData[nodeIndex].normalAndActive.w>0.0f)++activeSelfContactVertices;
            maxSelfContactVertices=std::max(maxSelfContactVertices,activeSelfContactVertices);
            std::size_t activeSideSeamContacts=0u;
            for(std::size_t coinIndex=0u;coinIndex<kCoinCount;++coinIndex){
                if(sideSeamData[coinIndex].nodesAndActive.w==0u)continue;
                ++activeSideSeamContacts;
                maxSideSeamPenetration=std::max(maxSideSeamPenetration,
                    static_cast<double>(sideSeamData[coinIndex].barycentricAndDepth.w));
                windowMaxSideSeamPenetration=std::max(windowMaxSideSeamPenetration,
                    static_cast<double>(sideSeamData[coinIndex].barycentricAndDepth.w));
                const auto impulse=sideSeamData[coinIndex].correctionCoin;
                maxSideSeamImpulseNs=std::max(maxSideSeamImpulseNs,
                    std::sqrt(static_cast<double>(impulse.x)*impulse.x+
                        static_cast<double>(impulse.y)*impulse.y+
                        static_cast<double>(impulse.z)*impulse.z));
                windowMaxSideSeamImpulseNs=std::max(windowMaxSideSeamImpulseNs,
                    std::sqrt(static_cast<double>(impulse.x)*impulse.x+
                        static_cast<double>(impulse.y)*impulse.y+
                        static_cast<double>(impulse.z)*impulse.z));
            }
            maxSideSeamContacts=std::max(maxSideSeamContacts,activeSideSeamContacts);
            const double shellFloor=kGroundHeight+0.5*kFilmGauge;
            for(const auto& node:nodes)
                maxShellGroundPenetration=std::max(maxShellGroundPenetration,
                    shellFloor-physicalPosition(node)[1]);
            for(const auto& coin:coins){
                const auto linear=coin.velocityAndMass;
                maxCoinLinearSpeed=std::max(maxCoinLinearSpeed,
                    std::sqrt(static_cast<double>(linear.x)*linear.x+
                        static_cast<double>(linear.y)*linear.y+
                        static_cast<double>(linear.z)*linear.z));
                windowMaxCoinLinearSpeed=std::max(windowMaxCoinLinearSpeed,
                    std::sqrt(static_cast<double>(linear.x)*linear.x+
                        static_cast<double>(linear.y)*linear.y+
                        static_cast<double>(linear.z)*linear.z));
                const auto omega=coin.angularVelocity;
                maxCoinAngularSpeed=std::max(maxCoinAngularSpeed,
                    std::sqrt(static_cast<double>(omega.x)*omega.x+
                        static_cast<double>(omega.y)*omega.y+
                        static_cast<double>(omega.z)*omega.z));
                windowMaxCoinAngularSpeed=std::max(windowMaxCoinAngularSpeed,
                    std::sqrt(static_cast<double>(omega.x)*omega.x+
                        static_cast<double>(omega.y)*omega.y+
                        static_cast<double>(omega.z)*omega.z));
                const auto q=coin.orientation;
                const double axisY=1.0-2.0*(static_cast<double>(q.x)*q.x+
                    static_cast<double>(q.z)*q.z);
                const double extent=kCoinRadius*std::sqrt(std::max(0.0,1.0-axisY*axisY))+
                    0.5*kCoinThickness*std::abs(axisY);
                maxCoinGroundPenetration=std::max(maxCoinGroundPenetration,
                    kGroundHeight+extent-physicalPosition(coin)[1]);
            }
            const auto* contactData=static_cast<const PlasticBagContactGPU*>(contactBuffer.contents);
            std::memcpy(candidateCounts.data(),candidateCountBuffer.contents,
                candidateCounts.size()*sizeof(candidateCounts[0]));
            std::memcpy(candidateOverflow.data(),candidateOverflowBuffer.contents,
                candidateOverflow.size()*sizeof(candidateOverflow[0]));
            for(std::size_t coinIndex=0u;coinIndex<kCoinCount;++coinIndex){
                if(candidateOverflow[coinIndex]!=0u)
                    throw std::runtime_error("coin-film swept broad phase overflow at frame "+std::to_string(step)+" coin "+std::to_string(coinIndex)+" capacity "+std::to_string(contactCandidateCapacity));
                maxContactCandidates=std::max(maxContactCandidates,
                    static_cast<std::size_t>(candidateCounts[coinIndex]));
            }
            std::size_t activeContacts=0u;
            for(std::size_t coinIndex=0u;coinIndex<kCoinCount;++coinIndex){
                for(std::uint32_t sampleIndex=0u;sampleIndex<kContactSamples;++sampleIndex){
                    if(contactData[coinIndex*kContactSamples+sampleIndex].normalAndActive.w>0.0f){
                        ++activeContacts;
                        break;
                    }
                }
            }
            maxContacts=std::max(maxContacts,activeContacts);
            for(const auto& edge:edges){
                const Vec3 a{nodes[edge.nodes.x].positionAndInverseMass.x,nodes[edge.nodes.x].positionAndInverseMass.y,nodes[edge.nodes.x].positionAndInverseMass.z};
                const Vec3 b{nodes[edge.nodes.y].positionAndInverseMass.x,nodes[edge.nodes.y].positionAndInverseMass.y,nodes[edge.nodes.y].positionAndInverseMass.z};
                maxStretch=std::max(maxStretch,norm(sub(a,b))/edge.material.x);
            }
            for(const auto& node:nodes){
                const auto p=node.positionAndInverseMass;
                if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))
                    throw std::runtime_error("non-finite shell state at frame "+std::to_string(step));
            }
            const PouchCounts pouch=classifyCoins(coins,scene,nodes);
            escapedCoinsFinal=pouch.escaped;
            maxEscapedCoins=std::max(maxEscapedCoins,pouch.escaped);
            maxProjectedDepthSpanExceeded=std::max(maxProjectedDepthSpanExceeded,
                pouch.projectedDepthSpanExceeded);
            if(static_cast<double>(step)*kFrameDt>=0.5)
                minInsideAfterDrop=std::min(minInsideAfterDrop,pouch.inside);
            if(step%recordEvery==0u||step==steps)
                writeFrame(trajectory,step,scene,nodes,coins,pouch,firstFrame);
            if(step%60u==0u){
                const double minimumCoinY=std::min_element(coins.begin(),coins.end(),
                    [](const auto& a,const auto& b){return a.positionAndInverseMass.y<b.positionAndInverseMass.y;})
                    ->positionAndInverseMass.y;
                const auto lowestCoin=std::min_element(coins.begin(),coins.end(),
                    [](const auto& a,const auto& b){return a.positionAndInverseMass.y<b.positionAndInverseMass.y;});
                const std::size_t lowestCoinIndex=static_cast<std::size_t>(lowestCoin-coins.begin());
                const auto* lowestCoinContacts=contactData+lowestCoinIndex*kContactSamples;
                const auto* deepestCoinContact=std::max_element(lowestCoinContacts,
                    lowestCoinContacts+kContactSamples,[](const auto& a,const auto& b){
                        return a.barycentricAndDepth.w<b.barycentricAndDepth.w;});
                double minimumBottomSeamY=std::numeric_limits<double>::infinity();
                for(std::size_t u=0u;u<scene.grids[0].nu;++u){
                    const auto node=scene.grids[0].nodes[gridIndex(scene.grids[0],u,0u)];
                    minimumBottomSeamY=std::min(minimumBottomSeamY,
                        static_cast<double>(nodes[node].positionAndInverseMass.y));
                }
                std::cout<<"accepted_frames="<<acceptedSteps<<" simulated_s="
                    <<static_cast<double>(step)*kFrameDt<<" max_edge_stretch="<<maxStretch
                    <<" active_contacts="<<activeContacts<<" min_coin_y="<<minimumCoinY
                    <<" window_max_seam_penetration_m="<<windowMaxSideSeamPenetration
                    <<" window_max_seam_impulse_Ns="<<windowMaxSideSeamImpulseNs
                    <<" window_max_coin_speed_m_s="<<windowMaxCoinLinearSpeed
                    <<" window_max_coin_angular_speed_rad_s="<<windowMaxCoinAngularSpeed
                    <<" inside_coins="<<pouch.inside<<" escaped_coins="<<pouch.escaped
                    <<" max_contact_candidates="<<maxContactCandidates
                    <<" lowest_coin="<<lowestCoinIndex<<" min_bottom_seam_y="<<minimumBottomSeamY
                    <<" contact_depth="<<deepestCoinContact->barycentricAndDepth.w
                    <<" contact_normal_y="<<deepestCoinContact->normalAndActive.y<<std::endl;
                windowMaxSideSeamPenetration=0.0;
                windowMaxSideSeamImpulseNs=0.0;
                windowMaxCoinLinearSpeed=0.0;
                windowMaxCoinAngularSpeed=0.0;
            }
        }
        trajectory<<"]}\n";
        trajectory.flush();
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream summary(output/"run_summary.json",std::ios::binary|std::ios::trunc);
        summary<<std::setprecision(10)<<"{\n"
            <<"  \"device\":\""<<device.name.UTF8String<<"\",\n"
            <<"  \"solver\":\"Metal elastic damped membrane, objective dihedrals, inertial coin-film contact\",\n"
            <<"  \"requested_steps\":"<<steps<<",\n"
            <<"  \"start_step\":"<<startStep<<",\n"
            <<"  \"accepted_steps\":"<<acceptedSteps<<",\n"
            <<"  \"simulated_time_s\":"<<acceptedSteps*kFrameDt<<",\n"
            <<"  \"simulation_rate_hz\":240,\n"
            <<"  \"drop_clearance_m\": "<<kDropHeight<<",\n"
            <<"  \"initial_tilt_deg\":12,\n"
            <<"  \"release_event\":\"loaded bag free-falls 0.25 m; Panda approaches actual landed corner, pinches, lifts 0.32 m, opens fingers and drops again\",\n"
            <<"  \"coin_count\":"<<kCoinCount<<",\n"
            <<"  \"coin_mass_kg_each\":"<<kCoinMass<<",\n"
            <<"  \"total_coin_mass_kg\":"<<kCoinCount*kCoinMass<<",\n"
            <<"  \"film_nodes\":"<<nodes.size()<<",\n"
            <<"  \"film_triangles\":"<<scene.triangles.size()<<",\n"
            <<"  \"degenerate_triangles_skipped\":"<<scene.skippedDegenerateTriangles<<",\n"
            <<"  \"membrane_constraints\":"<<edges.size()<<",\n"
            <<"  \"max_edge_stretch_ratio\":"<<maxStretch<<",\n"
            <<"  \"max_coins_with_film_contact\":"<<maxContacts<<",\n"
            <<"  \"contact_samples_per_coin\":"<<kContactSamples<<",\n"
            <<"  \"max_triangle_candidates_per_coin\":"<<maxContactCandidates<<",\n"
            <<"  \"contact_candidate_capacity_per_coin\":"<<contactCandidateCapacity<<",\n"
            <<"  \"full_triangle_contact_slots_per_coin\":"<<scene.triangles.size()<<",\n"
            <<"  \"full_triangle_closure_sweeps\":8,\n"
            <<"  \"film_table_transmitted_load_friction\":true,\n"
            <<"  \"ccd_model\":\"GJK support-plane lower-bound conservative advancement; linear triangle vertices, quaternion slerp cylinder; 32 iterations; conservative uncertain constraints retained\",\n"
            <<"  \"shell_self_contact_gap_m\":"<<kFilmGauge<<",\n"
            <<"  \"max_shell_self_contact_vertices\":"<<maxSelfContactVertices<<",\n"
            <<"  \"side_seam_contact_model\":\"shared welded edges; triangle film contact only; no geometric containment projection\",\n"
            <<"  \"max_coins_with_side_seam_contact\":"<<maxSideSeamContacts<<",\n"
            <<"  \"maximum_side_seam_preprojection_penetration_m\":"<<maxSideSeamPenetration<<",\n"
            <<"  \"maximum_side_seam_impulse_Ns\":"<<maxSideSeamImpulseNs<<",\n"
            <<"  \"maximum_coin_linear_speed_m_s\":"<<maxCoinLinearSpeed<<",\n"
            <<"  \"maximum_coin_angular_speed_rad_s\":"<<maxCoinAngularSpeed<<",\n"
            <<"  \"angular_speed_cap_rad_s\":"<<config.geometry.w<<",\n"
            <<"  \"containment_test\":\"coin-center odd/even ray crossings through current front/back triangle sheets; sampled finite-cylinder film clearance is separate\",\n"
            <<"  \"final_coins_inside_pouch\":"<<kCoinCount-escapedCoinsFinal<<",\n"
            <<"  \"final_escaped_coins\":"<<escapedCoinsFinal<<",\n"
            <<"  \"maximum_escaped_coins\":"<<maxEscapedCoins<<",\n"
            <<"  \"maximum_coins_exceeding_flat_panel_depth_proxy\":"<<maxProjectedDepthSpanExceeded<<",\n"
            <<"  \"minimum_inside_coins_after_0_5_s\":"<<minInsideAfterDrop<<",\n"
            <<"  \"max_shell_ground_penetration_m\":"<<std::max(0.0,maxShellGroundPenetration)<<",\n"
            <<"  \"max_coin_ground_penetration_m\":"<<std::max(0.0,maxCoinGroundPenetration)<<",\n"
            <<"  \"ground_contact_tolerance_m\":0.0001,\n"
            <<"  \"ground_contact_gate_passed\":"<<((maxShellGroundPenetration<=1.0e-4&&maxCoinGroundPenetration<=1.0e-4)?"true":"false")<<",\n"
            <<"  \"frame_dt_s\":"<<kFrameDt<<",\n"
            <<"  \"substeps_per_frame\":"<<kSubsteps<<",\n"
            <<"  \"constraint_iterations\":"<<kConstraintIterations<<",\n"
            <<"  \"maximum_adaptive_substeps_per_frame\":"<<maximumSubsteps<<",\n"
            <<"  \"energy_rejected_frame_attempts\":"<<energyRejectedFrames<<",\n"
            <<"  \"contact_accuracy_rejected_frame_attempts\":"<<contactRejectedFrames<<",\n"
            <<"  \"required_film_mid_surface_clearance_m\":"<<(.5*kFilmGauge-2.e-6)<<",\n"
            <<"  \"film_contact_slop_m\":0.000002,\n"
            <<"  \"geometry_error_control\":\"FP64 full triangle / cylinder distance bounds; retry output interval at smaller Metal substeps; no state clamping\",\n"
            <<"  \"initial_mechanical_energy_J\":"<<initialEnergy<<",\n"
            <<"  \"peak_accepted_mechanical_energy_J\":"<<peakAcceptedEnergy<<",\n"
            <<"  \"final_mechanical_energy_J\":"<<acceptedEnergy.total()<<",\n"
            <<"  \"final_native_kinetic_energy_J\":"<<acceptedEnergy.kinetic<<",\n"
            <<"  \"gpu_time_ms\":"<<gpuMilliseconds<<",\n"
            <<"  \"wall_time_s\":"<<elapsed<<",\n"
            <<"  \"film_mass_kg\":"<<std::accumulate(scene.mass.begin(),scene.mass.end(),0.0)<<",\n"
            <<"  \"film_strain_viscosity_time_s_assumption\":"<<kStrainViscosityTime<<",\n"
            <<"  \"film_bending_viscosity_time_s_assumption\":"<<kBendViscosityTime<<",\n"
            <<"  \"objective_bending_hinges\":"<<scene.hinges.size()<<",\n"
            <<"  \"film_thickness_m\":"<<kFilmGauge<<",\n"
            <<"  \"film_areal_density_kg_m2\":"<<kFilmDensity*kFilmGauge<<",\n"
            <<"  \"film_youngs_modulus_Pa\":"<<kYoungsModulus<<",\n"
            <<"  \"film_poisson_ratio_assumption\":"<<kPoissonRatio<<",\n"
            <<"  \"table_height_m\":"<<kGroundHeight<<",\n"
            <<"  \"coin_film_friction_coefficient_assumption\":"<<kCoinContactFriction<<",\n"
            <<"  \"reeded_coin_edge_friction_coefficient_assumption\":"
            <<static_cast<double>(PLASTIC_BAG_REEDED_EDGE_FRICTION)<<",\n"
            <<"  \"bag_construction\":\"continuous 75 um soft LDPE sheet folded at bottom; two welded side edges; open top; no separate floor panel\",\n"
            <<"  \"maximum_initial_mouth_gap_m\":"<<kBagDepth<<",\n"
            <<"  \"loaded_body_panel_separation_m\":"<<kBagBodyDepth<<",\n"
            <<"  \"status\":\""<<"full native trajectory; exact triangle coverage and conservative swept contact; material calibration and CCD qualification remain open"<<"\"\n}\n";
        summary.flush();
        std::cout<<"completed frames="<<acceptedSteps<<" simulated_s="<<acceptedSteps*kFrameDt
            <<" gpu_ms="<<gpuMilliseconds<<" wall_s="<<elapsed<<" output="<<output<<'\n';
    }
    return 0;
}

} // namespace

int main(int argc,char** argv){
  @autoreleasepool {
    id activity=[[NSProcessInfo processInfo] beginActivityWithOptions:(NSActivityUserInitiated|NSActivityLatencyCritical) reason:@"Requested native Numi bag simulation"];
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
    int status=0;
    try{status=run(argc,argv);}catch(const std::exception& error){
        std::cerr<<"plastic bag scene failed: "<<error.what()<<'\n';
        status=1;
    }
    [[NSProcessInfo processInfo] endActivity:activity];
    return status;
  }
}
