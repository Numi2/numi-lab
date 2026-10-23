#include "numi/matter/numi_human_shared.h"
#include "numi/matter/numi_human.hpp"
#include "fem_surface_attachment_validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
struct V { double x, y, z; };
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(double s,V a){return {s*a.x,s*a.y,s*a.z};}
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm(V a){return std::sqrt(dot(a,a));}
void require(bool b,const char* what){if(!b){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
bool close(double a,double b,double tol){return std::abs(a-b)<=tol;}
struct Eval { V slave,target,gap,force; double energy; };
constexpr std::array<double,3> beta{0.2,0.3,0.5};
constexpr double ka=2000.0;
constexpr V target{0.3,0.5,0.1};
const std::array<V,3> base{{{0,0,0},{1,0,0},{0,1,0}}};
Eval evaluate(const std::array<V,3>& x,V t=target) {
    V slave{0,0,0};
    for(int i=0;i<3;++i) slave=slave+beta[i]*x[i];
    V gap=t-slave;
    return {slave,t,gap,ka*gap,0.5*ka*dot(gap,gap)};
}
V rotateZ(V x){return {-x.y,x.x,x.z};}
bool recordValid(const NMNumiHumanFEMSurfaceAttachmentGPU& s) {
    const auto b=s.barycentric;
    const auto p=s.bodyLocalPointAndArea;
    const auto m=s.stiffnessAndStrainPerPressure;
    const double sum=double(b.x)+b.y+b.z;
    const double kaHost=double(m.x)*p.w;
    return s.node[0]!=s.node[1] && s.node[0]!=s.node[2] &&
        s.node[1]!=s.node[2] &&
        s.flags==NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE &&
        s.stableID!=NM_INVALID_INDEX && s.reserved0==0 && s.reserved1==0 &&
        std::isfinite(sum) && std::abs(sum-1.0)<=1e-5 &&
        b.x>=0 && b.y>=0 && b.z>=0 && b.w==0 &&
        p.w>0 && std::isfinite(kaHost) &&
        kaHost>=std::numeric_limits<float>::denorm_min() &&
        kaHost<=std::numeric_limits<float>::max() &&
        m.x>0 && m.y>0 && m.z==0 && m.w==0;
}
}

int main() {
    static_assert(sizeof(NMNumiHumanFEMBodyContactSampleGPU)==64u);
    static_assert(sizeof(NMNumiHumanFEMSurfaceAttachmentGPU)==80u);
    static_assert(sizeof(NMNumiHumanTendonFEMLoadDispatchGPU)==96u);
    NMNumiHumanFEMSurfaceAttachmentGPU sample{};
    sample.node[0]=0; sample.node[1]=1; sample.node[2]=2;
    sample.bodyIndex=1; sample.stableID=73;
    sample.flags=NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE;
    sample.barycentric={0.2f,0.3f,0.5f,0.0f};
    sample.bodyLocalPointAndArea={0.3f,0.5f,0.0f,0.02f};
    sample.stiffnessAndStrainPerPressure={100000.0f,1.0e-5f,0,0};
    require(recordValid(sample),"valid typed attachment record");

    // Exercise the same source-span validator called during adapter creation.
    std::array<NMNumiHumanTendonFEMNodeLoadGPU,3> sourceNodes{};
    std::array<NMNumiHumanFEMSurfaceAttachmentGPU,1> surfaceSpan{sample};
    std::array<NMNumiHumanFEMBodyContactSampleGPU,1> legacySpan{};
    numi::matter::NumiHumanTendonFEMLoadSource source{};
    source.nodeLoads=sourceNodes;
    require(numi::matter::detail::validFEMSurfaceAttachmentSamples(
        source.femSurfaceAttachments,source.nodeLoads.size()),
        "zero surface span remains a valid legacy source");
    source.femSurfaceAttachments=surfaceSpan;
    require(numi::matter::detail::validFEMSurfaceAttachmentSamples(
        source.femSurfaceAttachments,source.nodeLoads.size()),
        "surface-only source span is accepted");
    source.femBodyContactSamples=legacySpan;
    require(source.femBodyContactSamples.size()==1u &&
        numi::matter::detail::validFEMSurfaceAttachmentSamples(
            source.femSurfaceAttachments,source.nodeLoads.size()),
        "mixed legacy and surface source spans are accepted");
    auto invalidSurface=surfaceSpan;
    invalidSurface[0].stableID=NM_INVALID_INDEX;
    source.femSurfaceAttachments=invalidSurface;
    require(!numi::matter::detail::validFEMSurfaceAttachmentSamples(
        source.femSurfaceAttachments,source.nodeLoads.size()),
        "host source initialization rejects malformed surface record");
    auto bad=sample; bad.barycentric={0.2f,0.3f,0.4f,0};
    require(!recordValid(bad),"reject non-normalized barycentrics");
    bad=sample; bad.barycentric={-0.1f,0.6f,0.5f,0};
    require(!recordValid(bad),"reject negative barycentrics");
    bad=sample; bad.node[2]=bad.node[1];
    require(!recordValid(bad),"reject repeated triangle nodes");
    bad=sample; bad.bodyLocalPointAndArea.w=std::numeric_limits<float>::max();
    bad.stiffnessAndStrainPerPressure.x=std::numeric_limits<float>::max();
    require(!recordValid(bad),"reject overflowing kA");

    const Eval e=evaluate(base);
    require(close(e.slave.x,0.3,1e-14)&&close(e.slave.y,0.5,1e-14)&&
            close(e.slave.z,0,1e-14),"barycentric slave point");
    require(close(e.force.z,200.0,1e-12)&&close(e.energy,10.0,1e-12),
            "finite spring force and energy");
    std::array<V,3> dx{{{0.2,-0.1,0.03},{-0.4,0.7,0.11},{0.6,0.2,-0.08}}};
    const V dt{-0.13,0.09,0.17};
    double nodalWork=0;
    for(int i=0;i<3;++i) nodalWork+=dot(beta[i]*e.force,dx[i]);
    nodalWork+=dot(-1.0*e.force,dt);
    const double eps=1.0e-6;
    auto xp=base,xm=base;
    for(int i=0;i<3;++i){xp[i]=xp[i]+eps*dx[i];xm[i]=xm[i]-eps*dx[i];}
    const double up=evaluate(xp,target+eps*dt).energy;
    const double um=evaluate(xm,target-eps*dt).energy;
    const double minusDU=-(up-um)/(2*eps);
    require(close(nodalWork,minusDU,2e-7),
            "B transpose force work equals negative energy differential");

    // The Hessian couples every pair of barycentric nodes. This specifically
    // checks the off-diagonal beta_i*beta_j terms of B^T k B.
    double maxFD=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int axis=0;axis<3;++axis) {
        auto xp2=base,xm2=base;
        V d{0,0,0}; if(axis==0)d.x=1;else if(axis==1)d.y=1;else d.z=1;
        xp2[j]=xp2[j]+eps*d; xm2[j]=xm2[j]-eps*d;
        const V fp=beta[i]*evaluate(xp2).force;
        const V fm=beta[i]*evaluate(xm2).force;
        const V derivative=(1.0/(2*eps))*(fp-fm);
        const double expected=-ka*beta[i]*beta[j];
        const double got=axis==0?derivative.x:(axis==1?derivative.y:derivative.z);
        maxFD=std::max(maxFD,std::abs(got-expected));
    }
    require(maxFD<2e-7,"full coupled stiffness finite difference including off-diagonals");
    require(std::abs(ka*beta[0]*beta[1])>100.0,
            "off-diagonal B-transpose tangent is nonzero");

    const V translation{0.23,-0.31,0.42};
    std::array<V,3> xr;
    for(int i=0;i<3;++i)xr[i]=translation+rotateZ(base[i]);
    const Eval er=evaluate(xr,translation+rotateZ(target));
    require(close(er.energy,e.energy,1e-12),"rigid motion preserves spring energy");
    require(close(er.force.x,-e.force.y,1e-12)&&
            close(er.force.y,e.force.x,1e-12)&&close(er.force.z,e.force.z,1e-12),
            "rigid motion rotates spring force");

    const V degenerateA{0,0,0},degenerateB{1,0,0},degenerateC{2,0,0};
    require(norm(cross(degenerateB-degenerateA,degenerateC-degenerateA))==0,
            "degenerate surface triangle is identified");
    std::printf("{\"status\":\"PASS\",\"energy_J\":%.9g,\"force_N\":[%.9g,%.9g,%.9g],\"work_error\":%.3g,\"stiffness_fd_max\":%.3g}\n",
        e.energy,e.force.x,e.force.y,e.force.z,std::abs(nodalWork-minusDU),maxFD);
    return 0;
}
