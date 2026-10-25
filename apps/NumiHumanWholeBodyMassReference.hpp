#pragma once
#include <metalrobo/EngineModel.hpp>
#include <metalrobo/NumiHumanTissueMass.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace metalrobo::human {
struct WholeBodyMassReference {
    EngineModel model;
    std::vector<NumiHumanTissueMassNode> nodes;
    std::vector<NumiHumanTissueMassContribution> contributions;
    std::vector<std::array<double, 3>> offsets;
    double originalMassKg = 0.0;
    double tissueMassKg = 0.0;
    double residualScale = 0.0;
    double maximumRelativeInertiaCorrection = 0.0;
    std::string error;
    [[nodiscard]] bool succeeded() const noexcept { return error.empty(); }
};

namespace whole_body_mass_reference_detail {
using M = std::array<double, 9>;
using V = std::array<double, 3>;
inline M inertia(const MRBodyPropertiesGPU& b) {
    return {b.inertiaRow0.x,b.inertiaRow0.y,b.inertiaRow0.z,
            b.inertiaRow1.x,b.inertiaRow1.y,b.inertiaRow1.z,
            b.inertiaRow2.x,b.inertiaRow2.y,b.inertiaRow2.z};
}
inline M second(const M& i) {
    M s{}; const double h=.5*(i[0]+i[4]+i[8]);
    for(std::size_t k=0;k<9;++k)s[k]=(k%4==0?h:0.0)-i[k];
    return s;
}
inline M toInertia(const M& s) {
    M i{}; const double t=s[0]+s[4]+s[8];
    for(std::size_t k=0;k<9;++k)i[k]=(k%4==0?t:0.0)-s[k];
    return i;
}
inline bool finite(const M& a) {
    return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});
}
inline bool eigen(const M& in,V& e,M& q) {
    if(!finite(in))return false;
    M a{};
    for(std::size_t r=0;r<3;++r)for(std::size_t c=0;c<3;++c){
        const double x=in[3*r+c],y=in[3*c+r];
        if(std::abs(x-y)>64*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(x),std::abs(y)}))return false;
        a[3*r+c]=.5*(x+y);
    }
    q={1,0,0,0,1,0,0,0,1};
    for(unsigned it=0;it<48;++it){
        std::size_t p=0,r=1; double z=std::abs(a[1]);
        if(std::abs(a[2])>z){p=0;r=2;z=std::abs(a[2]);}
        if(std::abs(a[5])>z){p=1;r=2;z=std::abs(a[5]);}
        const double scale=std::max({std::abs(a[0]),std::abs(a[4]),std::abs(a[8]),1e-300});
        if(z<=8*std::numeric_limits<double>::epsilon()*scale)break;
        const double apq=a[3*p+r]; if(apq==0)continue;
        const double tau=(a[3*r+r]-a[3*p+p])/(2*apq);
        const double t=std::copysign(1.0,tau)/(std::abs(tau)+std::sqrt(1+tau*tau));
        const double c=1/std::sqrt(1+t*t),s=t*c,app=a[3*p+p],arr=a[3*r+r];
        a[3*p+p]=app-t*apq; a[3*r+r]=arr+t*apq; a[3*p+r]=a[3*r+p]=0;
        for(std::size_t k=0;k<3;++k)if(k!=p&&k!=r){
            const double kp=a[3*k+p],kr=a[3*k+r];
            a[3*k+p]=a[3*p+k]=c*kp-s*kr;
            a[3*k+r]=a[3*r+k]=s*kp+c*kr;
        }
        for(std::size_t k=0;k<3;++k){
            const double kp=q[3*k+p],kr=q[3*k+r];
            q[3*k+p]=c*kp-s*kr; q[3*k+r]=s*kp+c*kr;
        }
    }
    e={a[0],a[4],a[8]};
    return std::all_of(e.begin(),e.end(),[](double x){return std::isfinite(x);});
}
inline M reconstruct(const V& e,const M& q){
    M a{};for(std::size_t r=0;r<3;++r)for(std::size_t c=0;c<3;++c)
        for(std::size_t k=0;k<3;++k)a[3*r+c]+=q[3*r+k]*e[k]*q[3*c+k];
    return a;
}
inline bool floorSource(const M& src,M& dst,double& correction,std::string& err){
    V e{};M q{};if(!eigen(src,e,q)){err="source central second moment is non-finite or asymmetric";return false;}
    const double max=*std::max_element(e.begin(),e.end());
    if(!(max>0)&&std::isfinite(max)){err="source central second moment has no positive eigenvalue";return false;}
    if(!std::isfinite(max)){err="source central second moment is non-finite";return false;}
    constexpr double kUncertainty=64.0*static_cast<double>(std::numeric_limits<float>::epsilon());
    const double floor=kUncertainty*max;
    for(double& x:e){
        if(x < -floor){err="source central second moment has a materially negative eigenvalue";return false;}
        if(x<floor){correction=std::max(correction,(floor-x)/max);x=floor;}
    }
    dst=reconstruct(e,q);if(!finite(dst)){err="floored source moment is non-finite";return false;}
    return true;
}
inline bool pd(const M& a){
    V e{};M q{};if(!eigen(a,e,q))return false;
    const double hi=*std::max_element(e.begin(),e.end()),lo=*std::min_element(e.begin(),e.end());
    return hi>0&&lo>64*std::numeric_limits<double>::epsilon()*hi;
}
inline M add(const M&a,const M&b){M c{};for(size_t i=0;i<9;++i)c[i]=a[i]+b[i];return c;}
inline M sub(const M&a,const M&b){M c{};for(size_t i=0;i<9;++i)c[i]=a[i]-b[i];return c;}
inline M mul(const M&a,double s){M c{};for(size_t i=0;i<9;++i)c[i]=a[i]*s;return c;}
inline M outer(const V&a,const V&b){M c{};for(size_t i=0;i<3;++i)for(size_t j=0;j<3;++j)c[3*i+j]=a[i]*b[j];return c;}
inline mr_float4 row(const M&a,size_t i){return {float(a[3*i]),float(a[3*i+1]),float(a[3*i+2]),0};}
inline M inverse(const M&a){
    M b{a[4]*a[8]-a[5]*a[7],a[2]*a[7]-a[1]*a[8],a[1]*a[5]-a[2]*a[4],
        a[5]*a[6]-a[3]*a[8],a[0]*a[8]-a[2]*a[6],a[2]*a[3]-a[0]*a[5],
        a[3]*a[7]-a[4]*a[6],a[1]*a[6]-a[0]*a[7],a[0]*a[4]-a[1]*a[3]};
    const double d=a[0]*b[0]+a[1]*b[3]+a[2]*b[6];
    for(double&x:b)x/=d;return b;
}
inline bool packBody(MRBodyPropertiesGPU& b,double mass,const M&s,std::string&err){
    if(!(mass>0)||!std::isfinite(mass)||!pd(s)){err="composed reference mass moment is not positive definite";return false;}
    const M i=toInertia(s);if(!pd(i)){err="composed reference inertia is not positive definite";return false;}
    b.massAndInverseMass={float(mass),float(1/mass),0,0};
    b.inertiaRow0=row(i,0);b.inertiaRow1=row(i,1);b.inertiaRow2=row(i,2);
    const M ip=inertia(b),iv=inverse(ip);
    if(!finite(ip)||!pd(ip)||!finite(iv)||!std::all_of(iv.begin(),iv.end(),[](double x){return std::isfinite(float(x));})||
       !std::isfinite(b.massAndInverseMass.x)||!std::isfinite(b.massAndInverseMass.y)||b.massAndInverseMass.x<=0||b.massAndInverseMass.y<=0){
        err="composed reference inertia is not representable in the engine ABI";return false;
    }
    b.inverseInertiaRow0=row(iv,0);b.inverseInertiaRow1=row(iv,1);b.inverseInertiaRow2=row(iv,2);return true;
}
} // namespace whole_body_mass_reference_detail

[[nodiscard]] inline WholeBodyMassReference prepareWholeBodyMassReference(
    const EngineModel& source,std::span<const NumiHumanTissueMassNode> nodes) {
    using namespace whole_body_mass_reference_detail;
    WholeBodyMassReference out;
    auto fail=[&](std::string why){out.error=std::move(why);return out;};
    std::string why;if(!source.valid(&why))return fail("invalid source EngineModel: "+why);
    if(nodes.empty())return fail("whole-body mass reference requires cooked FEM nodes");
    if(!source.constraintProgram.empty())return fail("spatial ConstraintIR must be recooked after whole-body mass reference construction");

    double originalMass=0,tissueMass=0;
    std::vector<M> floored(source.bodies.size());
    std::set<std::uint32_t> seen;
    std::map<std::uint32_t,NumiHumanTissueMassPartition> donor;
    for(std::size_t i=0;i<source.bodies.size();++i){
        const auto& b=source.bodies[i];if(b.motionType!=MR_MOTION_DYNAMIC)continue;
        const double m=b.massAndInverseMass.x;if(!(m>0)||!std::isfinite(m))return fail("dynamic source body has invalid mass");
        originalMass+=m;M s{};if(!floorSource(second(inertia(b)),s,out.maximumRelativeInertiaCorrection,why)){
            out.error="body "+std::to_string(i)+": "+why;return out;
        }floored[i]=s;
    }
    if(!(originalMass>0)||!std::isfinite(originalMass))return fail("source EngineModel has no positive dynamic mass");
    for(const auto& n:nodes){
        if(n.nodeIndex==MR_INVALID_INDEX||!seen.insert(n.nodeIndex).second)return fail("duplicate or invalid cooked FEM node ownership");
        if(n.donorBody>=source.bodies.size()||!(n.massKg>0)||!std::isfinite(n.massKg)||
           !std::all_of(n.localPosition.begin(),n.localPosition.end(),[](double x){return std::isfinite(x);}))
            return fail("invalid cooked FEM mass node");
        if(source.bodies[n.donorBody].motionType!=MR_MOTION_DYNAMIC)return fail("cooked FEM donor is not dynamic");
        auto&p=donor[n.donorBody];p.donorBody=n.donorBody;
        if(p.nodeCount==std::numeric_limits<std::uint32_t>::max())return fail("cooked FEM node count exceeds ABI");
        ++p.nodeCount;p.tissueMassKg+=n.massKg;tissueMass+=n.massKg;
        for(size_t r=0;r<3;++r){p.tissueFirstMomentKgM[r]+=n.massKg*n.localPosition[r];
            for(size_t c=0;c<3;++c)p.tissueSecondMomentKgM2[3*r+c]+=n.massKg*n.localPosition[r]*n.localPosition[c];}
    }
    if(!std::isfinite(tissueMass)||!(tissueMass<originalMass))return fail("cooked FEM mass must be finite and less than total dynamic rigid mass");
    const double scale=(originalMass-tissueMass)/originalMass;
    if(!(scale>0)||!std::isfinite(scale))return fail("global residual rigid mass scale is not positive");

    std::vector<NumiHumanTissueMassPartition> parts;parts.reserve(source.bodies.size());
    out.offsets.assign(source.bodies.size(),{});
    for(size_t i=0;i<source.bodies.size();++i){
        const auto& orig=source.bodies[i];if(orig.motionType!=MR_MOTION_DYNAMIC)continue;
        NumiHumanTissueMassPartition p{};auto it=donor.find(std::uint32_t(i));if(it!=donor.end())p=it->second;
        p.donorBody=std::uint32_t(i);p.sourceBody=orig;
        const double residual=scale*orig.massAndInverseMass.x;
        const double composed=residual+p.tissueMassKg;
        V c{};if(p.tissueMassKg>0)for(size_t k=0;k<3;++k)c[k]=p.tissueFirstMomentKgM[k]/composed;
        M s=add(mul(floored[i],scale),M{});
        s=add(s,p.tissueSecondMomentKgM2);
        s=sub(s,mul(outer(c,c),composed));
        if(!finite(s)||!pd(s))return fail("body "+std::to_string(i)+": composed rigid/FEM reference moment is not positive definite");
        p.remainingCOMOffsetM=c;p.remainingBody=orig;
        p.remainingBody.centerOfMass.x=float(double(orig.centerOfMass.x)+c[0]);
        p.remainingBody.centerOfMass.y=float(double(orig.centerOfMass.y)+c[1]);
        p.remainingBody.centerOfMass.z=float(double(orig.centerOfMass.z)+c[2]);
        if(!std::isfinite(p.remainingBody.centerOfMass.x)||!std::isfinite(p.remainingBody.centerOfMass.y)||!std::isfinite(p.remainingBody.centerOfMass.z)||
           !packBody(p.remainingBody,composed,s,why)){
            out.error="body "+std::to_string(i)+": "+(why.empty()?std::string("composed COM is not representable"):why);return out;
        }
        parts.push_back(p);out.offsets[i]=c;
    }
    if(!rebaseNumiHumanTissueMassPartition(source,parts,out.model,why))
        return fail("could not construct whole-body reference EngineModel: "+why);
    out.nodes.assign(nodes.begin(),nodes.end());
    for(auto& n:out.nodes)for(size_t k=0;k<3;++k)n.localPosition[k]-=out.offsets[n.donorBody][k];
    out.originalMassKg=originalMass;out.tissueMassKg=tissueMass;out.residualScale=scale;
    return out;
}

// Explicit source-mass path. The first span remains one row per actual FEM node
// and owns its registered frame/anchor; the second span carries source-body
// moments independently, including multiple source bodies for one FEM node.
[[nodiscard]] inline WholeBodyMassReference prepareWholeBodyMassReference(
    const EngineModel& source,
    std::span<const NumiHumanTissueMassNode> nodeFrames,
    std::span<const NumiHumanTissueMassContribution> contributions) {
    WholeBodyMassReference out;
    const auto fail=[&](std::string why){out.error=std::move(why);return out;};
    if(nodeFrames.empty()||contributions.empty())
        return fail("explicit whole-body reference requires node frames and source contributions");
    std::map<std::uint32_t,double> massByNode;
    for(const auto& node:nodeFrames){
        if(node.nodeIndex==MR_INVALID_INDEX||node.donorBody>=source.bodies.size()||
           !std::isfinite(node.massKg)||node.massKg<=0.0||
           !std::all_of(node.localPosition.begin(),node.localPosition.end(),[](double x){return std::isfinite(x);})||
           !massByNode.emplace(node.nodeIndex,node.massKg).second)
            return fail("explicit whole-body reference has an invalid or duplicate node/frame row");
    }
    std::map<std::uint32_t,double> attributed;
    std::vector<NumiHumanTissueMassNode> sourceRows;sourceRows.reserve(contributions.size());
    for(std::size_t i=0;i<contributions.size();++i){
        const auto& c=contributions[i];
        if(!massByNode.contains(c.nodeIndex)||c.donorBody>=source.bodies.size()||
           !std::isfinite(c.massKg)||c.massKg<=0.0||
           !std::all_of(c.localPosition.begin(),c.localPosition.end(),[](double x){return std::isfinite(x);}))
            return fail("explicit whole-body reference has an invalid source contribution");
        if(i>=std::numeric_limits<std::uint32_t>::max())
            return fail("explicit whole-body source-contribution count exceeds index range");
        attributed[c.nodeIndex]+=c.massKg;
        sourceRows.push_back({static_cast<std::uint32_t>(i),c.donorBody,c.massKg,c.localPosition});
    }
    const auto close=[](double a,double b){
        return std::isfinite(a)&&std::isfinite(b)&&
            std::abs(a-b)<=1.0e-9*std::max({std::abs(a),std::abs(b),1.0e-12});
    };
    for(const auto& [node,mass]:massByNode){
        const auto it=attributed.find(node);
        if(it==attributed.end()||!close(mass,it->second))
            return fail("explicit whole-body source contributions do not close a FEM node mass");
    }
    if(attributed.size()!=massByNode.size())
        return fail("explicit whole-body source contributions contain an unknown FEM node");
    out=prepareWholeBodyMassReference(source,sourceRows);
    if(!out.succeeded())return out;
    out.nodes.assign(nodeFrames.begin(),nodeFrames.end());
    out.contributions.assign(contributions.begin(),contributions.end());
    for(auto& node:out.nodes)for(size_t axis=0;axis<3;++axis)
        node.localPosition[axis]-=out.offsets[node.donorBody][axis];
    for(auto& c:out.contributions)for(size_t axis=0;axis<3;++axis)
        c.localPosition[axis]-=out.offsets[c.donorBody][axis];
    return out;
}
} // namespace metalrobo::human
