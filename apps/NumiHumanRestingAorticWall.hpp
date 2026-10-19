#pragma once
// Reduced pressure-coupled abdominal-aortic wall source.
//
// Geometry source: NHA stable ID 9, source body 7, source payload SHA
// 1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108.
// The 87-edge section is the BodyParts3D EXTERNAL SURFACE section, not a
// patent lumen. Its inward contour and wall thickness below are explicit
// population-level proxies. This creates a closed, capped lumped reservoir
// segment: CVSim remains the sole blood-volume/flow owner; the caps are
// pressure-transfer boundaries, not spatial blood ports or a patent lumen.
// Its reference coordinates are derived from an axisymmetric finite-Lame
// thick-wall equilibrium at the admitted initial compartment pressure. The
// source geometry remains the initial current mesh; axial stretch and isotropic
// material parameters are explicit inference, not patient measurements.
#include "numi/matter/matter.hpp"
#include "metalrobo/NumiHumanTissueMass.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace numi::human::aortic_wall {
using Vec = std::array<double,3>;
using Face = std::array<std::uint32_t,3>;
using Tet = std::array<std::uint32_t,4>;
using namespace numi::matter;

inline Vec add(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]+=b[i];return a;}
inline Vec sub(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]-=b[i];return a;}
inline Vec mul(Vec a,double s){for(double& x:a)x*=s;return a;}
inline double dot(const Vec&a,const Vec&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec cross(const Vec&a,const Vec&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double norm(const Vec&a){return std::sqrt(dot(a,a));}
inline Vec unit(Vec a){const double n=norm(a);if(!(n>1e-12))throw std::runtime_error("aortic wall axis is degenerate");return mul(a,1.0/n);}
inline Vec rotate(const mr_float4 q,const Vec& p){
    const Vec u{q.x,q.y,q.z};
    const Vec uv=cross(u,p),uuv=cross(u,uv);
    return add(p,add(mul(uv,2.0*q.w),mul(uuv,2.0)));
}
inline double signedTet6(const Vec&a,const Vec&b,const Vec&c,const Vec&d){
    return dot(sub(b,a),cross(sub(c,a),sub(d,a)));
}
inline void appendTet(ObjectSource& o,Tet t){
    const double d=signedTet6(o.femNodes[t[0]],o.femNodes[t[1]],o.femNodes[t[2]],o.femNodes[t[3]]);
    if(!std::isfinite(d)||std::abs(d)<1e-15)throw std::runtime_error("aortic wall tetrahedron is degenerate");
    if(d<0)std::swap(t[1],t[2]);
    o.tetrahedra.push_back({t});
}
inline std::vector<Vec> resampleClosedContour(std::span<const Vec> p,std::size_t count){
    if(p.size()<12||count<12)throw std::runtime_error("aortic section contour is too small");
    std::vector<double> arc(p.size()+1,0.0);
    for(std::size_t i=0;i<p.size();++i)arc[i+1]=arc[i]+norm(sub(p[(i+1)%p.size()],p[i]));
    const double perimeter=arc.back();
    if(!(perimeter>1e-5)||!std::isfinite(perimeter))throw std::runtime_error("aortic section perimeter is invalid");
    std::vector<Vec> r;r.reserve(count);
    for(std::size_t j=0;j<count;++j){
        const double s=perimeter*double(j)/double(count);
        const auto hi=std::upper_bound(arc.begin(),arc.end(),s);
        const std::size_t i=std::min<std::size_t>(p.size()-1,std::size_t(hi-arc.begin()-1));
        const double f=(s-arc[i])/(arc[i+1]-arc[i]);
        r.push_back(add(p[i],mul(sub(p[(i+1)%p.size()],p[i]),f)));
    }
    return r;
}
inline double polygonArea(std::span<const Vec> p,const Vec& axis){
    Vec twice{};
    for(std::size_t i=0;i<p.size();++i)twice=add(twice,cross(p[i],p[(i+1)%p.size()]));
    return 0.5*std::abs(dot(twice,axis));
}
inline double cavityVolume(const std::vector<Vec>& x,std::span<const Face> faces){
    if(faces.empty())return 0.0;
    const Vec origin=x[faces.front()[0]];
    double v=0.0;
    for(const Face& f:faces){
        const Vec a=sub(x[f[0]],origin),b=sub(x[f[1]],origin),c=sub(x[f[2]],origin);
        v+=dot(a,cross(b,c))/6.0;
    }
    return v;
}
struct SectionProof {
    std::uint32_t stableId=0;
    std::uint32_t sourceBody=0;
    std::string sourcePayloadSHA256;
    Vec planePoint{};
    Vec planeNormal{0,1,0};
    std::vector<Vec> orderedExternalContour;
    double externalAreaM2=0.0;
};
struct Config {
    // Labrosse et al. (J Biomech 2009, PMID 19345356) reported median
    // unpressurized abdominal wall thickness 1.2 mm and median circumferential
    // stretch 1.26 at 13.3 kPa in eight male sexagenarians. This is a
    // population reference, not a measurement of this anatomy.
    double wallThicknessM=0.0012;
    double densityKgM3=1066.0; // fresh normal human aorta material reference
    double shearModulusPa=130000.0; // inferred homogeneous isotropic reference parameter
    double bulkPenaltyPa=1300000.0; // first Lamé parameter lambda; K=lambda+2*mu/3
    // Finite compressibility is retained in the loaded-reference radial solve.
    std::uint32_t circumferentialSegments=24; // coarse area-preserving section proxy
    std::uint32_t axialSegments=8;
    // Opt-in exterior-only contact on the outer lateral annulus. Cap and lumen
    // surfaces remain excluded; the default preserves the contact-free wall.
    bool outerLateralContact=false;
    std::uint32_t attachmentStableBase=910000u;
    // Explicit inferred axial stretch used by the cylindrical loaded-reference
    // solve. Unity is a reference assumption, not a measured axial stretch.
    double inferredAxialStretch=1.0;
    std::uint32_t equilibriumIntegrationSegments=2048u;
};
struct LoadedReferenceEquilibrium {
    double currentInnerRadiusM=0.0;
    double currentOuterRadiusM=0.0;
    double referenceInnerRadiusM=0.0;
    double referenceOuterRadiusM=0.0;
    double innerPressurePa=0.0;
    double outerPressurePa=0.0;
    double axialStretch=1.0;
    double innerBoundaryResidualPa=0.0;
    double outerBoundaryResidualPa=0.0;
    double maxRadialEquilibriumResidualPaPerM=0.0;
    double minimumJacobian=0.0;
    double maximumJacobian=0.0;
    double incompressibleReferenceInnerRadiusM=0.0;
    double incompressibleInnerHoopStretch=0.0;
    double finiteKInnerHoopStretch=0.0;
    double lameLambdaPa=0.0;
    double shearModulusPa=0.0;
};
struct Built {
    ObjectSource object;
    VascularCavitySource cavity;
    std::vector<Vec> donorCOMLocalNodes;
    std::vector<Vec> attachmentCOMLocalNodes;
    double sourceExternalAreaM2=0.0;
    double inferredLumenAreaM2=0.0;
    double segmentLengthM=0.0;
    double wallMassKg=0.0;
    std::uint32_t attachmentBody=MR_INVALID_INDEX;
    std::uint32_t donorBody=MR_INVALID_INDEX;
    std::uint32_t objectIndex=MR_INVALID_INDEX;
    std::uint32_t circumferentialSegments=0;
    std::uint32_t axialSegments=0;
    LoadedReferenceEquilibrium loadedReference;
};
inline double neoHookeanCauchyPrincipalStress(
    double stretch,double jacobian,double mu,double lameLambda){
    if(!(stretch>0.0&&jacobian>0.0&&mu>0.0&&lameLambda>0.0))
        throw std::runtime_error("loaded-reference constitutive state is not positive");
    return (mu*(stretch*stretch-1.0)+lameLambda*std::log(jacobian))/jacobian;
}
inline double radialStretchForCauchyStress(
    double radialStress,double currentRadius,double referenceRadius,
    double axialStretch,double mu,double lameLambda){
    const double hoopStretch=currentRadius/referenceRadius;
    const double hoopAxial=hoopStretch*axialStretch;
    auto residual=[&](double radialStretch){
        const double jacobian=radialStretch*hoopAxial;
        return neoHookeanCauchyPrincipalStress(radialStretch,jacobian,mu,lameLambda)-radialStress;
    };
    double lo=0.05,hi=8.0;
    const double flo0=residual(lo),fhi=residual(hi);
    double flo=flo0;
    if(!std::isfinite(flo0)||!std::isfinite(fhi)||flo0*fhi>0.0)
        throw std::runtime_error("finite-Lame radial stress could not be bracketed");
    for(unsigned i=0;i<72u;++i){
        const double mid=0.5*(lo+hi);
        const double fm=residual(mid);
        if(!std::isfinite(fm))throw std::runtime_error("finite-Lame radial stress became nonfinite");
        if((flo<0.0)==(fm<0.0)){lo=mid;flo=fm;}else{hi=mid;}
    }
    return 0.5*(lo+hi);
}
struct RadialEquilibriumSample { double radius=0.0,referenceRadius=0.0,radialStress=0.0; };
struct RadialEquilibriumIntegration {
    std::vector<RadialEquilibriumSample> samples;
    double outerResidualPa=0.0,minimumJacobian=0.0,maximumJacobian=0.0;
    double maximumBalanceResidualPaPerM=0.0;
};
inline RadialEquilibriumIntegration integrateFiniteLaméCylinder(
    double currentInnerRadius,double currentOuterRadius,double referenceInnerRadius,
    double innerPressure,double outerPressure,double mu,double lameLambda,
    double axialStretch,std::uint32_t segments){
    if(!(currentInnerRadius>0&&currentOuterRadius>currentInnerRadius&&
         referenceInnerRadius>0&&innerPressure>=outerPressure&&outerPressure>=0&&
         mu>0&&lameLambda>0&&axialStretch>0&&segments>=32u))
        throw std::runtime_error("finite-Lame cylinder inputs are outside the admitted domain");
    struct State { double referenceRadius=0.0,radialStress=0.0; };
    auto derivative=[&](double r,const State& y){
        if(!(r>0&&y.referenceRadius>0))throw std::runtime_error("cylindrical radial integration left its positive domain");
        const double lambdaR=radialStretchForCauchyStress(
            y.radialStress,r,y.referenceRadius,axialStretch,mu,lameLambda);
        const double lambdaTheta=r/y.referenceRadius;
        const double J=lambdaR*lambdaTheta*axialStretch;
        const double sigmaTheta=neoHookeanCauchyPrincipalStress(
            lambdaTheta,J,mu,lameLambda);
        return State{1.0/lambdaR,(sigmaTheta-y.radialStress)/r};
    };
    const double h=(currentOuterRadius-currentInnerRadius)/double(segments);
    RadialEquilibriumIntegration result;
    result.samples.reserve(std::size_t(segments)+1u);
    State y{referenceInnerRadius,-innerPressure};
    result.samples.push_back({currentInnerRadius,y.referenceRadius,y.radialStress});
    result.minimumJacobian=std::numeric_limits<double>::infinity();
    result.maximumJacobian=0.0;
    for(std::uint32_t i=0;i<segments;++i){
        const double r=currentInnerRadius+double(i)*h;
        const State k1=derivative(r,y);
        const State y2{y.referenceRadius+0.5*h*k1.referenceRadius,
                       y.radialStress+0.5*h*k1.radialStress};
        const State k2=derivative(r+0.5*h,y2);
        const State y3{y.referenceRadius+0.5*h*k2.referenceRadius,
                       y.radialStress+0.5*h*k2.radialStress};
        const State k3=derivative(r+0.5*h,y3);
        const State y4{y.referenceRadius+h*k3.referenceRadius,
                       y.radialStress+h*k3.radialStress};
        const State k4=derivative(r+h,y4);
        y.referenceRadius+=h*(k1.referenceRadius+2*k2.referenceRadius+
            2*k3.referenceRadius+k4.referenceRadius)/6.0;
        y.radialStress+=h*(k1.radialStress+2*k2.radialStress+
            2*k3.radialStress+k4.radialStress)/6.0;
        if(!(y.referenceRadius>0)||!std::isfinite(y.referenceRadius)||
           !std::isfinite(y.radialStress))
            throw std::runtime_error("finite-Lame cylinder integration became nonfinite");
        result.samples.push_back({currentInnerRadius+double(i+1u)*h,
                                  y.referenceRadius,y.radialStress});
    }
    result.outerResidualPa=y.radialStress+outerPressure;
    for(const auto& sample:result.samples){
        const double lambdaTheta=sample.radius/sample.referenceRadius;
        const double lambdaR=radialStretchForCauchyStress(
            sample.radialStress,sample.radius,sample.referenceRadius,
            axialStretch,mu,lameLambda);
        const double J=lambdaR*lambdaTheta*axialStretch;
        result.minimumJacobian=std::min(result.minimumJacobian,J);
        result.maximumJacobian=std::max(result.maximumJacobian,J);
    }
    // Independent centered finite-difference check of d(sigma_r)/dr =
    // (sigma_theta-sigma_r)/r along the integrated solution.
    for(std::size_t i=1;i+1<result.samples.size();++i){
        const auto& a=result.samples[i-1u];
        const auto& c=result.samples[i];
        const auto& b=result.samples[i+1u];
        const double ds=(b.radialStress-a.radialStress)/(b.radius-a.radius);
        const double lambdaTheta=c.radius/c.referenceRadius;
        const double lambdaR=radialStretchForCauchyStress(
            c.radialStress,c.radius,c.referenceRadius,axialStretch,mu,lameLambda);
        const double J=lambdaR*lambdaTheta*axialStretch;
        const double sigmaTheta=neoHookeanCauchyPrincipalStress(
            lambdaTheta,J,mu,lameLambda);
        const double residual=ds-(sigmaTheta-c.radialStress)/c.radius;
        result.maximumBalanceResidualPaPerM=std::max(
            result.maximumBalanceResidualPaPerM,std::abs(residual));
    }
    return result;
}
inline double incompressibleCylinderPressure(
    double ri,double ro,double referenceRi,double mu,double axialStretch,
    std::uint32_t segments){
    const double h=(ro-ri)/double(segments);
    double sum=0.0;
    for(std::uint32_t i=0;i<=segments;++i){
        const double r=ri+double(i)*h;
        const double referenceR=std::sqrt(referenceRi*referenceRi+
            axialStretch*(r*r-ri*ri));
        const double lambdaTheta=r/referenceR;
        const double lambdaR=referenceR/(axialStretch*r);
        const double y=(lambdaTheta*lambdaTheta-lambdaR*lambdaR)/r;
        sum+=(i==0u||i==segments?1.0:(i%2u?4.0:2.0))*y;
    }
    return mu*sum*h/3.0;
}
inline LoadedReferenceEquilibrium solveLoadedReference(
    double currentInnerRadius,double currentOuterRadius,double innerPressure,
    double outerPressure,double mu,double lameLambda,double axialStretch,
    std::uint32_t segments){
    if(segments%2u)++segments;
    LoadedReferenceEquilibrium out;
    out.currentInnerRadiusM=currentInnerRadius;out.currentOuterRadiusM=currentOuterRadius;
    out.innerPressurePa=innerPressure;out.outerPressurePa=outerPressure;
    out.axialStretch=axialStretch;out.lameLambdaPa=lameLambda;out.shearModulusPa=mu;
    double lo=currentInnerRadius*0.25,hi=currentInnerRadius*0.999999;
    auto integrate=[&](double refRi){
        return integrateFiniteLaméCylinder(currentInnerRadius,currentOuterRadius,refRi,
            innerPressure,outerPressure,mu,lameLambda,axialStretch,segments);
    };
    auto low=integrate(lo),high=integrate(hi);
    if(!(low.outerResidualPa>0.0&&high.outerResidualPa<0.0))
        throw std::runtime_error("finite-Lame reference-radius shooting did not bracket the outer traction");
    RadialEquilibriumIntegration solved;
    for(unsigned i=0;i<64u;++i){
        const double mid=0.5*(lo+hi);
        auto trial=integrate(mid);
        if(trial.outerResidualPa>0.0){lo=mid;}else{hi=mid;}
        solved=std::move(trial);
    }
    const double referenceRi=0.5*(lo+hi);
    solved=integrate(referenceRi);
    out.referenceInnerRadiusM=referenceRi;
    out.referenceOuterRadiusM=solved.samples.back().referenceRadius;
    out.innerBoundaryResidualPa=solved.samples.front().radialStress+innerPressure;
    out.outerBoundaryResidualPa=solved.outerResidualPa;
    out.maxRadialEquilibriumResidualPaPerM=solved.maximumBalanceResidualPaPerM;
    out.minimumJacobian=solved.minimumJacobian;out.maximumJacobian=solved.maximumJacobian;
    const double pressureTolerance=std::max(1.0e-7,innerPressure*1.0e-10);
    if(std::abs(out.innerBoundaryResidualPa)>pressureTolerance||
       std::abs(out.outerBoundaryResidualPa)>pressureTolerance||
       !(out.minimumJacobian>0.0&&out.maximumJacobian<10.0)||
       !std::isfinite(out.maxRadialEquilibriumResidualPaPerM))
        throw std::runtime_error("finite-Lame loaded-reference boundary residual did not converge");
    double ilo=currentInnerRadius*0.25,ihi=currentInnerRadius*0.999999;
    const std::uint32_t simpsonSegments=std::max<std::uint32_t>(segments,512u);
    for(unsigned i=0;i<64u;++i){
        const double mid=0.5*(ilo+ihi);
        const double pressure=incompressibleCylinderPressure(
            currentInnerRadius,currentOuterRadius,mid,mu,axialStretch,simpsonSegments);
        if(pressure>innerPressure-outerPressure)ilo=mid;else ihi=mid;
    }
    out.incompressibleReferenceInnerRadiusM=0.5*(ilo+ihi);
    out.incompressibleInnerHoopStretch=currentInnerRadius/
        out.incompressibleReferenceInnerRadiusM;
    out.finiteKInnerHoopStretch=currentInnerRadius/out.referenceInnerRadiusM;
    return out;
}
inline MaterialProgram makeMaterial(const Config& c){
    const std::string source =
        "material numi_reduced_abdominal_aortic_wall_v1 {\n"
        "  parameter density : kg/m^3 = "+std::to_string(c.densityKgM3)+" in [900,1250];\n"
        "  parameter mu : Pa = "+std::to_string(c.shearModulusPa)+" in [50000,300000];\n"
        "  parameter lambda : Pa = "+std::to_string(c.bulkPenaltyPa)+" in [500000,5000000];\n"
        "  model neo_hookean; energy = neo_hookean(mu,lambda); valid = J() - 0.1;\n"
        "  supports fem;\n"
        "  interface {static_friction=0;dynamic_friction=0;restitution=0;adhesion=0;}\n"
        "  limits {minimum_J=0.1;maximum_J=10;maximum_stress=1e9;maximum_energy_density=1e9;}\n"
        "}\n";
    auto parsed=parseMatter(source);
    if(!parsed.succeeded())throw std::runtime_error("reduced aortic material source failed Matter parsing");
    return parsed.material;
}
inline Built build(const SectionProof& proof,const MRBodyStateGPU& attachmentPose,
                   const Vec& attachmentBodyCOM,std::uint32_t runtimeAttachmentBody,
                   const MRBodyStateGPU& massDonorPose,
                   std::uint32_t massDonorRuntimeBody,double targetBloodVolumeM3,
                   double internalPressurePa,const Config& cfg={}){
    constexpr std::string_view expectedSHA="1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108";
    if(proof.stableId!=9u||proof.sourceBody!=7u||proof.sourcePayloadSHA256!=expectedSHA||
       proof.orderedExternalContour.size()!=87u||runtimeAttachmentBody==MR_INVALID_INDEX||
       massDonorRuntimeBody==MR_INVALID_INDEX||!(targetBloodVolumeM3>0)||
       !(internalPressurePa>0)||!std::isfinite(internalPressurePa)||
       cfg.circumferentialSegments<12u||cfg.axialSegments<8u||
       !(cfg.wallThicknessM>0&&cfg.densityKgM3>0&&cfg.shearModulusPa>0&&cfg.bulkPenaltyPa>0&&
         cfg.inferredAxialStretch>0.8&&cfg.inferredAxialStretch<1.2&&
         cfg.equilibriumIntegrationSegments>=128u))
        throw std::runtime_error("aortic wall requires the exact admitted stable-ID-9/body-7 contour and positive declared parameters");
    const Vec axis=unit(proof.planeNormal);
    auto outer=resampleClosedContour(proof.orderedExternalContour,cfg.circumferentialSegments);
    Vec centroid{};for(const Vec& p:outer)centroid=add(centroid,mul(p,1.0/double(outer.size())));
    const double sampledArea=polygonArea(outer,axis);
    if(!(sampledArea>0)||!(proof.externalAreaM2>0)||
       std::abs(sampledArea-proof.externalAreaM2)>proof.externalAreaM2*0.05)
        throw std::runtime_error("coarse aortic section proxy differs materially from the pinned 87-edge contour proof");
    // Preserve the measured external cross-sectional area after reducing the
    // 87-point source contour to a 24-point functional mesh.
    const double areaScale=std::sqrt(proof.externalAreaM2/sampledArea);
    for(Vec& p:outer)p=add(centroid,mul(sub(p,centroid),areaScale));
    const double outerArea=polygonArea(outer,axis);
    if(!(outerArea>0)||std::abs(outerArea-proof.externalAreaM2)>proof.externalAreaM2*2e-12)
        throw std::runtime_error("area-preserving aortic section resampling failed");
    const double equivalentRadius=std::sqrt(outerArea/3.14159265358979323846);
    const double innerScale=(equivalentRadius-cfg.wallThicknessM)/equivalentRadius;
    if(!(innerScale>0.5&&innerScale<1.0))throw std::runtime_error("inferred aortic wall thickness is incompatible with source section scale");
    std::vector<Vec> inner;inner.reserve(outer.size());
    for(const Vec& p:outer)inner.push_back(add(centroid,mul(sub(p,centroid),innerScale)));
    const double innerArea=polygonArea(inner,axis);
    const double length=targetBloodVolumeM3/innerArea;
    if(!(length>0.02&&length<0.30))throw std::runtime_error("aortic reduced reservoir length is outside the declared anatomical segment bounds");
    const double currentOuterRadius=equivalentRadius;
    const double currentInnerRadius=std::sqrt(innerArea/3.14159265358979323846);
    const LoadedReferenceEquilibrium loaded=solveLoadedReference(
        currentInnerRadius,currentOuterRadius,internalPressurePa,0.0,
        cfg.shearModulusPa,cfg.bulkPenaltyPa,cfg.inferredAxialStretch,
        cfg.equilibriumIntegrationSegments);

    Built out;out.sourceExternalAreaM2=outerArea;out.inferredLumenAreaM2=innerArea;
    out.circumferentialSegments=cfg.circumferentialSegments;out.axialSegments=cfg.axialSegments;
    out.loadedReference=loaded;
    out.segmentLengthM=length;out.attachmentBody=runtimeAttachmentBody;out.donorBody=massDonorRuntimeBody;
    auto& object=out.object;object.name="Human/BodyParts3D/stable-9/reduced-cavity-wall-v1";
    object.representation=Representation::fem;object.automaticRepresentation=false;
    // Matter explicit reference coordinates are admitted only for immutable,
    // non-mixed FEM; the current mesh remains the initialized position.
    object.mixedFEM=false;
    object.twoWayCoupling=true;object.adaptive=false;object.identifiable=false;
    object.deformableContact=cfg.outerLateralContact;object.deformableSelfContact=false;object.characteristicLength=std::max(cfg.wallThicknessM,length/double(cfg.axialSegments));
    // Stable derivation-contract identity; the compiled world fingerprint also
    // hashes the actual reference coordinate array and constitutive parameters.
    object.femReferenceSourceIdentity={
        0x4152544552594c44ULL,0x5245464552454e43ULL,
        0x4e48414c44303136ULL,0x46494e4954454b31ULL};
    const std::size_t n=outer.size();const std::uint32_t axial=cfg.axialSegments;
    const double cap=cfg.wallThicknessM;
    const Vec com=attachmentBodyCOM;
    auto bodyPoint=[&](const Vec& surface,double axialOffset){return sub(add(surface,mul(axis,axialOffset)),com);};
    const double axialMidpoint=0.5*length;
    auto addNode=[&](const Vec& surface,double axialOffset,double wallFraction){
        const Vec attachmentBodyLocal=bodyPoint(surface,axialOffset);
        const Vec world=add(Vec{attachmentPose.position.x,attachmentPose.position.y,attachmentPose.position.z},
                            rotate(attachmentPose.orientation,attachmentBodyLocal));
        const Vec donorLocal=rotate({-massDonorPose.orientation.x,-massDonorPose.orientation.y,
            -massDonorPose.orientation.z,massDonorPose.orientation.w},
            sub(world,Vec{massDonorPose.position.x,massDonorPose.position.y,massDonorPose.position.z}));
        const double referenceRadius=wallFraction<=0.0?loaded.referenceInnerRadiusM:
            (wallFraction>=1.0?loaded.referenceOuterRadiusM:
             loaded.referenceInnerRadiusM+wallFraction*(loaded.referenceOuterRadiusM-loaded.referenceInnerRadiusM));
        Vec radial=sub(surface,centroid);
        radial=sub(radial,mul(axis,dot(radial,axis)));
        const double currentEquivalentRadius=wallFraction<=0.0?currentInnerRadius:currentOuterRadius;
        if(currentEquivalentRadius>1.0e-12)radial=mul(radial,referenceRadius/currentEquivalentRadius);
        const double referenceAxial=(axialOffset-axialMidpoint)/cfg.inferredAxialStretch+axialMidpoint;
        const Vec referenceBodyLocal=sub(add(add(centroid,mul(axis,referenceAxial)),radial),com);
        const Vec referenceWorld=add(Vec{attachmentPose.position.x,attachmentPose.position.y,attachmentPose.position.z},
                            rotate(attachmentPose.orientation,referenceBodyLocal));
        const std::uint32_t id=static_cast<std::uint32_t>(object.femNodes.size());
        object.femNodes.push_back(world);object.femReferenceNodes.push_back(referenceWorld);
        out.donorCOMLocalNodes.push_back(donorLocal);
        out.attachmentCOMLocalNodes.push_back(attachmentBodyLocal);return id;
    };
    // Main annular sidewall rings at z = 0..L.
    std::vector<std::vector<std::uint32_t>> sideInner(axial+1,std::vector<std::uint32_t>(n));
    std::vector<std::vector<std::uint32_t>> sideOuter(axial+1,std::vector<std::uint32_t>(n));
    for(std::uint32_t z=0;z<=axial;++z){
        const double s=length*double(z)/double(axial);
        for(std::size_t i=0;i<n;++i){
            sideInner[z][i]=addNode(inner[i],s,0.0);
            sideOuter[z][i]=addNode(outer[i],s,1.0);
        }
    }
    if(cfg.outerLateralContact){
        object.femContactNodes.reserve(std::size_t(axial+1u)*n);
        for(const auto& ring:sideOuter)
            object.femContactNodes.insert(object.femContactNodes.end(),ring.begin(),ring.end());
        std::sort(object.femContactNodes.begin(),object.femContactNodes.end());
        if(object.femContactNodes.size()!=std::size_t(axial+1u)*n||
           std::adjacent_find(object.femContactNodes.begin(),object.femContactNodes.end())!=object.femContactNodes.end())
            throw std::runtime_error("aortic outer-lateral contact node mask is not exact and unique");
    }
    const std::uint32_t lowCenter=addNode(centroid,0.0,0.0);
    const std::uint32_t highCenter=addNode(centroid,length,0.0);
    std::vector<std::uint32_t> lowInner(n),lowOuter(n),highInner(n),highOuter(n);
    std::uint32_t lowCapCenter=addNode(centroid,-cap,0.0);
    std::uint32_t highCapCenter=addNode(centroid,length+cap,0.0);
    for(std::size_t i=0;i<n;++i){
        lowInner[i]=addNode(inner[i],-cap,0.0);lowOuter[i]=addNode(outer[i],-cap,1.0);
        highInner[i]=addNode(inner[i],length+cap,0.0);highOuter[i]=addNode(outer[i],length+cap,1.0);
    }
    const auto hex=[&](std::array<std::uint32_t,8> h){
        static constexpr std::array<std::array<unsigned,4>,6> split{{
            {{0,1,2,6}},{{0,2,3,6}},{{0,3,7,6}},{{0,7,4,6}},{{0,4,5,6}},{{0,5,1,6}}
        }};
        for(const auto& s:split)appendTet(object,{h[s[0]],h[s[1]],h[s[2]],h[s[3]]});
    };
    const auto prism=[&](std::array<std::uint32_t,6> p){
        appendTet(object,{p[0],p[1],p[2],p[5]});
        appendTet(object,{p[0],p[1],p[4],p[5]});
        appendTet(object,{p[0],p[3],p[4],p[5]});
    };
    // Annular wall and both closed, pressure-bearing cut-end membranes.
    for(std::uint32_t z=0;z<axial;++z)for(std::size_t i=0;i<n;++i){
        const std::size_t j=(i+1)%n;
        hex({sideInner[z][i],sideOuter[z][i],sideOuter[z][j],sideInner[z][j],
             sideInner[z+1][i],sideOuter[z+1][i],sideOuter[z+1][j],sideInner[z+1][j]});
    }
    for(std::size_t i=0;i<n;++i){
        const std::size_t j=(i+1)%n;
        // proximal cap slab from z=-t to the source cut plane z=0
        hex({lowInner[i],lowOuter[i],lowOuter[j],lowInner[j],
             sideInner[0][i],sideOuter[0][i],sideOuter[0][j],sideInner[0][j]});
        prism({lowCapCenter,lowInner[i],lowInner[j],lowCenter,sideInner[0][i],sideInner[0][j]});
        // distal cap slab from z=L to z=L+t
        hex({sideInner[axial][i],sideOuter[axial][i],sideOuter[axial][j],sideInner[axial][j],
             highInner[i],highOuter[i],highOuter[j],highInner[j]});
        prism({highCenter,sideInner[axial][i],sideInner[axial][j],
               highCapCenter,highInner[i],highInner[j]});
    }
    struct Occ{unsigned count=0;Face oriented{};};std::map<Face,Occ> boundary;
    auto sorted=[](Face f){std::sort(f.begin(),f.end());return f;};
    for(const auto& tet:object.tetrahedra){
        const auto t=tet.nodes;
        for(const Face f:std::array<Face,4>{{{t[1],t[2],t[3]},{t[0],t[3],t[2]},{t[0],t[1],t[3]},{t[0],t[2],t[1]}}}){
            auto& e=boundary[sorted(f)];++e.count;e.oriented=f;
        }
    }
    if(cfg.outerLateralContact) {
        const std::set<std::uint32_t> outerNodes(
            object.femContactNodes.begin(),object.femContactNodes.end());
        std::size_t enabledFaces=0u;
        for(std::size_t ti=0;ti<object.tetrahedra.size();++ti) {
            const auto& t=object.tetrahedra[ti].nodes;
            for(unsigned opposite=0;opposite<4u;++opposite) {
                Face face{};unsigned k=0;
                for(unsigned c=0;c<4u;++c)if(c!=opposite)face[k++]=t[c];
                if(boundary.at(sorted(face)).count!=1u)continue;
                const bool outerFace=std::all_of(face.begin(),face.end(),
                    [&](auto node){return outerNodes.contains(node);});
                if(outerFace)++enabledFaces;
                else object.femDisabledContactSurfaceFaces.push_back(
                    static_cast<std::uint32_t>(4u*ti+opposite));
            }
        }
        if(enabledFaces!=2u*n*axial ||
           object.femDisabledContactSurfaceFaces.empty())
            throw std::runtime_error("aortic source contact-face selection is incomplete");
    }
    std::set<std::uint32_t> innerSet;for(const auto& ring:sideInner)innerSet.insert(ring.begin(),ring.end());
    std::set<Face> capKeys;
    for(std::size_t i=0;i<n;++i){const std::size_t j=(i+1)%n;
        capKeys.insert(sorted({lowCenter,sideInner[0][i],sideInner[0][j]}));
        capKeys.insert(sorted({highCenter,sideInner[axial][i],sideInner[axial][j]}));
    }
    std::vector<Face> cavityFaces;
    for(const auto& [key,e]:boundary)if(e.count==1){
        const bool innerSide=std::all_of(key.begin(),key.end(),[&](auto node){return innerSet.contains(node);});
        if(innerSide||capKeys.contains(key)){Face f=e.oriented;std::swap(f[1],f[2]);cavityFaces.push_back(f);}
    }
    if(cavityFaces.size()!=2u*n*axial+2u*n)
        throw std::runtime_error("aortic cavity boundary failed closed sidewall/capped-port topology admission");
    double signedV=cavityVolume(object.femNodes,cavityFaces);
    if(signedV<0){for(auto& f:cavityFaces)std::swap(f[1],f[2]);signedV=-signedV;}
    if(!(signedV>0)||std::abs(signedV-targetBloodVolumeM3)>targetBloodVolumeM3*2e-4)
        throw std::runtime_error("aortic closed-reservoir geometry does not match the existing CVSim blood volume");
    const std::uint32_t materialIndex=0u;object.materialIndex=materialIndex;
    // The source section ring is attached to the source abdomen body (7).
    // Its mechanical mass is separately attributed to the registered torso
    // mass pool by the shared NumiHumanTissueMass partition.
    for(std::size_t i=0;i<n;i+=std::max<std::size_t>(1,n/24)){
        FEMHumanAttachmentSource a;a.node=sideOuter[0][i];a.bodyIndex=runtimeAttachmentBody;
        a.stableIdentifier=cfg.attachmentStableBase+static_cast<std::uint32_t>(i);
        a.localPoint=out.attachmentCOMLocalNodes[a.node];object.femHumanAttachments.push_back(a);
    }
    out.wallMassKg=0.0; // Replaced with the exact post-cook FEM nodal-mass sum by massNodes().
    out.cavity.stableIdentifier=9007u;out.cavity.compartment=7u;out.cavity.objectIndex=0u;
    out.cavity.pressureScale=13300.0;out.cavity.geometryResidualTolerance=1e-5;
    out.cavity.sourceIdentity={0x1a191fe2f6c4a1cbULL,0x03ee290bc850b800ULL,0x11cc267577992ea8ULL,0x6abb3a4cb39f4108ULL};
    out.cavity.mechanicalIdentity={0x61726f727469632dULL,0x77616c6c2d763100ULL,0x6c6162726f737365ULL,0x3139333435333536ULL};
    for(std::size_t i=0;i<cavityFaces.size();++i)
        out.cavity.faces.push_back({90070000u+static_cast<std::uint32_t>(i),cavityFaces[i],VascularCavityFaceRole::materialWall});
    return out;
}
struct AnnularMechanicalWall {
    ObjectSource object;
    std::vector<std::array<float,4>> externalForces;
    double pressurePa=0.0;
    double sidewallAreaM2=0.0;
    double endForceN=0.0;
    Vec netExternalForceN{};
};
// Mechanical-only annular control. It deliberately omits vascular state and
// hydraulic closure triangles; the two cut rings receive equal/opposite
// pressure resultants through the existing FEM external-force API.
inline AnnularMechanicalWall makeAnnularMechanicalWall(const Built& built,
    const SectionProof& proof,double pressurePa) {
    if(!(pressurePa>0.0)||!std::isfinite(pressurePa))
        throw std::runtime_error("annular wall pressure must be finite and positive");
    AnnularMechanicalWall result;result.object=built.object;
    const Vec axis=unit(proof.planeNormal);
    const std::size_t n=built.object.femNodes.size();
    if(n==0||built.object.femReferenceNodes.size()!=n)
        throw std::runtime_error("annular wall source node/reference arrays disagree");
    const std::size_t circum=built.circumferentialSegments;
    const std::size_t axial=built.axialSegments;
    if(circum<3u||axial==0u)
        throw std::runtime_error("annular wall grid dimensions are absent");
    // The builder emits exactly six Freudenthal tetrahedra per annular
    // circumferential-sector/axial cell before adding its artificial end caps.
    const std::size_t sideTetCount=6u*circum*axial;
    if(built.object.tetrahedra.size()<sideTetCount)
        throw std::runtime_error("annular wall source is missing the sidewall tet prefix");
    std::set<std::uint32_t> retained;
    ObjectSource source=built.object;
    source.tetrahedra.resize(sideTetCount);
    retained.clear();
    for(const auto& tet:source.tetrahedra)for(auto id:tet.nodes)retained.insert(id);
    std::vector<std::uint32_t> remap(n,MR_INVALID_INDEX);
    std::vector<Vec> current,reference;current.reserve(retained.size());reference.reserve(retained.size());
    for(std::uint32_t old=0;old<n;++old)if(retained.contains(old)) {
        remap[old]=std::uint32_t(current.size());
        current.push_back(source.femNodes[old]);reference.push_back(source.femReferenceNodes[old]);
    }
    for(auto& tet:source.tetrahedra)for(auto& id:tet.nodes)id=remap.at(id);
    source.femFixedNodes.clear();
    for(const auto& attachment:source.femHumanAttachments) {
        if(attachment.node>=remap.size()||remap[attachment.node]==MR_INVALID_INDEX)
            throw std::runtime_error("proximal source attachment is not on the annular wall");
        source.femFixedNodes.push_back(remap[attachment.node]);
    }
    std::sort(source.femFixedNodes.begin(),source.femFixedNodes.end());
    source.femFixedNodes.erase(std::unique(source.femFixedNodes.begin(),source.femFixedNodes.end()),source.femFixedNodes.end());
    source.femHumanAttachments.clear();
    source.femNodes=std::move(current);source.femReferenceNodes=std::move(reference);
    source.twoWayCoupling=false;
    result.object=std::move(source);result.externalForces.resize(result.object.femNodes.size());
    result.pressurePa=pressurePa;
    const auto addForce=[&](std::uint32_t id,const Vec& f) {
        if(id>=result.externalForces.size())throw std::runtime_error("annular pressure force escaped compact nodes");
        auto& out=result.externalForces[id];
        for(unsigned k=0;k<3;++k){out[k]+=static_cast<float>(f[k]);result.netExternalForceN[k]+=f[k];}
    };
    Vec centroid{};for(const auto& p:proof.orderedExternalContour)centroid=add(centroid,mul(p,1.0/double(proof.orderedExternalContour.size())));
    for(const auto& face:built.cavity.faces) {
        if(std::any_of(face.nodes.begin(),face.nodes.end(),[&](auto id){return id>=remap.size()||remap[id]==MR_INVALID_INDEX;}))continue;
        const auto a=remap[face.nodes[0]],b=remap[face.nodes[1]],c=remap[face.nodes[2]];
        const Vec av=mul(cross(sub(result.object.femNodes[b],result.object.femNodes[a]),
                               sub(result.object.femNodes[c],result.object.femNodes[a])),0.5);
        const double magnitude=norm(av);
        if(!(magnitude>0.0))throw std::runtime_error("annular pressure boundary has a degenerate triangle");
        if(std::abs(dot(av,axis))/magnitude>0.5)continue;
        result.sidewallAreaM2+=magnitude;
        const Vec nodal=mul(av,pressurePa/3.0);
        addForce(a,nodal);addForce(b,nodal);addForce(c,nodal);
    }
    result.endForceN=pressurePa*built.inferredLumenAreaM2;
    for(const int end:{0,1}) {
        std::vector<std::pair<double,std::uint32_t>> ring;
        const double target=end?built.segmentLengthM:0.0;
        for(std::uint32_t i=0;i<result.object.femNodes.size();++i) {
            Vec rel=sub(result.object.femNodes[i],centroid);const double axial=dot(rel,axis);
            if(std::abs(axial-target)>1e-8)continue;
            rel=sub(rel,mul(axis,axial));ring.push_back({norm(rel),i});
        }
        if(ring.size()!=2u*circum)throw std::runtime_error("annular port ring does not contain one inner and outer node per source sector");
        std::sort(ring.begin(),ring.end());
        for(std::size_t i=0;i<circum;++i)
            addForce(ring[i].second,mul(axis,(end?1.0:-1.0)*result.endForceN/12.0));
    }
    if(norm(result.netExternalForceN)>std::max(1e-8,result.endForceN*1e-8))
        throw std::runtime_error("annular pressure boundary resultants do not balance");
    return result;
}

inline Built appendToWorld(WorldSource& source,const SectionProof& proof,
                            const MRBodyStateGPU& attachmentPose,const Vec& attachmentCOM,
                            std::uint32_t attachmentBody,const MRBodyStateGPU& donorPose,
                            std::uint32_t donorBody,const Config& cfg={}){
    auto comp=std::find_if(source.vascular.compartments.begin(),source.vascular.compartments.end(),
        [](const VascularCompartmentSource& c){return c.stableIdentifier==7u;});
    if(comp==source.vascular.compartments.end()||comp->storageKind!=VascularStorageKind::absoluteVolume||
       comp->pressureLaw!=VascularPressureLaw::linearCompliance||!(comp->compliance>0)||
       !(comp->initialVolume>0)||comp->initialSpeciesAmounts.size()!=source.vascular.species.size())
        throw std::runtime_error("stable CVSim compartment 7 must be the admitted absolute-volume linear-compliance aortic reservoir");
    for(const auto& cavity:source.vascular.cavities)if(cavity.compartment==7u)
        throw std::runtime_error("CVSim compartment 7 already has a pressure-wall owner");
    const double pressure=comp->referencePressure+(comp->initialVolume-comp->referenceVolume)/comp->compliance;
    if(!(pressure>0)||!std::isfinite(pressure))throw std::runtime_error("CVSim aortic reference pressure is invalid");
    Built result=build(proof,attachmentPose,attachmentCOM,attachmentBody,donorPose,
                       donorBody,comp->initialVolume,pressure,cfg);
    auto material=makeMaterial(cfg);
    result.objectIndex=static_cast<std::uint32_t>(source.objects.size());
    result.object.materialIndex=static_cast<std::uint32_t>(source.materials.size());
    result.cavity.objectIndex=result.objectIndex;result.cavity.initialPressure=pressure;
    source.materials.push_back(std::move(material));
    source.objects.push_back(result.object);
    source.vascular.cavities.push_back(result.cavity);
    // The existing CVSim compartment keeps its volume and species state. Its
    // constitutive law is replaced by the same-step deforming FEM cavity row.
    comp->referenceVolume=0;comp->referencePressure=0;comp->compliance=0;
    comp->pressureLaw=VascularPressureLaw::deformingCavity;
    comp->elastanceMin=0;comp->elastanceMax=0;comp->periodSeconds=0;
    comp->activationStart=0;comp->activationEnd=0;comp->sourcePi=0;
    comp->maximumVolumeDisplacement=0;comp->phaseDelay=0;
    comp->periodNumeratorSeconds=0;comp->periodDenominator=0;
    return result;
}
inline std::vector<metalrobo::NumiHumanTissueMassNode> cookedMassNodes(
    const CompiledWorld& compiled,std::uint32_t objectIndex,std::uint32_t donorBody,
    std::span<const Vec> donorCOMLocalNodes){
    if(objectIndex>=compiled.objects.size())throw std::runtime_error("aortic FEM object index is outside the compiled Matter world");
    const auto& object=compiled.objects[objectIndex];
    if(object.stateOffset>compiled.fem.nodes.size()||object.stateCount>compiled.fem.nodes.size()-object.stateOffset||
       object.stateCount!=donorCOMLocalNodes.size())throw std::runtime_error("aortic cooked nodal mass span does not match its source mesh");
    std::vector<metalrobo::NumiHumanTissueMassNode> result;result.reserve(object.stateCount);
    for(std::uint32_t i=0;i<object.stateCount;++i){
        const double mass=compiled.fem.nodes[object.stateOffset+i].positionAndMass.w;
        if(!(mass>0)||!std::isfinite(mass))throw std::runtime_error("aortic wall has a nonpositive cooked nodal mass");
        result.push_back({object.stateOffset+i,donorBody,mass,donorCOMLocalNodes[i]});
    }
    return result;
}
} // namespace numi::human::aortic_wall
