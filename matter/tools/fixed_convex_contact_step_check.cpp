#include "numi/matter/fixed_convex_contact_step.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using Vec=std::array<double,3>;
static double norm(Vec v){return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
static double dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static double sdf(unsigned shape,Vec p,Vec& normal){
    if(shape==NM_RIGID_PLANE){normal={0,1,0};return p[1];}
    if(shape==NM_RIGID_CAPSULE)p[2]-=std::clamp(p[2],-.005,.005);
    if(shape==NM_RIGID_SPHERE||shape==NM_RIGID_CAPSULE){double n=norm(p);for(unsigned i=0;i<3;++i)normal[i]=p[i]/n;return n-.005;}
    Vec delta{},outside{};for(unsigned i=0;i<3;++i){delta[i]=std::abs(p[i])-.005;outside[i]=std::max(delta[i],0.0);}
    const double n=norm(outside);
    if(n>0){for(unsigned i=0;i<3;++i)normal[i]=std::copysign(outside[i]/n,p[i]);return n;}
    unsigned axis=0;for(unsigned i=1;i<3;++i)if(delta[i]>delta[axis])axis=i;
    normal={0,0,0};normal[axis]=std::copysign(1.0,p[axis]);return delta[axis];
}
int main(){try{
    // The z-plane signed distance does not depend on tangential world x/y.
    // Keep the authored physical clearance while bounding roundoff from the
    // actual dot/offset arithmetic instead of the unrelated world origin.
    constexpr float eps=1.1920928955078125e-7f;
    constexpr float authoredClearance=1.6163281770786853e-6f;
    constexpr float contactSlop=1.6163282e-2f;
    const float planeScale=nmFixedPlaneContactArithmeticScale(
        0.0f,0.0f,1.0f,0.25f,-0.4f,1.7e-6f,0.0f,contactSlop);
    const float translatedPlaneScale=nmFixedPlaneContactArithmeticScale(
        0.0f,0.0f,1.0f,1.6f,-1.6f,1.7e-6f,0.0f,contactSlop);
    if(planeScale!=translatedPlaneScale)
        throw std::runtime_error("horizontal plane guard depends on tangential translation");
    const float thickness=std::max(authoredClearance,16.0f*eps*planeScale);
    const float roundoff=8.0f*eps*std::max(thickness,planeScale);
    if(thickness<authoredClearance || thickness!=authoredClearance)
        throw std::runtime_error("plane guard reduced the authored physical clearance");
    const float feasible=thickness+roundoff;
    const float nearGap=feasible+4.0f*roundoff;
    const float nearTravel=1.0e-7f;
    if(!nmFixedConvexContactOutward(0,NM_RIGID_PLANE,nearGap,feasible,
                                    nearTravel,nearTravel,roundoff))
        throw std::runtime_error("near-boundary outward plane motion was not certified");
    if(nmFixedConvexContactOutward(0,NM_RIGID_PLANE,nearGap,feasible,
                                   nearTravel,-nearTravel,roundoff))
        throw std::runtime_error("near-boundary inward plane motion was certified");
    const float crossingTravel=2.0f*nearGap;
    const float crossingEndpoint=nearGap-crossingTravel;
    if(!(crossingEndpoint<0.0f) ||
       nmFixedConvexContactOutward(0,NM_RIGID_PLANE,nearGap,feasible,
                                   crossingTravel,-crossingTravel,roundoff))
        throw std::runtime_error("actual plane crossing was not rejected");
    if(nmFixedPlaneContactArithmeticScale(0.0f,0.0f,1.0f,
            1.6f,-1.6f,1.7e-6f,0.0f,contactSlop)!=planeScale)
        throw std::runtime_error("translated plane used a different clearance threshold");
    std::mt19937 gen(42);std::uniform_real_distribution<double> rand(-1,1);
    unsigned checked=0,certified=0;double worst=0;
    for(unsigned shape:{NM_RIGID_PLANE,NM_RIGID_SPHERE,NM_RIGID_CAPSULE,NM_RIGID_BOX}){
        for(unsigned i=0;i<6000;++i){
            Vec p{.012*rand(gen),.012*rand(gen),.012*rand(gen)},n{};
            const double initial=sdf(shape,p,n);if(initial<=1e-8)continue;
            Vec motion{.03*rand(gen),.03*rand(gen),.03*rand(gen)};
            const double along=dot(n,motion),length=norm(motion);
            const float sep=float(initial),travel=float(length),projection=float(along);
            const float roundoff=32*0x1p-23f*.03f;
            // The guard is tested with adversarial normal/dot uncertainty,
            // independent of its support-plane proof, not exact-double dots.
            const float noisyProjection=projection+float(8*roundoff/initial*length);
            const bool safe=nmFixedConvexContactOutward(0,shape,sep,1e-9f,travel,noisyProjection,roundoff);
            ++checked;if(!safe)continue;++certified;
            if(along<0)throw std::runtime_error("rounding guard certified inward motion");
            for(unsigned k=0;k<=128;++k){Vec x{};for(unsigned j=0;j<3;++j)x[j]=p[j]+double(k)/128*motion[j];Vec m{};
                const double loss=initial-sdf(shape,x,m);worst=std::max(worst,loss);
                if(loss>1e-12)throw std::runtime_error("certified fixed-convex segment approaches obstacle");}
            for(unsigned flag:{NM_RIGID_DYNAMIC,NM_RIGID_ARTICULATED,NM_RIGID_SUTURE_STRAND,NM_RIGID_PUNCTURE_TIP,NM_RIGID_PUNCTURE_DILATOR})
                if(nmFixedConvexContactOutward(flag,shape,sep,1e-9f,travel,noisyProjection,roundoff))throw std::runtime_error("moving/nonconvex topology bypassed travel guard");
        }
    }
    if(certified<1000)throw std::runtime_error("insufficient independent segment coverage");
    if(nmFixedConvexContactOutward(0,NM_RIGID_ARC,.001f,1e-9f,.01f,.01f,1e-9f))throw std::runtime_error("nonconvex arc admitted");
    if(nmFixedConvexContactOutward(0,NM_RIGID_BOX,1e-9f,1e-9f,.01f,.01f,1e-12f))throw std::runtime_error("boundary equality admitted");
    if(nmFixedConvexContactOutward(0,NM_RIGID_BOX,.001f,1e-9f,.01f,0,1e-9f))throw std::runtime_error("unresolved tangent admitted");
    if(nmFixedConvexContactOutward(0,NM_RIGID_BOX,1e-8f,1e-9f,.01f,.01f,1e-6f))throw std::runtime_error("unresolved coordinate/gap ratio admitted");
    std::cout<<"PASS fixed convex CCD supporting plane checks="<<checked<<" certified_segments="<<certified<<" worst_distance_loss="<<worst<<" final_trial_ccd_unchanged=true\n";
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
