#include "numi/matter/canonical_neo_hookean.h"
#include "cardiac_material_reference.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
namespace ref=numi_cardiac_reference;
void require(bool ok){if(!ok)throw std::runtime_error("canonical neo-Hookean numerical regression");}
int main(){try{
    constexpr float shear=13000,bulk=650000;
    float high[9]={1,0,0,0,1,0,0,0,1},low[9]{},p[9],t[9];
    require(nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,nullptr));
    for(float v:p)require(v==0);
    double maximum=0;
    for(double eps:{1e-10,1e-9,1e-8,1e-7,1e-6,1e-4,.01}){
        high[0]=float(1+eps);low[0]=float((1+eps)-double(high[0]));
        // The sub-ULP cases intentionally store eps in the low word directly.
        if(high[0]==1)low[0]=float(eps);
        const double e=(double(high[0])-1)+double(low[0]),a=1+e,l=std::log1p(e);
        require(nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,nullptr));
        const double expected=shear*(e+e/a)+bulk*l/a;
        maximum=std::max(maximum,std::abs(p[0]-expected)/std::abs(expected));
        require(std::abs(p[0]-expected)<1e-5*std::abs(expected));
    }
    std::uint32_t rng=0x5a6f7261u;
    const auto random=[&](){rng=1664525u*rng+1013904223u;return double(rng)/4294967296.0-.5;};
    double tangentError=0,stressError=0;
    for(unsigned sample=0;sample<64;++sample){
        ref::Matrix f{},h{};
        float d[9];
        for(unsigned i=0;i<9;++i){
            const double exact=(i%4==0?1.:0.)+.3*random();
            high[i]=float(exact);low[i]=float(exact-double(high[i]));
            f[i]=double(high[i])+low[i];d[i]=float(random());h[i]=d[i];
        }
        require(nmCanonicalNeoHookeanResponse(high,low,shear,bulk,d,p,t));
        const auto fit=ref::inverseTranspose(f);
        const double logj=std::log(ref::determinant(f)),v=ref::contract(fit,h);
        const auto ep=ref::add(ref::scale(ref::sub(f,fit),shear),ref::scale(fit,bulk*logj));
        const auto et=ref::add(ref::add(ref::scale(h,shear),ref::scale(fit,bulk*v)),
            ref::scale(ref::multiply(ref::multiply(fit,ref::transpose(h)),fit),shear-bulk*logj));
        double ps=0,pe=0,ts=0,te=0;
        for(unsigned i=0;i<9;++i){
            ps=std::max(ps,std::abs(ep[i]));pe=std::max(pe,std::abs(p[i]-ep[i]));
            ts=std::max(ts,std::abs(et[i]));te=std::max(te,std::abs(t[i]-et[i]));
        }
        stressError=std::max(stressError,pe/ps);tangentError=std::max(tangentError,te/ts);
        require(pe/ps<1e-5&&te/ts<1e-5);
    }
    // Invalid orientation, nonfinite inputs and a missing tangent direction
    // must fail, without introducing a positive-J floor or input clamping.
    for(unsigned i=0;i<9;++i){high[i]=i%4==0?1:0;low[i]=0;}
    high[0]=-1;require(!nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,nullptr));
    high[0]=std::numeric_limits<float>::infinity();
    require(!nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,nullptr));
    high[0]=1;low[4]=std::numeric_limits<float>::quiet_NaN();
    require(!nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,nullptr));
    low[4]=0;require(!nmCanonicalNeoHookeanResponse(high,low,shear,bulk,nullptr,p,t));
    std::cout<<"canonical_neo_hookean=PASS sub_ulp_max_relative="<<maximum
        <<" random_stress_relative="<<stressError<<" random_tangent_relative="<<tangentError<<"\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

