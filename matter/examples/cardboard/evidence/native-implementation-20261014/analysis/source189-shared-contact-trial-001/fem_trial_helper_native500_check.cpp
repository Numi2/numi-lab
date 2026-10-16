#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
struct float3{float x,y,z;float3&operator+=(const float3&b){x+=b.x;y+=b.y;z+=b.z;return *this;}};
inline float3 operator*(float a,const float3&b){return {a*b.x,a*b.y,a*b.z};}
inline float3 operator+(const float3&a,const float3&b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline float3 operator-(const float3&a,const float3&b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
struct alignas(16) float4{float3 xyz;float w;float4()=default;float4(float a,float b,float c,float d):xyz{a,b,c},w(d){}};
struct alignas(16) NMFEMNodeStateGPU{float4 positionAndMass,velocityAndInverseMass,restAndFixed,deltaVelocity,referenceDisplacementAndMode;};
static_assert(sizeof(float4)==16&&sizeof(NMFEMNodeStateGPU)==80);
inline NMFEMNodeStateGPU femTrialNodeAtAlpha(
    const NMFEMNodeStateGPU acceptedState,
    NMFEMNodeStateGPU next,
    const float3 correction,
    const float alpha,
    const float dt,
    const bool updateEligible
) {
    if (!updateEligible || next.restAndFixed.w > 0.5f || !(alpha > 0.0f))
        return next;
    next.velocityAndInverseMass.xyz += alpha * correction;
    next.deltaVelocity.xyz = next.velocityAndInverseMass.xyz -
        acceptedState.velocityAndInverseMass.xyz;
    if (next.referenceDisplacementAndMode.w == 1.0f) {
        next.referenceDisplacementAndMode.xyz =
            acceptedState.referenceDisplacementAndMode.xyz +
            dt * next.velocityAndInverseMass.xyz;
        next.positionAndMass.xyz = next.restAndFixed.xyz +
            next.referenceDisplacementAndMode.xyz;
    } else {
        next.positionAndMass.xyz = acceptedState.positionAndMass.xyz +
            dt * next.velocityAndInverseMass.xyz;
    }
    return next;
}

static std::vector<unsigned char> rd(const char*p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open");return {std::istreambuf_iterator<char>(f),{}};}
static float fl(const std::vector<unsigned char>&b,size_t o){float x;std::memcpy(&x,b.data()+o,4);return x;}
static uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
static uint32_t ord(uint32_t u){return (u&0x80000000u)?~u:(u^0x80000000u);}
static void ck(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(int c,char**v){try{if(c!=7)throw std::runtime_error("args");auto ab=rd(v[1]),cb=rd(v[2]),xb=rd(v[3]),al=rd(v[4]),tb=rd(v[5]),ob=rd(v[6]);
ck(ab.size()==2880&&cb.size()==2880&&xb.size()==576&&al.size()==96&&tb.size()==4&&ob.size()==5376,"extent");
std::array<NMFEMNodeStateGPU,36>A{},C{};std::array<float4,36>X{};std::memcpy(A.data(),ab.data(),2880);std::memcpy(C.data(),cb.data(),2880);std::memcpy(X.data(),xb.data(),576);
uint32_t nodes[]={10,27,28,31};float dt=fl(tb,0), expected[]={1,.5f,.25f,.125f,.0625f,.03125f,.015625f,.0078125f};size_t k=0,exact=0;uint32_t maxUlp=0;
for(size_t row=0;row<8;row++){float alpha=fl(al,4*row);ck(bits(alpha)==bits(expected[row]),"alpha");
for(size_t j=0;j<4;j++){uint32_t n=nodes[j];auto q=femTrialNodeAtAlpha(A[n],C[n],{X[n].xyz.x,X[n].xyz.y,X[n].xyz.z},alpha,dt,true);float got[]={q.positionAndMass.xyz.x,q.positionAndMass.xyz.y,q.positionAndMass.xyz.z};size_t o=row*224+128+j*16;
for(size_t xyz=0;xyz<3;xyz++){uint32_t e;std::memcpy(&e,ob.data()+o+4*xyz,4);uint32_t g=bits(got[xyz]);uint32_t og=ord(g),oe=ord(e);uint32_t ulp=og>oe?og-oe:oe-og;maxUlp=std::max(maxUlp,ulp);exact+=(g==e);ck(ulp<=1,"native endpoint exceeds 1 ULP");k++;}}}
auto s=C[10];auto q=femTrialNodeAtAlpha(A[10],s,{1,2,3},1,dt,false);ck(!std::memcmp(&s,&q,80),"ineligible");
s=C[10];s.restAndFixed.w=1;q=femTrialNodeAtAlpha(A[10],s,{1,2,3},1,dt,true);ck(!std::memcmp(&s,&q,80),"fixed");
s=C[10];q=femTrialNodeAtAlpha(A[10],s,{1,2,3},0,dt,true);ck(!std::memcmp(&s,&q,80),"alpha zero");
NMFEMNodeStateGPU old{},n{};old.velocityAndInverseMass={.25f,.5f,-1,1};old.referenceDisplacementAndMode={.25f,-.5f,1,1};n.velocityAndInverseMass={1,2,3,1};n.restAndFixed={10,20,30,0};n.referenceDisplacementAndMode={.25f,-.5f,1,1};q=femTrialNodeAtAlpha(old,n,{2,-4,8},.5f,.125f,true);
ck(bits(q.velocityAndInverseMass.xyz.x)==bits(2)&&bits(q.velocityAndInverseMass.xyz.y)==bits(0)&&bits(q.velocityAndInverseMass.xyz.z)==bits(7),"ref velocity");
ck(bits(q.deltaVelocity.xyz.x)==bits(1.75f)&&bits(q.deltaVelocity.xyz.y)==bits(-.5f)&&bits(q.deltaVelocity.xyz.z)==bits(8),"ref delta");
ck(bits(q.referenceDisplacementAndMode.xyz.x)==bits(.5f)&&bits(q.referenceDisplacementAndMode.xyz.y)==bits(-.5f)&&bits(q.referenceDisplacementAndMode.xyz.z)==bits(1.875f),"ref displacement");
ck(bits(q.positionAndMass.xyz.x)==bits(10.5f)&&bits(q.positionAndMass.xyz.y)==bits(19.5f)&&bits(q.positionAndMass.xyz.z)==bits(31.875f),"ref position");
std::cout<<"extracted_helper_native500_endpoint_components_within_1ulp="<<k<<"/96 exact="<<exact<<" max_ulp="<<maxUlp<<"\nineligible_fixed_zero_alpha_reference_displacement_controls=pass\n";return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
