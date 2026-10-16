import argparse,hashlib,json,pathlib,subprocess
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def extract(t,m):
 s=t.find(m); b=t.find("{",s); d=0
 if s<0: raise RuntimeError("helper missing")
 for i in range(b,len(t)):
  if t[i]=="{": d+=1
  elif t[i]=="}":
   d-=1
   if d==0:return t[s:i+1]
 raise RuntimeError("unbalanced helper")
ap=argparse.ArgumentParser()
for x in ("source","case","output","package"): ap.add_argument("--"+x,required=True)
a=ap.parse_args(); src=pathlib.Path(a.source); case=pathlib.Path(a.case); pkg=pathlib.Path(a.package)
names=("accepted-nodes.bin","candidate-nodes.bin","correction.bin","alphas.bin","timestep.bin","manifest.json")
fs={n:case/n for n in names}; j=json.loads(fs[names[-1]].read_text())
r=next((r for r in j["records"] if r.get("contact_slot")==7),None)
if not r or r["nodes"]!=[10,27,28,31]: raise SystemExit("slot7 case mismatch")
eh={"accepted-nodes.bin":"928da924ab7b58bfad074124a460aa807687c6466c6d8e2295a8af349503f39d","candidate-nodes.bin":"cf79a4e8b241e221a321f52124b5c691a2dbfc5e2833e3ca6a165eeafac4a146","correction.bin":"04dacef2468fa1216b86444d21bb0ab9a4145086c5cc8acd55b0681401b3cf08","alphas.bin":"ae4c873080e3027604a3d81ec76c7fd26db9f464f3a769b7a0642a2af0aa1e95","timestep.bin":"0f2f3591427956b004ca17393783192e0f81858c06bb5496aa2cea004f1ee863"}
for n,h in eh.items():
 if sha(fs[n])!=h: raise SystemExit("hash mismatch "+n)
if sha(a.output)!="cb1f014b70d5bd21e08f385e4fbea2c6d857328b2fd4a9c0506d64cb18ff0237": raise SystemExit("output hash mismatch")
fem=src/"matter/src/metal/fem.metalinc"; text=fem.read_text(); fn=extract(text,"inline NMFEMNodeStateGPU femTrialNodeAtAlpha(")
(pkg/"extracted_femTrialNodeAtAlpha.inc").write_text(fn+"\n")
pre=r"""#include <array>
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
"""
post=r"""
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
"""
cpp=pkg/"fem_trial_helper_native500_check.cpp";cpp.write_text(pre+fn+"\n"+post);exe=pkg/"fem_trial_helper_native500_check"
cc=["clang++","-std=c++17","-O2","-fno-fast-math","-ffp-contract=off","-Wall","-Wextra","-Werror",str(cpp),"-o",str(exe)]
subprocess.run(cc,check=True)
run=[str(exe),*[str(fs[n]) for n in names[:5]],a.output]
res=subprocess.run(run,check=True,capture_output=True,text=True);print(res.stdout,end="")
ct=src/"matter/src/metal/contact.metalinc"; rt=src/"matter/src/runtime.mm"; ctext=ct.read_text(); rtext=rt.read_text()
checks={"fem_apply":text.count("femTrialNodeAtAlpha(accepted[global], next")==2,"contact_limiter":"femTrialNodeAtAlpha(\n                accepted[nodeGlobal]" in ctext,"endpoint_sweep":"certifyDeformableNewtonSweepToEndpoint(\n            contact.identity.z, featurePosition, featureEndpoint" in ctext,"armijo_trials":"femTrialNodeAtAlpha(accepted[globalNode],\n                                candidate[globalNode]" in ctext,"runtime17":"atIndex:17u];" in rtext,"runtime18":"atIndex:18u];" in rtext,"runtime19":"atIndex:19u];" in rtext,"compensated_vt":"contactCompensatedVertexTriangleSeparation(" in ctext}
if not all(checks.values()): raise SystemExit("source contract failure "+repr(checks))
receipt={"schema":"source189.extracted_fem_trial_helper_check.v1","source_hashes":{"fem":sha(fem),"contact":sha(ct),"runtime":sha(rt)},"helper_sha256":hashlib.sha256(fn.encode()).hexdigest(),"case_manifest_sha256":sha(fs["manifest.json"]),"native500_output_sha256":sha(a.output),"input_sha256":{n:sha(p) for n,p in fs.items()},"support_nodes":r["nodes"],"endpoint_components_checked":96,"source_contract":checks,"compile_argv":cc,"run_argv":run,"stdout":res.stdout}
(pkg/"helper-check-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
