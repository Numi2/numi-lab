#!/usr/bin/env python3
"""Native Metal parity of actual generated scalar programs, including failures."""
from pathlib import Path
import argparse,re,subprocess,json,shutil
a=argparse.ArgumentParser()
a.add_argument("before",type=Path);a.add_argument("after",type=Path)
a.add_argument("--bundles",action="store_true",help="compare exact shared read-only derivative bundles")
a.add_argument("--include",type=Path,required=True)
a.add_argument("--output",type=Path,required=True)
v=a.parse_args();v.output.mkdir(parents=True,exist_ok=True)
def programs(path):
    text=path.read_text();found={}
    for m in re.finditer(r"template <typename CachePointer>\ninline float (nmNative(?:Direct)?P\d+)\(",text):
        first=text.index("{",m.end());depth=1;end=first+1
        while depth:depth+=(text[end]=="{")-(text[end]=="}");end+=1
        found[m.group(1)]=text[m.start():end]
    return found
before,after=programs(v.before),programs(v.after)
assert before.keys()==after.keys() and before
pc=max(16,max(int(x) for s in before.values() for x in re.findall(r"p\[(\d+)\]",s))+1)
cc=max(16,max(int(x) for s in before.values() for x in re.findall(r"c\[(\d+)\]",s))+1)
fused={}
if v.bundles:
    text=v.after.read_text()
    for m in re.finditer(r"// Exact read-only output bundle:([^\n]+)\ntemplate <typename CachePointer>\ninline void (nmNativeFusedColumn\w+)\(",text):
        first=text.index("{",m.end());end=first+1;depth=1
        while depth:depth+=(text[end]=="{")-(text[end]=="}");end+=1
        rows=m.group(1).split()
        assert len(rows)==9 and all(row in before for row in rows)
        fused[m.group(2)]=(rows,text[m.start():end])
    assert fused, "no eligible fused bundles"
    after={name:item[1] for name,item in fused.items()}
shader='#include <metal_stdlib>\n#include "numi/matter/fiber_exp_linear.h"\nusing namespace metal;\nconstant float kGeometryFloor=1e-12f;\n'
for namespace,src in [("Before",before),("After",after)]:
    shader+='\nnamespace '+namespace+' {\n'+'\n'.join(src.values())+'\n}\n'
wrapper=r"""
kernel void guardINDEX(device const float* params [[buffer(0)]],
 device uint4* out [[buffer(1)]],uint trial [[thread_position_in_grid]]) {
 if(trial>=128u)return;
 device const float* p=params+trial*PCu;
 float ca[CC],cb[CC],f[9],d[9],r[9];
 for(uint i=0;i<CCu;++i)ca[i]=.01f+float((i+trial)%11u)*.001f;
 for(uint i=0;i<9u;++i){
  f[i]=(i%4u==0u?1.0f:0.0f)+float(int((i+trial)%7u)-3)*.001f;
  d[i]=float(int((i+trial)%5u)-2)*.01f;
  r[i]=float(int((i+trial)%3u)-1)*.001f;
 }
 float dt=1e-4f,temp=310.0f;
 if(trial>=16u){
  const float bad[8]={as_type<float>(0x7fc00000u),INFINITY,-INFINITY,
   as_type<float>(0x7f7fffffu),as_type<float>(0xff7fffffu),1e-35f,-1e-35f,0.0f};
  const float poison=bad[(trial/7u)%8u];
  switch(trial%7u){
  case 0:f[(trial/7u)%9u]=poison;break;
  case 1:d[(trial/7u)%9u]=poison;break;
  case 2:r[(trial/7u)%9u]=poison;break;
  case 4:ca[(trial/7u)%CCu]=poison;break;
  case 5:dt=poison;break;case 6:temp=poison;break;
  }
 }
 for(uint i=0;i<CCu;++i)cb[i]=ca[i];
 bool va=true,vb=true,na=false,nb=false;
 const float x=Before::NAME(p,f,d,r,dt,temp,ca,va,na);
 const float y=After::NAME(p,f,d,r,dt,temp,cb,vb,nb);
 bool exact=va==vb&&na==nb&&as_type<uint>(x)==as_type<uint>(y);
 for(uint i=0;i<CCu;++i)exact=exact&&as_type<uint>(ca[i])==as_type<uint>(cb[i]);
 out[trial]=uint4(exact?1u:0u,va?1u:0u,va?0u:1u,na?1u:0u);
}
"""
cases=4096 if v.bundles else 128
entries=list(fused) if v.bundles else list(before)
if v.bundles:
    wrapper=wrapper.replace("float ca[CC],cb[CC],f[9],d[9],r[9];",
        "float ca[CC],cb[CC],f[9],d[9],r[9];\n"
        "uint rng=trial*747796405u+2891336453u;\n")
    wrapper=wrapper.replace("float dt=1e-4f,temp=310.0f;",
        "for(uint i=0;i<9u;++i){rng=rng*1664525u+1013904223u;"
        "d[i]=float(int(rng>>16u)-32768)*1e-4f;"
        "rng=rng*1664525u+1013904223u;"
        "f[i]+=float(int(rng>>16u)-32768)*1e-5f;}\n"
        "float dt=1e-4f,temp=310.0f;")
    wrapper=wrapper.replace("trial>=16u","trial>=2048u")
    oldcall=""" const float x=Before::NAME(p,f,d,r,dt,temp,ca,va,na);
 const float y=After::NAME(p,f,d,r,dt,temp,cb,vb,nb);
 bool exact=va==vb&&na==nb&&as_type<uint>(x)==as_type<uint>(y);"""
    newcall=""" float xa[9],xb[9];
 for(uint i=0;i<9u;++i)xa[i]=xb[i]=as_type<float>(0x7fc00100u+i);
 CALLS
 After::NAME(p,f,d,r,dt,temp,cb,xb,vb,nb);
 bool exact=va==vb&&na==nb;
 for(uint i=0;i<9u;++i)exact=exact&&as_type<uint>(xa[i])==as_type<uint>(xb[i]);"""
    assert oldcall in wrapper
    wrapper=wrapper.replace(oldcall,newcall)
for idx,name in enumerate(entries):
    code=wrapper
    if v.bundles:
        calls="do{"
        for row,program in enumerate(fused[name][0]):
            calls+=f"xa[{row}]=Before::{program}(p,f,d,r,dt,temp,ca,va,na);if(!va)break;"
        calls+="}while(false);"
        code=code.replace("CALLS",calls)
    shader+=code.replace("INDEX",str(idx)).replace("NAME",name).replace("PC",str(pc)).replace("CC",str(cc)).replace("128u",str(cases)+"u")
metal=v.output/'finite-guards.metal';metal.write_text(shader)
host=r"""
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
struct R{unsigned exact,valid,rejected,nonfinite;};
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==2,"metallib required");
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
 need([[device name] hasPrefix:@"Apple"],"physical Apple GPU required");
 NSError* error=nil;
 id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:
  [NSString stringWithUTF8String:argv[1]]] error:&error];need(lib!=nil,"cannot load test library");
 id<MTLCommandQueue> queue=[device newCommandQueue];
 std::vector<float> params(CASES u*PCu);
 float bad[]={std::numeric_limits<float>::quiet_NaN(),INFINITY,-INFINITY,
  std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),1e-35f,-1e-35f,0.0f};
 for(unsigned t=0;t<CASES;++t){
  for(unsigned i=0;i<PCu;++i)params[t*PCu+i]=1.01f+float((i+t)%7)*.013f;
  if(t>=THRESHOLD&&t%7==3)params[t*PCu+(t/7)%PCu]=bad[(t/7)%8];
 }
 id<MTLBuffer> input=[device newBufferWithBytes:params.data()
  length:params.size()*sizeof(float) options:MTLResourceStorageModeShared];
 id<MTLBuffer> output=[device newBufferWithLength:COUNTu*CASES u*sizeof(R)
  options:MTLResourceStorageModeShared];need(input!=nil&&output!=nil,"buffer allocation failed");
 id<MTLCommandBuffer> cmd=[queue commandBuffer];
 id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
 for(unsigned i=0;i<COUNTu;++i){
  NSString* name=[NSString stringWithFormat:@"guard%u",i];
  id<MTLFunction> fn=[lib newFunctionWithName:name];need(fn!=nil,"kernel missing");
  id<MTLComputePipelineState> pso=[device newComputePipelineStateWithFunction:fn error:&error];
  need(pso!=nil,"test pipeline creation failed");
  [enc setComputePipelineState:pso];
  [enc setBuffer:input offset:0 atIndex:0];
  [enc setBuffer:output offset:i*CASES u*sizeof(R) atIndex:1];
  [enc dispatchThreads:MTLSizeMake(CASES,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 }
 [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
 need(cmd.status==MTLCommandBufferStatusCompleted,"Metal command failed");
 auto* result=static_cast<const R*>(output.contents);
 unsigned valid=0,rejected=0,nonfinite=0;
 for(unsigned i=0;i<COUNTu*CASES u;++i){
  if(result[i].exact!=1u){std::cerr<<"program_index="<<i/CASES u<<" trial="<<i%CASES u<<"\n";
    throw std::runtime_error("native finite-guard result/flag/cache parity failed");}
  valid+=result[i].valid;rejected+=result[i].rejected;nonfinite+=result[i].nonfinite;
 }
 std::cout<<"PASS Metal MODE parity device="<<[device name].UTF8String
  <<" programs=COUNT cases="<<COUNTu*CASES u<<" valid="<<valid<<" rejected="<<rejected
  <<" nonfinite="<<nonfinite<<" exact_return_flags_and_cache=true\n";
 return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}}
""".replace("PC",str(pc)).replace("COUNT",str(len(entries))).replace("CASES u",str(cases)+"u").replace("CASES",str(cases)).replace("THRESHOLD","2048" if v.bundles else "16").replace("MODE","read-only bundle" if v.bundles else "scalar finite-guard")
mm=v.output/'finite-guards.mm';mm.write_text(host)
located=shutil.which('metal')
binpath=Path(located).parent if located else Path.home()/'Library/Developer/Toolchains/Metal.xctoolchain/usr/bin'
assert (binpath/'metal').is_file() and (binpath/'metallib').is_file(), 'Metal toolchain is required'
air=v.output/'finite-guards.air';lib=v.output/'finite-guards.metallib';exe=v.output/'finite-guards'
subprocess.run([str(binpath/'metal'),'-std=metal3.1','-fno-fast-math',
 '-I'+str(v.include),'-c',str(metal),'-o',str(air)],check=True)
subprocess.run([str(binpath/'metallib'),str(air),'-o',str(lib)],check=True)
subprocess.run(['/usr/bin/clang++','-std=c++20','-O2','-fobjc-arc',str(mm),
 '-framework','Foundation','-framework','Metal','-o',str(exe)],check=True)
subprocess.run([str(exe),str(lib)],check=True)
print(json.dumps({"scope":("actual generated read-only tangent bundles" if v.bundles else "actual generated scalar Metal programs")+", not a simulation benchmark",
 "programs":len(entries),"mode":"bundles" if v.bundles else "scalar","parameter_slots":pc,"cache_slots":cc}))
