from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/source143-host-policy-parser-check-001'
p=B/'analysis/base-cgls-primary-policy-overlay-143-001/overlay/matter/src/runtime.mm'
s=p.read_text();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=s.index('            bool femBaseCGLSPrimaryPolicyRequested = false;');b=s.index('            std::uint32_t femBaseCGLSIterations = 0u;',a)
c=s.index('            if (femBaseCGLSPrimaryPolicyRequested &&');d=s.index('            const auto emitFEMNewtonTraceSkip',c)
first=s[a:b];second=s[c:d]
O.mkdir(exist_ok=False)
head='''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
struct D { std::string message; bool requested=false; std::uint32_t index=999; };
struct Ownership { bool preDynamicsOpen=true; } globalOwnership;
struct Request { unsigned controlStep=1; } request;
D run(unsigned iterations,bool fallback,bool all) {
 D diagnostics; globalOwnership={}; auto* ownership=&globalOwnership;
'''
mid='''
 const std::uint32_t femBaseCGLSIterations=iterations;
 const bool femBaseCauchyFallbackRequested=fallback;
 const bool femBaseCauchyFallbackAllNewtonIterations=all;
'''
tail='''
 diagnostics.requested=femBaseCGLSPrimaryPolicyRequested;
 diagnostics.index=femBaseCGLSPrimaryStartNewton;
 return diagnostics;
}
int main() {
 unsigned checks=0;
 auto test=[&](const char* value,unsigned iterations,bool fallback,bool all,bool valid,bool requested,unsigned index) {
  if(value) setenv("NM_FEM_BASE_CGLS_PRIMARY_AFTER",value,1); else unsetenv("NM_FEM_BASE_CGLS_PRIMARY_AFTER");
  D result=run(iterations,fallback,all);
  bool actualValid=result.message.empty() && globalOwnership.preDynamicsOpen;
  bool okay=actualValid==valid && (!valid || (result.requested==requested && (!requested || result.index==index)));
  if(!valid)okay=okay&&!globalOwnership.preDynamicsOpen&&!result.message.empty();
  if(!okay){std::cerr<<"FAIL value="<<(value?value:"<absent>")<<" message="<<result.message<<"\\n";std::exit(1);} ++checks;
 };
 test(nullptr,0,false,false,true,false,0);
 test("0",64,true,true,true,true,0);
 test("48",64,true,true,true,true,48);
 test("256",32,true,true,true,true,256);
 for(const char* value:{"","00","048","+48","-1","257","48x"," 48","48 ","999999999999999999999999999999999999999999"})
  test(value,64,true,true,false,false,0);
 test("48",0,true,true,false,false,0);
 test("48",64,false,true,false,false,0);
 test("48",64,true,false,false,false,0);
 std::cout<<"PASS actual extracted host parser and admission guard; checks="<<checks<<"\\n";
}
'''
cpp=O/'actual_host_parser.cpp';cpp.write_text(head+first+mid+second+tail)
commands=[['/usr/bin/c++','-std=c++17','-Wall','-Wextra','-Werror',str(cpp),'-o',str(O/'actual_host_parser')],[str(O/'actual_host_parser')]]
results=[]
for i,cmd in enumerate(commands):
 r=subprocess.run(cmd,capture_output=True,text=True);(O/f'stdout-{i}.log').write_text(r.stdout);(O/f'stderr-{i}.log').write_text(r.stderr);results.append({'argv':cmd,'exit_code':r.returncode});assert r.returncode==0,r.stderr
report={'schema':'numi.cardboard.actual_host_policy_parser_check.v1','source':str(p),'source_sha256':sha(p),'extracted_blocks_sha256':hashlib.sha256((first+second).encode()).hexdigest(),'commands':results,'checks':17,'pass':True,'coverage':'Actual unchanged extracted new host parsing/admission blocks; stub ownership/diagnostic carriers only. Existing broader CGLS scope parser and GPU sequencing are separate.','physical_steps':0,'physical_validation':False}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'check.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.host_parser_check_seal.v1','files':{x.name:sha(x) for x in O.iterdir() if x.is_file()}},indent=2)+'\n');print(results[-1],(O/'stdout-1.log').read_text(),sha(O/'manifest.json'))
