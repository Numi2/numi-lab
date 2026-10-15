#include <cstdint>
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
            bool femBaseCGLSPrimaryPolicyRequested = false;
            std::uint32_t femBaseCGLSPrimaryStartNewton = 0u;
            if (const char* primaryAfter = std::getenv(
                    "NM_FEM_BASE_CGLS_PRIMARY_AFTER")) {
                bool digitsOnly = primaryAfter[0] != '\0';
                for (const char* cursor = primaryAfter;
                     *cursor != '\0'; ++cursor) {
                    digitsOnly = digitsOnly && *cursor >= '0' && *cursor <= '9';
                }
                char* end = nullptr;
                const unsigned long parsed = std::strtoul(primaryAfter, &end, 10);
                if (!digitsOnly || end == primaryAfter || *end != '\0' ||
                    parsed > 256ul || std::to_string(parsed) != primaryAfter) {
                    std::fprintf(stderr,
                        "fem_base_cgls_primary_policy_rejected={\"schema\":1,\"control_step\":%u,\"reason\":\"primary_after_must_be_canonical_newton_index_0_to_256\"}\n",
                        request.controlStep);
                    diagnostics.message =
                        "base CGLS primary start must be a canonical Newton index from 0 through 256";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
                femBaseCGLSPrimaryPolicyRequested = true;
                femBaseCGLSPrimaryStartNewton =
                    static_cast<std::uint32_t>(parsed);
            }

 const std::uint32_t femBaseCGLSIterations=iterations;
 const bool femBaseCauchyFallbackRequested=fallback;
 const bool femBaseCauchyFallbackAllNewtonIterations=all;
            if (femBaseCGLSPrimaryPolicyRequested &&
                (femBaseCGLSIterations == 0u ||
                 !femBaseCauchyFallbackRequested ||
                 !femBaseCauchyFallbackAllNewtonIterations)) {
                std::fprintf(stderr,
                    "fem_base_cgls_primary_policy_rejected={\"schema\":1,\"control_step\":%u,\"reason\":\"requires_cgls_iterations_and_control_step_all_newton_selector\"}\n",
                    request.controlStep);
                diagnostics.message =
                    "base CGLS primary policy requires CGLS and the selected control-step all-Newton selector";
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }

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
  if(!okay){std::cerr<<"FAIL value="<<(value?value:"<absent>")<<" message="<<result.message<<"\n";std::exit(1);} ++checks;
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
 std::cout<<"PASS actual extracted host parser and admission guard; checks="<<checks<<"\n";
}
