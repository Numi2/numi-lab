from pathlib import Path
import hashlib
import json
import subprocess

O = Path(__file__).resolve().parents[1]
source = O/'candidate/matter/src/runtime.mm'
text = source.read_text()
start = text.index('                const char* separator = std::strchr(baseCauchySelector')
stop = text.index('                    if (request.controlStep != femBaseCauchyFallbackRoot)', start)
body = text[start:stop] + '                }\n'
tr_start = text.index('            if (const char* selector = std::getenv("NM_FEM_TRUE_RESIDUAL_ROOT")) {')
tr_stop = text.index('                    femTrueResidualNoticeLogged = true;', tr_start)
tr_body = text[tr_start:tr_stop]
tr_body = tr_body[:tr_body.index('                    std::fprintf(stderr,')]
# This harness compiles the actual source selector code, preserving its control
# flow. Mock callbacks record admission only; it does not emulate any GPU solve.
cpp = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
struct Parsed { bool valid; std::uint32_t root, iteration; bool allNewton; };
Parsed parse(const char* baseCauchySelector, std::uint32_t control,
             std::uint32_t femBaseCGLSIterations, bool femBaseCGLSPrimaryPolicyRequested) {
    struct { std::uint32_t controlStep; } request{control};
    bool valid = true;
    std::uint32_t femBaseCauchyFallbackRoot=0, femBaseCauchyFallbackIteration=0;
    bool femBaseCauchyFallbackAllNewtonIterations = false;
    auto rejectBaseCauchy = [&](const char*) { valid=false; };
BODY
    return {valid, femBaseCauchyFallbackRoot, femBaseCauchyFallbackIteration,
            femBaseCauchyFallbackAllNewtonIterations};
}
bool trueRoot(const char* value, std::uint32_t control) {
    setenv("NM_FEM_TRUE_RESIDUAL_ROOT", value, 1);
    struct { std::uint32_t controlStep; } request{control};
    TR_BODY
        return false;
    } else { return parsed == control; }
  }
  return false;
}
int main() {
    unsigned checks=0;
    auto check=[&](bool value) { ++checks; assert(value); };
    for (auto control: {0u, 1u, 29u, 4294967295u}) {
        auto p=parse("all:all",control,64,true);
        check(p.valid && p.root==control && p.iteration==0 && p.allNewton);
        check(trueRoot("all",control));
    }
    for (auto value: {"all:0", "all:48", "all:256", "all:257", "all:",
                      "All:all", "alll:all", "all:All", "all:all:all",
                      " all:all", "all:all ", "all", ":all", "", "-1:all",
                      "4294967296:all", "1:257", "1:-1", "1:48:2"})
        check(!parse(value,1,64,true).valid);
    check(!parse("all:all",1,0,true).valid);
    check(!parse("all:all",1,64,false).valid);
    check(!parse("all:all",1,0,false).valid);
    for (auto value: {"0:all", "1:all", "4294967295:all", "1:0", "1:256", "01:048"}) {
        auto p=parse(value,29,0,false);
        check(p.valid);
    }
    auto legacy=parse("19:3",42,0,false);
    check(legacy.valid && legacy.root==19 && legacy.iteration==3 && !legacy.allNewton);
    check(trueRoot("19",19)); check(!trueRoot("19",20));
    for (auto value: {"All", "alll", "all:all", "", "4294967296"})
        check(!trueRoot(value,1));
    std::cout<<"selector_checks="<<checks<<" passed\n";
}
'''.replace('TR_BODY', tr_body).replace('BODY', body)
cpp_path = O/'validation/extracted-selectors.cpp'
cpp_path.write_text(cpp)
compiler = ['/usr/bin/clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
            str(cpp_path),'-o',str(O/'validation/extracted-selectors')]
completed = subprocess.run(compiler, text=True, capture_output=True)
(O/'validation/compile.stdout').write_text(completed.stdout)
(O/'validation/compile.stderr').write_text(completed.stderr)
assert completed.returncode == 0, completed.stderr
run = subprocess.run([str(O/'validation/extracted-selectors')], text=True, capture_output=True)
(O/'validation/run.stdout').write_text(run.stdout)
(O/'validation/run.stderr').write_text(run.stderr)
assert run.returncode == 0, run.stderr
(O/'validation/extracted-selectors.json').write_text(json.dumps({
    'schema':'numi.cardboard.extracted_selector_cpu_contract.v1',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'harness_sha256':hashlib.sha256(cpp_path.read_bytes()).hexdigest(),
    'compile':compiler,'compile_exit_code':completed.returncode,
    'run_exit_code':run.returncode,'stdout':run.stdout,
    'scope':'actual extracted source parser only; no native solve or physical qualification',
},indent=2)+'\n')
print(run.stdout,end='')
