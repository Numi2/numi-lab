from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,os,sys,difflib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-qualification-checkpoint-proof-221-003';s=q/'snapshots/source-diagnostic-epsilon-replay-229-001';d=q/'analysis/source229-diagnostic-epsilon-replay-001';b=q/'build-source229-diagnostic-epsilon-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());assert sha(base/'frozen-source.json')=='fb885540a2f096ee70df5a03701a8d2710832c6df402e8e097ff95c736e7c1a5';s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in f['source_sha256'].items():
 assert sha(base/n)==h;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
p=s/'matter/tools/cardboard_coupon_resume.hpp';text=p.read_text()
text=text.replace('#include <cstdint>','#include <cstdint>\n#include <cmath>\n#include <cstdlib>')
text=text.replace('change only these five selectors','change only these six observer controls')
text=text.replace('std::array<std::string_view, 5u>\n    kRegisteredDiagnosticReplaySelectorKeys','std::array<std::string_view, 6u>\n    kRegisteredDiagnosticReplaySelectorKeys')
needle='        "NM_FEM_OPERATOR_FD_SELECTOR",';assert text.count(needle)==1;text=text.replace(needle,'        "NM_FEM_OPERATOR_FD_EPSILON_SCALE",\n'+needle)
needle='    if (const std::string* state = find("NM_FEM_BASE_CGLS_STATE_CAPTURE")) {'
addition="""    if (const std::string* scale = find("NM_FEM_OPERATOR_FD_EPSILON_SCALE")) {
        if (find("NM_FEM_OPERATOR_FD_SELECTOR") == nullptr ||
            find("NM_FEM_OPERATOR_FD_CAPTURE_PATH") == nullptr)
            throw std::invalid_argument(
                "operator-FD epsilon scale requires its paired capture path and selector");
        char* end = nullptr;
        const float parsed = std::strtof(scale->c_str(), &end);
        bool allowed = !scale->empty() &&
            scale->find_first_of(" \\t\\r\\n") == std::string::npos &&
            end == scale->c_str() + scale->size() &&
            std::isfinite(parsed);
        if (allowed) {
            allowed = false;
            for (int exponent = -10; exponent <= 4; ++exponent)
                if (parsed == std::ldexp(1.0f, exponent)) {
                    allowed = true;
                    break;
                }
        }
        if (!allowed)
            throw std::invalid_argument(
                "operator-FD epsilon scale must be a finite power of two in [1/1024,16]");
    }

"""
assert text.count(needle)==1;text=text.replace(needle,addition+needle);p.write_text(text)
p=s/'matter/tools/cardboard_probe.mm';text=p.read_text();assert text.count('five registered observer selectors')==1;text=text.replace('five registered observer selectors','six registered observer controls');p.write_text(text)
p=s/'matter/tools/cardboard_coupon_resume_check.cpp';text=p.read_text().replace('kRegisteredDiagnosticReplaySelectorKeys.size() == 5u','kRegisteredDiagnosticReplaySelectorKeys.size() == 6u')
needle='    auto wrongPathOrder = sourcePaths;'
addition="""    const std::vector<std::pair<std::string, std::string>> fdReplay{
        {"NM_FEM_OPERATOR_FD_SELECTOR", "19:3"},
        {"NM_FEM_OPERATOR_FD_CAPTURE_PATH", "/fresh/fd"},
    };
    for (const char* allowedScale : {
            "0.0009765625", "0.001953125", "0.00390625", "0.0078125",
            "0.015625", "0.03125", "0.0625", "0.125", "0.25",
            "0.5", "1", "2", "4", "8", "16"}) {
        auto changed = fdReplay;
        changed.emplace_back("NM_FEM_OPERATOR_FD_EPSILON_SCALE", allowedScale);
        validateDiagnosticReplaySelectors(changed, 15u, 2u);
        assert(couponDiagnosticReplaySettingsMatch(fdReplay, changed));
        assert(canonicalCouponRuntimeSettings(fdReplay) !=
               canonicalCouponRuntimeSettings(changed));
    }
    for (const char* invalidScale : {
            "", "0", "-1", "nan", "inf", "0.003", "32",
            "0.00048828125", "0.00390625x", " 0.00390625"}) {
        auto changed = fdReplay;
        changed.emplace_back("NM_FEM_OPERATOR_FD_EPSILON_SCALE", invalidScale);
        mustReject([&] { validateDiagnosticReplaySelectors(changed, 15u, 2u); });
    }
    mustReject([&] { validateDiagnosticReplaySelectors({
        {"NM_FEM_OPERATOR_FD_EPSILON_SCALE", "0.00390625"}}, 15u, 2u); });
    mustReject([&] { validateDiagnosticReplaySelectors({
        {"NM_FEM_OPERATOR_FD_SELECTOR", "19:3"},
        {"NM_FEM_OPERATOR_FD_EPSILON_SCALE", "0.00390625"}}, 15u, 2u); });
    auto fdWithWrongSolver = fdReplay;
    fdWithWrongSolver.emplace_back("NM_FEM_STRICT_INNER_FORCING", "0");
    assert(!couponDiagnosticReplaySettingsMatch(fdReplay, fdWithWrongSolver));
    auto fdWithDuplicateScale = fdReplay;
    fdWithDuplicateScale.emplace_back("NM_FEM_OPERATOR_FD_EPSILON_SCALE", "1");
    fdWithDuplicateScale.emplace_back("NM_FEM_OPERATOR_FD_EPSILON_SCALE", "0.5");
    mustReject([&] { validateDiagnosticReplaySelectors(
        fdWithDuplicateScale, 15u, 2u); });

"""
assert text.count(needle)==1;text=text.replace(needle,addition+needle);p.write_text(text)
changed=[n for n,h in f['source_sha256'].items() if sha(s/n)!=h];assert len(changed)==3
f['source_sha256']={n:sha(s/n) for n in sorted(f['source_sha256'])}
f['source229_epsilon_replay']={'parent':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'changed':changed,'reason':'Native564 failed before resumed transactions because observer epsilon scale was omitted from diagnostic-replay allowlist; allow only existing bounded power-of-two epsilon control with paired future-root capture. Ordinary resume remains strict. Runtime/Metal/physical settings unchanged.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
(d/'source.patch').write_text(''.join(''.join(difflib.unified_diff((base/n).read_text().splitlines(True),(s/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in changed))
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','2','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-coupon-resume-check']
(d/'command.json').write_text(json.dumps({'argv':cmd,'source_frozen_sha256':sha(s/'frozen-source.json'),'changed':changed,'physical_operator_or_material_law_changed':False,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=s)
tests=[]
if r.returncode==0:
 commands=[['/usr/bin/clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(s/'matter/tools/cardboard_coupon_resume_check.cpp'),'-o',str(d/'assert-enabled-coupon-resume-check')],[str(d/'assert-enabled-coupon-resume-check')],[str(b/'numi-matter-cardboard-probe'),'--verify-coupon-source-runner',str(q/'solver/fresh-source210-strict-inner-full-coupon-545')]]
 for i,a in enumerate(commands):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:rc=subprocess.run(a,stdout=out,stderr=err,cwd=s,env=env).returncode
  tests.append({'argv':a,'exit_code':rc,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if rc:break
rr={'schema':'numi.cardboard.source229-epsilon-replay.v1','build_exit_code':r.returncode,'tests':tests,'all_pass':r.returncode==0 and len(tests)==3 and all(x['exit_code']==0 for x in tests),'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'source_patch_sha256':sha(d/'source.patch'),'changed':changed,'physical_operator_or_material_law_changed':False,'all_source_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'gpu_execution':False,'physical_transactions':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr),flush=True);sys.exit(int(not rr['all_pass']))
