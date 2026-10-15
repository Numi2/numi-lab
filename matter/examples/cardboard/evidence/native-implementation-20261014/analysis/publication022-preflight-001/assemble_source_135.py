from pathlib import Path
import difflib, hashlib, json, subprocess
B=Path('/Users/home/cardboard-implementation-20261014'); R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-base-state-cauchy-armijo-133'; N=B/'snapshots/source-cauchy-positive-mass-admission-135'
O=B/'analysis/source135-cauchy-mass-admission-001'; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text()); u=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**f['source_sha256'],**u['files']}.items():assert sha(S/p)==h,p
assert not N.exists(); O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True); (O/'clone.log').write_text(q.stdout+q.stderr); assert q.returncode==0
p='matter/src/runtime.mm'; before=(N/p).read_text()
needle='''                    } else if (const char* reason = femResidualArmijoScope()) {
                        rejectBaseCauchy(reason);'''
replacement='''                    } else if (
                        state.femAuthoredMassLayout.size() !=
                            state.dispatch.femNodeCount ||
                        std::any_of(state.femAuthoredMassLayout.begin(),
                            state.femAuthoredMassLayout.end(),
                            [](const auto& mass) {
                                return !std::isfinite(mass[0]) ||
                                    !(mass[0] > 0.0f);
                            })) {
                        // The residual eliminates nonpositive masses while
                        // the operator eliminates constraint tags. Keep the
                        // fallback's free row spaces identical, using the
                        // immutable authored masses bound by runtime restore.
                        rejectBaseCauchy(
                            "requires_positive_finite_immutable_nodal_masses");
                    } else if (const char* reason = femResidualArmijoScope()) {
                        rejectBaseCauchy(reason);'''
assert before.count(needle)==1; after=before.replace(needle,replacement); (N/p).write_text(after)
f['source_sha256'][p]=sha(N/p); f['base_snapshot']=str(S)
f['scope']+=' Source135 adds only fail-closed positive-finite immutable mass admission to the opt-in fallback; disabled production and already admitted positive-mass fixtures are unchanged.'
(N/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
for name,h in {**f['source_sha256'],**u['files']}.items():assert sha(N/name)==h,name
(O/'source135.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+p,tofile='b/'+p)))
(O/'integration.json').write_text(json.dumps(dict(source=str(N),base=str(S),changed_files={p:dict(base_sha256=sha(S/p),candidate_sha256=sha(N/p))},metal_sources_unchanged=True,scope='Reject zero, negative and nonfinite authored mass layouts only when the bounded Cauchy fallback is requested on its selected root. No operator or tolerance changes.',physical_steps=0),indent=2)+'\n')
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-135'),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe']
(B/'build-command-135.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build135-runner.log').open('x') as out:code=subprocess.call(cmd,stdout=out,stderr=subprocess.STDOUT)
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_build_sha256=sha(R/'build-cardboard-provenance-135/source-build.json')),indent=2)+'\n');print('build135',code,flush=True);raise SystemExit(code)
