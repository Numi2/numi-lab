from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent; repo=root.parents[2]
base=repo/"docs/evidence/mixed-full-operator-20261108/probe.mm"
work=Path("/Users/numi/Work/mixed-out-of-band-artifacts-20261108");work.mkdir(parents=True,exist_ok=True)
prior=repo/"docs/evidence/mixed-joint-long-rod-20261108"
art=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108")
lib=Path("/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib")
arms={"ordinary":art/"long.metallib","witness":art/"witness.metallib","forced":art/"forced-fallback.metallib","old":Path("/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/joint.metallib")}
expected={"ordinary":prior/"artifacts.sha256","witness":prior/"witness/artifacts.sha256","forced":prior/"fallback/artifacts.sha256","old":prior.parent/"mixed-joint-contact-candidate-20261108/v2-artifacts.sha256"}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for key,p in arms.items():
 assert any(line.split()[0]==digest(p) and line.split(maxsplit=1)[1]==str(p) for line in expected[key].read_text().splitlines()),key
assert digest(lib)=="d8bd1d10177b72004f29360e2ec979b4c72ff3b8da2a708dfddfd0aa85e64e61"
assert all(v.get("match") and v.get("repeat") and v.get("exact_trace") and v.get("capture_noninvasive") for v in json.loads((root/"compat-analysis.json").read_text()).values())
s=base.read_text(); old='rod.model=metalrobo::makeStraightSutureRod(3,.04,mat);rod.defaultState=metalrobo::makeDiscreteElasticRodDefaultState(rod.model);for(auto&p:rod.defaultState.positions){p[0]-=.02;p[1]=.0228;}'
new="""rod.model=metalrobo::makeStraightSutureRod(9,.16,mat);rod.defaultState=metalrobo::makeDiscreteElasticRodDefaultState(rod.model);rod.defaultState.positions={
 {-0.01,0.0228,0.0},{0.01,0.0228,0.0},{0.03,0.0228,0.0},
 {0.04,0.0228,0.01732},{0.029,0.0228,0.034},{0.01,0.0228,0.025},
 {0.01,0.0209,0.005},{-0.01,0.0209,0.005},{-0.03,0.0209,0.005}};"""
assert s.count(old)==1;s=s.replace(old,new)
source=work/"probe.mm";source.write_text(s);exe=work/"probe"
inputs=[base,lib]+list(arms.values())
(root/"inputs.sha256").write_text(''.join(digest(p)+'  '+str(p)+'\n' for p in inputs))
records=[]
def run(label,command):
 p=subprocess.run([str(x) for x in command],cwd=repo,capture_output=True,text=True)
 (root/(label+'.stdout')).write_text(p.stdout);(root/(label+'.stderr')).write_text(p.stderr)
 records.append({'label':label,'command':[str(x) for x in command],'returncode':p.returncode})
 (root/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('build',['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',repo/'include',source,'-L',lib.parent,'-lmetalrobo','-Wl,-rpath,'+str(lib.parent),'-framework','Foundation','-framework','Metal','-o',exe])
(root/'artifacts.sha256').write_text(''.join(digest(p)+'  '+str(p)+'\n' for p in (source,exe)))
for name,shader in arms.items():run(name,[exe,shader,'2'])
