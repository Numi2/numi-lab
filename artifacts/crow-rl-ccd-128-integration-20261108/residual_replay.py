from pathlib import Path
import csv,hashlib,json,os,subprocess,time
import numpy as np
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-128-integration-20261108'
T=R/'.numi/retained-runtime/ccd-event-capture-20261108/integrated128'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
source=(B/'fixture128.mm').read_text()
old='    std::array<std::array<double,3>,lanes> anchor{};'
new='''    std::ofstream residual(dir/"solver.tsv");residual<<"tick\\tlane\\tsolver_iterations\\timpulse_delta\\tnormal_residual\\tcone_violation\\tfactor_residual\\tmaximum_penetration\\tccd_advances\\tzero_replays\\tpitch_velocity\\tjaw_velocity\\n"<<std::setprecision(12);
    std::array<std::array<double,3>,lanes> anchor{};'''
assert source.count(old)==1;source=source.replace(old,new)
old='            const unsigned tick=begin+count-1;const auto rel=relative(k[head],pos);'
new='''            const unsigned tick=begin+count-1;const auto rel=relative(k[head],pos);
            const auto& st=result.contactStatuses[lane];residual<<tick<<'\\t'<<lane<<'\\t'<<st.solverIterations<<'\\t'<<st.residuals.x<<'\\t'<<st.residuals.y<<'\\t'<<st.residuals.z<<'\\t'<<st.residuals.w<<'\\t'<<st.diagnostics.y<<'\\t'<<st.ccdAdvanceCount<<'\\t'<<st.zeroTimeCCDReplays<<'\\t'<<dv[dv.size()-2]<<'\\t'<<dv.back()<<'\\n';'''
assert source.count(old)==1;source=source.replace(old,new)
(B/'fixture-residual.mm').write_text(source)
plan={'purpose':'Observe accepted final-substep contact solver residuals and joint velocities with exact replay of retained32/64 traces. Diagnose unresolved iteration error before any solver/model change.','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'input_source_sha256':sha(B/'fixture128.mm'),'instrument_source_sha256':sha(B/'fixture-residual.mm'),'native_dylib_sha256':sha(T/'libmetalrobo.dylib'),'native_shader_sha256':sha(T/'metalrobo/MetalRobo.metallib'),'changes':'Two host output-only additions, no world, step, task, action or native changes.','prediction':'All trace, final q/v/scene, complete scene trajectory and summaries byte exact to existing32/64. Residual sizes are exploratory; no claim that a global maximum localizes to jaw.','budget':{'runs':2,'timeout_seconds_each':100,'outputs_less_than_bytes':1000000},'stopping':'First mismatch or native failure stops. No parameter trials, retries or relaxed thresholds.','limits':'Published residuals are final-substep solver diagnostics, not full microstep history or an independent contact certificate.'}
write(A/'residual-replay-plan.json',plan)
subprocess.run(['git','add',str(A/'residual-replay-plan.json'),str(A/'residual_replay.py'),str(A/'results.json'),str(A/'release-analysis.json')],check=True)
subprocess.run(['git','commit','-m','Record native object controls and seal solver diagnostic replay'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
commands=[]
def run(name,cmd,env=None):
    start=time.monotonic()
    with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=100).returncode
    commands.append({'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start})
    write(B/'residual-replay-execution.json',commands);print(json.dumps(commands[-1]),flush=True);assert code==0
run('compile-residual',['clang++','-std=c++20','-O3','-fno-fast-math','-fobjc-arc','-I',R/'include',B/'fixture-residual.mm',T/'libmetalrobo.dylib','-Wl,-rpath,@loader_path/runtime','-framework','Foundation','-o',B/'probe-residual'])
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
result={}
for sub in (32,64):
    name='residual'+str(sub);reference='integrated'+str(sub)
    run(name,[B/'probe-residual',B/name,'0','1',str(sub)],env)
    for file in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):
        assert (B/name/file).read_bytes()==(B/reference/file).read_bytes(),(name,file)
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'solver.tsv').open(),delimiter='\t')]
    result[name]={'physical_replay_exact':True,'phases':{}}
    for lo,hi in ((0,31),(32,63),(64,95),(96,127),(128,159)):
        phase=[r for r in rows if lo<=r['tick']<=hi]
        result[name]['phases'][str((lo,hi))]={k:{'maximum':max(r[k] for r in phase),'median':float(np.median([r[k] for r in phase]))} for k in ('normal_residual','cone_violation','factor_residual','maximum_penetration','ccd_advances','zero_replays')}
write(A/'residual-replay-results.json',{'commands':commands,'arms':result})
print(json.dumps(result,indent=2),flush=True)
