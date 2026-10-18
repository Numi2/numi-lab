#!/usr/bin/env python3
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/Users/numi/numi-human-separation-20261018/C/humanmatter-extension-181/aortic-wall-budget-033')
build=root/'build'
binary=build/'wall_probe'
app=root/'source/wall_probe.mm'
header=root/'source/NumiHumanRestingAorticWall.hpp'
contour=root/'source/actual_nha_contour.inc'
runtime=Path('/Users/numi/numi-human-separation-20261018/C/blood-transfer-208/build/libmetalrobo.dylib')
shader=Path('/Users/numi/numi-human-separation-20261018/C/aortic-vascular-control-020/source/MatterSource.metal')
wrapper=Path('/Users/numi/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py')
python=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python3')
fraction=os.environ.get('NUMI_AORTIC_WALL_FORCE_BUDGET_FRACTION','0.0005')
assert fraction in ('0.0005','0.00025')
out=build/('native-budget-0005' if fraction=='0.0005' else 'native-budget-00025')
out.mkdir(parents=True,exist_ok=True)
trajectory=out/'trajectory'
trajectory.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=os.environ.copy()
env['NUMI_MATTER_TEST_SOURCE']=str(shader)
env['NUMI_MATTER_TEST_SOURCE_SHA256']=sha(shader)
env['DYLD_LIBRARY_PATH']=str(runtime.parent)+(':'+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
env['NM_FEM_NEWTON_TRACE_ROOT']='0'
for k in ('NM_MATTER_FIRST_FAILURE_CAPTURE_ROOT','NM_MATTER_FIRST_FAILURE_CAPTURE_PATH','NM_MATTER_FIRST_FAILURE_ACCEPTED_SNAPSHOT_PATH','NM_MATTER_FIRST_FAILURE_ACCEPTED_SNAPSHOT_SHA256'):
    env.pop(k,None)
env['NUMI_AORTIC_WALL_FORCE_BUDGET_FRACTION']=fraction
env['NUMI_AORTIC_WALL_TRAJECTORY_DIRECTORY']=str(trajectory)
cmd=[str(python),str(wrapper),'--',str(binary)]
inv={'schema':'numi.aortic_wall.force_budget_invocation.v1','app_source':str(app),'app_source_sha256':sha(app),'wall_header':str(header),'wall_header_sha256':sha(header),'contour_include':str(contour),'contour_include_sha256':sha(contour),'binary':str(binary),'binary_sha256':sha(binary),'runtime_library':str(runtime),'runtime_library_sha256':sha(runtime),'matter_source':str(shader),'matter_source_sha256':env['NUMI_MATTER_TEST_SOURCE_SHA256'],'wrapper':str(wrapper),'python':str(python),'command':cmd,'environment':{'NUMI_MATTER_TEST_SOURCE':env['NUMI_MATTER_TEST_SOURCE'],'NUMI_MATTER_TEST_SOURCE_SHA256':env['NUMI_MATTER_TEST_SOURCE_SHA256'],'DYLD_LIBRARY_PATH':env['DYLD_LIBRARY_PATH'],'NM_FEM_NEWTON_TRACE_ROOT':env['NM_FEM_NEWTON_TRACE_ROOT'],'NUMI_AORTIC_WALL_FORCE_BUDGET_FRACTION':fraction,'NUMI_AORTIC_WALL_TRAJECTORY_DIRECTORY':str(trajectory)},'budget_fraction':float(fraction),'budget_semantics':'dt * external nodal force L2 * fraction; local wall-only solver parameter; not a global solver gate','requested_consecutive_steps':8,'first_failure_capture':'disabled; status/snapshot after each attempted step','timeout_seconds':240}
(out/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
t=time.time()
try:
    p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    output=p.stdout; code=p.returncode; timed=False
except subprocess.TimeoutExpired as e:
    output=e.stdout or ''; output=output.decode(errors='replace') if isinstance(output,bytes) else output; code=124; timed=True
(out/'launcher.log').write_text(output)
result={'schema':'numi.aortic_wall.force_budget_result.v1','exit_code':code,'timed_out':timed,'wall_seconds':time.time()-t,'launcher_log_sha256':hashlib.sha256(output.encode()).hexdigest(),'physical_run_result':next((line.split('=',1)[1] for line in output.splitlines() if line.startswith('result=')),None)}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
for line in output.splitlines():
    if line.startswith('wall_step=') or line.startswith('external_nodal_force_l2_n=') or line.startswith('local_force_budget_fraction=') or line.startswith('derived_impulse_budget_ns=') or line.startswith('snapshot_fem_') or line.startswith('max_nodal_displacement_m=') or line.startswith('cumulative_accepted_frozen_load_external_work_j=') or line.startswith('result='):
        print(line)
raise SystemExit(code)
