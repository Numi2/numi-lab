#!/usr/bin/env python3
"""Static verifier for prepared 466/467 source154 commands; never runs the native tool."""
from pathlib import Path
import hashlib, json, subprocess
BASE=Path('/Users/home/cardboard-implementation-20261014')
A=BASE/'analysis'
S=BASE/'snapshots/source-cgls-all-controls-newton128-154'
B=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-154')
P466=A/'native466-source154-default64-parity-command-001'
P467=A/'native467-source154-allcontrol-outer128-command-001'
P463=A/'native462-463-right-scale-command-package-001/native-463-command.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def cmd(pkg, n): return load(pkg/f'native-{n}-command.json')
def inputs(argv):
    stop=argv.index('--')
    out=[]
    for i,v in enumerate(argv[:stop]):
        if v=='--input-file': out.append(Path(argv[i+1]))
    return out
def envmap(argv):
    e={}
    for i,v in enumerate(argv[:-1]):
        if v=='--env':
            k,sep,val=argv[i+1].partition('=')
            assert sep and k not in e,(k,argv[i+1])
            e[k]=val
    return e
def argval(argv,k):return argv[argv.index(k)+1]
def norm_probe(argv, basecase, newcase, build_prefix, source_prefix):
    x=list(argv[argv.index('--'):])
    # These identity/path operands legitimately change between immutable runs.
    x[1]='<PROBE>'
    for flag,token in [('--metallib','<METALLIB>'),('--output','<OUTPUT>'),('--liner-material','<LINER>'),('--medium-material','<MEDIUM>')]:
        x[x.index(flag)+1]=token
    x=[v.replace(basecase,'<CASE>').replace(newcase,'<CASE>') if isinstance(v,str) else v for v in x]
    x=[v.replace(str(build_prefix),'<BUILD>') if isinstance(v,str) else v for v in x]
    x=[v.replace(str(source_prefix),'<SOURCE>') if isinstance(v,str) else v for v in x]
    return x
errors=[]
source=json.loads((S/'frozen-source.json').read_text())
build=load(B/'source-build.json')
assert build['success'] and build['targets']==['numi-matter-cardboard-probe']
assert build['source_snapshot']==str(S)
assert build['artifacts_sha256'][str(B/'numi-matter-cardboard-probe')]==sha(B/'numi-matter-cardboard-probe')
assert build['artifacts_sha256'][str(B/'shaders/NumiMatter.metallib')]==sha(B/'shaders/NumiMatter.metallib')
assert source['source_sha256']['matter/tools/cardboard_probe.mm']==sha(S/'matter/tools/cardboard_probe.mm')
commands={466:cmd(P466,466),467:cmd(P467,467)}
for n,pkg in [(466,P466),(467,P467)]:
    a=commands[n]; pre=load(pkg/'preflight.json'); binding=load(pkg/'input-bindings.json')
    assert a[1]==str(S/'matter/tools/cardboard_source_bound_run.py')
    assert argval(a,'--source-snapshot')==str(S)
    assert argval(a,'--runtime-dir')==str(B)
    assert argval(a,'--build-provenance')==str(B/'source-build.json')
    assert a[a.index('--')+1]==str(B/'numi-matter-cardboard-probe')
    assert argval(a,'--metallib')==str(B/'shaders/NumiMatter.metallib')
    assert Path(argval(a,'--case')).name.endswith(str(n))
    assert int(argval(a,'--timeout'))==(4800 if n==466 else 14400)
    files=inputs(a)
    assert len(files)==binding['input_files'].__len__()==pre['input_file_count']
    assert len({str(p) for p in files})==len(files)
    hashes={r['path']:r for r in binding['input_files']}
    for f in files:
        assert f.is_file(),f
        row=hashes[str(f)]
        assert row['sha256']==sha(f) and row['bytes']==f.stat().st_size,(n,f)
    assert not Path(argval(a,'--case')).exists(), 'prepared command must not overwrite an existing case'
    assert pre['all_input_files_present']
    assert pre['native_or_gpu_execution']=='not launched'
    assert binding['runner_not_invoked'] is True
    e=envmap(a)
    assert e['NM_FEM_BASE_CGLS_ITERATIONS']=='64'
    assert e['NM_FEM_BASE_CGLS_RIGHT_SCALE']=='block3'
    assert e['NM_FEM_BASE_CGLS_PRIMARY_AFTER']=='48'
    assert e['NM_FEM_TRUE_RESIDUAL_CERTIFY']=='1'
# 466 exactly repeats numeric selector/capture values from463, with path roots moved.
a463=load(P463); a466=commands[466]
assert envmap(a466)=={k:v.replace('/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463',argval(a466,'--case')) for k,v in envmap(a463).items()}
assert argval(a466,'--newton-iterations')=='64'
assert '--post-release-relax-steps' not in a466
# Compare physical probe options; only binary/material/metallib/output identities are different.
p463=norm_probe(a463,'/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463',argval(a466,'--case'),Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-149'),Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149'))
p466=norm_probe(a466,'/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463',argval(a466,'--case'),B,S)
assert p463==p466,(p463,p466)
# 467 exact all-control values and absence of every singleton capture family.
a467=commands[467]; e467=envmap(a467)
assert e467['NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR']=='all:all'
assert e467['NM_FEM_TRUE_RESIDUAL_ROOT']=='all'
assert e467['NM_FEM_RESIDUAL_ARMIJO_ROOT']=='all'
for key in e467:
    assert not key.startswith('NM_FEM_BASE_CGLS_CAPTURE')
    assert key!='NM_FEM_BASE_CGLS_STATE_CAPTURE'
    assert not key.startswith('NM_FEM_COUPLED_CAPTURE')
    assert not key.startswith('NM_FEM_STATUS5_CAPTURE')
    assert key!='NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'
assert argval(a467,'--newton-iterations')=='128'
assert argval(a467,'--timeout')=='14400'
assert argval(a467,'--post-release-relax-steps')=='3'
assert argval(a467,'--release-steps')=='3'
assert Path(argval(a467,'--output')).is_relative_to(Path(argval(a467,'--case')))
# Runtime guard is external to the runner; record current preflight without dispatch.
df=subprocess.check_output(['df','-k','/Users/home'],text=True).splitlines()[1].split()
free=int(df[3])*1024
minfree=1610612736
assert load(P467/'preflight.json')['disk_policy']['dispatch_preflight_minimum_free_bytes']==minfree
report={'schema':'numi.cardboard.native466_467_static_preflight.v1','status':'pass','runner_invoked':False,'native_or_gpu_execution':'not launched','source_snapshot':str(S),'source_frozen_sha256':sha(S/'frozen-source.json'),'source_build_receipt_sha256':sha(B/'source-build.json'),'probe_sha256':sha(B/'numi-matter-cardboard-probe'),'metallib_sha256':sha(B/'shaders/NumiMatter.metallib'),'466':{'command_sha256':sha(P466/'native-466-command.json'),'numeric_selectors_exactly_match463_after_path_rebinding':True,'outer_newton':'64'},'467':{'command_sha256':sha(P467/'native-467-command.json'),'all_control_selectors':{'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR':'all:all','NM_FEM_TRUE_RESIDUAL_ROOT':'all','NM_FEM_RESIDUAL_ARMIJO_ROOT':'all'},'forbidden_singleton_capture_keys_absent':True,'post_release_relax_steps':3,'timeout_seconds':14400},'live_disk_preflight':{'available_bytes':free,'required_bytes':minfree,'passes':free>=minfree,'abort_below_bytes':536870912,'protected_reserve_bytes':134217728},'input_files':{'466':len(inputs(a466)),'467':len(inputs(a467))}}
print(json.dumps(report,indent=2))
if not report['live_disk_preflight']['passes']: raise SystemExit('live disk preflight is below the required threshold; do not dispatch')
