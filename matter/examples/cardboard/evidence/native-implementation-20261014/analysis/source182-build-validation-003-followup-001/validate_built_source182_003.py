#!/usr/bin/env python3
"""Validate the already-built source182-003 without overwriting failed attempt-001."""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, platform, re, subprocess, time

B = pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
S = B / 'snapshots/source-admission-encoder-182-003'
D = B / 'build-source182-003'
O = B / 'analysis/source182-build-validation-003-followup-001'
P = B / 'analysis/coupon-diagnostic-replay-overlay-182-003'
PYTHON = pathlib.Path('/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13')
TARGETS = ['numi-matter-cardboard-probe','numi-matter-cardboard-warm-start-check',
           'numi-matter-cardboard-shipping-snapshot-check','numi-matter-cardboard-coupon-resume-check']
CTEST_REGEX = r'^matter\.(compiler\.cardboard_(shipping_snapshot|coupon_resume)|cli\.cardboard_argument_admission)$'

def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)
def run(argv: list[str], env: dict[str,str], timeout: int) -> dict:
    start=time.monotonic(); p=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,check=False)
    data=p.stdout
    return {'argv':argv,'returncode':p.returncode,'wall_seconds':time.monotonic()-start,
            'output':data.decode('utf-8','replace'),'output_sha256':hashlib.sha256(data).hexdigest()}
def file_map(root: pathlib.Path, mapping: dict[str,str], label: str) -> None:
    for rel,expected in mapping.items():
        p=root/rel
        require(p.is_file() and not p.is_symlink(),f'{label} missing/nonregular: {rel}')
        require(sha(p)==expected,f'{label} SHA mismatch: {rel}')
def main() -> int:
    require(not O.exists(),'follow-up validation directory already exists')
    require(S.is_dir() and D.is_dir() and P.is_dir(),'source/build/overlay path missing')
    frozen=json.loads((S/'frozen-source.json').read_text())
    support=json.loads((S/'configure-support-source.json').read_text())
    build=json.loads((D/'source-build.json').read_text())
    require(len(frozen['source_sha256'])==805 and len(support['files'])==1034,'source inventory counts changed')
    file_map(S,frozen['source_sha256'],'selected source')
    file_map(S,support['files'],'configure-support source')
    require(build['success'] and build['source_snapshot']==str(S) and build['build_dir']==str(D) and build['targets']==TARGETS,'source-build provenance mismatch')
    require(build['source_sha256']==frozen['source_sha256'],'build/source selected hashes differ')
    for raw,expected in build['artifacts_sha256'].items():
        p=pathlib.Path(raw); require(p.is_file() and not p.is_symlink() and sha(p)==expected,f'build artifact mismatch: {raw}')
    overlay=json.loads((P/'overlay-manifest.json').read_text())
    require(sha(P/'overlay-manifest.json')=='9df6c9c865c7e2accd400b74f0f529feb1f4c68d3620aae7c76c59839a822826','overlay manifest changed')
    require(overlay['native_gpu_run']=='NOT_RUN','overlay native status changed')
    O.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not(k.startswith('NM_') or k.startswith('NUMI_MATTER_'))}
    env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
    ctest=['/opt/homebrew/bin/ctest','--test-dir',str(D),'-R',CTEST_REGEX,'--output-on-failure','-j','1']
    c=run(ctest,env,180); (O/'ctest.json').write_text(json.dumps(c,indent=2)+'\n'); (O/'ctest.log').write_text(c['output'])
    require(c['returncode']==0,'targeted CTest returned failure')
    m=re.search(r'100% tests passed(?:, 0 tests failed)? out of (\d+)',c['output'])
    require(m and int(m.group(1))==3,'targeted CTest did not establish exactly three passing tests')
    envcmd=[str(PYTHON),'-B',str(S/'matter/tools/test_cardboard_solver_environment.py')]
    e=run(envcmd,env,120); (O/'environment-tests.json').write_text(json.dumps(e,indent=2)+'\n'); (O/'environment-tests.log').write_text(e['output'])
    require(e['returncode']==0 and re.search(r'\b12\s+(?:tests|cases)\b|Ran 12 tests|12 passed',e['output'],re.I),'12-case environment tests failed or count missing')
    contract_compile=['clang++','-std=c++20','-Wall','-Wextra','-Werror','-I',str(P/'candidate/matter/tools'),str(P/'tests/coupon_resume_contract_test.cpp'),'-o',str(O/'coupon_resume_contract_test')]
    cc=run(contract_compile,env,60); (O/'contract-compile.json').write_text(json.dumps(cc,indent=2)+'\n')
    require(cc['returncode']==0,'standalone contract compile failed')
    cr=run([str(O/'coupon_resume_contract_test')],env,30); (O/'contract-run.json').write_text(json.dumps(cr,indent=2)+'\n')
    require(cr['returncode']==0 and 'contract checks passed' in cr['output'],'standalone contract run failed')
    file_map(S,frozen['source_sha256'],'selected source post-tests'); file_map(S,support['files'],'configure support post-tests')
    report={'schema':'numi.cardboard.source182-003-build-validation-followup.v1','host':'macmini','platform':platform.platform(),
      'source_snapshot':str(S),'build_dir':str(D),'source_frozen_sha256':sha(S/'frozen-source.json'),
      'source_build_json_sha256':sha(D/'source-build.json'),'compile_commands_sha256':sha(D/'compile_commands.json'),
      'overlay_manifest_sha256':sha(P/'overlay-manifest.json'),'overlay_patch_sha256':sha(P/'overlay.patch'),
      'selected_source_files_verified':805,'configure_support_files_verified':1034,'targets':TARGETS,
      'artifacts_sha256':build['artifacts_sha256'],'ctest_count':3,'environment_cases':12,
      'standalone_contract':'PASS','original_runner_attempt':'preserved; stopped after all three CTests passed because its summary regex expected older CTest wording',
      'gpu_dispatch':False,'native_resume_replay':'NOT_RUN','trajectory_parity':False,'physical_qualification':False,
      'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    inv={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.source182-003-build-validation-followup-manifest.v1','files':inv,'report_sha256':sha(O/'report.json')},indent=2)+'\n')
    print(json.dumps(report,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as ex:
        print(f'FAIL: {ex}')
        raise
