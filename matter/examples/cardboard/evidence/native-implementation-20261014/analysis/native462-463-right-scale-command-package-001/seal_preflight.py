#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys
from pathlib import Path
pkg=Path(__file__).resolve().parent
root=Path('/Users/home/cardboard-implementation-20261014')
source=root/'snapshots/source-cgls-block-right-preconditioned-149'
build=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-149')
probe=build/'matter/numi-matter-cardboard-probe'
metal=build/'matter/shaders/NumiMatter.metallib'
receipt=build/'source-build.json'
failed=root/'analysis/native461-block-right-cgls-command-002/native-461-command.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
cmds=[]
for label,name in [('enabled463','native-463-command.json'),('disabled462','native-462-command.json')]:
 p=pkg/name; argv=json.loads(p.read_text())
 case=Path(argv[argv.index('--case')+1])
 inputs=[]
 for i,v in enumerate(argv[:-1]):
  if v=='--input-file':
   q=Path(argv[i+1]); inputs.append({'path':str(q),'exists':q.is_file(),'sha256':sha(q) if q.is_file() else None,'size_bytes':q.stat().st_size if q.is_file() else None})
 cmds.append({'name':label,'argv_path':str(p),'argv_sha256':sha(p),'case_path':str(case),'selector':'block3' if 'NM_FEM_BASE_CGLS_RIGHT_SCALE=block3' in argv else None,'input_files':inputs})
report={
 'schema':'numi.cardboard.native-cgls-right-scale-command-preflight.v1',
 'source_snapshot':{'path':str(source),'exists':source.is_dir(),'source_file_count':len(json.loads(receipt.read_text())['source_sha256']),'build_receipt_sha256':sha(receipt)},
 'build':{'path':str(build),'exists':build.is_dir(),'probe':{'path':str(probe),'sha256':sha(probe)},'metallib':{'path':str(metal),'sha256':sha(metal)},'receipt':{'path':str(receipt),'sha256':sha(receipt)}},
 'commands':cmds,
 'failed461':{'command_path':str(failed),'command_sha256':sha(failed),'reason':'preserved pre-execution input-seal refusal; probe executable argv[117] selected build145 despite source/runtime/metallib binding149'},
 'verification_command':'/Users/home/.pyenv/versions/3.10.12/bin/python3 '+str(pkg/'verify_commands.py'),
 'native_execution_performed':False,
 'claim_limit':'Prepared commands only; no solver result or physical qualification.'}
assert report['source_snapshot']['exists'] and report['build']['exists']
assert all(c['argv_sha256'] and all(i['exists'] for i in c['input_files']) for c in cmds)
(pkg/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('preflight_sha256',sha(pkg/'preflight.json'))
print('commands',[(c['name'],c['argv_sha256'],len(c['input_files'])) for c in cmds])
print('source_files_verified',report['source_snapshot']['source_file_count'])
