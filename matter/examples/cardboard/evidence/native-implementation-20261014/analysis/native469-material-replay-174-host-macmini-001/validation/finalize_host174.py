#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,datetime
ROOT=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PKG=ROOT/'analysis/native469-material-replay-174-host-macmini-001'
SRC=ROOT/'snapshots/source-checkpoint-warm-breakdown-173'
BUILD=ROOT/'build-source173-001'
V=PKG/'validation'
EXPECTED={
 'frozen_source':'6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3',
 'source_build':'96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229',
 'compiler_archive':'65703d9d857ec15d2d1e8ae93cac605ef5f46964b5a1d836f1af1b0eaf916047',
 'host_wrapper':'69dc60e47321ccca5ab09e5133d9f08620aca13a0bf15395e349e411329a0475',
 'original_wrapper':'f4c81f0217e1a528b982cba1293a7868e7d5c3f43bce97d586a1675494350f7c',
 'rebase_patch':'6e3ef4b7d3fb89cd09cfb93bef45355ab61f1edf5b2730f008824e51cc779aa0',
 'shader':'23f6f758cda7ccc0e0651e45a298644e6b71a3c25cd5ef2f7c9cbe4ceca40628',
 'helper':'865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81',
 'air':'5e6ab9c4122b2be8b272dd7aa8b1b040efb09cc2d8f7a9aeb877c065406bcf85',
 'metallib':'3996691d7674278d3cfffd4d6444b2412948824b78a28437af2da3c322a4fecd',
 'object':'dce7df1d06053d788ceca9e59a96d1d6c931333aa3d3dbf47a455b235d3f7e3f',
 'executable':'5dc0b921b41527b973b498d682f84b010b6f41f8acf4c11b1eec4a7305eef3b2',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def require(path,key):
 got=sha(path)
 if got!=EXPECTED[key]:raise SystemExit(f'{key} hash mismatch: {path} {got}')
 return got
def main():
 if (V/'host-build.json').exists():raise SystemExit('refusing to overwrite host-build.json')
 require(SRC/'frozen-source.json','frozen_source')
 require(BUILD/'source-build.json','source_build')
 require(BUILD/'libnumi_matter_compiler.a','compiler_archive')
 require(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm','host_wrapper')
 require(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check_source174_exact.mm','original_wrapper')
 require(PKG/'patch/source173-binding-rebase.patch','rebase_patch')
 require(PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic.metal','shader')
 require(PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm','helper')
 require(PKG/'shaders/cardboard_bilinear15_capture_diagnostic.air','air')
 require(PKG/'shaders/cardboard_bilinear15_capture_diagnostic.metallib','metallib')
 require(PKG/'build/cardboard_source469_material_replay_check.mm.o','object')
 require(PKG/'tools/numi-matter-cardboard-source469-material-replay-check','executable')
 steps=['metal-air-build','metallib-build','objcxx-build','link']
 step_rows=[]
 for name in steps:
  command=V/f'{name}.command.json';out=V/f'{name}.stdout.log';err=V/f'{name}.stderr.log'
  if not command.is_file() or not out.is_file() or not err.is_file():raise SystemExit(f'missing recorded build evidence for {name}')
  argv=json.loads(command.read_text()).get('argv')
  if not isinstance(argv,list) or not argv:raise SystemExit(f'invalid argv record for {name}')
  if out.stat().st_size or err.stat().st_size:raise SystemExit(f'unexpected compiler output requires review for {name}')
  step_rows.append({'step':name,'command_record_sha256':sha(command),'stdout_bytes':0,'stderr_bytes':0,'success_basis':'build_host174.py reached post-link receipt assembly; its run() raises immediately on any nonzero subprocess result'})
 failure={'schema':'numi.cardboard.native469-host-build-finalization-incident.v1','build_script':'build_host174.py','failure':'FileNotFoundError while hashing patch/source173-binding-rebase.patch after all four run() subprocess calls returned successfully','reason':'patch existed at package root but not at the path expected by build_host174.py; copied byte-identical patch into the declared patch/ directory before finalizing','original_patch_path_sha256':require(PKG/'patch/source173-binding-rebase.patch','rebase_patch'),'prior_build_script_exit_code':1,'native_or_gpu_execution':False}
 (V/'build-script-finalization-incident.json').write_text(json.dumps(failure,indent=2,sort_keys=True)+'\n')
 records={'schema':'numi.cardboard.native469-host-build.v1','status':'build_steps_passed_receipt_finalized_after_packaging_path_repair','host':'macmini','package':str(PKG),'source_snapshot':str(SRC),'source_frozen_manifest_sha256':EXPECTED['frozen_source'],'source_build_provenance_sha256':EXPECTED['source_build'],'compiler_archive_sha256':EXPECTED['compiler_archive'],'wrapper_source_sha256':EXPECTED['host_wrapper'],'exact_source174_wrapper_sha256':EXPECTED['original_wrapper'],'source173_binding_patch_sha256':EXPECTED['rebase_patch'],'diagnostic_shader_source_sha256':EXPECTED['shader'],'helper_source_sha256':EXPECTED['helper'],'air_sha256':EXPECTED['air'],'metallib_sha256':EXPECTED['metallib'],'object_sha256':EXPECTED['object'],'executable_sha256':EXPECTED['executable'],'build_script_exit_code':1,'build_script_exit_explanation':'Compilation/link steps succeeded; the original script failed only during receipt assembly because the patch directory path was missing. This finalizer verifies and records the already-created outputs without rerunning or overwriting them.','build_steps':step_rows,'gpu_execution':False,'native_replay_executed':False,'numpy_exporter_executed':False,'numpy_note':'Reference JSONL was precomputed and hash-verified on the source169 host; no NumPy exporter ran on macmini.'}
 (V/'host-build.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
 print(json.dumps(records,indent=2,sort_keys=True))
if __name__=='__main__':main()
