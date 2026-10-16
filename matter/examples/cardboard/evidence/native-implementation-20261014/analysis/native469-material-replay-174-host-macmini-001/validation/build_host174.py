#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, pathlib, shlex, shutil, subprocess, sys

ROOT=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
SRC=ROOT/'snapshots/source-checkpoint-warm-breakdown-173'
BUILD=ROOT/'build-source173-001'
PKG=ROOT/'analysis/native469-material-replay-174-host-macmini-001'
VALID=PKG/'validation'; OUT=PKG/'build'; SHADERS=PKG/'shaders'; BIN=PKG/'tools'
EXPECTED_SOURCE='6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
EXPECTED_BUILD='96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229'
EXPECTED_ARCHIVE='65703d9d857ec15d2d1e8ae93cac605ef5f46964b5a1d836f1af1b0eaf916047'
EXPECTED_TOOL_SHA='69dc60e47321ccca5ab09e5133d9f08620aca13a0bf15395e349e411329a0475'
EXPECTED_METAL='23f6f758cda7ccc0e0651e45a298644e6b71a3c25cd5ef2f7c9cbe4ceca40628'
EXPECTED_HELPER='865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def run(name,cmd,cwd=None):
    (VALID/f'{name}.command.json').write_text(json.dumps({'cwd':str(cwd) if cwd else None,'argv':cmd},indent=2)+'\n')
    p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True)
    (VALID/f'{name}.stdout.log').write_text(p.stdout)
    (VALID/f'{name}.stderr.log').write_text(p.stderr)
    if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}); see {VALID}/{name}.stderr.log')
    return p

def main():
    for p in (VALID,OUT,SHADERS,BIN):p.mkdir(parents=True,exist_ok=True)
    frozen=SRC/'frozen-source.json'; build_json=BUILD/'source-build.json'; archive=BUILD/'libnumi_matter_compiler.a'
    if sha(frozen)!=EXPECTED_SOURCE:raise RuntimeError('source173 frozen manifest hash mismatch')
    if sha(build_json)!=EXPECTED_BUILD:raise RuntimeError('source173 build provenance hash mismatch')
    if sha(archive)!=EXPECTED_ARCHIVE:raise RuntimeError('source173 compiler archive hash mismatch')
    transfer=json.loads((VALID/'remote-input-hash-audit.json').read_text())
    if transfer.get('status')!='PASS':raise RuntimeError('transferred input hash audit was not PASS')
    for rel,want in [('matter/tools/cardboard_source469_material_replay_check.mm',EXPECTED_TOOL_SHA),('matter/tools/cardboard_bilinear15_capture_diagnostic.metal',EXPECTED_METAL),('matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm',EXPECTED_HELPER)]:
        if sha(PKG/'candidate'/rel)!=want:raise RuntimeError(f'host source hash mismatch: {rel}')
    shader=PKG/'tools/cardboard_bilinear15_capture_diagnostic.metal'
    if sha(shader)!=EXPECTED_METAL:raise RuntimeError('copied source173 shader mismatch')
    metallib=SHADERS/'cardboard_bilinear15_capture_diagnostic.metallib'
    air=SHADERS/'cardboard_bilinear15_capture_diagnostic.air'
    obj=OUT/'cardboard_source469_material_replay_check.mm.o'
    executable=BIN/'numi-matter-cardboard-source469-material-replay-check'
    if any(p.exists() for p in (metallib,air,obj,executable)):raise RuntimeError('refusing to overwrite a prior build artifact')
    shader_cmd=['/usr/bin/xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(SRC/'matter/include'),'-I',str(SRC/'include'),'-c',str(shader),'-o',str(air)]
    run('metal-air-build',shader_cmd,SRC/'matter')
    run('metallib-build',['/usr/bin/xcrun','-sdk','macosx','metallib',str(air),'-o',str(metallib)],SRC/'matter')
    compile_db=json.loads((BUILD/'compile_commands.json').read_text())
    source173_file=str(SRC/'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
    row=next(x for x in compile_db if x['file']==source173_file)
    compile_cmd=shlex.split(row['command'])
    replaced=[];i=0
    while i<len(compile_cmd):
        a=compile_cmd[i]
        if a=='-o':replaced.extend(['-o',str(obj)]);i+=2;continue
        if a==source173_file:replaced.append(str(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm'));i+=1;continue
        if a.startswith('-DNUMI_CARDBOARD_BASE_SNAPSHOT_SHA256='):replaced.append(f'-DNUMI_CARDBOARD_BASE_SNAPSHOT_SHA256="{EXPECTED_SOURCE}"');i+=1;continue
        if a.startswith('-DNUMI_CARDBOARD_PAPER_METALLIB='):replaced.append(f'-DNUMI_CARDBOARD_PAPER_METALLIB="{metallib}"');i+=1;continue
        replaced.append(a);i+=1
    if str(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm') not in replaced:raise RuntimeError('compile source path did not bind host wrapper')
    if not any(a==f'-DNUMI_CARDBOARD_BASE_SNAPSHOT_SHA256="{EXPECTED_SOURCE}"' for a in replaced):raise RuntimeError('compile source macro did not bind source173')
    if not any(a==f'-DNUMI_CARDBOARD_PAPER_METALLIB="{metallib}"' for a in replaced):raise RuntimeError('compile metallib macro did not bind package output')
    run('objcxx-build',replaced,pathlib.Path(row['directory']))
    link_file=BUILD/'CMakeFiles/numi-matter-cardboard-bilinear15-capture-diagnostic-check.dir/link.txt'
    link_cmd=shlex.split(link_file.read_text().strip())
    out=[];i=0
    while i<len(link_cmd):
        if link_cmd[i]=='-o':out.extend(['-o',str(executable)]);i+=2;continue
        if link_cmd[i].endswith('cardboard_bilinear15_capture_diagnostic_check.mm.o'):out.append(str(obj));i+=1;continue
        out.append(link_cmd[i]);i+=1
    if not any(a.endswith('libnumi_matter_compiler.a') for a in out):raise RuntimeError('link command lost compiled source173 archive')
    run('link',out,BUILD)
    records={'schema':'numi.cardboard.native469-host-build.v1','host':'macmini','package':str(PKG),'source_snapshot':str(SRC),'source_frozen_manifest_sha256':sha(frozen),'source_build_provenance_sha256':sha(build_json),'compiler_archive_sha256':sha(archive),'wrapper_source_sha256':sha(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm'),'exact_source174_wrapper_sha256':'f4c81f0217e1a528b982cba1293a7868e7d5c3f43bce97d586a1675494350f7c','source173_binding_patch_sha256':sha(PKG/'patch/source173-binding-rebase.patch'),'diagnostic_shader_source_sha256':sha(shader),'helper_source_sha256':sha(PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm'),'air_sha256':sha(air),'metallib_sha256':sha(metallib),'object_sha256':sha(obj),'executable_sha256':sha(executable),'build_steps':['Metal AIR compile only','metallib link only','Objective-C++ compile','static compiler archive/framework link'],'gpu_execution':False,'native_replay_executed':False,'numpy_exporter_executed':False,'numpy_note':'The selected reference JSONL was precomputed and hash-verified on the source169 host. Remote Python lacks NumPy, so exporter was not run here.'}
    (VALID/'host-build.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
    print(json.dumps(records,indent=2,sort_keys=True))
if __name__=='__main__':main()
