#!/usr/bin/env python3
"""Reproduce source175 overlay CPU contracts and host syntax checks; never dispatches Metal."""
from pathlib import Path
import hashlib
import json
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
OUT = B / 'analysis/source175-sparse-private-scan-integration-001'
BASE = B / 'snapshots/source-checkpoint-warm-breakdown-173'
CAND = OUT / 'candidate'
SRC170 = B / 'analysis/hierarchical-sparse-private-contact-scan-overlay-170-001/candidate'
SCAN_PKG = B / 'analysis/native-sparse-private-scan-check-170-001'
VAL = OUT / 'validation'
VAL.mkdir(parents=True, exist_ok=True)
clang = '/usr/bin/clang++'
inc = ['-I',str(CAND/'matter/include'),'-I',str(BASE/'matter/include'),'-I',str(BASE/'include')]
syntax_inc = inc + ['-I',str(CAND/'matter/tools'),'-I',str(BASE/'matter/tools'),'-I',str(CAND/'matter/src'),'-I',str(BASE/'matter/src')]
metallib = str(SCAN_PKG/'NumiMatter-O3.metallib')
commands = [
 ('sparse_admission_compile', [clang,'-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror',*inc,str(CAND/'matter/tools/cardboard_sparse_private_contact_admission_check.cpp'),'-o',str(VAL/'sparse-admission-check')]),
 ('sparse_admission_execute', [str(VAL/'sparse-admission-check')]),
 ('runtime_objc_syntax', [clang,'-std=c++20','-x','objective-c++','-fobjc-arc','-fblocks','-Wall','-Wextra','-Wpedantic','-fsyntax-only',*syntax_inc,'-DNUMI_MATTER_DEFAULT_METALLIB="'+metallib+'"',str(CAND/'matter/src/runtime.mm')]),
 ('full_contact_tool_objc_syntax', [clang,'-std=c++20','-x','objective-c++','-fobjc-arc','-fblocks','-Wall','-Wextra','-Wpedantic','-Werror','-Wno-unused-function','-Wno-unused-const-variable','-fsyntax-only',*syntax_inc,'-DNUMI_MATTER_METALLIB="'+metallib+'"',str(CAND/'matter/tools/cardboard_full_contact_armijo_check.mm')]),
 ('shipping_memory_tool_syntax', [clang,'-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror','-fsyntax-only',*syntax_inc,str(CAND/'matter/tools/cardboard_shipping_memory_check.cpp')]),
 ('sparse_native_checker_compile', [clang,'-std=c++17','-x','objective-c++','-fobjc-arc','-fblocks','-Wall','-Wextra','-Wpedantic','-Werror',*inc,'-DNUMI_MATTER_METALLIB="'+metallib+'"',str(CAND/'matter/tools/cardboard_sparse_private_contact_scan_native_check.mm'),'-framework','Foundation','-framework','Metal','-o',str(VAL/'sparse-scan-checker')]),
 ('sparse_native_checker_cpu_only', [str(VAL/'sparse-scan-checker'),'--cpu-only']),
]
rows=[]
for name, argv in commands:
    result=subprocess.run(argv,cwd=OUT,text=True,capture_output=True)
    log=VAL/(name+'.log')
    log.write_text('argv='+json.dumps(argv)+'\nexit_code='+str(result.returncode)+'\n--- stdout ---\n'+result.stdout+'\n--- stderr ---\n'+result.stderr)
    rows.append({'name':name,'argv':argv,'exit_code':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr.encode()).hexdigest(),'log':str(log.relative_to(OUT))})
    if result.returncode:
        raise SystemExit(f'{name} failed; see {log}')
report={'schema':'numi.cardboard.source175-cpu-validation.v1','metal_dispatch':False,'physical_transactions':0,'results':rows}
(VAL/'cpu-validation.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'passed':len(rows),'metal_dispatch':False,'physical_transactions':0},sort_keys=True))
