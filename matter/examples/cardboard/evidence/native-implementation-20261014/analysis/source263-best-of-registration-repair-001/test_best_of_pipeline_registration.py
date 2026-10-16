#!/usr/bin/env python3
from pathlib import Path
import argparse,re
p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
a=p.parse_args()
root=Path(a.source)
runtime=(root/'matter/src/runtime.mm').read_text()
metal=(root/'matter/src/metal/base_cgls_runtime.metalinc').read_text()
m=re.search(r'const char\* kernelNames\[\]\{(.*?)\n\s*\};',runtime,re.S)
assert m, 'Runtime kernelNames registration array missing'
registered=set(re.findall(r'"([^"]+)"',m.group(1)))
names=(
'nm_fgmres_base_cgls_best_of_save',
'nm_fgmres_base_cgls_best_of_save_direction',
'nm_fgmres_base_cgls_best_of_resolve',
'nm_fgmres_base_cgls_best_of_restore_direction',
'nm_fgmres_base_cgls_best_of_measure_committed',
)
for name in names:
    assert name in registered, f'dispatched best-of pipeline absent from Runtime registration: {name}'
    assert re.search(r'kernel\s+void\s+'+re.escape(name)+r'\s*\(',metal), f'Metal implementation absent: {name}'
    assert len(re.findall(r'dispatchThreads\(\s*"'+re.escape(name)+r'"',runtime)) == 1, f'expected one dispatch site for {name}'
print('PASS: all five best-of CGLS dispatches are registered and have Metal implementations')
