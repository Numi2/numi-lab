from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014'); R=Path('/Users/home/numi-cardboard-20261006'); OLD=B/'snapshots/source-explicit-material-budget-109'; S=B/'snapshots/source-private-ccd-fracture-115'; O=B/'analysis/source115-main-adoption-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text())['source_sha256']; b=json.loads((S/'frozen-source.json').read_text())['source_sha256']; changed={k:v for k,v in b.items() if a.get(k)!=v}; assert len(changed)==5,len(changed)
for p,h in changed.items(): assert sha(R/p)==a[p] and sha(S/p)==h,p
cpu=json.loads((B/'cpu-checks-115/receipt.json').read_text()); native=json.loads((B/'native/deterministic-private-ccd-producer-reducer-381/receipt.json').read_text())
assert cpu['exit_code']==0 and cpu['selected_count']==74
assert native['exit_code']==0 and native['source_inputs_unchanged'] and not native['timed_out']
cmake_before=sha(R/'CMakeLists.txt'); head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(); O.mkdir(exist_ok=False)
for p in changed: shutil.copy2(S/p,R/p)
assert sha(R/'CMakeLists.txt')==cmake_before
for p,h in changed.items(): assert sha(R/p)==h,p
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.source115-main-adoption.v1',git_head_before=head,root_cmake_unchanged_sha256=cmake_before,source_build_sha256=sha(R/'build-cardboard-provenance-115/source-build.json'),changes=changed,cpu_checks=74,native_status_routing_checks=169,physical_qualification=False,limits='Full-contact parity is adjudicated independently; no complete coupon/box claim.'),indent=2)+'\n'); shutil.copy2(__file__,O/'adopt.py'); print('adopted',len(changed),'exact source115 files')
