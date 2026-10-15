from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-explicit-material-budget-109';OLD=B/'snapshots/source-solver-capture-publication-104'
if not OLD.exists():
 matches=[p for p in (B/'snapshots').glob('*104') if (p/'frozen-source.json').exists()];assert len(matches)==1,matches;OLD=matches[0]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text())['source_sha256'];b=json.loads((S/'frozen-source.json').read_text())['source_sha256']
changed={k:v for k,v in b.items() if a.get(k)!=v};assert len(changed)==15,len(changed)
P=B/'analysis/cardboard-convergence-metric-correction-110/sealed';assert sha(P/'manifest.json')=='47c5379fc0cc527d866f59807e705e8a3556d633408682eeaa3f5ee9deb38f20'
m=json.loads((P/'manifest.json').read_text());assert sha(P/'patch.diff')==m['patch_sha256']
for k,v in changed.items():
 assert sha(S/k)==v,k
 assert (sha(R/k) if (R/k).exists() else None)==a.get(k),k
for k,v in m['base_changed_files'].items():assert sha(R/k)==v,k
root_hash=sha(R/'CMakeLists.txt');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
O=B/'analysis/source109-110-main-adoption-001';O.mkdir(exist_ok=False)
for k in changed:(R/k).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/k,R/k)
for k,v in m['replacement_files'].items():
 src=P/'files'/k;assert sha(src)==v,k;shutil.copy2(src,R/k)
assert sha(R/'CMakeLists.txt')==root_hash
for k,v in {**changed,**m['replacement_files']}.items():assert sha(R/k)==v
receipt=dict(schema='numi.cardboard.source109-110-main-adoption.v1',git_head=head,root_cmake_unchanged_sha256=root_hash,source109_files=changed,evaluator110_files=m['replacement_files'],limits='Build109/CPU74 cover exact native files; evaluator tests run separately. No full physics qualification.')
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(__file__,O/'adopt.py');print(json.dumps(receipt,indent=2))
