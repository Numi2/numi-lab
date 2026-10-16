from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');P=B/'snapshots/source-shared-contact-budget-184';S=B/'snapshots/source-portable-paper-reference-185';O=B/'analysis/source185-portable-reference-snapshot-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not S.exists() and not O.exists();O.mkdir()
f=json.loads((P/'frozen-source.json').read_text());changed=['matter/CMakeLists.txt'];added=['matter/tools/'+x for x in ['cardboard_compensated_fp64_base.py','cardboard_compensated_fp64_fixture.py','cardboard_compensated_fp64_reference.py','cardboard_dd_fp64.py','test_cardboard_compensated_fp64_reference.py']]
assert sorted(n for n,h in f['source_sha256'].items() if sha(R/n)!=h)==changed
shutil.copytree(P,S)
for n in changed+added:
 t=S/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,t);f['source_sha256'][n]=sha(t)
f['base_snapshot']=str(P);f['scope']='Source184 plus additive compensated-binary64 CPU reference, explicit source173 equations, strict projected residual and correction gates, sibling MP fixture and CTest registration. No production Metal/operator/physical changes.'
(S/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
assert len(f['source_sha256'])==812
for n,h in f['source_sha256'].items(): assert sha(S/n)==h
(O/'source-binding.json').write_text(json.dumps({'parent_frozen_sha256':sha(P/'frozen-source.json'),'frozen_sha256':sha(S/'frozen-source.json'),'selected_files':812,'changed':{n:sha(S/n) for n in changed},'added':{n:sha(S/n) for n in added},'gpu_dispatch':False},indent=2)+'\n');(O/'prepare.py').write_bytes(Path(__file__).read_bytes());print(sha(S/'frozen-source.json'))
