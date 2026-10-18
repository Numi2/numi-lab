from pathlib import Path
import json,shutil,hashlib
old=Path('/Users/n/numi-human-rope-swing-build-20261018-002');out=Path('/Users/n/numi-human-rope-swing-build-20261018-003');out.mkdir();(out/'bin').mkdir();shutil.copytree(old/'source',out/'source');shutil.copyfile(old/'rope.metal',out/'rope.metal')
p=out/'source/NumiHumanRestingWindow.hpp';s=p.read_text();assert '_lastSimulatedSeconds/3.0' in s;s=s.replace('_lastSimulatedSeconds/3.0','_lastSimulatedSeconds/1.25');p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((old/'build-plan-abi41.json').read_text())
for k in ['compile_argv','link_argv']:plan[k]=[v.replace(str(old),str(out)) for v in plan[k]]
plan['application_header_sha256']={str(out/'source'/Path(x).name):sha(out/'source'/Path(x).name) for x in plan['application_header_sha256']}
plan['source_sha256']=plan['new_source_sha256']=sha(out/'source/numilab_human_myosim_visual_probe.mm')
(out/'build-plan-abi41.json').write_text(json.dumps(plan,indent=2));(out/'execute_build_abi41.py').write_text((old/'execute_build_abi41.py').read_text().replace(str(old),str(out)))
