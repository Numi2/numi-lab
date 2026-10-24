#!/usr/bin/env python3
import pathlib,json,hashlib,shutil,subprocess,time,shlex
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");O=E/'native-candidate020-mechanical-source-024';A=O/'apps';B=O/'build-002';B.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
olddeps=shlex.split((O/'build/candidate.d').read_text().split(':',1)[1].replace('\\\n',''))
resolved=[p for p in olddeps if p.endswith('NumiHumanWholeBodySkinReduction.hpp')]
assert len(resolved)==1 and pathlib.Path(resolved[0])!=A/'NumiHumanWholeBodySkinReduction.hpp'
(O/'build/dependency-rejection.json').write_text(json.dumps({'status':'REJECTED_WRONG_NESTED_HEADER_OWNER','resolved_reduction':resolved,'expected_reduction':str(A/'NumiHumanWholeBodySkinReduction.hpp'),'native_run':False,'physical_qualification':False},indent=2)+'\n')
src=E/'native-app-main-header-integration-151/apps-008/NumiHumanSkinFEMIntegration.wholebody.hpp'
assert sha(src)==sha(E/'wholebody-attachment-source-rebind-037-candidate020-identity004-002/source/apps/NumiHumanSkinFEMIntegration.wholebody.hpp')
shutil.copy2(src,A/src.name)
(O/'wrapper-include-resolution-fix.json').write_text(json.dumps({'copied_unchanged_wrapper_sha256':sha(src),'reason':'quoted nested include must resolve candidate reduction in same overlay directory','input':str(src),'output':str(A/src.name)},indent=2)+'\n')
for label in ('compile','link'):
 argv=json.loads((O/(label+'-command.json')).read_text())
 argv=[str(B/pathlib.Path(x).name) if x.startswith(str(O/'build')+'/') else x for x in argv]
 (B/(label+'-command.json')).write_text(json.dumps(argv,indent=2)+'\n')
 t=time.monotonic()
 with (B/(label+'.log')).open('wb') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 result={'returncode':r.returncode,'wall_seconds':time.monotonic()-t}
 (B/(label+'-result.json')).write_text(json.dumps(result,indent=2)+'\n');print(label,json.dumps(result),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
deps=shlex.split((B/'candidate.d').read_text().split(':',1)[1].replace('\\\n',''))
expected=('NumiHumanSkinFEMIntegration.wholebody.hpp','NumiHumanWholeBodySkinReduction.hpp','NumiHumanWholeBodySkinTransferAncestry.hpp','NumiHumanFinal020CandidateRegistration.hpp','NumiHumanSkinFatReductionLabels.hpp','NumiHumanUnpairedSkinFatVolume.hpp','NumiHumanSkinBoundaryNodeMap.hpp')
for name in expected:
 got=[pathlib.Path(p) for p in deps if p.endswith('/'+name)]
 assert got==[A/name],(name,got)
pins={p:sha(p) for p in deps if pathlib.Path(p).is_file() and (p.startswith('/Users/') or p.startswith('/tmp/'))}
(B/'dependency-certification.json').write_text(json.dumps({'status':'PASS_EXACT_CANDIDATE_AND_EXISTING_CONTACT_OWNERS','expected_overlay_headers':list(expected),'input_hashes':pins,'binary_sha256':sha(B/'numi-human-native'),'physical_qualification':False},indent=2)+'\n')
print('dependencies certified; binary',sha(B/'numi-human-native'),flush=True)
