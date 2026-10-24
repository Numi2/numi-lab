#!/usr/bin/env python3
import pathlib,json,hashlib,subprocess,time,difflib,shutil
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
P=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013'
D=E/'wholebody-attachment-source-rebind-037-candidate020-identity004-002'
O=E/'native-candidate020-mechanical-source-024'
O.mkdir(exist_ok=False)
A=O/'apps';A.mkdir()
B=O/'build';B.mkdir()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
baseapp=P/'apps/numilab_human_myosim_visual_probe.mm'
labels=P/'apps/NumiHumanSkinFatReductionLabels.hpp'
for name in ('NumiHumanSkinFatReductionLabels.hpp','NumiHumanUnpairedSkinFatVolume.hpp','NumiHumanSkinBoundaryNodeMap.hpp','NumiHumanMaterialColumnSelfContact.hpp'):
 shutil.copy2(P/'apps'/name,A/name)
for name in ('NumiHumanWholeBodySkinReduction.hpp','NumiHumanWholeBodySkinTransferAncestry.hpp','NumiHumanFinal020CandidateRegistration.hpp'):
 shutil.copy2(D/'source/apps'/name,A/name)
id0=json.loads((D/'D37-candidate-identity.json').read_text())
identity=dict(id0)
identity.update(schema='numi.human.skin.attachment-candidate-identity.d38.v1',candidate_id='candidate020-d37-current-unpaired163-owner-integration-038',base_candidate_id=id0['candidate_id'],source_label_owner_sha256=sha(labels),source_app_base_sha256=sha(baseapp),regeneration_candidate_identity_sha256=sha(D/'D37-candidate-identity.json'),physical_qualification=False)
ip=O/'D38-candidate-identity.json';ip.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
old=baseapp.read_text();app=old
repls=[('#include "NumiHumanFinal128CandidateRegistration.hpp"','#include "NumiHumanFinal020CandidateRegistration.hpp"'),('wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001',identity['candidate_id']),('68b4b407c269f992b4ca4e214e272090f14d1fd1e80b70b387c757ef6c3e91cd',sha(ip))]
for a,b in repls:
 assert app.count(a)==1,(a,app.count(a))
 app=app.replace(a,b)
ap=A/baseapp.name;ap.write_text(app)
(O/'app-integration.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),app.splitlines(True),fromfile=str(baseapp),tofile=str(ap))))
cmd=json.loads((P/'build-014/compile-command.json').read_text())
for i,x in enumerate(cmd):
 if x==str(P/'apps'):cmd[i]=str(A)
 elif x==str(baseapp):cmd[i]=str(ap)
 elif x.startswith(str(P/'build-014')):cmd[i]=str(B/pathlib.Path(x).name)
cmd[1:1]=['-iquote',str(P/'apps')]
link=json.loads((P/'build-014/link-command.json').read_text())
link=[str(B/pathlib.Path(x).name) if x.startswith(str(P/'build-014')) else x for x in link]
(O/'compile-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(O/'link-command.json').write_text(json.dumps(link,indent=2)+'\n')
pins={str(p):sha(p) for p in [baseapp,labels,D/'D37-candidate-identity.json',D/'D37-manifest.json',ip,*A.iterdir()] if p.is_file()}
(O/'source-integration.json').write_text(json.dumps({'status':'PRIVATE_UNADMITTED_DIAGNOSTIC','physical_qualification':False,'base_native_source':'identity004 app-preflight013 build014','candidate_regeneration':'D37','input_hashes':pins,'changes':['registered candidate metadata and donor frames from D37','one registration include and exact private candidate guard identity','unchanged unpaired163 contact/source/ancestry owner'],'native_execution':False},indent=2)+'\n')
for label,argv in [('compile',cmd),('link',link)]:
 t=time.monotonic()
 with (B/(label+'.log')).open('wb') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 result={'returncode':p.returncode,'wall_seconds':time.monotonic()-t,'argv_sha256':sha(O/(label+'-command.json'))}
 (B/(label+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(label,json.dumps(result),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
print('binary',str(B/'numi-human-native'),sha(B/'numi-human-native'),flush=True)
