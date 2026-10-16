from pathlib import Path
import json,re,hashlib,subprocess,struct,datetime
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');S=Q/'snapshots/source-contact-damped-cgls-memory-195-001';P=Q/'analysis/native520-independent-binary-rollback-001';P.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
header=S/'matter/tools/cardboard_shipping_snapshot.hpp';src=S/'matter/tools/cardboard_full_contact_armijo_check.mm'
text=header.read_text();macro=text.split('#define NUMI_CARDBOARD_SNAPSHOT_VECTORS(X)',1)[1].split('inline void writeSnapshotPayload',1)[0]
names=re.findall(r'X\((\w+)\)',macro)
suffix=['femRegionalGalerkinLinearizationNodes','femRegionalGalerkinBasisP','femRegionalGalerkinEffectiveBasisP','femRegionalGalerkinOperatorAP','femRegionalGalerkinPivotedRAP','femRegionalGalerkinPivots','femRegionalGalerkinFactorValid','femRegionalGalerkinNodeBlockInverse']
cpp='#include "numi/matter/matter.hpp"\n#include <cstdio>\nint main(){using S=numi::matter::RuntimeStateSnapshot;\n'
for n in names+suffix:cpp+='std::printf("'+n+' %zu\\n",sizeof(decltype(S::'+n+')::value_type));\n'
cpp+='}\n';(P/'sizeof_snapshot_records.cpp').write_text(cpp)
cmd=['/usr/bin/xcrun','clang++','-std=c++20','-O2','-I',str(S/'matter/include'),'-I',str(S/'include'),str(P/'sizeof_snapshot_records.cpp'),'-o',str(P/'sizeof_snapshot_records')]
with (P/'compile.log').open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
assert r.returncode==0
raw=subprocess.check_output([str(P/'sizeof_snapshot_records')]);(P/'abi-strides.txt').write_bytes(raw);strides={n:int(v) for n,v in (line.split() for line in raw.decode().splitlines())}
scalars=[('sourcePhysicsFingerprint','Q'),('deviceProgramFingerprint','Q'),('controlStep','I'),('physicsSubstep','I'),('identificationGeneration','I'),('identificationCheckpoint','I'),('identificationAdvanced','B'),('sutureProxyBindingRevision','Q'),('coupledTimestepMultiplier','I'),('coupledTimestepDivisor','I'),('fgmresIterationBudgetOverride','I'),('newtonIterationBudgetOverride','I'),('allocationGeneration','I'),('learnedWeightRevision','I'),('materialStateStride','I')]
def decode(path):
 b=path.read_bytes();offset=0
 def read(fmt):
  nonlocal offset
  n=struct.calcsize('<'+fmt);assert offset+n<=len(b)
  value=struct.unpack_from('<'+fmt,b,offset)[0];offset+=n;return value
 def string():
  nonlocal offset
  n=read('I');assert offset+n<=len(b);v=b[offset:offset+n];offset+=n;return v.decode()
 message=string();state={n:read(fmt) for n,fmt in scalars};vectors={}
 def vector(n):
  nonlocal offset
  count=read('Q');length=count*strides[n];assert length<=len(b)-offset
  v=b[offset:offset+length];info={'count':count,'stride':strides[n],'offset':offset,'bytes':length,'sha256':hashlib.sha256(v).hexdigest()};offset+=length
  vectors[n]=info
 for n in names:vector(n)
 assert b[offset:offset+8]==b'NMRAPDI1';offset+=8
 state['available']=read('B');state['femRegionalGalerkinDiagnosticsAvailable']=read('B');state['femRegionalGalerkinModeCount']=read('I')
 regionalMessage=string()
 for n in suffix:vector(n)
 assert offset==len(b),(offset,len(b))
 return {'file_sha256':sha(path),'file_bytes':len(b),'message':message,'scalars':state,'vectors':vectors,'regionalMessage':regionalMessage}
C=Q/'native/integrated195-contact-rejection-520/baseline'
a=decode(C/'transaction-000001-before.bin');b=decode(C/'transaction-000001-after.bin')
meta=json.loads((C/'transaction-000001.json').read_text())
for k,s in [('before',a),('after',b)]:
 for n,v in meta[k]['accepted_state_vector_sha256'].items():assert s['vectors'][n]['sha256']==v,(k,n)
selected=list(meta['before']['accepted_state_vector_sha256'])
for n in selected:assert a['vectors'][n]['sha256']==b['vectors'][n]['sha256'],n
changed=[n for n in names+suffix if a['vectors'][n]['sha256']!=b['vectors'][n]['sha256']]
assert a['scalars']==b['scalars']
assert a['scalars']['controlStep']==1 and a['scalars']['physicsSubstep']==0
report={'schema':'numi.cardboard.independent-binary-rollback.v1','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Python decodes original raw payload independently of native readSnapshotPayload and native JSON hash generation. CPU sizeof helper supplies source ABI strides only. Complete byte-consumption, counts and original metadata hashes checked. Failed root2 Newton9 remains failed.','snapshot_header_sha256':sha(header),'fixture_serializer_sha256':sha(src),'before':a,'after':b,'accepted_history_vectors_equal':selected,'all_scalars_equal':True,'all_vectors_changed':changed,'accepted':False,'certificate_invalid_after':meta['after']['solver_certificates'][0]['validity'][3]==0,'physical_steps_qualified':0}
assert report['certificate_invalid_after']
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(P/'commands.json').write_text(json.dumps({'compile':cmd,'abi_run':[str(P/'sizeof_snapshot_records')]},indent=2)+'\n')
(P/'runner.py').write_bytes(Path(__file__).read_bytes())
(P/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file()}},indent=2)+'\n')
print(json.dumps({'accepted_vectors_verified':len(selected),'decoded_vectors':len(names+suffix),'changed':changed,'all_scalars_equal':True,'report_sha256':sha(P/'report.json')}))
