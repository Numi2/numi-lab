from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');C=B/'solver/two-stage-first-failure-rich-fd-419/operator-capture';I=B/'analysis/native-residual-fd-sweep-result-424';O=B/'analysis/native424-residual-sweep-audit-001';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
recs=[json.loads(x,parse_int=lambda s:-0.0 if s=='-0' else int(s)) for x in C.read_text().splitlines()]
nodes=sorted([x for x in recs if x['record']=='node'],key=lambda x:(x['environment'],x['node_id']))
assert len(nodes)==1764
f32=lambda x:np.asarray(x,dtype=np.float32)
capnodes=lambda stage:f32([x[stage]['raw_state_float4s'] for x in nodes]).reshape(2,882,5,4)
capres=lambda pair:f32([x['epsilon_pair_residual_impulses_Ns'][pair] for x in nodes]).reshape(2,882,4)
read=lambda name:np.fromfile(I/'result'/name,dtype='<f4').reshape(2,882,4)
center=read('sample-0-0-residual.bin');capcenter=f32([x['center_residual_impulse_Ns'] for x in nodes]).reshape(2,882,4)
fresh=read('fresh-action.float4.bin');free=f32([x['center']['constraint_tag'] for x in nodes]).reshape(2,882)==0
parity=[]
def words(a,b):
 assert a.shape==b.shape
 return int(np.count_nonzero(a.view('<u4')!=b.view('<u4')))
parity.append(dict(stage='center',node_word_mismatches=words(np.fromfile(I/'result/sample-0-0-nodes.bin',dtype='<f4').reshape(2,882,5,4),capnodes('center')),residual_word_mismatches=words(center,capcenter)))
for k,pair in [(1,0),(2,1)]:
 for side,suffix in [(0,'plus'),(1,'minus')]:
  parity.append(dict(stage=f'epsilon_pair_{pair}_{suffix}',node_word_mismatches=words(np.fromfile(I/f'result/sample-{k}-{side}-nodes.bin',dtype='<f4').reshape(2,882,5,4),capnodes(f'epsilon_pair_{pair}_{suffix}')),residual_word_mismatches=words(read(f'sample-{k}-{side}-residual.bin'),capres(2*pair+side))))
rows=[json.loads(x) for x in (I/'result/residual-sweep.jsonl').read_text().splitlines()];samples={}
def metric(v,ref):
 norm=float(np.linalg.norm(ref));err=float(np.linalg.norm(v-ref))
 return dict(vector_l2=float(np.linalg.norm(v)),reference_l2=norm,absolute_l2_error=err,relative_error=None if norm==0 else err/norm,noise_floor_resolved=False)
for k in range(1,12):
 eps=next(x['epsilon'] for x in rows if x['sample']==k);rp=read(f'sample-{k}-0-residual.bin').astype(float);rm=read(f'sample-{k}-1-residual.bin').astype(float)
 stages=[x for x in rows if x['sample']==k];envs=[]
 for e in range(2):
  mask=free[e];a=fresh[e,mask,:3].astype(float);c=center[e,mask,:3].astype(float);p=rp[e,mask,:3];m=rm[e,mask,:3]
  envs.append(dict(environment=e,status=[x['status'][e] for x in stages],centered=metric(-(p-m)/(2*eps),a),plus_one_sided=metric(-(p-c)/eps,a),minus_one_sided=metric(-(c-m)/eps,a),center_residual_l2=float(np.linalg.norm(c)),plus_residual_l2=float(np.linalg.norm(p)),minus_residual_l2=float(np.linalg.norm(m))))
 samples[k]=dict(epsilon=eps,environments=envs)
report=dict(schema='numi.cardboard.native-residual-fd-sweep.v1',scope='Exact source125 control1/Newton48 native candidate, geometry, internal force and residual replay. No physical transactions.',capture_sha256=sha(C),native_receipt_sha256=sha(I/'receipt.json'),parity=parity,all_retained_endpoints_bitwise_equal=all(x['node_word_mismatches']==0 and x['residual_word_mismatches']==0 for x in parity),residual_sign='A(delta_v) = -dR/dvelocity; displacements follow production prepare candidate with dx=dt*dv, fixed accepted history.',samples=samples,derivative_qualified=False,limitations=['Invalid endpoints are inadmissible derivatives.','Branch changes and absolute FP32 noise floor require independent material adjudication.','Zero environment1 direction gives an inconclusive relative comparison.'])
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n')
print('endpoint parity',report['all_retained_endpoints_bitwise_equal'],parity)
for k,row in samples.items():
 e=row['environments'][0];print(k,row['epsilon'],e['status'],e['centered']['relative_error'],e['plus_one_sided']['relative_error'],e['minus_one_sided']['relative_error'],e['plus_residual_l2'],e['minus_residual_l2'])
