from pathlib import Path
import struct,csv,json,numpy as np,hashlib
from scipy.spatial import cKDTree
D=Path(__file__).parent;S=D.parent;H=S.parent;e=H/'current-native-1024-root/current-433-1044/volume-export'
skin=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/left-ankle-rest-preserving-weight-repair-128/candidate.nhskin');data=skin.read_bytes();assert hashlib.sha256(data).hexdigest()=='4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5'
magic,abi,nb,nv,ni,fingerprint,sourcehash=struct.unpack_from('<8s5I32s',data);assert abi==5
b=np.frombuffer(data,dtype=np.dtype([('body','<u4'),('translation','<f4',3),('quaternion','<f4',4),('scale','<f4')]),count=nb,offset=60)
v=np.frombuffer(data,dtype=np.dtype([('position','<f4',3),('normal','<f4',3),('binding','<u4',4),('weight','<f4',4)]),count=nv,offset=60+nb*36)
w=np.frombuffer(data,dtype='<f4',count=nv*nb,offset=60+nb*36+nv*56+ni*4).reshape(nv,nb)
posepath=H.parent/'native-registered-volume-parent-identity-physical-control-365/output/registered-skin-poses-100.json';pose=json.loads(posepath.read_text());assert pose['registration_fingerprint32']==fingerprint
frame={int(x['donor']):x for x in csv.DictReader((e/'registered/donor-frames.csv').open())};poses={x['body_index']:x['current'] for x in pose['bodies']}
for i in b['body']:
 f=frame[int(i)];p=poses[int(i)];assert np.array_equal(np.array([float(f[f'current_{k}']) for k in 'xyz'],dtype='f4'),np.array(p['position_world_m'],dtype='f4'));assert np.array_equal(np.array([float(f[f'current_R{a}{c}']) for a in range(3) for c in range(3)]),np.array(p['local_to_world_row_major_f32']))
records=list(csv.DictReader((D/'source-records.csv').open()));rows=np.array([int(x['source_row']) for x in records]);
def rotate(q,p):
 axis=q[:3];twice=np.float32(2)*np.cross(axis,p);return p+q[3]*twice+np.cross(axis,twice)
points=np.zeros((len(rows),3),dtype='f4')
for lane,binding in enumerate(b):
 local=binding['translation']+rotate(binding['quaternion'],v['position'][rows])*binding['scale'];p=poses[int(binding['body'])];world=np.array(p['position_world_m'],dtype='f4')+rotate(np.array(p['orientation_world_xyzw'],dtype='f4'),local);points+=w[rows,lane,None]*world
known={37607:[.047828856855630875,-.41048908233642578,.00059818720910698175],3412:[-.30670645833015442,-.88691836595535278,.12189969420433044]};errors={}
for r,xyz in known.items():
 k=int(np.flatnonzero(rows==r)[0]);errors[r]=float(np.linalg.norm(points[k]-xyz));assert errors[r]<1e-6,(r,points[k],errors[r])
with (S/'self-audit-055/O.bin').open('rb') as f:
 n,nf=map(int,np.fromfile(f,dtype='<u8',count=2));op=np.fromfile(f,dtype='<f8',count=n*3).reshape(-1,3);faces=np.fromfile(f,dtype='<u4',count=nf*3).reshape(-1,3)
donors=np.array([int(r['donor']) for r in csv.DictReader((S/'dominant-frame-registration-042/registered-O.csv').open())]);weights=np.load(H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz');active=weights['current_typed_role_active_source_vertex_rows_O'];lookup=weights['current_typed_role_source_row_to_joint_node_O'];ow=weights['full86_weights'][lookup[active]]
results=[];tree=cKDTree(op)
for j,row in enumerate(rows):
 donor=int(b['body'][np.argmax(w[row])]);mask=np.flatnonzero(donors==donor);sameD,idx=cKDTree(op[mask]).query(points[j]);anyD,anyRow=tree.query(points[j]);lane=np.flatnonzero(b['body']==donor);weight=float(ow[anyRow,int(lane[0])]) if len(lane) else 0
 results.append({'source_row':int(row),'dominant_donor':donor,'nhcnt_body':int(records[j]['donor']),'posed_point':points[j].astype(float).tolist(),'nearest_same_donor_node_gap_m':float(sameD),'nearest_any_node_gap_m':float(anyD),'nearest_any_node':int(anyRow),'nearest_any_donor':int(donors[anyRow]),'source_donor_weight_at_nearest_any':weight})
out={'diagnostic_only':True,'native_witness_agreement_m':errors,'source_sha256':hashlib.sha256(data).hexdigest(),'records':results,'count':len(results),'node_bound_failures':[r for r in results if r['nearest_same_donor_node_gap_m']>.03]};(D/'source-pose-and-ownership-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
