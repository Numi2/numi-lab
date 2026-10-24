"""Bind the final empty-bag delivery to its actual native state and inputs."""
from pathlib import Path
import json,struct,hashlib,numpy as np
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_empty_franka_20261024';O=ROOT/'outputs/plastic_bag_empty_franka_20261024'
def read(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read('STATUS.json')['status']=='complete';a=read('assessment.json');assert a['full_native_run']
data=(O/'combined/latest_checkpoint.bin').read_bytes();h=struct.unpack_from('<QQQdddQ',data);assert h[0]==0x4e554d4942473031 and h[1]==1920
o=56;bs=[]
for _ in range(h[2]):
 n=struct.unpack_from('<Q',data,o)[0];o+=8;bs.append(data[o:o+n]);o+=n
assert o==len(data)
nodes=np.frombuffer(bs[14],dtype='<f4').reshape(1081,24);counts=np.frombuffer(bs[-1],dtype='<u4')[:4];assert list(counts)==[1081,4246,2112,0]
assert np.isfinite(nodes).all() and (nodes[:,3]>0).all() and (nodes[:,11]==0).all()
assert np.max(abs(nodes[:,3].astype(float)*nodes[:,7]-1))<2e-7
rows=np.loadtxt(W/'grip.csv',delimiter=',',skiprows=1);path=np.frombuffer(bs[0],dtype='<f4').reshape(-1,16);expected=np.zeros_like(path);expected[:,:3]=rows[:,1:4];expected[:,3]=rows[:,7];expected[:,4:7]=rows[:,4:7];expected[:,8:11]=rows[:,8:11];expected[:,12:15]=np.cross(rows[:,8:11],rows[:,4:7]);assert np.array_equal(path,expected)
identity={}
for line in (O/'launch_identity_sha256.txt').read_text().splitlines():
 digest,name=line.split(None,1);identity[name.strip()]=digest
for name in ['source/PlasticBag.metal','source/plastic_bag_coins_scene.mm','grip.csv','robot.json','initial_state.json']:assert sha(W/name)==identity[name],name
v=read('video_record.json');assert sha(O/'trajectory.json')==v['trajectory_sha256'];assert sha(O/'empty_bag_air_drop_high_pickup_drop_8s_normal.mp4')==v['video_sha256'];assert v['decode_pass'] and v['end_state_included']
r={'native_hz':240,'duration_s':8,'native_states':1921,'coin_count_from_gpu':int(counts[3]),'dynamic_nodes_from_gpu':int((nodes[:,3]>0).sum()),'fixed_nodes_from_gpu':int((nodes[:,3]<=0).sum()),'film_mass_kg_from_gpu':float(nodes[:,7].astype(float).sum()),'native_path_bitwise_matches_launch_csv':True,'source_and_inputs_match_launch_manifest':True,'checkpoint_sha256':sha(O/'combined/latest_checkpoint.bin'),'trajectory_sha256':v['trajectory_sha256'],'video_sha256':v['video_sha256'],'actual_corner_lift_m':a['actual_corner_lift_m'],'limitations':a['limitations']}
(O/'delivery_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
