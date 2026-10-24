"""Full native sequence and genuine gripper support handoff, from saved states."""
from pathlib import Path
import json,sys,numpy as np,csv,struct,hashlib
O=Path(sys.argv[1]);fs=json.loads((O/'trajectory.json').read_text())['frames'];assert [f['step'] for f in fs]==list(range(1921));assert all(not f['coins'] for f in fs)
contact=np.array([f['grip_contact_nodes_by_robot'] for f in fs]);ts=np.arange(len(fs))/240
both=(contact[:,0]>0)&(contact[:,1]>0);receiver=(contact[:,0]==0)&(contact[:,1]>0)
assert both.any();assert receiver[-1] and receiver[ts>=7].all();assert contact[-1,0]==0
first2=int(np.where(contact[:,1]>0)[0][0]);last1=int(np.where(contact[:,0]>0)[0][-1]);assert first2<last1
p=np.array([f['front']+f['back'] for f in fs]);assert np.isfinite(p).all();floor=float(p[:,:,1].min()-.108-.0000375);assert floor>=-1e-7
cp=(O/'finish/latest_checkpoint.bin').read_bytes();h=struct.unpack_from('<QQQdddQ',cp);assert h[1]==1920 and h[2]==38
bs=[];off=56
for j in range(h[2]):n=struct.unpack_from('<Q',cp,off)[0];off+=8;bs.append(cp[off:off+n]);off+=n
nodes=np.frombuffer(bs[14],'<f4').reshape(1081,24);counts=np.frombuffer(bs[33],'<u4')[:4];assert list(counts)==[1081,4246,2112,0] and (nodes[:,3]>0).all() and (nodes[:,11]==0).all();mass=float(nodes[:,7].astype(float).sum())
metrics={'duration_s':8,'native_hz':240,'native_states':len(fs),'coins':0,'dynamic_film_nodes':1081,'fixed_film_nodes':0,'film_mass_kg':mass,'receiver_first_contact_s':first2/240,'donor_last_contact_s':last1/240,'overlap_grip_duration_s':float(both.sum()/240),'receiver_only_supported_duration_s':float(receiver.sum()/240),'receiver_continuous_only_support_after_s':(last1+1)/240,'last_grip_nodes':contact[-1].tolist(),'min_film_table_outer_gap_m':floor,'receiver_corner_displacement_after_donor_release_m':float(np.linalg.norm(p[-1,550]-p[last1+1,550])),'final_film_corner_550_m':p[-1,550].tolist(),'final_checkpoint_sha256':hashlib.sha256(cp).hexdigest(),'limitations':['Prescribed native Franka joint trajectories, not motor dynamics.','Finite compliant fingertip model; housing/film and inter-arm mesh crossing audited at saved states, not continuous whole-robot CCD.','Same uncalibrated LDPE parameters as approved empty bag; no air drag.','Finite inward pad patches, not volumetric rubber or complete back/side surface contact.','Finite pad patch entry lacks a smooth tangential elastic edge potential; reduced preload is a scene choice, not a repair of that model limit.']}
(O/'handoff_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
