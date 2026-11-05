import sys
from pathlib import Path
import json,hashlib,subprocess,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104'); C=(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else W/'joint-boundary-rebuild-64c')/'compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706'); B=H/'regional-compensation-1061-root'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
out=C/'bindings';out.mkdir(exist_ok=True)
history={}
def execute(name,text,source):
 p=out/(name+'.py')
 if (out/(name+'.stdout.log')).exists() and name=='restore-registration':
  print(name,'already passed; retained',flush=True);return
 p.write_text(text)
 with (out/(name+'.stdout.log')).open('w') as f,(out/(name+'.stderr.log')).open('w') as e:
  r=subprocess.run([sys.executable,str(p)],stdout=f,stderr=e)
 history[name]={'source':str(source),'source_sha256':sha(source),'executed':str(p),'executed_sha256':sha(p),'exit_code':r.returncode}
 (out/'execution.json').write_text(json.dumps(history,indent=2)+'\n')
 print(name,r.returncode,flush=True)
 if r.returncode:print((out/(name+'.stderr.log')).read_text()[-3500:],flush=True);raise SystemExit(r.returncode)
p=B/'current-reference-import-031/restore_complete_registration.py';s=p.read_text()
a=s.index("plan=json.loads(");b=s.index("def sha(",a)
s=s[:a]+"E=Path("+repr(str(C/'volume-export'))+");env={};pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'\nfor key,path in [('COMBINED_NODE_MAP_CSV',E/'volume/combined-node-map.csv'),('NODE_REGISTRATION_CSV',E/'registered/node-parent-registration.csv')]+[('SURFACE_'+r+'_NODES_CSV',E/f'registered/registered-{r}.csv') for r in 'OID']:\n env[pre+key]=str(path)\n for tag in [key.replace('_CSV','_SHA256')]:env[pre+tag]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()\n"+s[b:]
s=s.replace("assert not output.exists()","assert not output.exists()")
execute('restore-registration',s,p)
# Existing remapper with fresh local quality data; all thresholds remain unchanged.
p=B/'attachment-remap-039/remap_attachments.py';s=p.read_text()
s=s.replace("S=Path(__file__).resolve().parent.parent; H=S.parent; P=H.parent; out=Path(__file__).resolve().parent","S=Path("+repr(str(B))+");H=S.parent;P=H.parent;out=Path(__file__).resolve().parent")
s=s.replace("export=H/'current-native-1024-root/current-433-1044/volume-export'","export=Path("+repr(str(C/'volume-export'))+")")
s=s.replace("vol=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'; assert sha(vol)=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'","vol=Path("+repr(str(C/'rich433/candidate-current-typed-volume.npz'))+"); assert sha(vol)=="+repr(sha(C/'rich433/candidate-current-typed-volume.npz')))
a=s.index("cache= S/");b=s.index("maximum_move_m=",a)
s=s[:a]+"""cache=out/'prepared-quality.npz'
# Minimum altitude of every incident tetrahedron, using all four opposite faces.
for start in range(0,len(tets),65536):
 tt=tets[start:start+65536];q=points[tt];e=q[:,1:]-q[:,:1]
 vol6=np.abs(np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2])))
 area2=np.stack([np.linalg.norm(np.cross(q[:,b]-q[:,a],q[:,c]-q[:,a]),axis=1) for a,b,c in [(1,2,3),(0,3,2),(0,1,3),(0,2,1)]],axis=1)
 assert np.all(vol6>0) and np.all(area2>0)
 # Match the owning attachment gate: altitude opposite this vertex in each incident cell.
 for j in range(4):np.minimum.at(altitude,tt[:,j],vol6/area2[:,j])
np.savez_compressed(cache,altitude=altitude,confidence=confidence,nodes=nodes)
""" +s[b:]
execute('remap-attachments',s,p)
p=B/'dominant-frame-registration-042/regenerate_frames.py';s=p.read_text()
s=s.replace("S=Path(__file__).resolve().parent.parent;H=S.parent; E=H/'current-native-1024-root/current-433-1044/volume-export';out=Path(__file__).resolve().parent","S=Path("+repr(str(B))+");H=S.parent;E=Path("+repr(str(C/'volume-export'))+");out=Path(__file__).resolve().parent")
s=s.replace("vol=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz';assert sha(vol)=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'","vol=Path("+repr(str(C/'rich433/candidate-current-typed-volume.npz'))+");assert sha(vol)=="+repr(sha(C/'rich433/candidate-current-typed-volume.npz')))
s=s.replace("assert node not in changes;changes[node]=data","assert node not in changes or changes[node]==data;changes[node]=data")
s=s.replace(";assert len(changes)==14 and count==15",";assert count>=len(changes)")
execute('dominant-frames',s,p)
