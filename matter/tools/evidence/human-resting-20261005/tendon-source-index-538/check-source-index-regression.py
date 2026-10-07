import sys,json,csv,hashlib,importlib.util
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
O=E/'native-tendon-source-precision-audit-537'
spec=importlib.util.spec_from_file_location('reader','/Users/n/numi-human-resting-lab-20261005/matter/tools/accepted_mrvpack_surface_audit.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-049/src')
from numilab_human import cardiac_cavity_intersections as p
def sha(f): return hashlib.file_digest(open(f,'rb'),'sha256').hexdigest()
def pack(f):
 m,s,o,surfaces=reader.read_pack(f)
 return m,s,o,surfaces
def rows(m,o,ids,n=3,start=0):
 return np.array([np.frombuffer(m,dtype='<f4',count=n,offset=o+80*int(i)+start) for i in ids])
oldpath=E/'integrated-anatomy-failure-capture-535/accepted-geometry/step-0.mrvpack'
newdir=E/'native-tendon-source-index-regression-538'
old=pack(oldpath);new=pack(newdir/'accepted-geometry/step-0.mrvpack')
assert old[3].keys()==new[3].keys()
changed=[]
for key in sorted(new[3]):
 f=np.asarray(new[3][key]['faces'],dtype=np.int64)
 assert np.array_equal(f,np.asarray(old[3][key]['faces']))
 ids=np.unique(f)
 a=rows(old[0],old[2],ids,8);b=rows(new[0],new[2],ids,8)
 if not np.array_equal(a,b): changed.append({'semantic':key[0],'stable_id':key[1],'position_vertices_changed':int(np.any(a[:,:3]!=b[:,:3],axis=1).sum()),'normal_vertices_changed':int(np.any(a[:,4:7]!=b[:,4:7],axis=1).sum())})
print('changed surfaces',changed,flush=True)
result={'scope':'Actual Float32 native viewer geometry; observational change only. Tendon source open boundaries are retained. This does not qualify the complete body or long horizon.','old_pack_sha256':sha(oldpath),'new_initial_pack_sha256':sha(newdir/'accepted-geometry/step-0.mrvpack'),'surface_count':len(new[3]),'changed_surfaces':changed,'audits':[]}
assert [(r['semantic'],r['stable_id']) for r in changed]==[(51006,7),(51006,8)]
del old
for step in (0,31):
 fpath=newdir/f'accepted-geometry/step-{step}.mrvpack'
 m,s,o,surfaces=new if step==0 else pack(fpath)
 for stable in (7,8):
  fs=np.asarray(surfaces[(51006,stable)]['faces'])
  ids=np.unique(fs);pts=rows(m,o,ids).astype(np.float64)
  # Exact coordinate quotient identifies UV duplicated vertices only.
  verts,inverse=np.unique(pts,axis=0,return_inverse=True)
  local=inverse[np.searchsorted(ids,fs)]
  tris=verts[local];cross=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
  zeros=np.flatnonzero(np.all(cross==0,axis=1)).tolist()
  assert np.isfinite(pts).all()
  den=max(float(v).as_integer_ratio()[1] for v in verts.flat)
  iv=[tuple(int(float(x).as_integer_ratio()[0]*(den//float(x).as_integer_ratio()[1])) for x in row) for row in verts]
  assert not zeros, (step,stable,zeros)
  rec=p._records(iv,local.tolist());audit=p._audit_pair(rec,rec,same_surface=True)
  item={'step':step,'stable_id':stable,'face_count':len(fs),'referenced_vertices':len(ids),'coordinate_quotient_vertices':len(verts),'zero_area_face_count':len(zeros),'nonfinite_vertex_count':0,'exact_self_audit':audit}
  result['audits'].append(item);print('tendon',step,stable,'self',audit['count'],'candidates',audit['aabb_candidate_pairs'],flush=True)
base=E/'integrated-anatomy-native-cycle-532'
parity={}
for name in ('resting-coupled.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv'):
 aa=list(csv.reader(open(base/name)));bb=list(csv.reader(open(newdir/name)))
 assert aa[0]==bb[0],name
 diffs=[]
 for i,b in enumerate(bb[1:],1):
  if aa[i]!=b: diffs.append({'row':i,'columns':[aa[0][j] for j,(x,y) in enumerate(zip(aa[i],b)) if x!=y]})
 parity[name]={'compared_rows':len(bb)-1,'column_count':len(bb[0]),'mismatches':diffs}
result['physical_prefix_comparison']=parity
result['passed_surface_index_fix']=all(r['zero_area_face_count']==0 and r['exact_self_audit']['count']==0 for r in result['audits'])
result['passed_physical_noninterference']=all(not r['mismatches'] for r in parity.values())
out=O/'native-source-index-regression-report.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(out,sha(out),result['passed_surface_index_fix'],result['passed_physical_noninterference'],flush=True)

