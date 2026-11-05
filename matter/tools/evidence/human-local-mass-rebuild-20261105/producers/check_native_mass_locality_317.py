from pathlib import Path
import gzip,json,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
def nodes(name):
 objs={};keys=[];mass=[];u=[]
 with gzip.open(W/name/'install-motion.stdout.log.gz','rt') as f:
  for line in f:
   if not line.startswith('resting_FEM_node,'):continue
   d=dict(x.split('=',1) for x in line.strip().split(',')[1:]);ob=objs.setdefault(d['object'],len(objs))
   xyz=np.array([d['rest_'+a] for a in 'xyz'],np.float32)
   keys.append((ob,int(d.get('anatomical_region_id',-1)),*xyz.view(np.uint32)))
   mass.append(float(np.float32(d['mass_kg'])))
   u.append([float(np.float32(d['reference_u_hi_'+a]))+float(np.float32(d['reference_u_lo_'+a])) if float(d['reference_mode']) else float(np.float32(d['accepted_'+a]))-float(xyz[i]) for i,a in enumerate('xyz')])
 dtype=np.dtype([(k,np.int64) for k in ['object','region','x','y','z']])
 return objs,np.array(keys,dtype),np.array(mass),np.array(u)
a,ka,ma,ua=nodes('shared-bundle-native-307');b,kb,mb,ub=nodes('prescribed-local-mass-native-317');assert a==b
va,ia,ca=np.unique(ka,return_index=True,return_counts=True);vb,ib,cb=np.unique(kb,return_index=True,return_counts=True)
_,aa,bb=np.intersect1d(va[ca==1],vb[cb==1],return_indices=True);ai=ia[ca==1][aa];bi=ib[cb==1][bb]
diff=np.abs(mb[bi]-ma[ai]);rel=diff/ma[ai]
out={'status':'EXACT_REST_COORDINATE_DIAGNOSTIC_NOT_CANONICAL_ANCESTRY_ADMISSION','matched_unique_nodes':len(ai),'reference_nodes':len(ma),'candidate_nodes':len(mb),'reference_duplicate_coordinate_keys':int((ca>1).sum()),'candidate_duplicate_coordinate_keys':int((cb>1).sum()),'matched_reference_mass_kg':float(ma[ai].sum()),'matched_mass_l1_difference_kg':float(diff.sum()),'matched_mass_l1_relative':float(diff.sum()/ma[ai].sum()),'nodes_mass_changed_over_1percent':int((rel>.01).sum()),'nodes_mass_changed_over_10percent':int((rel>.1).sum()),'maximum_absolute_mass_change_kg':float(diff.max()),'maximum_relative_mass_change':float(rel.max()),'maximum_displacement_difference_m':float(np.linalg.norm(ub[bi]-ua[ai],axis=1).max()),'mass_change_per_region':{}}
for r in np.unique(ka['region'][ai]):
 m=ka['region'][ai]==r
 out['mass_change_per_region'][str(int(r))]={'matched_nodes':int(m.sum()),'reference_mass_kg':float(ma[ai[m]].sum()),'l1_difference_kg':float(diff[m].sum()),'changed_over_1percent':int((rel[m]>.01).sum())}
(W/'prescribed-local-mass-native-317-mass-locality.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='mass_change_per_region'},indent=2))
