
"""Prepare pinned cell-mass constraints for the existing source-carrier allocator.
Preserve the retained conditioned baseline outside changed topology and a declared
one-ring halo. This is not geometric ancestry or physical/native admission.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
V=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
CAV=H/'regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz'
FLOW=H/'regional-compensation-1061-root/full-allocation-003/allocator-output/flow.csv'
LABELS=H/'current-native-1024-root/current-433-1044/volume-export/volume/child.labels.csv'
CAND=W/'joint-boundary-rebuild-64c'
RICH=CAND/'compiler/rich433/candidate-current-typed-volume.npz'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def volume(p,t):
 x=p[t];return np.einsum('ij,ij->i',x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))/6
def keys(t):return np.ascontiguousarray(np.sort(t,axis=1),dtype=np.int32).view('V16').ravel()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--halo-rings',type=int,choices=[0,1,2],default=1);a=ap.parse_args()
 a.output.mkdir(exist_ok=False);start=time.time()
 pins={str(p):sha(p) for p in [V,CAV,FLOW,LABELS,RICH,CAND/'report.json',Path(__file__)]}
 assert pins[str(V)]=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'
 assert pins[str(CAV)]=='4be98e7d5710360407a561d10efaa02b86702efdaf1b4312c4b035d6d9a19cb8'
 assert pins[str(FLOW)]=='ab1b7b75d211ce2bdd94668914b59de2f8a3c908416a100a4455e7ca5419dfeb'
 with np.load(V) as z:
  p=z['points_reference_world_m_f64'].astype(np.float32).astype(float);t=z['volume_tetrahedra'];mat=z['material_index']
  oldmaps={r:z['current_typed_role_source_row_to_joint_node_'+r] for r in 'OID'}
 original_labels=np.loadtxt(LABELS,delimiter=',',skiprows=1,dtype=np.int64)
 assert original_labels.shape==(len(t),5) and np.array_equal(original_labels[:,0],np.arange(len(t)))
 assert np.array_equal(original_labels[:,1],mat)
 original_labels=original_labels[:,1:4]
 f=np.loadtxt(FLOW,delimiter=',',skiprows=1,usecols=(0,2))
 mass=np.bincount(f[:,0].astype(np.int64),weights=f[:,1],minlength=len(t));del f
 assert np.all(np.isfinite(mass)&(mass>0))
 with np.load(CAV) as z:
  kept=z['retained_input_cell_rows'];new=z['replacement_tetrahedra'];po=z['replacement_parent_offsets'];pr=z['replacement_parent_cell_rows'];patch=z['replacement_tet_offsets']
  v=volume(p,new);newmass=np.empty(len(new))
  for first,last in zip(patch[:-1],patch[1:]):
   parents=pr[po[first]:po[first+1]]
   assert np.all(mat[parents]==mat[parents[0]])
   total=float(mass[parents].sum());part=v[first:last]/v[first:last].sum()*total
   part[-1]=total-float(part[:-1].sum());assert np.all(part>0)
   newmass[first:last]=part
  ct=np.vstack((t[kept],new));cm=np.r_[mass[kept],newmass]
  cmat=np.r_[mat[kept],mat[pr[po[:-1]]]]
  clabel=np.vstack((original_labels[kept],original_labels[pr[po[:-1]]]))
 del t,mass,new,newmass
 with np.load(RICH) as z:
  x=z['points_reference_world_m_f64'];nt=z['volume_tetrahedra'];nmat=z['material_index']
  nlabel=np.column_stack([z[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
  node_map=np.full(len(p),-1,np.int64)
  for r in 'OID':
   aold=oldmaps[r];anew=z['current_typed_role_source_row_to_joint_node_'+r]
   hit=(aold>=0)&(anew>=0);old=aold[hit];new=anew[hit]
   already=node_map[old]>=0;assert np.array_equal(node_map[old[already]],new[already])
   node_map[old]=new
  # Interior vertices have no O/I/D surface row. Use only unique exact emitted
  # coordinates there; complete cell topology and all semantic labels are
  # independently checked below. Ambiguous keys never select a guessed owner.
  surface_node=np.zeros(len(p),bool)
  for r in 'OID':surface_node[oldmaps[r][oldmaps[r]>=0]]=True
  interior=np.flatnonzero((node_map<0)&~surface_node)
  nk=np.ascontiguousarray(x,dtype=np.float32).view('V12').ravel()
  order=np.argsort(nk,kind='stable');ks=nk[order]
  pk=np.ascontiguousarray(p[interior],dtype=np.float32).view('V12').ravel()
  left=np.searchsorted(ks,pk,side='left');right=np.searchsorted(ks,pk,side='right')
  unique=right-left==1
  node_map[interior[unique]]=order[left[unique]]
  interior_mapped=int(unique.sum());interior_ambiguous=int((~unique).sum())
  valid=node_map>=0;assert np.array_equal(p[valid].astype(np.float32),x[node_map[valid]].astype(np.float32))
 operations=json.loads((CAND/'report.json').read_text())['operations']
 free=np.zeros(len(ct),bool)
 for op in operations:free[np.asarray(op['source_cell_rows'],np.int64)]=True
 direct=int(free.sum())
 for _ in range(a.halo_rings):
  nodes=np.zeros(len(p),bool);nodes[ct[free].ravel()]=True;free|=nodes[ct].any(axis=1)
 unmapped=np.any(node_map[ct]<0,axis=1)
 ambiguous_outside=int((unmapped&~free).sum())
 free|=unmapped
 oldrows=np.flatnonzero(~free);mapped=node_map[ct[oldrows]]
 assert np.all(mapped>=0)
 nk=keys(nt);order=np.argsort(nk,kind='stable');sortedkeys=nk[order]
 assert np.all(sortedkeys[1:]!=sortedkeys[:-1])
 k=keys(mapped);pos=np.searchsorted(sortedkeys,k)
 assert np.all(pos<len(nk)) and np.array_equal(sortedkeys[pos],k)
 targetrows=order[pos];assert len(np.unique(targetrows))==len(targetrows)
 assert np.array_equal(cmat[oldrows],nmat[targetrows])
 differences=clabel[oldrows]!=nlabel[targetrows]
 mismatch=np.flatnonzero(differences.any(axis=1))
 if len(mismatch):
  diagnostic={"status":"STOP_UNCHANGED_GEOMETRY_HAS_CHANGED_CELL_LABELS",
   "unchanged_geometry_cells_checked":len(oldrows),"changed_labels":len(mismatch),
   "changed_by_field":dict(zip(["material","geometric_donor","anatomical_region"],map(int,differences.sum(0)))),
   "witnesses":[{"old_conditioned_cell":int(oldrows[i]),"candidate_cell":int(targetrows[i]),
    "old_labels":clabel[oldrows[i]].tolist(),"new_labels":nlabel[targetrows[i]].tolist()} for i in mismatch[:16]]}
  (a.output/"label-witnesses.json").write_text(json.dumps(diagnostic,indent=2)+chr(10))
  # Retain this independent identity witness. Cell-mass constraints bind exact
  # geometry and material; they do not establish primary-label equivalence.
  # All original semantic/source checks remain mandatory in the allocator.
  print(json.dumps({"primary_label_differences":len(mismatch)}),flush=True)
 # Every prescribed cell has exactly the same retained Float32 rest tetrahedron.
 oldv=volume(p,ct[oldrows]);newv=volume(x,nt[targetrows])
 assert np.allclose(oldv,newv,rtol=1e-11,atol=0)
 targetmass=cm[oldrows];s=np.argsort(targetrows);targetrows=targetrows[s];targetmass=targetmass[s]
 csv=a.output/'prescribed-child-mass.csv'
 np.savetxt(csv,np.column_stack((targetrows,targetmass)),delimiter=',',header='child_row,mass_kg',comments='',fmt=['%d','%.17g'])
 report={'scope':'CONSTRUCTION_CONSTRAINTS_NOT_SOURCE_FLOW_OR_NATIVE_ADMISSION','input_pins':pins,'halo_rings':a.halo_rings,
  'baseline_cells':len(ct),'candidate_cells':len(nt),'direct_changed_star_cells':direct,'baseline_free_cells_including_halo':int(free.sum()),
  'prescribed_cells':len(targetrows),'candidate_free_cells':len(nt)-len(targetrows),'prescribed_mass_kg':float(targetmass.sum()),
  'unique_interior_nodes_mapped':interior_mapped,'ambiguous_interior_nodes_unprescribed':interior_ambiguous,
  'additional_cells_unprescribed_for_ambiguous_mapping':ambiguous_outside,
  'prescribed_material_labels_unchanged':True,'primary_donor_region_label_differences':len(mismatch),
  'primary_label_equivalence':'NOT_QUALIFIED' if len(mismatch) else 'PASS',
  'baseline_mass_kg':float(cm.sum()),'all_prescribed_rest_tetrahedra_identical_f32':True,
  'physical_density_transport_source_and_semantic_gates':'UNCHANGED_MUST_PASS_EXISTING_ALLOCATOR',
  'prescribed_relative_tolerance':1e-10,'output':{'path':str(csv),'sha256':sha(csv)},'wall_s':time.time()-start}
 (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_pins'},indent=2))
if __name__=='__main__':main()
