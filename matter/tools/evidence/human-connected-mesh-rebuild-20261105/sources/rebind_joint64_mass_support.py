import sys
"""Rebuild existing mass witness inputs after a proven topology quotient.
Source quadrature is unchanged. Receiver references are proposals; the unchanged
mass allocator must replay all witnesses and solve conservation/density gates.
"""
from pathlib import Path
import csv,gzip,json,hashlib,time,importlib.util,sys
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else W/'joint-boundary-rebuild-64c')/'compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
B=H/'regional-compensation-1061-root';O=C/'mass-support';O.mkdir(exist_ok=True)
start=time.time();pins={};stats={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p);r={'path':str(p),'sha256':sha(p)};pins[str(p)]=r['sha256'];return r
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
manifest=json.loads((B/'full-allocation-003/run-manifest.json').read_text())
env=dict(x.split('=',1) for x in manifest['argv'][1:] if x.startswith('NUMI_'))
for stem in ['P1_SKIN','P1_FAT','CARRIERS_SKIN','CARRIERS_FAT','CARRIER_WITNESS_SKIN','CARRIER_WITNESS_FAT','EXACT_PROJECTION','SOURCE_CENTROID']:
 ext='NPZ' if stem.startswith(('P1_','CARRIERS_')) else 'CSV'
 p=Path(env['NUMI_H706_FRACTIONAL_'+stem+'_'+ext])
 assert pin(p)['sha256']==env['NUMI_H706_FRACTIONAL_'+stem+'_SHA256'],('input pin',stem)
oldp=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
assert pin(oldp)['sha256']=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'
with np.load(oldp) as z:
 oldpts=z['points_reference_world_m_f64'];oldt=z['volume_tetrahedra'];oldcent=oldpts[oldt].mean(axis=1)
 oldmat=z['material_index']
with np.load(C/'rich433/candidate-current-typed-volume.npz') as z:
 pts=z['points_reference_world_m_f64'];t=z['volume_tetrahedra'];cent=pts[t].mean(axis=1)
 newmat=z['material_index'];newlab=np.column_stack([z[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
nold=len(oldt);nnew=len(t);oldskin=int((oldmat==15).sum());newskin=int((newmat==15).sum())
assert np.all(oldmat[:oldskin]==15) and np.all(oldmat[oldskin:]==16)
assert np.all(newmat[:newskin]==15) and np.all(newmat[newskin:]==16)
def build_ancestry_receiver_maps(role_unions, cent, oldcent, newmat, oldmat, maximum_chunk_records=2000000):
 """Same deterministic nearest-ancestry policy, with bounded temporary arrays."""
 nold=len(oldcent);nnew=len(cent);maximum=np.iinfo(np.int64).max
 chosen_old=np.full(nnew,-1,np.int64);nearest=np.full(nold,maximum,np.int64)
 best=np.full(nold,np.inf);allparents=[];alloffsets=[np.array([0],np.int64)];nnz=0
 for off,rr,oldshift,newshift in role_unions:
  assert off[0]==0 and off[-1]==len(rr) and np.all(np.diff(off)>0)
  count=len(off)-1;first=0
  allparents.append((rr+oldshift).astype(np.int32))
  alloffsets.append(off[1:]+nnz);nnz+=len(rr)
  while first<count:
   last=min(count,max(first+1,int(np.searchsorted(off,off[first]+maximum_chunk_records,side='right')-1)))
   counts=np.diff(off[first:last+1]);begin=int(off[first]);end=int(off[last])
   pr=rr[begin:end]+oldshift
   nr=np.repeat(np.arange(first+newshift,last+newshift,dtype=np.int64),counts)
   assert np.all(newmat[nr]==oldmat[pr])
   delta=cent[nr]-oldcent[pr];dist2=np.einsum('ij,ij->i',delta,delta)
   starts=off[first:last]-begin;minimum=np.minimum.reduceat(dist2,starts)
   selected=np.where(dist2==np.repeat(minimum,counts),pr,maximum)
   chosen_old[first+newshift:last+newshift]=np.minimum.reduceat(selected,starts)
   previous=best.copy();np.minimum.at(best,pr,dist2)
   nearest[best<previous]=maximum
   hit=dist2==best[pr];np.minimum.at(nearest,pr[hit],nr[hit])
   first=last
 assert np.all(chosen_old>=0) and np.all(nearest!=maximum) and np.all(np.isfinite(best))
 parent_indices=np.concatenate(allparents);parent_offsets=np.concatenate(alloffsets)
 full=csc_matrix((np.ones(len(parent_indices),np.uint8),parent_indices,parent_offsets),shape=(nold,nnew)).tocsr()
 full.sort_indices()
 assigned=np.bincount(chosen_old,minlength=nold);missing=np.flatnonzero(assigned==0)
 src=np.r_[chosen_old,missing];dst=np.r_[np.arange(nnew),nearest[missing]]
 mapping=csr_matrix((np.ones(len(src),np.uint8),(src,dst)),shape=(nold,nnew));mapping.sort_indices()
 assert np.all(np.diff(mapping.indptr)>0) and len(np.unique(mapping.indices))==nnew
 return chosen_old,mapping,full

unions=[]
for role,oldshift,newshift in [('skin',0,0),('fat',oldskin,newskin)]:
 p=C/f'canonical/{role}-canonical-parent-union.npz';pin(p)
 with np.load(p) as z:
  unions.append((z['canonical_parent_union_offsets'],z['canonical_parent_union_parent_rows'],oldshift,newshift))
chosen_old,mapping,full=build_ancestry_receiver_maps(unions,cent,oldcent,newmat,oldmat)
mo=mapping.indptr;mr=mapping.indices;fo=full.indptr;fr=full.indices
del unions,oldpts,oldt,t
np.savez_compressed(O/'receiver-rebind.npz',old_to_new_offsets=mo,old_to_new_rows=mr,new_support_parent_old_rows=chosen_old)
bits=np.zeros((nnew,4),dtype=np.uint64)
for role,shift in [('skin',0),('fat',newskin)]:
 with np.load(C/f'p1-{role.title()}/{role}-current-p1-143-support.npz') as z:
  off=z['cell_admissible_region_offsets'];rr=z['cell_admissible_region_ids'];cc=np.repeat(np.arange(len(off)-1)+shift,np.diff(off))
  assert np.all((rr>=0)&(rr<256))
  for word in range(4):
   mask=rr//64==word
   np.bitwise_or.at(bits[:,word],cc[mask],np.left_shift(np.uint64(1),(rr[mask]%64).astype(np.uint64)))
def allowed(c,region):return bool(bits[c,region//64]&np.uint64(1<<(region%64)))
def targets(old):return mr[mo[old]:mo[old+1]]
def fs(v):return format(float(v),'.17g')
def coords(c):return [fs(v) for v in cent[c]]
def distance(p,c):return float(np.linalg.norm(p-cent[c]))
stats['mapping']={'old_cells':nold,'new_cells':nnew,'support_links':len(mr),'new_cells_covered':len(np.unique(mr)),'maximum_receivers_per_old_cell':int(np.diff(mo).max())}
print('MAPPING',stats['mapping'],flush=True)
capture=H.parent/'native-physical-parent-mass-330/native-source-identity-control-003'
with (capture/'compiled-parent-cell-mass.csv.nodes.csv').open() as f:
 nn=list(csv.DictReader(f));reference=np.array([[float(x[f'rest_{a}_m']) for a in 'xyz'] for x in nn]);current=np.array([[float(x[f'current_{a}_m']) for a in 'xyz'] for x in nn])
with (capture/'compiled-parent-cell-mass.csv.tets.csv').open() as f:st=np.array([[int(x[f'parent_node{k}']) for k in range(4)] for x in csv.DictReader(f)])
with (capture/'compiled-parent-cell-mass.csv.labels.csv').open() as f:sl=np.array([[int(x[k]) for k in ['material_index','geometric_donor_body','anatomical_region_id']] for x in csv.DictReader(f)])
# Centroid witnesses: preserve source barycentres, replay all new receivers, deduplicate.
srcfile=Path(env['NUMI_H706_FRACTIONAL_SOURCE_CENTROID_CSV']);dest=O/'source-centroid-slots.csv';seen=set();nread=0;rejected=0
with srcfile.open() as f,dest.open('w') as out:
 rd=csv.reader(f);wr=csv.writer(out);wr.writerow(next(rd))
 for row in rd:
  nread+=1;p=int(row[0]);old=int(row[1]);source=np.array(list(map(float,row[9:12])))
  assert np.linalg.norm(oldcent[old]-np.array(list(map(float,row[12:15]))))<5e-9
  for c0 in targets(old):
   c=int(c0);key=(p<<32)|c;region=int(row[4])
   if key in seen:continue
   d=distance(source,c)
   if not allowed(c,region) or d>.187444:rejected+=1;continue
   seen.add(key);r=row.copy();r[1]=str(c);r[12:15]=coords(c);r[15]=fs(d);r[18:21]=list(map(str,newlab[c]));wr.writerow(r)
stats['centroid']={'input_rows':nread,'output_rows':len(seen),'inadmissible_or_out_of_bound':rejected};del seen
print('CENTROID',stats['centroid'],flush=True)
# Carrier source quadrature is invariant. Change only receiver metadata and witness slots.
for role,oshift,nshift in [('skin',0,0),('fat',oldskin,newskin)]:
 pp=Path(env['NUMI_H706_FRACTIONAL_CARRIERS_'+role.upper()+'_NPZ'])
 with np.load(pp) as z:packet={k:z[k].copy() for k in z.files}
 po=packet['parent_offsets'];global_parent=packet['global_source330_tet_rows'];carrier_parent=np.repeat(global_parent,np.diff(po))
 source=np.einsum('ni,nij->nj',packet['carrier_barycentric'],reference[st[carrier_parent]])
 labels=sl[carrier_parent];oldtarget=packet['carrier_target_child_rows'].astype(np.int64)+oshift
 primary=np.empty(len(oldtarget),dtype=np.int64);primarydist=np.empty(len(primary))
 for k,old in enumerate(oldtarget):
  rr=fr[fo[old]:fo[old+1]];region=int(labels[k,2])
  eligible=[int(c) for c in rr if allowed(int(c),region)]
  if not eligible:raise ValueError(('primary has no admissible descendant',role,k,int(old)))
  dd=np.linalg.norm(cent[eligible]-source[k],axis=1);i=int(dd.argmin());c=eligible[i]
  assert dd[i]<=.187444,('primary beyond unchanged radius',role,k,float(dd[i]))
  primary[k]=c;primarydist[k]=dd[i]
 packet['carrier_target_child_rows']=(primary-nshift).astype(np.uint32);packet['carrier_target_distance_m']=primarydist
 if 'regional_compensation_child_rows' in packet:
  rr=packet['regional_compensation_child_rows'].astype(np.int64)+oshift
  mapped=np.unique(np.concatenate([targets(int(c)) for c in rr]))
  packet['regional_compensation_child_rows']=(mapped-nshift).astype(np.uint32)
 np.savez_compressed(O/(role+'-source-carriers.npz'),**packet)
 # Replay primary construction from prior evidence; generate extra exact carrier-point slots.
 oldfile=Path(env['NUMI_H706_FRACTIONAL_CARRIER_WITNESS_'+role.upper()+'_CSV']);dest=O/(role+'-carrier-witnesses.csv.gz')
 seen=set();primary_seen=set();nread=0;rejected=0
 with gzip.open(oldfile,'rt') as f,gzip.open(dest,'wt',compresslevel=1) as out:
  rd=csv.reader(f);wr=csv.writer(out);wr.writerow(next(rd))
  for row in rd:
   nread+=1
   if row[0]=='centroid_route_fallback':continue
   k=int(row[3]);old=int(row[7]);assert 0<=k<len(primary)
   assert np.linalg.norm(source[k]-np.array(list(map(float,row[19:22]))))<5e-9
   assert np.linalg.norm(oldcent[old]-np.array(list(map(float,row[22:25]))))<5e-9
   isprimary=row[0]=='carrier';rr=[int(primary[k])] if isprimary else list(map(int,targets(old)))
   if isprimary:assert k not in primary_seen;primary_seen.add(k)
   for c in rr:
    key=(k<<32)|c
    if key in seen:continue
    if not isprimary and c==primary[k]:continue
    d=distance(source[k],c)
    if not allowed(c,int(labels[k,2])) or d>.187444:rejected+=1;continue
    seen.add(key);r=row.copy();r[6]=str(c-nshift);r[7]=str(c);r[11]='CurrentCarrierPointProjection';r[12:14]=['-1','-1'];r[22:25]=coords(c);r[25]=fs(d)
    if not isprimary:r[0]='carrier_child_slot';r[2]='current_source_carrier_edge_support'
    wr.writerow(r)
 assert len(primary_seen)==len(primary)
 # Quadrature weights and barycentric sites must remain bit-identical.
 with np.load(pp) as z:
  for k in ['source_parent_tet_rows','global_source330_tet_rows','parent_offsets','carrier_weights','carrier_barycentric']:assert np.array_equal(packet[k],z[k])
 stats[role]={'input_witness_rows':nread,'output_witness_rows':len(seen),'primary_carriers':len(primary),'dropped_inadmissible_or_out_of_bound_slots':rejected,'quadrature_unchanged':True}
 print('CARRIER',role,stats[role],flush=True)
# Exact group projections use the existing 217 nearest-tetrahedron owner.
regpath=H/'current-native-1024-root/current-217-1036/register_current_combined_volume_nodes_217_1036.py'
spec=importlib.util.spec_from_file_location('reg217',regpath);reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)
parent,owner=reg.read_parent_current(capture/'compiled-parent-cell-mass.csv',regpath.parent);lookup=reg.make_lookup(parent,owner)
oldproj=Path(env['NUMI_H706_FRACTIONAL_EXACT_PROJECTION_CSV']);proposals={}
with oldproj.open() as f:
 rd=csv.reader(f);header=next(rd)
 for row in rd:
  old=int(row[12]);key=tuple(map(int,row[:3]))
  for c0 in targets(old):
   c=int(c0)
   if allowed(c,key[2]):proposals[(c,key)]=None
print('EXACT_PROJECTION_QUERIES',len(proposals),flush=True)
for c,key in proposals:
 q=reg.nearest_for_tuple(cent[c],key,lookup,owner,.019353,1e-8)
 if q['distance']<=.019353:proposals[(c,key)]=q
proposals={k:v for k,v in proposals.items() if v is not None};wanted={k[0] for k in proposals};contexts={}
with (C/'p1-export/child.labels.csv.source.csv').open() as f:
 for row in csv.DictReader(f):
  c=int(row['child_tet_row'])
  if c in wanted:contexts[c]=row
with np.load(C/'p1-export/p1-anchor.npz') as z:anchor=pts[z['reference_anchor_carrier_child_node_row']]
# Current cells have exact F32 points, positive signed volume.
with np.load(C/'rich433/candidate-current-typed-volume.npz') as z:
 tt=z['volume_tetrahedra'];vv=pts[tt];vol=np.einsum('ij,ij->i',vv[:,1]-vv[:,0],np.cross(vv[:,2]-vv[:,0],vv[:,3]-vv[:,0]))/6
assert np.all(vol>0)
with (O/'exact-projection-support.csv').open('w') as f:
 wr=csv.writer(f);wr.writerow(header)
 for (c,key),q in sorted(proposals.items()):
  x=contexts[c]
  ctx=[x[k] for k in ['material_index','geometric_donor_body','anatomical_region_id','selected_parent_tet_row']]
  ctx += [x[f'centroid_registered_reference_{a}_m'] for a in 'xyz']+[x[f'centroid_reference_residual_{a}_m'] for a in 'xyz']+[x[f'centroid_barycentric{k}'] for k in range(4)]+[x['reference_registration_distance_m']]
  wr.writerow([*key,q['tet'],*map(fs,q['beta']),*map(fs,q['registered']),fs(q['distance']),c,*map(int,newlab[c]),*coords(c),*ctx,*map(fs,anchor[c]),fs(np.linalg.norm(cent[c]-anchor[c])),fs(vol[c]),'PASS_EXACT_CURRENT_CENTROID_SOURCE_TUPLE_PROJECTION'])
stats['projections']={'qualified_rows':len(proposals)}
outputs={p.name:pin(p) for p in O.iterdir() if p.is_file() and p.suffix in ['.npz','.csv','.gz']}
save(O/'receipt.json',{'schema':'numi.h706.current-source-centroid-slot-execution.v1','status':'PREPARED_REBOUND_SUPPORT_REQUIRES_UNCHANGED_ALLOCATOR','inputs':pins,'outputs':outputs,'stats':stats,'wall_s':time.time()-start,'physical_admission':False,'native_steps':0})
print('MASS_SUPPORT_READY',time.time()-start,flush=True)
