#!/usr/bin/env python3
"""Remove unreferenced reduced-surface rows without changing used geometry.

Preparation for the existing Numi Human shell compiler. Does not repair invalid
cells, retarget attachments, change mass/density, or admit a physical model.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def compact(surface,layers,source_map,baseline_map):
 n=len(surface['world']);f=np.asarray(surface['faces'],np.int64)
 if f.ndim!=2 or f.shape[1]!=3 or f.size==0 or f.min()<0 or f.max()>=n:raise ValueError('invalid surface faces')
 keep=np.unique(f);old_to_new=np.full(n,-1,np.int64);old_to_new[keep]=np.arange(len(keep),dtype=np.int64)
 # Sorted selection preserves global-ID prism diagonals and face parity.
 sf={k:(v[keep].copy() if v.ndim and len(v)==n else v.copy()) for k,v in surface.items()}
 sf['faces']=old_to_new[f]
 lm={}
 for k,v in layers.items():
  if k=='tetrahedra':
   t=np.asarray(v,np.int64)
   if t.size and (t.min()<0 or t.max()>=3*n):raise ValueError('tet outside three shared surface layers')
   local=old_to_new[t%n]
   if np.any(local<0):raise ValueError('tet uses a row not referenced by its surface')
   lm[k]=(t//n)*len(keep)+local
  else:lm[k]=v[keep].copy() if v.ndim and len(v)==n else v.copy()
 sm={k:(v[keep].copy() if v.ndim and len(v)==n else v.copy()) for k,v in source_map.items()}
 bm={k:v.copy() for k,v in baseline_map.items()}
 original=np.asarray(bm['baseline_row_to_candidate'],np.int64)
 if np.any(original>=n):raise ValueError('baseline map exceeds old node domain')
 mapped=np.full(original.shape,-1,np.int64);ok=original>=0;mapped[ok]=old_to_new[original[ok]]
 bm['baseline_row_to_candidate']=mapped
 bm['baseline_active_mask']=np.asarray(bm['baseline_active_mask'],bool)&(mapped>=0)
 assert np.array_equal(sf['world'][sf['faces']],surface['world'][f])
 assert len(np.unique(sf['faces']))==len(keep)
 if 'tetrahedra' in layers:
  before=np.concatenate([layers[k] for k in ('outer_world','inner_world','deep_world')])
  after=np.concatenate([lm[k] for k in ('outer_world','inner_world','deep_world')])
  assert np.array_equal(before[layers['tetrahedra']],after[lm['tetrahedra']])
 return sf,lm,sm,bm,keep,old_to_new

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
 files=['coarse-surface.npz','candidate-layers.npz','coarse-to-source.npz','baseline-row-map.npz']
 data=[read(a.input/k) for k in files];s,l,c,b,keep,remap=compact(*data)
 a.output.mkdir(parents=True,exist_ok=False)
 for k,d in zip(files,(s,l,c,b)):np.savez_compressed(a.output/k,**d)
 # Face IDs, ancestry and barycentrics for detail->reduced transfer are unchanged.
 src=a.input/'source-to-reduced.npz'
 if src.is_file():(a.output/src.name).write_bytes(src.read_bytes())
 np.savez_compressed(a.output/'compaction-row-map.npz',old_to_new=remap,new_to_old=keep)
 from native_prism_checks_vectorized import signed_prisms,float32_tet_determinants
 checks=[]
 for label,o,i,t in [('skin','outer_world','inner_world','skin_thickness_m'),('fat','inner_world','deep_world','fat_thickness_m')]:
  old_bad,old_values=signed_prisms(data[1][o],data[1][i],data[1][t],data[0]['faces'])
  new_bad,new_values=signed_prisms(l[o],l[i],l[t],s['faces'])
  old_det,old_vol=float32_tet_determinants(data[1][o],data[1][i],data[0]['faces'])
  new_det,new_vol=float32_tet_determinants(l[o],l[i],s['faces'])
  same=all(np.array_equal(x,y) for x,y in [(old_bad,new_bad),(old_values,new_values),(old_det,new_det),(old_vol,new_vol)])
  assert same,'compaction changed signed geometry'
  checks.append({'layer':label,'signed_failure_mask_and_values_bit_exact':True,'float32_position_determinants_and_volumes_bit_exact':True,'invalid_prisms_retained':int(new_bad.sum()),'raw_absolute_volume_m3_diagnostic_not_admitted':float(np.abs(new_values[:,:3]).sum())})
 r={'scope':'Ordered reduced-surface orphan compaction only; invalid layers remain invalid. No native physical admission, attachment retarget, density or mass adjustment.', 'input_files':{k:digest(a.input/k) for k in files},'output_files':{p.name:digest(p) for p in a.output.glob('*.npz')},'allocated_before':len(remap),'referenced_after':len(keep),'removed_orphans':int(sum(remap<0)),'faces_unchanged_count':len(s['faces']),'tetrahedra_unchanged_count':len(l['tetrahedra']),'source_face_ids_and_barycentric_maps_unchanged':True,'surface_and_tetrahedron_position_tuples_bit_exact':True,'checks':checks,'removed_baseline_attachment_rows':[k for k in (297,373) if remap[k]<0]}
 (a.output/'compaction-report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
