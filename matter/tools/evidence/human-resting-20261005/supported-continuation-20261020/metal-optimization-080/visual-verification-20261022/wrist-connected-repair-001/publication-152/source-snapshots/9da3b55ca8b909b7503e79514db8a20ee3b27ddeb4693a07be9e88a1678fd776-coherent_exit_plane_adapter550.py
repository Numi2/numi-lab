"""Construction-only coherent exit-facet objective adapter.

Explicitly replaces routing planes for existing crossings. Already-clear
bone guards and all exact geometry/quality/identity gates remain owned by
the existing fitter and checkers. No native physical state is changed.
"""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
from scipy import sparse
AUDIT={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyed_selection(z):
 fields=['face_rows','bank_body','bank_stable_bone_id','bank_triangle_stable_id','selected_plane_triangle_stable_id','selected_plane_triangles_world_mm','selected_plane_eligible']
 if any(k not in z for k in fields):raise ValueError('selection fields missing')
 if any(np.asarray(z[k]).dtype.kind not in 'iu' for k in fields[:5]) or np.asarray(z['selected_plane_eligible']).dtype!=np.dtype(bool):raise ValueError('selection identity/eligibility types')
 n=len(z['face_rows'])
 if any(np.asarray(z[k]).shape!=(n,) for k in fields if k!='selected_plane_triangles_world_mm') or np.asarray(z['selected_plane_triangles_world_mm']).shape!=(n,3,3):raise ValueError('selection shape')
 if not np.isfinite(z['selected_plane_triangles_world_mm']).all():raise ValueError('nonfinite plane')
 out={}
 for i in range(n):
  key=tuple(int(z[k][i]) for k in fields[:4])
  if key in out:raise ValueError('duplicate original pair identity')
  out[key]={'triangle_id':int(z['selected_plane_triangle_stable_id'][i]),'triangle_mm':np.asarray(z['selected_plane_triangles_world_mm'][i],np.float64),'eligible':bool(z['selected_plane_eligible'][i])}
 return out

def build_bundle(original,selection,points,faces,support,anchors,pairs,bone_triangles_mm,total_weight=1.):
 pairs=list(pairs);keys=[(int(p['face_row']),int(p['body']),int(p['bone_id']),int(p['triangle_id'])) for p in pairs]
 if len(set(keys))!=len(keys) or set(keys)!=set(selection):raise ValueError('selection does not match complete current crossed-pair inventory')
 groups={}
 for i,key in enumerate(keys):groups.setdefault(key[:3],[]).append(i)
 parts=[];changed=0
 for group,indices in sorted(groups.items()):
  chosen=[selection[keys[i]] for i in indices]
  if any(c['eligible'] for c in chosen):
   if not all(c['eligible'] for c in chosen) or len({c['triangle_id'] for c in chosen})!=1 or any(not np.array_equal(chosen[0]['triangle_mm'],c['triangle_mm']) for c in chosen):raise ValueError('inconsistent coherent group')
  mp={}
  for i,c in zip(indices,chosen):
   key=keys[i];old=bone_triangles_mm[key[1:]]
   if not c['eligible'] and (c['triangle_id']!=key[3] or not np.allclose(c['triangle_mm'],old,rtol=0,atol=1e-10)):raise ValueError('fallback changes original plane')
   mp[key[1:]]=c['triangle_mm']
   changed+=int(c['triangle_id']!=key[3])
  local=[pairs[i] for i in indices]
  one=original(points,faces,support,anchors,local,mp,total_weight=1.)
  for table in ['guard_metadata','metadata']:
   for record in one[table]:
    j=int(record['pair_index']);gi=indices[j];key=keys[gi];c=selection[key]
    record['crossed_triangle_id']=int(key[3]);record['triangle_id']=c['triangle_id'];record['plane_triangle_id']=c['triangle_id'];record['pair_index']=gi
    record['plane_selection']='coherent_exit_facet_550' if c['eligible'] else 'original_crossed_facet_fallback'
    record['claim_scope']='selected_exit_facet_plane_only_not_original_pair_or_whole_bank_clearance_certificate'
  # Equal unique face/bank mass, independent of how many triangles the bank uses.
  parts.append(one)
 active=sum(bool(len(p['h'])) for p in parts)
 G=sparse.vstack([p['G'] for p in parts],format='csr');C=sparse.vstack([p['C'] for p in parts],format='csr')
 lower=np.concatenate([p['lower'] for p in parts]);h=np.concatenate([p['h'] for p in parts])
 w=np.concatenate([p['weights']*(float(total_weight)/active if active else 0.) for p in parts])
 gm=sum([p['guard_metadata'] for p in parts],[]);meta=sum([p['metadata'] for p in parts],[])
 anchored=[]
 for p in parts:anchored.extend(p['diagnostics']['anchored_positive_lowers'])
 diagnostics={'input_pair_count':len(pairs),'lower_bound_row_count':len(lower),'soft_pair_count':sum(p['diagnostics']['soft_pair_count'] for p in parts),'soft_row_count':len(h),'soft_weight_mass':float(w.sum()),'anchored_positive_lower_count':len(anchored),'anchored_positive_lowers':anchored,'construction_only':True,'coherent_exit_groups':len(groups),'active_group_mass_count':active,'changed_plane_pair_count':changed,'all_original_pair_identities_preserved':True,'original_pair_clearance_certificate':False}
 AUDIT.update(diagnostics)
 return dict(G=G,lower=lower,guard_metadata=gm,C=C,h=h,weights=w,metadata=meta,diagnostics=diagnostics)

def install_coherent_exit550(objective):
 if getattr(objective,'_coherent_exit550_installed',False):raise ValueError('adapter installed twice')
 path=Path(os.environ['NUMI_COHERENT_EXIT_SELECTION_550']);rp=path.with_name('report.json')
 report=json.loads(rp.read_text())
 base=Path(sys.argv[sys.argv.index('--base-candidate')+1]);targets=Path(sys.argv[sys.argv.index('--face-target-npz')+1])
 if report.get('candidate_sha256')!=sha(base) or report.get('targets_npz_sha256')!=sha(targets):raise ValueError('selection candidate/targets pins mismatch')
 if report.get('selection_npz_sha256')!=sha(path):raise ValueError('selection NPZ pin mismatch')
 with np.load(path,allow_pickle=False) as z:selection=keyed_selection(z)
 original=objective.build_fixed_bone_outward_plane_rows
 def build(*args,**kwargs):return build_bundle(original,selection,*args,**kwargs)
 objective.build_fixed_bone_outward_plane_rows=build;objective._coherent_exit550_installed=True
 AUDIT.update(selection_path=str(path),selection_sha256=sha(path),selection_report_sha256=sha(rp),adapter_path=str(Path(__file__).resolve()),adapter_sha256=sha(Path(__file__)))
