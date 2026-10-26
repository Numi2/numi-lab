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

def witness_plane_rows550(config,base_points,selection_arrays):
 """Add measured failed-node facets without relabeling nearest-point guards.

 Each row is a baseline-feasible outward halfspace at the exact registered
 facet named by an independently retained inside witness. Exact all-bank
 classification remains mandatory after construction.
 """
 fields=['bank_body','bank_stable_bone_id','bank_triangle_stable_id']
 ids=np.column_stack([selection_arrays[k] for k in fields])
 triangles=np.asarray(selection_arrays['original_crossed_triangles_world_mm'],np.float64)
 if ids.dtype.kind not in 'iu' or triangles.shape!=(len(ids),3,3) or not np.isfinite(triangles).all():
  raise ValueError('invalid original registered facets')
 points=np.asarray(base_points,np.float32)
 if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('invalid baseline points')
 rows=[];receipt=[];seen=set()
 for w in config['witnesses']:
  key=tuple(w[k] for k in ['node','body','bone','triangle'])
  if any(type(k)!=int for k in key) or key in seen:raise ValueError('duplicate or untyped witness')
  seen.add(key);node=key[0]
  if node<0 or node>=len(points):raise ValueError('witness node bounds')
  matches=np.flatnonzero(np.all(ids==key[1:],axis=1))
  if not len(matches):raise ValueError('witness facet absent from registered inventory')
  tri=triangles[matches[0]]
  if any(not np.allclose(tri,triangles[i],rtol=0,atol=1e-10) for i in matches) or not np.allclose(tri,np.asarray(w['triangle_world_mm']),rtol=0,atol=1e-10):
   raise ValueError('witness facet geometry mismatch')
  n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=float(np.linalg.norm(n))
  if norm<=0 or not np.isfinite(norm):raise ValueError('degenerate witness facet')
  n/=norm;p=points[node]
  signed=float((p.astype(np.float64)*1000.-tri[0])@n)
  ulp=(np.nextafter(np.abs(p),np.float32(np.inf))-np.abs(p)).astype(np.float64)*1000.
  epsilon=float(np.abs(n)@ulp);lower=epsilon-signed
  if not np.isfinite(lower) or lower>0:raise ValueError('witness facet excludes zero-step baseline')
  rows.append(dict(nodes=np.asarray([node],np.int64),coefficients=n[None,:],lower=lower,
   label='failed_point_registered_facet_550:'+':'.join(map(str,key)),kind='failed_point_registered_facet_plane_550'))
  receipt.append(dict(node=node,body=key[1],bone=key[2],triangle=key[3],baseline_signed_distance_mm=signed,emission_margin_mm=epsilon,lower_delta_mm=lower))
 return rows,receipt

def install_coherent_exit550(objective,owner=None):
 if getattr(objective,'_coherent_exit550_installed',False):raise ValueError('adapter installed twice')
 path=Path(os.environ['NUMI_COHERENT_EXIT_SELECTION_550']);rp=path.with_name('report.json')
 report=json.loads(rp.read_text())
 base=Path(sys.argv[sys.argv.index('--base-candidate')+1]);targets=Path(sys.argv[sys.argv.index('--face-target-npz')+1])
 if report.get('candidate_sha256')!=sha(base) or report.get('targets_npz_sha256')!=sha(targets):raise ValueError('selection candidate/targets pins mismatch')
 if report.get('selection_npz_sha256')!=sha(path):raise ValueError('selection NPZ pin mismatch')
 with np.load(path,allow_pickle=False) as z:selection_arrays={k:z[k] for k in z.files}
 selection=keyed_selection(selection_arrays)
 AUDIT['baseline_candidate_sha256']=sha(base)
 witness_path=os.environ.get('NUMI_COHERENT_EXIT_WITNESS_PLANES_550')
 if witness_path:
  if owner is None:raise ValueError('witness planes require construction owner')
  config=json.loads(Path(witness_path).read_text())
  if config.get('candidate_sha256')!=sha(base):raise ValueError('witness baseline pin mismatch')
  evidence=Path(config['inside_summary']);er=json.loads(evidence.read_text())
  if config['inside_summary_sha256']!=sha(evidence) or er['candidate_sha256']!=config['failed_candidate_sha256']:raise ValueError('witness evidence pin mismatch')
  exact={(int(w['output_node']),int(w['bank_body']),int(w['bank_stable_id']),int(w['nearest_triangle'])) for w in er['inside_witnesses'] if w['classification']=='inside'}
  if any(tuple(w[k] for k in ['node','body','bone','triangle']) not in exact for w in config['witnesses']):raise ValueError('unmeasured witness')
  with np.load(base,allow_pickle=False) as z:rows,wr=witness_plane_rows550(config,z['points_world_m'],selection_arrays)
  combine=owner.combine_barrier_rows
  def with_witness_rows(*args,**kwargs):return [*combine(*args,**kwargs),*rows]
  owner.combine_barrier_rows=with_witness_rows
  AUDIT.update(witness_config_sha256=sha(witness_path),witness_config_path=witness_path,failed_point_planes=wr,failed_point_plane_count=len(rows))
 original=objective.build_fixed_bone_outward_plane_rows
 def build(*args,**kwargs):return build_bundle(original,selection,*args,**kwargs)
 objective.build_fixed_bone_outward_plane_rows=build;objective._coherent_exit550_installed=True
 AUDIT.update(selection_path=str(path),selection_sha256=sha(path),selection_report_sha256=sha(rp),adapter_path=str(Path(__file__).resolve()),adapter_sha256=sha(Path(__file__)))
