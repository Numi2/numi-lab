#!/usr/bin/env python3
"""Recompute exact input admission and constitutive-branch coverage for 360/419.
CPU-only; this is a fixture coverage audit, not a native parity result.
"""
from __future__ import annotations
import hashlib, json, math, pathlib, struct, sys
ROOT=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CASES={
 '360': {
  'run':ROOT/'solver/reset-augmentation-first-failure-capture-360',
  'source_snapshot':ROOT/'snapshots/source-boundary-reset-order-101',
  'capture':'548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33',
  'package':'3c25f41c6c26654ccf3f8b4f4b510bbb64ab0ea0f52b94d05d8b1fa4060fafe5',
  'candidate':'dcd6b8ea62c05c3ef9a04dc4d46ece7e7fbad58101d8d55d2bee6dfd9d7e8614',
  'accepted':'e352376a5e63ae33802d6eedb225839e4e71d0245076679fe51b5fe938a45a2b',
  'condition':'f2775bdc8f1747dcda5136b2bbe890e1d0645804df304c66bfe3143e90d955ad',
  'run_manifest':'2a34ad4f3328616fb5660fe22022941ca9ace9383ca9f1f1b7262ecfd4350ec8',
  'source_build':'b40d30669b286449a3c5d50da2c865a052cd69c1e9f556132f0fc4bc66cd9fb0',
  'receipt':'d623591647a91ca07626e18fef6c78358e60c0d9ab7c38d465e79a2f6c18c474',
  'result':'4afcdb2310cf87130417399775dc132cb0509f805f537bfe0fa28b3c3ae7b2d5',
  'root':2,'newton':23,'step':3,'fingerprint':'6de3475316b7e8b9'},
 '419': {
  'run':ROOT/'solver/two-stage-first-failure-rich-fd-419',
  'source_snapshot':ROOT/'snapshots/source-status5-vtccd-diagnostics-125',
  'capture':'f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a',
  'package':'cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c',
  'candidate':'c4f2b8d790494949da5be98e876b911db2655dc88e7417416906c91412557269',
  'accepted':'fb270a40f77d2251451e1ca1ab2c09e1c4779775a9f073fde25a6b6218ef2ee4',
  'condition':'40e918b680ab35c6b5dd571796221311535f36215a43a80331b50b28fce4456d',
  'run_manifest':'bcc9d5c5085635ea845d4d34f37bbf4c26b5af9086c15f5e02bb4597d2c2c3f5',
  'source_build':'35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6',
  'receipt':'0826a69a65ea4a0d76ef95e56c5d0d7f5238f4de746dd6daf81f11a87907f67d',
  'result':'6600e9b15bba128d2ef4ed2c5c4a181e91fdc8600a4171ecdd7e1c2eb04efcba',
  'root':1,'newton':48,'step':2,'fingerprint':'75c1264bc96ac2d9'},
}
MATERIALS={0:'hajali2009_liner_hill_bilinear_fit',1:'hajali2009_medium_hill_bilinear_fit',2:'starch2007_finite_glue_elastic'}
STATE_CUTOFF=1e-12

def sha(path:pathlib.Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def f32(x):return struct.pack('<f',float(x))
def norm(mat):return math.sqrt(sum(float(v)*float(v) for row in mat for v in row))
def read_package_section_counts(path):
 blob=path.read_bytes()
 if len(blob)<56 or blob[:16]!=b'NUMIMATTERPKG\0\0\0':raise RuntimeError('compiled package header/magic mismatch')
 version,endian,abi,section_count=struct.unpack_from('<IIII',blob,16)
 if version!=19 or endian!=0x01020304 or section_count>256:raise RuntimeError('compiled package version/endian/section bound mismatch')
 offset=56;counts={};sizes={}
 for _ in range(section_count):
  if offset+32>len(blob):raise RuntimeError('truncated package section header')
  section_id,element_size,element_count,byte_count,_content_hash=struct.unpack_from('<IIQQQ',blob,offset)
  offset+=32
  if section_id in counts or byte_count!=element_size*element_count or offset+byte_count>len(blob):raise RuntimeError('invalid/duplicate package section')
  counts[section_id]=element_count;sizes[section_id]=element_size;offset+=byte_count
 if offset!=len(blob):raise RuntimeError('trailing package bytes')
 return {'abi':abi,'section_count':section_count,'objects_section_10':counts.get(10),'objects_element_size':sizes.get(10),'parameters_section_6':counts.get(6),'parameters_element_size':sizes.get(6)}
def audit(name,cfg):
 run=cfg['run'];out=run/'output';step=f"{cfg['step']:06d}"
 files={
  'operator-capture':cfg['capture'],
  'output/compiled.nmatterpack':cfg['package'],
  f'output/rejected-conditioning-{step}/candidate_nodes.bin':cfg['candidate'],
  f'output/rejected-conditioning-{step}/manifest.json':cfg['condition'],
  f'output/material_state_{step}_bent.json':cfg['accepted'],
  'output/manifest.json':cfg['run_manifest'],
  'inputs/external/source-build.json':cfg['source_build'],
  'receipt.json':cfg['receipt'],
  'output/result.json':cfg['result'],
 }
 for rel,expected in files.items():
  got=sha(run/rel)
  if got!=expected:raise RuntimeError(f'{name} hash mismatch {rel}: {got}')
 sb=json.loads((run/'inputs/external/source-build.json').read_text())
 if pathlib.Path(sb['source_snapshot']).resolve()!=cfg['source_snapshot'].resolve():
  raise RuntimeError(f'{name} source-build snapshot mismatch')
 manifest=json.loads((out/'manifest.json').read_text())
 package_shape=read_package_section_counts(out/'compiled.nmatterpack')
 if package_shape['objects_section_10']!=1 or package_shape['parameters_section_6']!=59:
  raise RuntimeError(f'{name} compiled package layout mismatch: {package_shape}')
 for key in ('liner','medium','glue'):
  pass
 asset_map={
  'liner':('liner_path','liner_sha256'),
  'medium':('medium_path','medium_sha256'),
  'glue':('glue_path','glue_sha256'),
 }
 for _,(pathkey,hashkey) in asset_map.items():
  asset=pathlib.Path(manifest['materials'][pathkey])
  if not asset.is_absolute():asset=cfg['source_snapshot']/asset
  if not asset.exists() or sha(asset)!=manifest['materials'][hashkey]:
   raise RuntimeError(f'{name} material asset binding failed: {pathkey}')
 capfile=(run/'operator-capture').open()
 header=None; counts={}; branch={}; node_rows=[]; tet_seen=set(); tangent_valid=0
 for line in capfile:
  if not line.strip():continue
  row=json.loads(line);kind=row['record']
  if kind=='header':
   if header is not None:raise RuntimeError('duplicate capture header')
   header=row
   continue
  if kind=='node':node_rows.append(row);continue
  if kind!='tetrahedron':continue
  env=row['environment'];mat=row['material_index'];tid=row['tet_id']
  if env not in (0,1) or mat not in MATERIALS or row['material_name']!=MATERIALS[mat]:raise RuntimeError(f'{name} unknown material/env row')
  key=(env,tid)
  if key in tet_seen:raise RuntimeError(f'{name} duplicate tet {key}')
  tet_seen.add(key)
  counts.setdefault(env,{m:0 for m in MATERIALS})[mat]+=1
  tangent_valid+=int(bool(row['cached_algorithmic_tangent_local']['valid']))
  q=row['material_frame_quaternion_xyzw']
  qbits=tuple(f32(x).hex() for x in q)
  c=row['center'];fdnorm=norm(c['Fdot_local_per_s'])
  st=c['accepted_state'];pst=c['projected_state']
  entry=branch.setdefault((env,mat),{'accepted_multiplier_gt_cutoff':0,'candidate_multiplier_increment_gt_cutoff':0,'no_candidate_multiplier_increment_gt_cutoff':0,'frame_quaternions':set(),'nonidentity_frames':0,'Fdot_nonzero':0,'Fdot_norm_min':None,'Fdot_norm_max':0.0,'stateful_rows':0})
  entry['frame_quaternions'].add(qbits)
  if max(abs(float(x)) for x in q[:3])>1e-6:entry['nonidentity_frames']+=1
  if fdnorm>1e-12:entry['Fdot_nonzero']+=1
  entry['Fdot_norm_min']=fdnorm if entry['Fdot_norm_min'] is None else min(entry['Fdot_norm_min'],fdnorm)
  entry['Fdot_norm_max']=max(entry['Fdot_norm_max'],fdnorm)
  if mat!=2:
   entry['stateful_rows']+=1
   oldlam=float(st[12]); newlam=float(pst[12]); delta=newlam-oldlam
   if abs(oldlam)>STATE_CUTOFF:entry['accepted_multiplier_gt_cutoff']+=1
   if delta>STATE_CUTOFF:entry['candidate_multiplier_increment_gt_cutoff']+=1
   if delta<=STATE_CUTOFF:entry['no_candidate_multiplier_increment_gt_cutoff']+=1
 if header is None:raise RuntimeError(f'{name} capture header missing')
 if header['schema']!='numi.cardboard.fem-newton-capture.v1' or header['root']!=cfg['root'] or header['newton_iteration']!=cfg['newton'] or header['world_fingerprint_hex']!=cfg['fingerprint']:
  raise RuntimeError(f'{name} capture identity mismatch')
 if header['environment_count']!=2 or header['node_count_per_environment']!=882 or header['tetrahedron_count_per_environment']!=1920:raise RuntimeError(f'{name} dimensions mismatch')
 if header['cached_algorithmic_tangent_capture']['policy_enabled'] is not True or header['cached_algorithmic_tangent_capture']['available'] is not True:raise RuntimeError(f'{name} missing cache capture')
 if len(node_rows)!=1764 or len(tet_seen)!=3840:raise RuntimeError(f'{name} incomplete capture rows')
 candidate=(out/f'rejected-conditioning-{step}/candidate_nodes.bin').read_bytes()
 if len(candidate)!=2*882*80:raise RuntimeError(f'{name} candidate binary shape mismatch')
 for row in node_rows:
  offset=(row['environment']*882+row['node_id'])*80
  values=[float(x) for vec in row['center']['raw_state_float4s'] for x in vec]
  if candidate[offset:offset+80]!=b''.join(f32(x) for x in values):raise RuntimeError(f'{name} candidate words differ at {row["environment"]}/{row["node_id"]}')
 side=json.loads((out/f'material_state_{step}_bent.json').read_text())
 if side['environment']!=0 or len(side['tetrahedra'])!=1920:raise RuntimeError(f'{name} accepted-sidecar shape mismatch')
 # The capture carries env0 and env1 accepted history; sidecar is an independent env0 serialization.
 # Read env0 state rows from capture in a second streaming pass to avoid retaining all matrices.
 env0={}
 with (run/'operator-capture').open() as f:
  for line in f:
   r=json.loads(line)
   if r.get('record')=='tetrahedron' and r['environment']==0:env0[r['tet_id']]=r
 for t,row in enumerate(side['tetrahedra']):
  tr=env0[t]
  if row['material_index']!=tr['material_index']:raise RuntimeError(f'{name} accepted sidecar material mismatch at {t}')
  if tr['material_state_count']==0:
   if row['state']:raise RuntimeError(f'{name} stateless sidecar row nonempty')
  elif [f32(x) for x in row['state']] != [f32(x) for x in tr['center']['accepted_state']]:
   raise RuntimeError(f'{name} accepted state differs at tet {t}')
 cov={}
 for (env,mat),v in sorted(branch.items()):
  cov[f'env{env}_material{mat}']={
   'tetrahedra':counts[env][mat],
   'stateful_rows':v['stateful_rows'],
   'accepted_accumulated_multiplier_abs_gt_1e-12_count':v['accepted_multiplier_gt_cutoff'],
   'candidate_multiplier_increment_gt_1e-12_count':v['candidate_multiplier_increment_gt_cutoff'],
   'candidate_multiplier_increment_le_1e-12_count':v['no_candidate_multiplier_increment_gt_cutoff'],
   'unique_material_frame_quaternion_count':len(v['frame_quaternions']),
   'nonidentity_frame_rows_xyz_norm_gt_1e-6':v['nonidentity_frames'],
   'nonzero_Fdot_rows_norm_gt_1e-12_per_s':v['Fdot_nonzero'],
   'Fdot_frobenius_norm_min_per_s':v['Fdot_norm_min'],
   'Fdot_frobenius_norm_max_per_s':v['Fdot_norm_max'],
  }
 # No actual Fdot identifier is present in the authored equation assets; Fdot is passed in the captured API but not used by these laws.
 code_files=[]
 for pathkey in ('liner_path','medium_path','glue_path'):
  asset=pathlib.Path(manifest['materials'][pathkey]); text=asset.read_text()
  active='\n'.join(line.split('//',1)[0] for line in text.splitlines())
  if 'Fdot' in active:raise RuntimeError(f'{name} unexpectedly has an Fdot constitutive identifier in {asset.name}')
  code_files.append({'path':str(asset),'sha256':manifest['materials'][pathkey.replace('_path','_sha256')], 'has_Fdot_identifier_in_noncomment_text':False})
 return {
  'run_id':name,'status':'admitted_exact_capture_fixture','source_snapshot':sb['source_snapshot'],
  'source_build_sha256':files['inputs/external/source-build.json'],'run_receipt_sha256':files['receipt.json'],
  'capture_sha256':files['operator-capture'],'package_sha256':files['output/compiled.nmatterpack'],
  'world_fingerprint_hex':header['world_fingerprint_hex'],'root':header['root'],'newton_iteration':header['newton_iteration'],
  'environment_count':header['environment_count'],'nodes_per_environment':header['node_count_per_environment'],
  'tets_per_environment':header['tetrahedron_count_per_environment'],'capture_node_rows':len(node_rows),
  'capture_tetrahedron_rows':len(tet_seen),'all_candidate_node_float_words_match_capture':True,
  'env0_accepted_state_sidecar_bitmatches_capture':True,'compiled_package_shape':package_shape,
  'capture_parameter_count':len(header['parameters_by_environment'][0]),'valid_cached_tangent_entries':tangent_valid,
  'coverage_threshold_note':'State classification uses |accepted lambda| > 1e-12 and projected lambda increment > 1e-12; the capture does not serialize a separate branch label. These counts are observable state thresholds, not an exact constitutive branch certificate.',
  'coverage':cov,
  'rate_coverage':'Fdot is nonzero in the active environment and the captured direction contract binds deltaFdot=deltaF/dt, but none of these three authored assets references Fdot in equation text. This does not cover a rate-dependent constitutive law or compare multiple physical rates.',
  'material_assets':code_files,
  'native_cache_comparison_scope':'The checker rebuilds old and projection-reuse cache kernels from one source125 metallib on identical fixture inputs; historical captured cache matrices are an auxiliary comparison, not the parity gate.',
  'physical_validation':False,'performance_claim':False,'production_qualification':False,
 }
def main():
 reports=[audit(k,v) for k,v in CASES.items()]
 doc={'schema':'numi.cardboard.tangent-cache-projection-reuse-fixture-coverage.v1','cpu_only':True,'multirate_constitutive_coverage':False,'reports':reports}
 out=pathlib.Path(__file__).resolve().parent/'fixture-coverage.json'
 out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'pass','output':str(out),'runs':[{'run_id':r['run_id'],'candidate_words_exact':r['all_candidate_node_float_words_match_capture'],'accepted_sidecar_exact':r['env0_accepted_state_sidecar_bitmatches_capture'],'valid_tangent_entries':r['valid_cached_tangent_entries']} for r in reports]},indent=2))
if __name__=='__main__':main()
