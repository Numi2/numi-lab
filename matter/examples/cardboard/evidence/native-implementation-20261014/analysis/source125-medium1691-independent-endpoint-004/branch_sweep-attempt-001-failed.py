#!/usr/bin/env python3
"""Strict FP64 endpoint branch census with recovered tet1691 plus root."""
import datetime,gzip,hashlib,importlib.util,json,pathlib,warnings
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014'); PKG=BASE/'analysis/source125-medium1691-independent-endpoint-004'
CAP=BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'; SWEEP=BASE/'analysis/native-residual-fd-sweep-result-424'
D=importlib.util.spec_from_file_location('diagnose',PKG/'diagnose.py'); d=importlib.util.module_from_spec(D); D.loader.exec_module(d)
a=d.load(d.SCRIPT003,'audit003_branch'); ref=d.load(d.REFPATH,'ref_branch'); e2=d.load(d.E2/'audit_capture.py','e2_branch'); fd=d.load(d.FDHELP,'fd_branch')
records=[json.loads(x) for x in CAP.read_text().splitlines() if x.strip()]; header=records[0]
tets=sorted((x for x in records if x.get('record')=='tetrahedron' and x.get('environment')==0),key=lambda x:int(x['tet_id']))
inv=np.asarray([a.f32(t['inverse_rest_rows_per_m']) for t in tets]); frames=np.asarray([a.frame_from_xyzw(t['material_frame_quaternion_xyzw']) for t in tets],dtype=np.float32)
active=np.asarray([int(a.f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0])==1 for t in tets])
ids=np.asarray([int(t['material_index']) for t in tets]); paper=np.flatnonzero(np.isin(ids,[0,1])); mats={int(m['index']):m for m in header['materials']}; ptab=a.f32(header['parameters_by_environment'][0])
p=np.stack([ptab[int(mats[ids[i]]['parameter_offset']):int(mats[ids[i]]['parameter_offset'])+len(mats[ids[i]]['parameter_names'])] for i in paper])
old=np.stack([a.f32(tets[i]['center']['accepted_state']) for i in paper])

def nodes(stem): return np.fromfile(SWEEP/'result'/f'{stem}-nodes.bin',dtype='<f4').reshape(2,882,5,4)[0]
Fs={}
for label,stem in [('center','sample-0-0'),('plus','sample-6-0'),('minus','sample-6-1')]:
 hi,lo=a.source_deformation_words(nodes(stem),tets,inv,frames,active); Fs[label]=(hi.astype(np.float64)+lo.astype(np.float64))[paper]
for i,t in enumerate(tets):
 if i in paper: Fs['center'][np.where(paper==i)[0][0]]=a.f32(t['center']['F_local'])+a.f32(t['center'].get('F_local_low',np.zeros((3,3))))
sol=json.loads((PKG/'result.json').read_text()); zplus1691=np.asarray(sol['recovered_endpoint']['state']); j1691=int(np.where(paper==1691)[0][0])

def strict_project(label):
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  z,it,rn,cn,ok,reasons,refine=e2.project_centers(ref,fd,p,Fs[label],old,p[:,26])
 warn=[{'category':w.category.__name__,'message':str(w.message),'filename':w.filename,'line':w.lineno} for w in caught]
 return z,it,rn,cn,ok,reasons,refine,warn
runs={}
for side in ('center','plus','minus'):
 z,it,rn,cn,ok,reasons,refine,warn=strict_project(side)
 if side=='plus':
  z[j1691]=zplus1691; ok[j1691]=True
  rn[j1691]=sol['recovered_endpoint']['scaled_residual_inf']; cn[j1691]=sol['recovered_endpoint']['dynamic_scaled_correction_inf']; it[j1691]=None
  reasons[j1691]=None
 assert np.all(ok),f'{side} unresolved row indices {np.flatnonzero(~ok)[:10]}'
 runs[side]={'z':z,'iterations':it,'residual':rn,'correction':cn,'warnings':warn}

def sig(side):
 z=runs[side]['z']; q,_=ref.hill(p,z[:,6:12]); lam=z[:,12]-old[:,12]
 radius=1+p[:,17]*p[:,19]*p[:,21]*z[:,13]/p[:,10]+p[:,18]*p[:,20]*p[:,22]*z[:,14]/p[:,10]
 gap=radius-q
 return lam,gap,q
branches={s:sig(s) for s in runs}
summary={'all_rows':len(p),'by_material':{},'transitions':{},'yield_gap_distribution':{},'strict_gates':{}}
for s in runs:
 lam,gap,q=branches[s]; summary['strict_gates'][s]={'valid_rows':int(len(runs[s]['residual'])),'max_scaled_residual':float(np.max(runs[s]['residual'])),'max_dynamic_correction':float(np.max(runs[s]['correction']))}
 summary['yield_gap_distribution'][s]={'minimum':float(np.min(gap)),'minimum_absolute':float(np.min(np.abs(gap))),'p01_absolute':float(np.quantile(np.abs(gap),.01)),'median_absolute':float(np.median(np.abs(gap))),'positive_count':int(np.count_nonzero(gap>0)),'negative_count':int(np.count_nonzero(gap<0)),'zero_count':int(np.count_nonzero(gap==0))}
 for mi,name in [(0,'liner'),(1,'medium')]:
  ix=ids[paper]==mi; summary['by_material'].setdefault(str(mi),{})[s]={'name':name,'rows':int(ix.sum()),'lambda_positive':int(np.count_nonzero(lam[ix]>0)),'lambda_nonpositive':int(np.count_nonzero(lam[ix]<=0)),'yield_gap_positive':int(np.count_nonzero(gap[ix]>0)),'yield_gap_nonpositive':int(np.count_nonzero(gap[ix]<=0)),'min_abs_yield_gap':float(np.min(np.abs(gap[ix])))}
for side in ('plus','minus'):
 cl,cg,_=branches['center']; sl,sg,_=branches[side]
 summary['transitions'][side]={'lambda_active_flip_count':int(np.count_nonzero((cl>0)!=(sl>0))),
   'yield_gap_sign_flip_count':int(np.count_nonzero((cg>0)!=(sg>0))),
   'lambda_flip_ids':[int(paper[i]) for i in np.flatnonzero((cl>0)!=(sl>0))],
   'yield_gap_flip_ids':[int(paper[i]) for i in np.flatnonzero((cg>0)!=(sg>0))]}

def hashfile(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
report={'schema':'numi.cardboard.source125-native424-endpoint-branch-census.v1','status':'all1728_strict_fp64_projections_available_with_tet1691_continuation_override',
 'scope':'paper rows only, exact source-transcribed native endpoints and unchanged accepted history; diagnostic branch census, no qualification',
 'bindings':{'source_runtime_sha256':hashfile(d.SOURCE/'matter/src/runtime.mm'),'capture419_sha256':hashfile(CAP),'native424_receipt_sha256':hashfile(SWEEP/'receipt.json'),
  'native424_plus_nodes_sha256':hashfile(SWEEP/'result/sample-6-0-nodes.bin'),'native424_minus_nodes_sha256':hashfile(SWEEP/'result/sample-6-1-nodes.bin'),
  'audit003_report_sha256':hashfile(BASE/'analysis/source125-kinematics-and-native424-force-audit-003/report.json'),
  'tet1691_continuation_sha256':hashfile(PKG/'result.json'),'fp64_reference_sha256':hashfile(d.REFPATH)},
 'summary':summary,'observed_numpy_warnings':sorted({(w['category'],w['message'],w['filename'],w['line']) for x in runs.values() for w in x['warnings']}),
 'tet1691':{s:{'accepted_state_unchanged':True,'lambda_increment':float(branches[s][0][j1691]),'lambda_active':bool(branches[s][0][j1691]>0),
   'yield_gap':float(branches[s][1][j1691]),'q_hill':float(branches[s][2][j1691]),'scaled_residual':float(runs[s]['residual'][j1691]),'dynamic_scaled_correction':float(runs[s]['correction'][j1691]),
   'iterations':None if runs[s]['iterations'][j1691] is None else int(runs[s]['iterations'][j1691])} for s in runs},
 'warning_note':'The reference computes both np.where branches; on some zero-denominator inactive rows NumPy reports divide-by-zero/invalid in the unselected rationalized branch. Outputs were checked finite and the strict gates are separately recorded; warning is retained, not silenced.',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out=PKG/'branch-sweep.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
with gzip.open(PKG/'branch-signatures.jsonl.gz','wt',compresslevel=6,encoding='utf-8') as f:
 for j,ti in enumerate(paper):
  f.write(json.dumps({'tet_id':int(ti),'material_index':int(ids[ti]),**{s:{'lambda_increment':float(branches[s][0][j]),'lambda_active':bool(branches[s][0][j]>0),'yield_gap':float(branches[s][1][j]),'q_hill':float(branches[s][2][j]),'scaled_residual':float(runs[s]['residual'][j]),'dynamic_scaled_correction':float(runs[s]['correction'][j]),'iterations':None if runs[s]['iterations'][j] is None else int(runs[s]['iterations'][j])} for s in runs}},sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'branch_report':str(out),'status':report['status'],'summary':summary,'tet1691':report['tet1691'],'warnings':report['observed_numpy_warnings']},indent=2,allow_nan=False))
