from pathlib import Path
import hashlib,importlib.util,json
import mpmath as mp
import numpy as np
b=Path('/Users/home/cardboard-implementation-20261014');o=b/'analysis/three-exact-captured-state-matrices-315-005';o.mkdir(exist_ok=False)
refpath=b/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py';npzpath=b/'analysis/retained-paper-allpoint-reference-296-001/reference.npz';case=b/'native/finer-controlled-material-sweep-315'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[refpath,npzpath,case/'receipt.json',case/'prediction.json',case/'inputs/external/source-build.json']+[case/'output'/f'paper-derivative-sweep-{p}.jsonl' for p in ('default','accurate')]
bindings=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashfile(p)) for p in inputs]
(o/'prediction.json').write_text(json.dumps(dict(schema='numi.cardboard.exact-center-matrix-prediction.v1',question='Do all nine native material actions at the eight selected inputs agree with independent70/90-digit exact two-word-center matrices?',scope='Eight recorded points comprise three unique parameter/F/history states; prove identity before deduplicating computation, retain each point result.',precision=[70,90],gate='nonzero column norm1e-3 native,1e-6 FP64; absolute near-zero floors remain inconclusive',stopping='All three unique states, both precisions, all nine columns, both native policies; preserve every failure.',inputs=bindings,physical_steps=0),indent=2)+'\n')
spec=importlib.util.spec_from_file_location('independent',refpath);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
data=np.load(npzpath);idxmap={tuple(k):i for i,k in enumerate(data['keys'])}
def f32(words):return np.asarray([int(x,16) for x in words],dtype='<u4').view('<f4').astype(float)
def metric(a,z):
 a=np.asarray(a,float);z=np.asarray(z,float);d=a-z;n=float(np.linalg.norm(z));return dict(reference_norm=n,absolute_l2=float(np.linalg.norm(d)),max_absolute=float(np.max(abs(d))),relative_l2=float(np.linalg.norm(d)/n) if n else None)
allrows={p:[json.loads(l) for l in (case/'output'/f'paper-derivative-sweep-{p}.jsonl').read_text().splitlines()][1:-1] for p in ('default','accurate')}
groups={};points=[]
for tet in sorted({r['point']['tet_id'] for r in allrows['accurate']}):
 centers=[r for r in allrows['accurate'] if r['point']['tet_id']==tet and r['kind']=='center_repeat'];assert len(centers)==3;c=centers[0];pt=c['point'];raw=f32(pt['input_f32_bits']);old=f32(pt['accepted_state_f32_bits']);keys=[k for k in idxmap if k[0]==pt['environment'] and k[1]==tet];assert len(keys)==1;idx=idxmap[keys[0]];p=np.asarray(data['parameters'][idx],dtype=np.float32).astype(float);assert np.array_equal(old,np.asarray(data['accepted'][idx],float))
 signature=json.dumps([p.tolist(),pt['input_f32_bits'][:18],pt['accepted_state_f32_bits']],separators=(',',':'));key=hashlib.sha256(signature.encode()).hexdigest();points.append(dict(tet_id=tet,unique_state_sha256=key))
 if key in groups:groups[key]['tets'].append(tet);continue
 record=dict(tets=[tet],parameter_f32=p.tolist(),input_f32_bits=pt['input_f32_bits'],accepted_state_f32_bits=pt['accepted_state_f32_bits']);precisions={}
 for digits in (70,90):
  mp.mp.dps=digits;fm=[ref.mpf(a)+ref.mpf(z) for a,z in zip(raw[:9],raw[9:18])];sys=ref.system(p,fm,old);seed=f32(c['output_f32_bits']);seed=seed[35:50]+seed[50:65]
  root,it,norm=ref.project(sys[4],seed);mat=ref.mp_tangent_matrix(p,fm,old,root);stress=sys[5](*list(root))
  precisions[str(digits)]=dict(iterations=it,scaled_residual=mp.nstr(norm,24),root=[mp.nstr(v,30) for v in root],stress=[mp.nstr(v,30) for v in stress],matrix=[[mp.nstr(mat[i,j],30) for j in range(9)] for i in range(9)],multiplier=mp.nstr(root[12]-ref.mpf(old[12]),24))
  print('reference',tet,digits,'iterations',it,flush=True)
 record['precisions']=precisions;m=np.asarray(precisions['90']['matrix'],float);record['mp70_vs_mp90']=metric(np.asarray(precisions['70']['matrix'],float),m)
 # The FP64 reference necessarily widens the exact pair into one FP64 F;
 # retain this input rounding distinction and compare rather than assuming equality.
 f=(raw[:9]+raw[9:18]).reshape(3,3);z,trace=ref.project_fp64_scaled_correction(p,f,old);fp=ref.tangent_matrix_fp64_stable(p,f,old,z);record['fp64_projection_trace']=trace;record['fp64_vs_exact_mp90_matrix']=metric(fp,m);record['fp64_vs_exact_mp90_columns']=[metric(fp[:,i],m[:,i]) for i in range(9)];record['native']={}
 for pol in ('default','accurate'):
  selected=[r for r in allrows[pol] if r['point']['tet_id']==tet and r['kind']=='basis_action'];assert len(selected)==9;selected=sorted(selected,key=lambda r:r['basis']);assert all(r['projection_valid'] and r['stress_valid'] and r['tangent_valid'] for r in selected)
  assert all(r['point']['input_f32_bits'][:18]==pt['input_f32_bits'][:18] and r['point']['accepted_state_f32_bits']==pt['accepted_state_f32_bits'] for r in selected)
  matrix=np.stack([f32(r['output_f32_bits'])[26:35] for r in selected],axis=1);record['native'][pol]=dict(matrix=matrix.tolist(),full=metric(matrix,m),columns=[metric(matrix[:,i],m[:,i]) for i in range(9)])
 groups[key]=record
assert len(groups)==3
# Verify all duplicate points have exactly identical nine-action results, not just equal inputs.
for key,g in groups.items():
 for pol in ('default','accurate'):
  first=[r['output_f32_bits'] for r in sorted([r for r in allrows[pol] if r['point']['tet_id']==g['tets'][0] and r['kind']=='basis_action'],key=lambda r:r['basis'])]
  for tet in g['tets'][1:]:assert first==[r['output_f32_bits'] for r in sorted([r for r in allrows[pol] if r['point']['tet_id']==tet and r['kind']=='basis_action'],key=lambda r:r['basis'])]
 g['duplicate_action_words_identical']=True
report=dict(schema='numi.cardboard.exact-center-matrices.v1',inputs=bindings,points=points,unique_states=groups,absolute_noise_floor=None,physical_steps=0,physical_validation=False,limits=['Only eight selected points/three unique states, not all3456captured derivatives.','Native law and reference are rate independent; rounded rate-direction errors elsewhere remain recorded.','FP64 reference uses rounded widened F; MP operands retain exact sum of both binary32 words at declared precision.','Nonzero column/matrix norm comparisons do not qualify near-zero components, derivative finite-difference resolution, element assembly or complete physical trajectories.'])
(o/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(o/'run.py').write_bytes(Path(__file__).read_bytes());(o/'README.md').write_text('# Three unique exact captured states\n\nIndependent70/90-digit roots and all nine material actions are evaluated from exact binary32 high-plus-low F operands, accepted history and parameter words. Eight source-selected points reduce to three unique states only after input identity is proved; all duplicate native action words are also checked. Two native policies and the separately rounded FP64 reference retain all nonzero-column errors. Absolute near-zero floors and trajectory qualification remain open.\n');(o/'manifest.json').write_text(json.dumps(dict(inputs=bindings,artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashfile(p)) for p in o.iterdir() if p.is_file()}),indent=2)+'\n')
print(json.dumps([dict(tets=g['tets'],fp64_worst=max(c['relative_l2'] for c in g['fp64_vs_exact_mp90_columns']),native={p:dict(matrix=v['full']['relative_l2'],worst=max(c['relative_l2'] for c in v['columns'])) for p,v in g['native'].items()}) for g in groups.values()],indent=2),flush=True)
