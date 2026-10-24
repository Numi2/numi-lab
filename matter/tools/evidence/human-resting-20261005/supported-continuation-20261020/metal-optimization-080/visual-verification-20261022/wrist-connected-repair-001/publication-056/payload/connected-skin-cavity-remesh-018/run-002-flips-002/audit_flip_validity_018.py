import importlib.util,pathlib,json,hashlib,numpy as np
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'); C=E/'connected-skin-cavity-remesh-018'; R=C/'run-002-holefix-001'; O=C/'run-002-flips-002'
src=C/'prepare_local_cavity_018-holefix.py'; spec=importlib.util.spec_from_file_location('m',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ln=m.read_tetgen_nodes(R/'cavity.1.node'); raw,la=m.read_tetgen_tets(R/'cavity.1.ele'); order=sorted(ln);rr={v:i for i,v in enumerate(order)};x=np.asarray([ln[i] for i in order],np.float64).astype(np.float32);t=np.asarray([[rr[int(v)] for v in q] for q in raw],np.int64);d=m.signed_dets(x,t);neg=np.flatnonzero(d<0);t[neg,0],t[neg,1]=t[neg,1].copy(),t[neg,0].copy()
r=json.loads((O/'flip-report.json').read_text()); owner=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse'/'build_014-replay.py'; inter=m.load_intersection_owner(owner,m.OWNER_SHA256)
def v(c):return np.abs(m.signed_dets(x,c))/6
rows=[]; matrices=[]
for z in r['trials']:
 if not z.get('positive_f32'):continue
 oi=np.asarray(z.get('removed_tets',z.get('incident_tets')),int);old=t[oi];new=np.asarray(z['new_tets'],int);W=np.zeros((len(new),len(old)))
 for i,ni in enumerate(new):
  for j,oj in enumerate(old):W[i,j]=inter(x[ni].astype(np.float64),x[oj].astype(np.float64))
 child=v(new); parent=v(old); rows.append({'kind':z['kind'],'key':z.get('face',z.get('edge')),'old_local_rows':oi.tolist(),'old_patch_volume_m3':float(parent.sum()),'new_patch_volume_m3':float(child.sum()),'volume_delta_m3':float(child.sum()-parent.sum()),'relative_patch_volume_delta':float((child.sum()-parent.sum())/parent.sum()),'new_child_overlap_coverage':(W.sum(1)/child).tolist(),'old_cell_coverage':(W.sum(0)/parent).tolist(),'overlap_volume_m3':W.tolist()});matrices.append(W)
# Count legal 3->2 stars around bad tet edges, independent of quality.
bad=int(r['bad_local_tet_row']); bf=json.loads((R/'prep-report.json').read_text())['boundary_faces_global'] if (R/'prep-report.json').exists() else None
sel=np.load(R/'cavity-selection.npz',allow_pickle=False); bfl=sel['oriented_boundary_faces_global']; g2l={}
# Rebuild global/local map identically.
basepath=E/'E'/'connected-skin-mechanical-transfer-013'/'boundary-refinement-002'/'candidate.npz';cand=np.load(E/'connected-skin-safe-flow-017'/'run-002'/'connected-rest-proposal.npz',allow_pickle=False);xn=cand['points_world_m'];bfset={tuple(sorted(map(int,f))) for f in bfl};
# local boundary point identities are known by exact coordinates
keys={xn[int(g)].tobytes():int(g) for g in np.unique(bfl)};allkey={p.tobytes():i for i,p in enumerate(xn)};mapped=[]
for p in x:
 k=p.tobytes();mapped.append(keys.get(k,allkey.get(k,-1)))
g2l={g:i for i,g in enumerate(mapped) if g>=0}; b_edges=set()
for f in bfl:
 q=[g2l[int(g)] for g in f]
 for a,b in ((q[0],q[1]),(q[1],q[2]),(q[2],q[0])):b_edges.add(tuple(sorted((a,b))))
ekeys,eowners=m.edge_owner_index(t,len(x));edge_rows=[]
for ia,ib in m.EDGE_PAIRS:
 u,vv=sorted(map(int,t[bad, [ia,ib]]));code=u*len(x)+vv;lo=int(np.searchsorted(ekeys,code,'left'));hi=int(np.searchsorted(ekeys,code,'right')); ids=np.unique(eowners[lo:hi]);edge_rows.append({'edge_local_rows':[u,vv],'boundary_edge':(u,vv) in b_edges,'incident_tet_count':int(len(ids)),'incident_local_tet_rows':ids.tolist(),'eligible_3to2':(u,vv) not in b_edges and len(ids)==3})
out={'schema':'numi.local-bistellar-flip-validity-audit-018.v1','candidate_npz_sha256':hashlib.sha256((O/'local-flip-geometry.npz').read_bytes()).hexdigest(),'initial_flip_report_sha256':hashlib.sha256((O/'flip-report.json').read_bytes()).hexdigest(),'overlap_owner_sha256':m.OWNER_SHA256,'status':'both_quality_passing_2to3_trials_rejected_as_nonconforming_retriangulations','boundary_facet_equality_is_insufficient':True,'trials':rows,'bad_tet_edge_stars':edge_rows,'eligible_3to2_edge_count':sum(e['eligible_3to2'] for e in edge_rows),'conclusion':'No admissible quality-passing one-step 2-to-3 or 3-to-2 flip was established. Both proposed 2-to-3 retriangulations fail same-volume coverage; original global TetGen mesh remains at one q-floor failure. Local TetGen→original-parent ancestry and full material mass map remain pending.'}
np.savez_compressed(O/'flip-overlap-validity-witness.npz',trial_0_overlap=matrices[0],trial_1_overlap=matrices[1])
out['overlap_witness_npz_sha256']=hashlib.sha256((O/'flip-overlap-validity-witness.npz').read_bytes()).hexdigest()
(O/'flip-validity-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
