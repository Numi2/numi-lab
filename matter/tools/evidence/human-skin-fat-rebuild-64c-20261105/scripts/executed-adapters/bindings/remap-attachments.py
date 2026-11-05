from pathlib import Path
import csv,json,hashlib,numpy as np,itertools,time
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root');H=S.parent;P=H.parent;out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldrun=P/'native-registered-volume-parent-identity-physical-control-365'; base=json.loads((oldrun/'launch.json').read_text()); env=base['environment']; oldcsv=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_NODES_CSV']); old=list(csv.DictReader(oldcsv.open()))
assert sha(oldcsv)==env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_NODES_SHA256']
reqpath=oldrun/'inputs/deep-attachment-requests.csv'; requests=list(csv.DictReader(reqpath.open())); assert len(requests)==14
export=Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/volume-export'); newcsv=export/'registered/registered-D.csv'; new=list(csv.DictReader(newcsv.open())); src=export/'roles/D-source.json'; doc=json.loads(src.read_text()); nodes=np.array([r['child_volume_node_row'] for r in doc['vertices']]); ids=np.array([int(r['source_vertex_row']) for r in new]); donor=np.array([int(r['donor']) for r in new]); xyz=np.array([[float(r[k]) for k in ['wx','wy','wz']] for r in new]); assert np.array_equal(ids,np.array([r['source_vertex_row'] for r in doc['vertices']]))
vol=Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/rich433/candidate-current-typed-volume.npz'); assert sha(vol)=='0152c7b15febfc81f4d9ce995820a80438204635deb425193212409e31c5c323'
z=np.load(vol,allow_pickle=False); points=z['points_reference_world_m_f64'].astype(np.float32).astype(np.float64); tets=z['volume_tetrahedra']; assert np.array_equal(xyz,points[nodes]); weights=z['full86_weights']; lanes={int(r['donor']):int(r['binding']) for r in csv.DictReader((export/'registered/donor-frames.csv').open())}; confidence=np.array([weights[n,lane] for n,lane in zip(nodes,[lanes[int(d)] for d in donor])]); altitude=np.full(len(points),np.inf)
cache=out/'prepared-quality.npz'
# Minimum altitude of every incident tetrahedron, using all four opposite faces.
for start in range(0,len(tets),65536):
 tt=tets[start:start+65536];q=points[tt];e=q[:,1:]-q[:,:1]
 vol6=np.abs(np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2])))
 area2=np.stack([np.linalg.norm(np.cross(q[:,b]-q[:,a],q[:,c]-q[:,a]),axis=1) for a,b,c in [(1,2,3),(0,3,2),(0,1,3),(0,2,1)]],axis=1)
 assert np.all(vol6>0) and np.all(area2>0)
 # Match the owning attachment gate: altitude opposite this vertex in each incident cell.
 for j in range(4):np.minimum.at(altitude,tt[:,j],vol6/area2[:,j])
np.savez_compressed(cache,altitude=altitude,confidence=confidence,nodes=nodes)
maximum_move_m=0.010; minimum_spacing_m=0.025; min_alt_m=0.00005; min_confidence=0.85
selected={};audit=[]
for body in sorted({int(r['donor_body']) for r in requests}):
 group=[(k,r) for k,r in enumerate(requests) if int(r['donor_body'])==body]; assert len(group)<=3; lists=[]; oldpoints=[]
 for k,r in group:
  v=old[int(r['stable_source_vertex'])];assert int(v['donor'])==body and v['source_domain']==r['source_domain'];x=np.array([float(v[c]) for c in ['wx','wy','wz']]);oldpoints.append(x);distance=np.linalg.norm(xyz-x,axis=1);valid=(donor==body)&(confidence>=min_confidence)&(altitude[nodes]>=min_alt_m)&(distance<=maximum_move_m);cand=np.flatnonzero(valid);cand=cand[np.lexsort((ids[cand],distance[cand]))];assert len(cand)>0,(r,'no eligible local site');lists.append([(int(j),float(distance[j])) for j in cand[:64]])
 best=None
 for combo in itertools.product(*lists):
  if any(np.linalg.norm(xyz[combo[i][0]]-xyz[combo[j][0]])<minimum_spacing_m for i in range(len(combo)) for j in range(i)):continue
  key=(sum(d*d for j,d in combo),tuple(int(ids[j]) for j,d in combo))
  if best is None or key<best[0]:best=(key,combo)
 assert best is not None,('cannot retain spacing within declared displacement',body)
 for (k,r),(j,d),oldpoint in zip(group,best[1],oldpoints):
  selected[k]={'stable_attachment_id':r['stable_attachment_id'],'donor_body':body,'stable_source_vertex':int(ids[j]),'source_domain':new[j]['source_domain']}
  others=[xyz[a[0]] for a in best[1] if a[0]!=j];audit.append({'stable_attachment_id':int(r['stable_attachment_id']),'donor_body':body,'old_source_vertex':int(r['stable_source_vertex']),'new_source_vertex':int(ids[j]),'new_compact_surface_row':j,'volume_node_row':int(nodes[j]),'old_world_m':oldpoint.tolist(),'current_world_m':xyz[j].tolist(),'displacement_m':d,'donor_weight':float(confidence[j]),'minimum_incident_altitude_m':float(altitude[nodes[j]]),'minimum_same_donor_spacing_m':min([float(np.linalg.norm(xyz[j]-v)) for v in others],default=None)})
path=out/'deep-attachment-requests.csv'
with path.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['stable_attachment_id','donor_body','stable_source_vertex','source_domain']);w.writeheader();w.writerows(selected[i] for i in range(14))
receipt={'status':'PASS_REFERENCE_ATTACHMENT_REMAP_ONLY','physical_admission':False,'native_steps':0,'method':'deterministic minimum-squared-displacement donor-preserving current D selection; all-pairs spacing checked jointly per donor; no hard constraints added','limits':{'maximum_displacement_m':maximum_move_m,'minimum_spacing_m':minimum_spacing_m,'minimum_donor_weight':min_confidence,'minimum_incident_altitude_m':min_alt_m,'maximum_attachments_per_donor':3},'inputs':{str(p):sha(p) for p in [oldrun/'launch.json',oldcsv,reqpath,src,newcsv,vol,export/'registered/donor-frames.csv',cache]},'output':{'path':str(path),'sha256':sha(path)},'rows':sorted(audit,key=lambda a:a['stable_attachment_id']),'geometry_changed':False,'mass_changed':False,'stable_attachment_ids_preserved':True}
(out/'remap-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
