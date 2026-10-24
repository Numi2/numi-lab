from pathlib import Path
import csv,json,hashlib,importlib.util
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';O=E/'source-cut-endpoint-distance-275';O.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('rb275',E/'batch-source-compose-252/tools/rebind_human_skin_reference.py');rb=importlib.util.module_from_spec(sp);sp.loader.exec_module(rb)
skin=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin';reg=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin.csv';report=E/'bilateral-source-cut-plane-271/report.json';d=rb.decode(skin.read_bytes());r=json.loads(report.read_text());rows=list(csv.DictReader(reg.open()));assert [int(v['vertex']) for v in rows]==list(range(d['nv']))
X=np.asarray([[float(v[k]) for k in ['wx','wy','wz']] for v in rows]);F=d['inds'];edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
n=np.array(r['plane']['normal_actual007']);origin=np.array(r['plane']['origin_actual007_m']);s=(X-origin)@n;es=s[edges];length=np.linalg.norm(X[edges[:,1]]-X[edges[:,0]],axis=1)
results=[]
for plane in [.650,r['plane']['distal_s0_m']]:
 mask=(es.min(axis=1)<plane)&(es.max(axis=1)>plane);active=edges[mask];params=(plane-es[mask,0])/(es[mask,1]-es[mask,0]);distance=np.minimum(params,1-params)*length[mask];order=np.argsort(distance)
 results.append({'plane_m':plane,'crossed_source_edges':int(mask.sum()),'minimum_edge_endpoint_distance_mm':float(distance.min()*1000),'counts_below_mm':{str(th):int(np.sum(distance<th/1000)) for th in [.05,.1,.2]},'worst_source_edges':active[order[:10]].astype(int).tolist(),'worst_distances_mm':(distance[order[:10]]*1000).tolist()})
out={'inputs_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [skin,reg,report]},'method':'Independent source-edge graph verification of old and selected construction cut plane; no new source vertex motion. No claim of continuous optimality.','results':results};(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

