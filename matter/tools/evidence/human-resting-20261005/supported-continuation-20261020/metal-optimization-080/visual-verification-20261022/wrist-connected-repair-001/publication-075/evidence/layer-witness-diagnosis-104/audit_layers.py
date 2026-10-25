from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
from scipy import sparse
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
K=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001'
P=D/'layer-witness-diagnosis-104';P.mkdir(exist_ok=True)
I=D/'connected-exterior-clearance-100/run-001/iteration-004'
INPUT=K/'candidate-tetrahedra-compacted.npz';z=np.load(INPUT,allow_pickle=False)
X=np.load(I/'candidate.npz')['points_world_f32'];O=z['current037_points_m_f32'];rows=z['source_input_row']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(INPUT)=='5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db'
assert np.array_equal(np.sort(rows[rows>=0]),np.arange(7233))
inverse=np.empty(7233,dtype=np.int64);direct=np.flatnonzero(rows>=0);inverse[rows[direct]]=direct
def layer(node):
 r=int(rows[int(node)])
 return 'generated' if r<0 else ['O','I','D'][r//2411]
sourceO=inverse[z['source_O_faces']]
outeredges={tuple(e) for e in np.sort(np.concatenate([sourceO[:,[0,1]],sourceO[:,[1,2]],sourceO[:,[2,0]]]),axis=1).tolist()}
samples=json.loads((I/'inside-edge-span-samples.json').read_text())['samples']
pairs=list(csv.DictReader((I/'midpoint-query/pairs.csv').open()))
lookup={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])):r['classification'] for r in pairs}
inside=[s for s in samples if lookup.get((int(s['sample_query_row']),int(s['body']),int(s['bone_stable_id'])))=='inside']
regional={}
for name,bodies in [('head_body_23',{23}),('selected_humerus_ulna_radius_bodies',{41,42,43,91,92,93})]:
 selected=[s for s in inside if int(s['body'])in bodies]
 counts=collections.Counter();oo=0;crosscol_id=0;samecol_id=0
 for s in selected:
  a=int(s['mechanics_node_a']);b=int(s['mechanics_node_b'])
  tags=[layer(a),layer(b)];key='-'.join(sorted(tags,key=lambda x:['O','I','D','generated'].index(x)));counts[key]+=1
  if tags==['O','O'] and tuple(sorted([a,b]))in outeredges:oo+=1
  if set(tags)=={'I','D'}:
   if rows[a]%2411==rows[b]%2411:samecol_id+=1
   else:crosscol_id+=1
 regional[name]={'target_inside_spans':len(selected),'by_endpoint_layers':dict(sorted(counts.items())),
    'O_O_spans_that_are_actual_source_O_face_edges':oo,'I_D_same_column':samecol_id,'I_D_cross_column':crosscol_id}
proj=[];slack=[]
for layeridx in [0,1]:
 a=inverse[layeridx*2411:(layeridx+1)*2411];b=inverse[(layeridx+1)*2411:(layeridx+2)*2411]
 d=O[a].astype(float)-O[b];length=np.linalg.norm(d,axis=1)
 distance=np.einsum('ij,ij->i',X[a].astype(float)-X[b],d/length[:,None])
 ratio=distance/length
 proj.append({'layer':'O-I' if layeridx==0 else 'I-D','min':float(ratio.min()),'max':float(ratio.max())})
 slack.extend((distance-.5*length).tolist());slack.extend((1.5*length-distance).tolist())
generated=sparse.load_npz(D/'connected-exterior-clearance-100/run-001/generated-layer-rows.npz')
gt=np.load(D/'connected-exterior-clearance-100/run-001/generated-layer-targets.npy')
gslack=generated@X.astype(float).ravel()-gt
report={'candidate_sha256':sha(I/'candidate.npz'),'candidate':str(I/'candidate.npz'),
 'input_sha256':sha(INPUT),'sample_map_sha256':sha(I/'inside-edge-span-samples.json'),
 'actual_target_query_pairs_sha256':sha(I/'midpoint-query/pairs.csv'),
 'source_input_row_is_authoritative':True,'source_global_node_ids_are_labels_only':True,
 'target_inside_span_count':len(inside),'regions':regional,
 'original_layer_projection_ratios':proj,'minimum_original_projected_thickness_slack_mm':float(min(slack)*1000),
 'minimum_generated_projected_thickness_slack_mm':float(gslack.min()*1000),
 'outer_nodes_displaced_over_one_micrometre':int(np.count_nonzero(np.linalg.norm(X[inverse[:2411]].astype(float)-O[inverse[:2411]],axis=1)>1e-6)),
 'maximum_outer_node_displacement_mm':float(np.linalg.norm(X[inverse[:2411]].astype(float)-O[inverse[:2411]],axis=1).max()*1000),
 'conclusion':'Head witnesses are predominantly deep-layer; arms include verified outer edges. Relative thickness bounds are inactive here. No global thickness scaling is justified by this diagnostic.',
 'accepted_physical_steps':0,'geometry_admitted':False,'visual_inspection_performed':False}
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
