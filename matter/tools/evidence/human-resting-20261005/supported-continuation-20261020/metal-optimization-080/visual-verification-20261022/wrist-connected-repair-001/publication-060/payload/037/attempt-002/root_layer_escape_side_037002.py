import numpy as np,json,hashlib
from pathlib import Path
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");f=e/'bone-clearance-fit-037-componentwise-035affine-002/candidate.npz';g=e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
a=np.load(f);b=np.load(g);nodes=a['witness_local_rows'];cols=nodes%2411;layers=nodes//2411;x0=b['points'];n=x0[cols+2411]-x0[cols+4822];n/=np.linalg.norm(n,axis=1)[:,None]
align=np.einsum('ij,ij->i',a['witness_normal_world'],n);move=np.einsum('ij,ij->i',a['points_world_f32'][nodes]-x0[nodes],n)*1000
ix=np.flatnonzero(nodes==5768)[0]
result={'status':'ANATOMICAL_SIDE_CONCERN_REQUIRES_EDGE_ENTRY_AUDIT','candidate_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'D_witnesses':int((layers==2).sum()),'D_escape_planes_opposing_baseline_D_to_I':int(((layers==2)&(align<0)).sum()),'D_inward_max_mm':float(-move[layers==2].min()),'head_remaining_local_node':5768,'head_remaining_global_node':7434,'head_plane_alignment_to_reference_outward':float(align[ix]),'head_displacement_outward_mm':float(move[ix]),'limits':['Column direction is a construction reference, not independently measured anatomical normal.','Negative alignment alone does not prove intracranial or wrong-side placement.','Nearest-face escape is not an anatomical-side guarantee; audit actual O-I/I-D material edges and outer-connected entry.']}
(e/'root-layer-escape-side-diagnostic-037002.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

