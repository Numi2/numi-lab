from pathlib import Path
import json,hashlib,numpy as np,importlib.util
N=Path(__file__).resolve().parent.parent;T=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location("helpers",N/"select_batch_endpoint_operations546.py");h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
st=json.loads((N/"topology/round-007-wide-chain-025/state.json").read_text());base=Path(st["candidate"]);assert sha(base)==st["candidate_sha256"]
raw=T/"raw-directions/raw-direction-001/candidate.npz";diagnostic=(T/"quality-projected/candidate.npz" if (T/"quality-projected/candidate.npz").exists() else T/"projection-failure-diagnostic/failed-iterate.npz");proof=(diagnostic.parent/"report.json")
pins={str(p):sha(p)for p in[base,raw,diagnostic,proof,Path(__file__),N/"select_batch_endpoint_operations546.py"]}
b=np.load(base,allow_pickle=False);y=np.load(diagnostic,allow_pickle=False)["points_world_m"];r=json.loads((T/"raw-directions/raw-direction-001/report.json").read_text());assert r["base_sha256"]==st["candidate_sha256"]
assert y.dtype==np.float32 and y.shape==b["points_world_m"].shape
anchors=b["typed_coarse_anchor_output_rows_018"];assert y[anchors].tobytes()==b["points_world_m"][anchors].tobytes()
free=np.array(r["free_nodes"],int);mask=np.ones(len(y),bool);mask[free]=False;assert y[mask].tobytes()==b["points_world_m"][mask].tobytes()
c=h.loadmod(N.parent/"contained-clearance-545/run_candidate_checks545.py","checks");owner=c.mod(c.OWNER);src,_=owner.compose_source_reference(b,c.R0)
w=h.arithmetic_witnesses(y,b["points_world_m"],b["source_O_faces"],b["area_reference_037_points_world_m_f64"],src,1.,12.);assert all(not len(v)for v in w.values()),str(w)
rows=np.flatnonzero(np.any(y!=b["points_world_m"],axis=1));payload={k:b[k]for k in b.files};payload["points_world_m"]=y;payload["local_witness_cluster_222_support_rows"]=rows;payload["local_witness_cluster_222_delta_m"]=y.astype(float)-b["points_world_m"].astype(float)
out=T/"projected-actual-gates";out.mkdir(exist_ok=False);np.savez_compressed(out/"candidate.npz",**payload)
j=dict(status="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED",inputs=dict(base_candidate_sha256=st["candidate_sha256"]),base_candidate=str(base),base_sha256=st["candidate_sha256"],candidate_sha256=sha(out/"candidate.npz"),source_pins=pins,free_nodes=free.tolist(),changed_nodes=len(rows),native_admission=False,accepted_physical_steps=0,semantics="Raw successful SCP field with bounded local quality projection; unchanged actual area .1, edge .5, trust1mm, source12mm, anchors and outside-free checks verified; exact geometry remains unadmitted",projector_failure_retained=str(proof))
assert all(sha(p)==v for p,v in pins.items());(out/"report.json").write_text(json.dumps(j,indent=2)+"\n");print(j["candidate_sha256"],len(rows))
