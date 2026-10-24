import pathlib,json,hashlib,csv,numpy as np,re
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");r=e/"root-reference-relative-j-062";p=r/"source-preflight/identity-reference-points.csv";c=e/"bone-clearance-fit-037-componentwise-035affine-002/candidate.npz";report=r/"material-specific-reference-J-062.json"
a=np.load(c);t=a["tetrahedra"].astype(np.int64);x=a["points_world_f32"].astype(np.float32).astype(float);mat=a["material_ids"]
rows=list(csv.DictReader(p.open()));assert [int(z["local_volume_node"]) for z in rows]==list(range(7233));ref=np.array([[z[k] for k in ["x_m","y_m","z_m"]] for z in rows],float)
def det(z):
 y=z[t];return np.linalg.det(y[:,1:]-y[:,:1])
J=det(x)/det(ref);expected=json.load(report.open());records={}
for m,name in [(1,"skin"),(2,"fat")]:
 z=expected["materials"][name];material=pathlib.Path(z["path"]);b=material.read_bytes();assert hashlib.sha256(b).hexdigest()==z["sha256"]
 text=b.decode();lo=float(re.search(r"minimum_J\s*=\s*([0-9.eE+-]+)",text).group(1));hi=float(re.search(r"maximum_J\s*=\s*([0-9.eE+-]+)",text).group(1));assert lo==z["minimum_J"] and hi==z["maximum_J"]
 v=J[mat==m];rec=dict(cells=len(v),below=int((v<lo).sum()),above=int((v>hi).sum()),inverted=int((v<=0).sum()),min_J=float(v.min()),max_J=float(v.max()),minimum_J=lo,maximum_J=hi)
 er=expected["results"]["candidate_037_002"][name];assert rec["below"]==er["below_minimum_J_count"] and rec["above"]==er["above_maximum_J_count"]
 records[name]=rec
out=dict(status="CONFIRMED_162_DESCRIPTOR_J_VIOLATIONS",tetrahedra=len(t),candidate_sha256=hashlib.sha256(c.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reviewed_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),materials=records,total_violations=sum(z["below"]+z["above"] for z in records.values()),limitations="Offline owner-reference determinant check, not a bitwise Metal replay; no physical admission.")
assert out["total_violations"]==162
(e/"root-independent-material-J-owner-bounds-062.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))

