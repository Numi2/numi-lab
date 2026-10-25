import csv,hashlib,json,socket,sys,time
from pathlib import Path
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"); W=Path("/Users/n/Work/294-source-frame-audit-001")
S=E/"accepted-transfer-row-export-075/run-001/source-preflight"; TR=S/"accepted-transfer-rows.csv"; CO=S/"accepted-coarse-reference.csv"; N35=S/"initial-skin-fem-nodes.csv"; NPZ=W/"reference-fit-layer-counterfactual-001/run-001/whole-layer-reference-fit.npz"; RPT=W/"reference-fit-layer-counterfactual-001/run-001/report.json"; V=S/"validation-report.json"; PAY=E/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"; OUT=Path(__file__).resolve().parent/"frame-audit.json"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(newline="") as f:return list(csv.DictReader(f))
def bounds(x):
 x=np.asarray(x,dtype=np.float64);return {"n":len(x),"min":x.min(0).tolist(),"max":x.max(0).tolist()}
def ds(x):
 d=np.linalg.norm(np.asarray(x,dtype=np.float64),axis=1);return {"n":len(d),"max_m":float(d.max()),"p99_m":float(np.quantile(d,.99)),"median_m":float(np.median(d)),"rms_m":float(np.sqrt(np.mean(d*d)))}
t0=time.time()
for p in [TR,CO,N35,NPZ,RPT,V,PAY]:
 if not p.is_file():raise FileNotFoundError(p)
tr=sorted(rows(TR),key=lambda r:int(r["detail_row"])); co=sorted(rows(CO),key=lambda r:int(r["coarse_row"])); n35=sorted((r for r in rows(N35) if int(r["layer"])==0),key=lambda r:int(r["row"]))
p075=np.array([[float(r[f"detail_rest_world_{a}_m"]) for a in "xyz"] for r in tr]); ids=np.array([int(r["detail_source_vertex_id"]) for r in tr])
c075=np.array([[float(r[f"world_{a}_m"]) for a in "xyz"] for r in co]); p035=np.array([[float(r[f"{a}_m"]) for a in "xyz"] for r in n35])
if not np.array_equal(ids,np.arange(61950)):raise RuntimeError("075 source IDs not identity rows")
z=np.load(NPZ,allow_pickle=False); pnat=z["source_current_native_world_m"]; pap=z["source_current_AP_world_m"]; cap=z["original_AP_outer_points_m"]
rpt=json.loads(RPT.read_text()); pl=rpt["checks"]["D36_to_AP_placement"]; R=np.array(pl["rotation"]);t=np.array(pl["translation_m"]); ptrans=pnat@R.T+t
samples=[]
for i in [0,1,12345,30975,61949]:
 samples.append({"id":i,"075_detail_AP":p075[i].tolist(),"294_AP":pap[i].tolist(),"294_native":pnat[i].tolist(),"D36_to_AP(native)":ptrans[i].tolist(),"075_minus_AP":(p075[i]-pap[i]).tolist(),"075_minus_declared_transform":(p075[i]-ptrans[i]).tolist()})
cs=[]
for i in [0,1,602,1205,1807,2410]:
 cs.append({"row":i,"075_coarse":c075[i].tolist(),"035_layer0":p035[i].tolist(),"294_AP_outer":cap[i].tolist(),"075_minus_035":(c075[i]-p035[i]).tolist()})
out={"schema":"numi.human.p1-frame-lineage-audit.v1","scope":"read-only exact-row/frame audit; no transform fit, geometry mutation, or simulation","host":socket.gethostname(),"inputs":{str(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in [TR,CO,N35,NPZ,RPT,V,PAY]},"declared_transform":{"label":"D36_to_AP_placement","rotation":pl["rotation"],"translation_m":pl["translation_m"],"report_path":str(RPT),"report_sha256":sha(RPT)},"lineage":{"075_detail_input_payload":str(PAY),"075_payload_sha256":sha(PAY),"075_status":json.loads(V.read_text())["status"],"294_source_positions_semantics":rpt["source_position_semantics"],"294_registered_skin_hash":rpt["inputs"].get("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/native-step-001/registered-skin.csv"),"294_NHSKIN_hash":rpt["inputs"].get("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")},"bounds":{"075_detail_AP":bounds(p075),"294_detail_native_D36":bounds(pnat),"294_detail_AP":bounds(pap),"075_coarse_world":bounds(c075),"035_initial_layer0_world":bounds(p035)},"comparisons":{"075_detail_vs_294_AP_by_identity_ID":ds(p075-pap),"075_detail_vs_declared_D36_to_AP_native":ds(p075-ptrans),"declared_D36_to_AP_vs_saved_AP":ds(ptrans-pap),"075_detail_vs_native_untransformed":ds(p075-pnat),"075_coarse_vs_035_initial_layer0":ds(c075-p035),"075_coarse_vs_294_original_AP_outer":ds(c075-cap),"075coarse_F32_bitwise_equal_035":bool(np.array_equal(c075.astype(np.float32),p035.astype(np.float32))),"075detail_F32_bitwise_equal_294_AP":bool(np.array_equal(p075.astype(np.float32),pap.astype(np.float32)))},"detail_samples":samples,"coarse_samples":cs,"interpretation":"The 075 source rows are in AP world coordinates. Comparing them directly to native D36 coordinates produces a metre-scale frame mismatch. The declared D36-to-AP placement is used as pinned; no best-fit transform was performed. Coarse-reference rows compare directly against 035 layer-0 FEM positions. This is coordinate-frame evidence only, not P1 physical qualification.","runtime":{"seconds":time.time()-t0,"numpy":np.__version__,"python":sys.version}}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"path":str(OUT),"sha256":sha(OUT),"report":out},indent=2,sort_keys=True,allow_nan=False))
