import hashlib,json,re
from pathlib import Path
import numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
B=E/"ankle-main-app-integration-001/native-candidate-off-root-control-007/stdout.log"
C=E/"ankle-main-app-integration-001/native-candidate-021-specialized-016/run-001/stdout.txt"
Z=E/"ankle-coupled-021-on-294-001/main294-commonmode-layer-candidate-001/candidate-layers.npz"
IR=E/"ankle-coupled-021-on-294-001/native-compileworld-candidate-001/candidate-input-report.json"
BP=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
CP=E/"ankle-coupled-021-on-294-001/ankle-coupled-021-on-294.nhskin"
O=E/"ankle-mass-ownership-readonly-001"
N,NF=2332,4619; OBJ="BodyParts3D whole-body skin-fat bonded shell"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for q in iter(lambda:f.read(1<<20),b""):h.update(q)
 return h.hexdigest()
def kv(s):return {k:v.rstrip(",") for k,v in re.findall(r"([A-Za-z0-9_]+)=([^\s]+)",s)}
def read(p):
 nd={}; prep=exact=massref=None
 for line in p.open():
  if line.startswith(f"resting_FEM_node,object={OBJ},"):
   d=dict(t.split("=",1) for t in line.rstrip().split(",")[2:] if "=" in t);i=int(d["local"])
   if i in nd:raise ValueError("duplicate FEM node")
   nd[i]=d
  elif line.startswith("resting_skin_fat_whole_body_preparation="):prep=kv(line)
  elif line.startswith("ankle021_exact_placed_layer_volume "):exact=kv(line)
  elif line.startswith("whole_body_tissue_mass_reference="):massref=kv(line)
 if len(nd)!=3*N or sorted(nd)!=list(range(3*N)) or prep is None:raise ValueError(f"bad shell dump {p}: {len(nd)}")
 return {"path":p,"f":{**prep,**(exact or {})},
  "x":np.array([[float(nd[i][f"rest_{a}"]) for a in "xyz"] for i in range(3*N)]),
  "m":np.array([float(nd[i]["mass_kg"]) for i in range(3*N)]),
  "r":np.array([int(nd[i].get("anatomical_region_id",-1)) for i in range(3*N)]),"massref":massref or {}}
def make_tets(F):
 a,b,c=np.sort(F,axis=1).T
 s=np.concatenate([np.stack((a,b,c,N+c),1),np.stack((a,b,N+b,N+c),1),np.stack((a,N+a,N+b,N+c),1)]).astype(np.int64)
 f=np.concatenate([np.stack((N+a,N+b,N+c,2*N+c),1),np.stack((N+a,N+b,2*N+b,2*N+c),1),np.stack((N+a,2*N+a,2*N+b,2*N+c),1)]).astype(np.int64)
 return s,f
def vol(x,T):
 q=x[T];return np.abs(np.einsum("ij,ij->i",np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0]))/6
def layer(run,name,T):
 d=run["f"];debit=float(d[name+"_mass_debit_kg"])
 V=float(d[name+"_m3"] if "ankle021_exact_placed_layer_volume" in d else d[name+"_volume_m3"])
 return debit,V,debit/V
with np.load(Z,allow_pickle=False) as z:F=np.array(z["faces"],dtype=np.int64)
if F.shape!=(NF,3) or F.min()!=0 or F.max()!=N-1:raise ValueError("face topology mismatch")
TS,TF=make_tets(F); b=read(B); c=read(C)
for q in (b,c):
 if q["f"]["source_vertices"]!=str(N) or q["f"]["source_faces"]!=str(NF):raise ValueError("run topology mismatch")
if b["f"]["reduction_sha256"]!=c["f"]["reduction_sha256"] or b["f"]["registered_csv_sha256"]!=c["f"]["registered_csv_sha256"]:raise ValueError("run source/frame identity mismatch")
if json.loads(IR.read_text()).get("measurements",{}).get("source_face_parents_equal") is not True:raise ValueError("source face ancestry parity not attested")
def analyze(name,T):
 mb,Vb,rb=layer(b,name,T);mc,Vc,rc=layer(c,name,T)
 vb=vol(b["x"],T);vc=vol(c["x"],T)
 changed=np.any(np.any(b["x"][T]!=c["x"][T],axis=2),axis=1)
 de=(rc-rb)*Vb; ge=rc*(Vc-Vb)
 dr=(rc-rb)*vb; gr=rc*(vc-vb)
 return {"base":{"owner_volume_m3":Vb,"stdout_reconstructed_volume_m3":float(vb.sum()),"volume_fit_scale":Vb/float(vb.sum()),"debit_kg":mb,"uniform_density_kg_m3":rb},
 "candidate":{"owner_volume_m3":Vc,"stdout_reconstructed_volume_m3":float(vc.sum()),"volume_fit_scale":Vc/float(vc.sum()),"debit_kg":mc,"uniform_density_kg_m3":rc},
 "volume_delta_m3":Vc-Vb,"volume_delta_percent":100*(Vc/Vb-1),"density_delta_kg_m3":rc-rb,"density_delta_percent":100*(rc/rb-1),
 "exact_debit_decomposition":{"uniform_density_change_kg":de,"geometry_volume_change_kg":ge,"net_kg":de+ge},
 "tet_reconstruction":{"element_count":len(T),"coordinate_changed_element_count":int(changed.sum()),"unchanged_element_count":int((~changed).sum()),"changed_face_ordinals":np.unique(np.flatnonzero(changed)%NF).astype(int).tolist(),
 "approx_density_term_sum_kg":float(dr.sum()),"approx_geometry_term_sum_kg":float(gr.sum()),"approx_net_sum_kg":float((dr+gr).sum()),"max_abs_net_tet_delta_kg":float(np.max(np.abs(rc*vc-rb*vb))),"max_abs_local_geometry_tet_term_kg":float(np.max(np.abs(gr)))},
 "arrays":(vb,vc)}
s=analyze("skin",TS);f=analyze("fat",TF)
def pred(run):
 out=np.zeros(3*N)
 for name,T in (("skin",TS),("fat",TF)):
  _,_,rho=layer(run,name,T);q=rho*vol(run["x"],T)/4
  for j in range(4):np.add.at(out,T[:,j],q)
 return out
pb,pc=pred(b),pred(c);dm=c["m"]-b["m"];dx=np.linalg.norm(c["x"]-b["x"],axis=1)
regions={}
for rid in sorted(set(b["r"].tolist())|set(c["r"].tolist())):
 z=float(c["m"][c["r"]==rid].sum()-b["m"][b["r"]==rid].sum())
 if z:regions[str(int(rid))]=z
top=[]
for i in np.argsort(np.abs(dm))[-12:][::-1]:
 top.append({"local_node":int(i),"layer":"outer_skin" if i<N else "skin_fat_interface" if i<2*N else "deep_fat","source_row":int(i%N),"region_id":int(c["r"][i]),"mass_base_kg":float(b["m"][i]),"mass_candidate_kg":float(c["m"][i]),"delta_kg":float(dm[i]),"rest_delta_m":float(dx[i])})
ds=layer(b,"skin",TS)[0];df=layer(b,"fat",TF)[0]
report={"schema":"numi.human.skin-fat.mass-ownership-reconstruction.v1","status":"read_only_diagnostic_not_candidate_admission",
"method":{"tet_mass_law":"rho_layer * abs(det)/6; one quarter of each tet mass is lumped to each incident node","tet_order":"sorted source face; exact three-tet prism order in preparePairedBoundaryFEM; all skin tets then all fat tets","interface":"skin inner and fat outer share nodes, so their two layer lump contributions add","precision":"global terms use exact owner-reported volumes/debits; per-element measures use stdout rest coordinates at finite text precision"},
"inputs_sha256":{"control007_stdout":sha(B),"candidate016_stdout":sha(C),"faces_npz":sha(Z),"input_report":sha(IR),"baseline_payload":sha(BP),"candidate_payload":sha(CP),"owner_header":sha(R/"apps/NumiHumanCoarseTissuePreparation.hpp"),"app_source":sha(R/"apps/numilab_human_myosim_visual_probe.mm")},
"run_identities":{"base_payload_sha256":b["f"].get("source_payload_sha256"),"candidate_payload_sha256":c["f"].get("source_payload_sha256"),"reduction_sha256":c["f"].get("reduction_sha256"),"registered_csv_sha256":c["f"].get("registered_csv_sha256")},
"topology":{"source_rows":N,"faces":NF,"skin_tets":len(TS),"fat_tets":len(TF),"combined_tets":len(TS)+len(TF),"fem_nodes":3*N,"source_face_parents_equal_receipt":True},
"layers":{"skin":{k:v for k,v in s.items() if k!="arrays"},"fat":{k:v for k,v in f.items() if k!="arrays"}},
"conservation":{"fixed_skin_debit_kg":ds,"fixed_fat_debit_kg":df,"fixed_total_debit_kg":ds+df,
"density_change_total_kg":s["exact_debit_decomposition"]["uniform_density_change_kg"]+f["exact_debit_decomposition"]["uniform_density_change_kg"],
"geometry_volume_change_total_kg":s["exact_debit_decomposition"]["geometry_volume_change_kg"]+f["exact_debit_decomposition"]["geometry_volume_change_kg"],
"nominal_1000_900_added_volume_mass_kg":1000*s["volume_delta_m3"]+900*f["volume_delta_m3"],
"printed_node_mass_sum_007_kg":float(b["m"].sum()),"printed_node_mass_sum_016_kg":float(c["m"].sum()),"printed_node_mass_sum_delta_kg":float(dm.sum()),
"cooked_tissue_mass_reference_007_kg":float(b["massref"]["cooked_tissue_kg"]),"cooked_tissue_mass_reference_016_kg":float(c["massref"]["cooked_tissue_kg"]),"massless_link_rerouted_kg_007":float(b["massref"]["massless_link_rerouted_kg"]),"massless_link_rerouted_kg_016":float(c["massref"]["massless_link_rerouted_kg"]),"massless_link_rerouted_kg_delta":float(c["massref"]["massless_link_rerouted_kg"])-float(b["massref"]["massless_link_rerouted_kg"]),"massless_link_rerouted_nodes_007":int(b["massref"]["massless_link_rerouted_nodes"]),"massless_link_rerouted_nodes_016":int(c["massref"]["massless_link_rerouted_nodes"])},
"nodes":{"mass_rows":len(dm),"nonzero_mass_delta_count":int(np.count_nonzero(dm)),"coordinate_changed_count":int(np.count_nonzero(dx>1e-8)),"coordinate_changed_local_nodes":np.flatnonzero(dx>1e-8).astype(int).tolist(),"max_reference_coordinate_delta_m":float(dx.max()),
"mass_delta_outer_skin_kg":float(dm[:N].sum()),"mass_delta_shared_interface_kg":float(dm[N:2*N].sum()),"mass_delta_deep_fat_kg":float(dm[2*N:].sum()),
"max_abs_lumped_mass_reconstruction_residual_007_kg":float(np.max(np.abs(b["m"]-pb))),"max_abs_lumped_mass_reconstruction_residual_016_kg":float(np.max(np.abs(c["m"]-pc))),
"region_mass_delta_kg":regions,"largest_absolute_node_mass_deltas":top},
"existing_api":{"uniform_density":"preparePairedBoundaryFEM computes rho=donorDebitKg/layer.volumeM3; each tetra adds rho*V/4 to its four nodal masses","per_tet_material":"appendRegisteredSkinFatFEMFromLayers uses femMaterialIndices but assigns one skin material to all skin tets and one fat material to all fat tets","shared_interface":"fat outer masses are remapped onto skin inner nodes and summed","downstream_mass_partition":"app gathers donorMassNodes; prepareWholeBodyMassReference/compileNumiHumanTissueMassPartition rebase body mass/inertia from node mass and donor-local positions","spatial_density_support":"this path has per-tet material identity but no per-tet scalar density map; local density changes require explicit materials or added prep/mass-partition support",
"source_refs":{"NumiHumanCoarseTissuePreparation.hpp":{"rho_and_lumping":"639-716","material_assignment_and_interface_merge":"719-900"},"numilab_human_myosim_visual_probe.mm":{"fixed_debits_and_layer_volume":"22390-22425","layer_append":"22502-22512","node_mass_rebase":"24192-24288"}}},
"limits":["Read-only: no production source, app, native, or GPU state was changed or rerun.","Per-tet estimates reconstruct exact owner topology from printed rest coordinates; stdout truncation makes their sums differ slightly from exact owner-reported volume.","Run 016 failed the separate force and regional-strain comparison gates; it is not admitted."]}
out=O/"mass-ownership-007-vs-016.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(out),"report_sha256":sha(out),"inputs":report["inputs_sha256"],"skin":report["layers"]["skin"],"fat":report["layers"]["fat"],"conservation":report["conservation"],"node_summary":{k:v for k,v in report["nodes"].items() if k not in ("region_mass_delta_kg","largest_absolute_node_mass_deltas")},"top_nodes":top},indent=2,sort_keys=True))
