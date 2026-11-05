"""Postprocess existing endpoint captures after a conforming mesh change.
Streaming arrays keep the analysis working set bounded. Regional aggregates are
diagnostics, never a substitute for canonical pointwise or energy qualification.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, re, time
import numpy as np

CHANNELS=("bulk_reaction","contact_force","free_internal_force","tag2_attachment_force")
def sha(path):
 with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def f32(value):return float(np.float32(value))
def kv(line,sep):return dict(x.split("=",1) for x in line.strip().split(sep) if "=" in x)
def capture(path):
 header=None;closed=False;body=None;objects={};seen_n=set();seen_t=set();summaries={}
 with gzip.open(path,"rt") as f:
  for line in f:
   if "resting_FEM_endpoint_capture=begin " in line:
    assert header is None
    header=kv(line.split("resting_FEM_endpoint_capture=begin ",1)[1]," ")
    assert header["version"]=="2" and header["coordinate_unit"]=="m" and header["force_unit"]=="N" and header["strain_unit"]=="1"
    n=int(header["nodes"]);nt=int(header["tetrahedra"])
    rest=np.zeros((n,3));u=np.zeros_like(rest);mass=np.zeros(n);region=np.full(n,-1,np.int32);obj=np.full(n,-1,np.int16)
    forces=np.zeros((n,len(CHANNELS),3));tag=np.zeros(n)
    tets=np.full((nt,4),-1,np.int32);treg=np.full(nt,-1,np.int32);tobj=np.full(nt,-1,np.int16)
   elif "resting_FEM_node," in line:
    assert header is not None
    row=kv(line.split("resting_FEM_node,",1)[1],",");i=int(row["global"]);assert 0<=i<n and i not in seen_n;seen_n.add(i)
    ob=objects.setdefault(row["object"],len(objects));obj[i]=ob;region[i]=int(row.get("anatomical_region_id",-1));mass[i]=float(row["mass_kg"]);tag[i]=float(row["constraint_tag"])
    rest[i]=[f32(row["rest_"+a]) for a in "xyz"]
    mode=int(float(row["reference_mode"]));assert mode in (0,1)
    u[i]=([f32(row["reference_u_hi_"+a])+f32(row["reference_u_lo_"+a]) for a in "xyz"] if mode else [f32(row["accepted_"+a])-rest[i,j] for j,a in enumerate("xyz")])
    for c,prefix in enumerate(CHANNELS):forces[i,c]=[float(row[prefix+"_"+a+"_N"]) for a in "xyz"]
   elif "resting_FEM_tet," in line:
    row=kv(line.split("resting_FEM_tet,",1)[1],",");i=int(row["global"]);assert 0<=i<nt and i not in seen_t;seen_t.add(i)
    tets[i]=[int(row["n"+str(j)]) for j in range(4)];treg[i]=int(row.get("anatomical_region_id",-1));tobj[i]=objects[row["object"]]
    assert math.isfinite(float(row["J"])) and float(row["J"])>0
   elif "resting_FEM_object_capture=summary " in line:
    row=kv(line.split("resting_FEM_object_capture=summary ",1)[1]," ");summaries[row["object"]]=row
   elif "resting_FEM_endpoint_capture=end" in line:closed=True
   elif "resting_integrated_body=completed " in line:body=kv(line.split("resting_integrated_body=completed ",1)[1]," ")
 assert header is not None and closed and body is not None and len(seen_n)==n and len(seen_t)==nt
 assert tets.min()>=0 and tets.max()<n and np.all(np.isfinite(rest)) and np.all(np.isfinite(u)) and np.all(np.isfinite(forces)) and np.all(mass>0)
 print("parsed",path.parent.name,n,nt,flush=True)
 return dict(header=header,body=body,objects=objects,rest=rest,u=u,mass=mass,region=region,obj=obj,forces=forces,tag=tag,tets=tets,treg=treg,tobj=tobj,summaries=summaries)
def strain_stats(z):
 results={};names={v:k for k,v in z["objects"].items()}
 for start in range(0,len(z["tets"]),8192):
  stop=min(start+8192,len(z["tets"]));t=z["tets"][start:stop];r=z["rest"][t];u=z["u"][t]
  dm=np.swapaxes(r[:,1:]-r[:,:1],1,2);du=np.swapaxes(u[:,1:]-u[:,:1],1,2)
  det=np.linalg.det(dm);assert np.all(np.isfinite(det)) and np.all(det!=0)
  f=np.eye(3)+du@np.linalg.inv(dm);j=np.linalg.det(f);assert np.all(np.isfinite(j)) and np.all(j>0)
  e=(np.swapaxes(f,1,2)@f-np.eye(3))*.5;en=np.linalg.norm(e,axis=(1,2));vol=np.abs(det)/6
  labels=z["treg"][start:stop];obs=z["tobj"][start:stop]
  # Original anatomical donor IDs remain independent of changing tetrahedron IDs.
  groups=np.where(labels>=0,labels,1000000+obs.astype(np.int32))
  for label in np.unique(groups):
   m=groups==label;key=("region:"+str(label) if label<1000000 else "object:"+names[int(label-1000000)])
   row=results.setdefault(key,dict(volume_m3=0.,volume_weighted_strain_norm=0.,volume_weighted_strain_squared=0.,maximum_green_strain_norm=0.,minimum_J=math.inf,cells=0))
   row["volume_m3"]+=float(vol[m].sum());row["volume_weighted_strain_norm"]+=float((vol[m]*en[m]).sum());row["volume_weighted_strain_squared"]+=float((vol[m]*en[m]**2).sum())
   row["maximum_green_strain_norm"]=max(row["maximum_green_strain_norm"],float(en[m].max()));row["minimum_J"]=min(row["minimum_J"],float(j[m].min()));row["cells"]+=int(m.sum())
 return results
def force_stats(z):
 results={};names={v:k for k,v in z["objects"].items()};groups=np.where(z["region"]>=0,z["region"],1000000+z["obj"].astype(np.int32))
 for label in np.unique(groups):
  m=groups==label;key=("region:"+str(label) if label<1000000 else "object:"+names[int(label-1000000)])
  values=z["forces"][m]
  results[key]={c:dict(resultant_N=values[:,i].sum(0).tolist(),nodal_l1_N=float(np.linalg.norm(values[:,i],axis=1).sum())) for i,c in enumerate(CHANNELS)}
 return results
def delta(a,b):
 return abs(b-a)/abs(a) if a else (0. if b==0 else None)
def compare(reference,candidate):
 a=capture(reference);b=capture(candidate)
 for key in ("steps","dt_s","endpoint_time_s","runtime_dt_s","runtime_coordinate_projection"):
  assert a["header"][key]==b["header"][key],key
 sa,sb=strain_stats(a),strain_stats(b);fa,fb=force_stats(a),force_stats(b);assert sa.keys()==sb.keys() and fa.keys()==fb.keys()
 strain={k:dict(reference=sa[k],candidate=sb[k],relative_change={name:delta(sa[k][name],sb[k][name]) for name in ("volume_weighted_strain_norm","volume_weighted_strain_squared","maximum_green_strain_norm")}) for k in sa}
 force={}
 for key in fa:
  force[key]={}
  for c in CHANNELS:
   x,y=fa[key][c],fb[key][c];diff=float(np.linalg.norm(np.array(y["resultant_N"])-x["resultant_N"]))
   force[key][c]=dict(reference=x,candidate=y,resultant_difference_N=diff,relative_resultant_difference_over_reference_nodal_l1=diff/x["nodal_l1_N"] if x["nodal_l1_N"] else (0. if diff==0 else None),relative_nodal_l1_change=delta(x["nodal_l1_N"],y["nodal_l1_N"]))
 physiological={}
 ar=list(csv.DictReader((reference.parent/"install-motion/resting-coupled.csv").open()));br=list(csv.DictReader((candidate.parent/"install-motion/resting-coupled.csv").open()))
 assert len(ar)==len(br)==int(a["header"]["steps"])
 for k in ar[0]:
  x=np.array([float(r[k]) for r in ar]);y=np.array([float(r[k]) for r in br]);assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
  physiological[k]=dict(max_abs_change=float(np.abs(x-y).max()),max_abs_reference=float(np.abs(x).max()),relative_max_abs_change=float(np.abs(x-y).max()/np.abs(x).max()) if np.any(x) else (0. if not np.any(y) else None))
 def telemetry(path):
  rows=[json.loads(s.split("=",1)[1]) for s in path.read_text().splitlines() if s.startswith("matter_fgmres_active_work=")]
  assert len(rows)==32
  return {k:sum(r[k] for r in rows) for k in ("cycles_with_work","used_columns_sum","budget_exhausted_solves")}
 wa=float(a["body"]["wall_s"]);wb=float(b["body"]["wall_s"])
 return dict(reference=dict(path=str(reference),sha256=sha(reference),header=a["header"],body=a["body"],active_work=telemetry(reference.parent/"install-motion.stderr.log")),candidate=dict(path=str(candidate),sha256=sha(candidate),header=b["header"],body=b["body"],active_work=telemetry(candidate.parent/"install-motion.stderr.log")),body_wall_reduction_fraction=1-wb/wa,body_throughput_ratio=wa/wb,total_fem_mass_kg=dict(reference=float(a["mass"].sum()),candidate=float(b["mass"].sum())),regional_strain=strain,regional_force_channels=force,coupled_trace=physiological,physical_fidelity_qualification=False,scope="Existing captured force channels and regional strain aggregates; changed topology. Not pointwise canonical-parent strain, material energy, full force-owner coverage, rest qualification, or sustained integrated throughput.")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("candidate",type=Path);p.add_argument("output",type=Path);args=p.parse_args()
 result=compare(args.reference,args.candidate);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k not in ("regional_strain","regional_force_channels","coupled_trace")},indent=2))
