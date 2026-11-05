
"""Matched-deformation diagnostics for retained-node remeshing.

Uses emitted Float32 rest keys and verified unchanged-object topology to resolve
coincident nodes. This is a field-restriction diagnostic, not ancestry admission,
a second constitutive solver, or a replacement for native physics certificates.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from compare_remeshed_native_310 import capture, strain_stats, delta, sha

def array_sha(x):
 a=np.ascontiguousarray(x)
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def node_keys(z,names):
 xyz=np.ascontiguousarray(z["rest"],dtype=np.float32).view(np.uint32).reshape(-1,3)
 out=np.empty(len(xyz),dtype=[("object","<i4"),("region","<i4"),("x","<u4"),("y","<u4"),("z","<u4")])
 canonical={name:i for i,name in enumerate(names)}
 owner=np.full(len(z["objects"]),-1,np.int32)
 for name,i in z["objects"].items():owner[i]=canonical[name]
 out["object"]=owner[z["obj"]];out["region"]=z["region"]
 for i,k in enumerate("xyz"):out[k]=xyz[:,i]
 return out

def retained_node_map(a,b):
 if set(a["objects"])!=set(b["objects"]):raise ValueError("object identity mismatch")
 names=sorted(a["objects"]);ka,kb=node_keys(a,names),node_keys(b,names)
 vals,first,counts=np.unique(ka,return_index=True,return_counts=True)
 loc=np.searchsorted(vals,kb)
 if np.any(loc==len(vals)) or np.any(vals[np.minimum(loc,len(vals)-1)]!=kb):raise ValueError("missing exact emitted rest/owner/region key")
 mapping=np.full(len(kb),-1,np.int64);unique=counts[loc]==1;mapping[unique]=first[loc[unique]]
 resolved=[]
 for name in names:
  bi=np.flatnonzero(b["obj"]==b["objects"][name])
  if not np.any(mapping[bi]<0):continue
  ai=np.flatnonzero(a["obj"]==a["objects"][name])
  at=a["tets"][a["tobj"]==a["objects"][name]];bt=b["tets"][b["tobj"]==b["objects"][name]]
  if len(ai)!=len(bi) or len(at)!=len(bt):raise ValueError("coincident keys require unchanged object topology: "+name)
  # Node rows need not be contiguous; all local incidences are checked.
  al=np.searchsorted(ai,at);bl=np.searchsorted(bi,bt)
  if np.any(al>=len(ai)) or np.any(bl>=len(bi)) or not np.array_equal(ai[al],at) or not np.array_equal(bi[bl],bt):raise ValueError("cross-owner tetrahedron")
  checks=(np.array_equal(ka[ai],kb[bi]),np.array_equal(al,bl),
          np.array_equal(a["treg"][a["tobj"]==a["objects"][name]],b["treg"][b["tobj"]==b["objects"][name]]),
          np.array_equal(a["tag"][ai],b["tag"][bi]),
          np.array_equal(a["mass"][ai],b["mass"][bi]))
  if not all(checks):raise ValueError("ambiguous coincident nodes in changed object: "+name)
  mapping[bi]=ai
  resolved.append(dict(object=name,nodes=len(ai),tetrahedra=len(at),node_key_sha256=array_sha(ka[ai]),local_topology_sha256=array_sha(al)))
 if np.any(mapping<0) or len(np.unique(mapping))!=len(mapping):raise ValueError("noninjective or incomplete map")
 if not np.array_equal(ka[mapping],kb):raise ValueError("mapping key mismatch")
 if not np.array_equal(a["tag"][mapping],b["tag"]):raise ValueError("constraint tag mismatch")
 return mapping,dict(method="exact Float32 rest coordinate + named object + anatomical region; duplicates require exact unchanged local topology, tags and mass",
                     unique_coordinate_nodes=int(unique.sum()),duplicate_nodes=int((~unique).sum()),
                     verified_unchanged_objects=resolved,mapping_sha256=array_sha(mapping),
                     canonical_ancestry_admission=False)

def regional_delta(a,b):
 if a.keys()!=b.keys():raise ValueError("regional identity mismatch")
 return {k:dict(reference=a[k],candidate=b[k],relative_change={n:delta(a[k][n],b[k][n]) for n in
          ("volume_weighted_strain_norm","volume_weighted_strain_squared","maximum_green_strain_norm")}) for k in a}

def gate_counts(rows):
 vals=[v for row in rows.values() for v in row["relative_change"].values()]
 return dict(metrics=len(vals),over_5percent=sum(v is None or v>.05 for v in vals),
             maximum_relative_change=max(v for v in vals if v is not None))

def tet_keys(t):
 return np.ascontiguousarray(np.sort(t,axis=1),dtype=np.int32).view(np.dtype([("a","<i4"),("b","<i4"),("c","<i4"),("d","<i4")])).reshape(-1)

def strain_fields(z,rows=None):
 rows=np.arange(len(z["tets"])) if rows is None else rows
 for start in range(0,len(rows),8192):
  rr=rows[start:start+8192];t=z["tets"][rr];r=z["rest"][t];u=z["u"][t]
  dm=np.swapaxes(r[:,1:]-r[:,:1],1,2);du=np.swapaxes(u[:,1:]-u[:,:1],1,2)
  f=np.eye(3)+du@np.linalg.inv(dm);e=(np.swapaxes(f,1,2)@f-np.eye(3))*.5
  yield rr,e,np.abs(np.linalg.det(dm))/6

def compare_matched(a,b):
 mapping,receipt=retained_node_map(a,b)
 c=dict(b,u=a["u"][mapping])
 sa,sb,sc=strain_stats(a),strain_stats(b),strain_stats(c)
 aa=regional_delta(sa,sb);ab=regional_delta(sa,sc);bc=regional_delta(sc,sb)
 ak,bk=tet_keys(a["tets"]),tet_keys(mapping[b["tets"]])
 vals,first,cnt=np.unique(ak,return_index=True,return_counts=True);loc=np.searchsorted(vals,bk)
 same=loc<len(vals);safe=np.minimum(loc,len(vals)-1);same&=(vals[safe]==bk)&(cnt[safe]==1)
 refrows=first[safe];same&=a["treg"][refrows]==b["treg"];same&=a["tobj"][refrows]==b["tobj"]
 # Object integer numbering is verified explicitly, independent of key canonicalization.
 if a["objects"]!=b["objects"]:raise ValueError("tet owner numbering differs; explicit canonicalization required")
 max_counterfactual=0.;max_actual=0.;maxw=None;total_sq=0.;total_v=0.
 for rows,e,vol in strain_fields(b,np.flatnonzero(same)):
  rr=refrows[rows]
  ar=a["rest"][a["tets"][rr]];au=a["u"][a["tets"][rr]]
  br=c["rest"][c["tets"][rows]];bu=c["u"][c["tets"][rows]]
  def green(r,u):
   f=np.eye(3)+np.swapaxes(u[:,1:]-u[:,:1],1,2)@np.linalg.inv(np.swapaxes(r[:,1:]-r[:,:1],1,2))
   return (np.swapaxes(f,1,2)@f-np.eye(3))*.5
  ae=green(ar,au);ce=green(br,bu)
  exact=np.linalg.norm(ce-ae,axis=(1,2));d=np.linalg.norm(e-ce,axis=(1,2))
  max_counterfactual=max(max_counterfactual,float(exact.max()))
  if float(d.max())>max_actual:
   j=int(d.argmax());max_actual=float(d[j]);maxw=dict(candidate_tet=int(rows[j]),reference_tet=int(rr[j]),region=int(b["treg"][rows[j]]),absolute_green_strain_difference=max_actual)
  total_sq+=float((vol*d*d).sum());total_v+=float(vol.sum())
 return dict(status="MATCHED_DEFORMATION_DIAGNOSTIC_NOT_PHYSICAL_ADMISSION",mapping=receipt,
             reference_actual_vs_candidate_actual=dict(summary=gate_counts(aa),regions=aa),
             discretization_at_reference_displacement=dict(summary=gate_counts(ab),regions=ab),
             candidate_actual_vs_restricted_reference_displacement=dict(summary=gate_counts(bc),regions=bc),
             identical_cells=dict(count=int(same.sum()),candidate_changed_cells=int((~same).sum()),
               removed_or_changed_reference_cells=int(len(a["tets"])-same.sum()),
               restricted_field_max_absolute_green_strain_difference=max_counterfactual,
               actual_field_max_absolute_green_strain_difference=max_actual,
               actual_field_volume_weighted_rms_green_strain_difference=float(np.sqrt(total_sq/total_v)),
               maximum_witness=maxw),
             maximum_matched_displacement_difference_m=float(np.linalg.norm(b["u"]-c["u"],axis=1).max()),
             physical_fidelity_qualification=False,
             limitations=["Counterfactual is kinematic strain only, not material energy or force.",
                          "Endpoint-field differences include model-induced trajectories and solver effects; this diagnostic alone does not distinguish them.",
                          "A small relative counterfactual discrepancy is not native mesh admission."])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("candidate",type=Path);p.add_argument("output",type=Path);args=p.parse_args()
 a,b=capture(args.reference),capture(args.candidate)
 result=compare_matched(a,b)
 result["inputs"]={k:dict(path=str(v),sha256=sha(v)) for k,v in [("reference",args.reference),("candidate",args.candidate)]}
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:({n:v[n] for n in v if n!="regions"} if isinstance(v,dict) else v) for k,v in result.items() if k!="inputs"},indent=2))
