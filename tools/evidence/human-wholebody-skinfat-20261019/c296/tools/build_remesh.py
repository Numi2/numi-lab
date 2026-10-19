import csv, hashlib, json, pathlib, re, shutil, subprocess, time
import numpy as np
from scipy.optimize import linprog
ROOT=pathlib.Path.cwd()
OUT=ROOT/"donor23-headstar-remesh-001"
AUDITOR=ROOT/"persistent-source-audit"
ACTIONS=[(27019,26031),(31512,32986)]

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def parse_candidate(path):
 t=pathlib.Path(path).read_text().split(); nv,nf=map(int,t[:2]); src=np.asarray(list(map(int,t[2:2+nv])),dtype=np.uint32); k=2+nv
 faces=np.asarray(list(map(int,t[k:k+3*nf])),dtype=np.uint32).reshape(nf,3); k+=3*nf
 parents=np.asarray(list(map(int,t[k:k+nf])),dtype=np.uint32)
 if len(src)!=nv or len(parents)!=nf or np.any(faces>=nv): raise ValueError("candidate text layout invalid")
 return src,faces,parents

def rot(face,a,b):
 f=list(map(int,face))
 for i in range(3):
  if f[i]==a and f[(i+1)%3]==b:return f[i:]+f[:i]
 return None

def flip(faces, donor, u, v):
 ids=np.where(np.sum(np.isin(faces,[u,v]),axis=1)==2)[0]
 if len(ids)!=2: raise ValueError(f"edge ({u},{v}) has incidence {len(ids)}")
 a=rot(faces[ids[0]],u,v); b=rot(faces[ids[1]],v,u)
 if a is None and b is not None:
  a=rot(faces[ids[1]],u,v); b=rot(faces[ids[0]],v,u); ids=ids[::-1]
 if a is None or b is None: raise ValueError("edge is not an oppositely oriented interior pair")
 x,y=a[2],b[2]
 if x==y or len({x,u,y})!=3 or len({x,y,v})!=3: raise ValueError("degenerate flip")
 if donor[x]!=donor[y] or donor[x]!=donor[u] or donor[x]!=donor[v]: raise ValueError("mixed-donor flip")
 if any(np.sum(np.isin(face,[x,y]))==2 for i,face in enumerate(faces) if i not in ids): raise ValueError("new diagonal already exists")
 out=faces.copy(); out[ids[0]]=[x,u,y]; out[ids[1]]=[x,y,v]
 return out,ids.tolist(),[[int(x) for x in faces[i]] for i in ids],[[int(x) for x in out[i]] for i in ids]

def topology(faces,nv):
 edges={}; face_keys=set(); face_adj=[set() for _ in range(nv)]
 for i,tri in enumerate(faces):
  key=tuple(sorted(map(int,tri)))
  if len(set(key))!=3 or key in face_keys: raise ValueError("duplicate or degenerate face")
  face_keys.add(key)
  a,b,c=map(int,tri)
  for u,v in ((a,b),(b,c),(c,a)):
   edge=tuple(sorted((u,v))); edges.setdefault(edge,[]).append(i)
 if any(len(v) not in (1,2) for v in edges.values()): raise ValueError("nonmanifold edge incidence")
 boundary=[e for e,v in edges.items() if len(v)==1]; deg=np.zeros(nv,dtype=np.int32); bad=[]
 graph=[set() for _ in range(nv)]
 for u,v in boundary: deg[u]+=1;deg[v]+=1;graph[u].add(v);graph[v].add(u)
 if np.any((deg!=0)&(deg!=2)): raise ValueError("boundary does not form loops")
 seen=set();loops=0
 for i in np.where(deg)[0]:
  if int(i) in seen:continue
  loops+=1;stack=[int(i)];seen.add(int(i))
  while stack:
   u=stack.pop()
   for v in graph[u]:
    if v not in seen:seen.add(v);stack.append(v)
 return {"edge_count":len(edges),"boundary_edges":len(boundary),"boundary_loops":loops,"components":None}

def cone(points,faces,v):
 inc=faces[np.any(faces==v,axis=1)]
 raw=np.cross(points[inc[:,1]]-points[inc[:,0]],points[inc[:,2]]-points[inc[:,0]])
 lengths=np.linalg.norm(raw,axis=1)
 if len(inc)==0 or np.any(~np.isfinite(lengths)) or np.any(lengths<=1e-15): return False,len(inc),None
 normals=raw/lengths[:,None]
 r=linprog(np.zeros(3),A_ub=-normals,b_ub=-np.ones(len(normals)),bounds=[(None,None)]*3,method="highs")
 return bool(r.success),len(inc), (r.x.tolist() if r.success else None), (float(np.min(normals@r.x)) if r.success else None)

def audit(points,faces):
 payload=f"{len(points)} {len(faces)}\n"+"\n".join(" ".join(f"{float(x):.17g}" for x in p) for p in points)+"\n"+"\n".join(" ".join(map(str,t)) for t in faces)+"\n"
 r=subprocess.run([str(AUDITOR)],input=payload,text=True,capture_output=True,check=False)
 return r.returncode==0,r.stdout.strip(),r.stderr.strip(),r.returncode

def write_text(path,src,faces,parents):
 with open(path,"w") as f:
  f.write(f"{len(src)} {len(faces)}\n")
  f.write(" ".join(map(str,map(int,src)))+"\n")
  for tri in faces:f.write(" ".join(map(str,map(int,tri)))+"\n")
  f.write(" ".join(map(str,map(int,parents)))+"\n")

def update_header(src_header,dst_header,faces):
 s=pathlib.Path(src_header).read_text()
 pat=r"(inline constexpr std::array<Face,\d+> kWholeBodySkinFaces=\{\n).*?(\n\};)"
 block=",\n".join("    Face{%du,%du,%du}"%tuple(map(int,t)) for t in faces)
 s2,n=re.subn(pat,lambda m:m.group(1)+block+m.group(2),s,count=1,flags=re.S)
 if n!=1:raise ValueError("could not uniquely replace face array")
 pathlib.Path(dst_header).write_text(s2)

report={"schema":"numi.donor23_two_headstar_flip.v1","actions_source_vertex_ids":[list(x) for x in ACTIONS],"targets":[27019,31512],"variants":{},"scope":"two local interior 2-to-2 flips only; no vertex positions/weights/donors changed"}
for label,rel in [("023","mechanical-edge-collapse-009/mechanical-layer-witness-fairing-023")]:
 base=ROOT/rel; dest=OUT/f"candidate-{label}"; dest.mkdir(parents=True,exist_ok=True)
 src,faces,parents=parse_candidate(base/"candidate.txt"); z=np.load(base/"candidate.npz")
 if not np.array_equal(src,z["sourceVertexIndices"]) or not np.array_equal(faces,z["faces"]) or not np.array_equal(parents,z["sourceFaceIndices"]):raise ValueError(label+" text/npz disagreement")
 if len(src)!=2333 or len(faces)!=4621:raise ValueError("unexpected base dimensions")
 if not np.array_equal(z["sourceFullWeights"],np.load(base/"candidate.npz")["sourceFullWeights"]):raise ValueError("weights changed")
 rows={}
 with open(base/"registered-skin-mechanical.csv",newline="") as f:
  rr=csv.reader(f); header=next(rr)
  for row in rr:rows[int(float(row[0]))]=row
 points64=np.asarray([[float(rows[int(s)][j]) for j in (2,3,4)] for s in src],dtype=np.float64)
 points32=z["registeredWorldPoints"].astype(np.float32)
 if not np.allclose(points64.astype(np.float32),points32,atol=0,rtol=0):raise ValueError(label+" csv/float32 point mismatch")
 donor=np.asarray([int(rows[int(s)][1]) for s in src],dtype=np.int32)
 if not np.array_equal(donor,z["donorBodyIndices"]):raise ValueError(label+" donor mismatch")
 before_top=topology(faces,len(src)); beforecones={}
 for source in (27019,31512):
  v=int(np.where(src==source)[0][0]); beforecones[str(source)]=cone(points32,faces,v)
 oldfaces=faces.copy(); flips=[]
 for source_u,source_v in ACTIONS:
  v=int(np.where(src==source_u)[0][0]);u=int(np.where(src==source_v)[0][0])
  faces,rows_changed,before,after=flip(faces,donor,u,v)
  flips.append({"edge_source_vertex_ids":[source_u,source_v],"face_rows":rows_changed,"old_local_faces":before,"new_local_faces":after,"source_face_parent_ids_preserved":[int(parents[i]) for i in rows_changed]})
 after_top=topology(faces,len(src)); aftercones={}
 for source in (27019,31512):
  v=int(np.where(src==source)[0][0]); aftercones[str(source)]=cone(points32,faces,v)
 print(label, beforecones, aftercones, flush=True)
 if not all(bool(x[0]) for x in aftercones.values()):raise ValueError(label+" target cone still infeasible")
 if before_top!=after_top:raise ValueError(label+" topology invariants changed: "+str((before_top,after_top)))
 if not np.array_equal(parents, z["sourceFaceIndices"]):raise ValueError("source ancestry changed")
 a64=audit(points64,faces);a32=audit(points32.astype(np.float64),faces)
 if not a64[0] or not a32[0]:raise ValueError(label+" persistent source audit failed "+repr((a64,a32)))
 # Preserve every non-topology payload byte-for-byte at array-value level.
 arrays={k:z[k] for k in z.files}; arrays["faces"]=faces.astype(z["faces"].dtype)
 npzpath=dest/"candidate.npz"; np.savez_compressed(npzpath,**arrays)
 txtpath=dest/"candidate.txt";write_text(txtpath,src,faces,parents)
 csvpath=dest/"registered-skin-mechanical.csv";shutil.copyfile(base/"registered-skin-mechanical.csv",csvpath)
 hdr=dest/"NumiHumanWholeBodySkinReduction.hpp";update_header(base/"NumiHumanWholeBodySkinReduction.hpp",hdr,faces)
 # Verify output roundtrip, and all non-face arrays exact.
 src2,f2,p2=parse_candidate(txtpath)
 if not np.array_equal(src,src2) or not np.array_equal(faces,f2) or not np.array_equal(parents,p2):raise ValueError("text roundtrip mismatch")
 z2=np.load(npzpath)
 for k in z.files:
  if k!="faces" and not np.array_equal(z[k],z2[k]):raise ValueError(label+" NPZ field changed: "+k)
 if not np.array_equal(z2["sourceFaceIndices"],parents):raise ValueError("parent IDs changed")
 report["variants"][label]={"input":{"npz_sha256":sha(base/"candidate.npz"),"txt_sha256":sha(base/"candidate.txt"),"csv_sha256":sha(base/"registered-skin-mechanical.csv"),"header_sha256":sha(base/"NumiHumanWholeBodySkinReduction.hpp")},"output":{"directory":str(dest),"npz_sha256":sha(npzpath),"txt_sha256":sha(txtpath),"csv_sha256":sha(csvpath),"header_sha256":sha(hdr)},"counts":{"vertices":len(src),"faces":len(faces),"unchanged_vertex_positions":True,"unchanged_donor_ids":True,"unchanged_full_weights":True,"unchanged_source_face_parent_per_row":True},"topology_before":before_top,"topology_after":after_top,"targets_before":{k:{"feasible":v[0],"incident_faces":v[1]} for k,v in beforecones.items()},"targets_after":{k:{"feasible":v[0],"incident_faces":v[1],"normal_cone_lp_witness":v[2],"minimum_normal_dot_on_witness":v[3]} for k,v in aftercones.items()},"flips":flips,"persistent_source_audit_float64":{"pass":a64[0],"stdout":a64[1],"stderr":a64[2],"exit":a64[3]},"persistent_source_audit_float32_positions":{"pass":a32[0],"stdout":a32[1],"stderr":a32[2],"exit":a32[3]}}
report["created_utc_note"]="Generated from frozen candidate 020 and 023 inputs; inputs not modified."
outjson=OUT/"topology-ancestry-receipt.json";outjson.write_text(json.dumps(report,indent=2)+"\n")
print(outjson)
print(json.dumps(report,indent=2))
