from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
source=D/"coupled-layer-trial-128/assemble_layers_128.py"
sp=importlib.util.spec_from_file_location("assembler",source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
C=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False,parents=True)
z=np.load(C);P=z["points_world_m"];F=z["source_O_faces"][:,[0,2,1]];n=len(P)
A=np.load(D/"fine-layer-fields-127/capacity-115-coarse-to-detail-layer-fields.npz")
I=A["capacity115_coarse_I_reference_m"];B=A["capacity115_coarse_D_reference_m"];nc=len(I)
Fc,_=m.face_csv(m.DEFAULT_FACE_CSV_GZ);Fc=Fc[:,[0,2,1]]
X=np.vstack((P,I,B));kind=z["node_kind"];sr=z["node_source_row"]
caps=[]
for e,occ in m.edges_of_faces(F).items():
 if len(occ)==1:
  _,a,b=occ[0];assert kind[a]==0 and kind[b]==0
  ai=n+int(sr[a]);bi=n+int(sr[b]);caps.extend(((b,a,ai),(b,ai,bi)))
skin=np.vstack((F,Fc[:,[0,2,1]]+n,np.array(caps)))
fat=m.shell_faces(Fc,np.array([(v[0][1],v[0][2])for v in m.edges_of_faces(Fc).values()if len(v)==1]),nc,n,n+nc)
report=dict(outer_sha256=m.sha256(C),script_sha256=m.sha256(__file__),accepted_physical_steps=0,geometry_admitted=False,
 point_count=len(X),outer_node_count=n,inner_node_count=nc,skin_topology=m.shell_topology(skin),fat_topology=m.shell_topology(fat),audits={})
for name,ff in (("O",F),("I",Fc+n),("D",Fc+n+nc),("skin_shell",skin),("fat_shell",fat)):
 r=m.exact_self_audit(X,ff);np.save(out/(name+"-pairs.npy"),r.pop("pairs_array"));report["audits"][name]=r
np.savez_compressed(out/"candidate.npz",points_world_m=X,outer_faces=F,inner_faces=Fc+n,deep_faces=Fc+n+nc,skin_shell_faces=skin,fat_shell_faces=fat,
 outer_node_kind=kind,outer_node_source_row=sr,inner_coarse_source_row=np.arange(nc),deep_coarse_source_row=np.arange(nc))
report["skin_volume_m3"]=m.signed_volume(X,skin);report["fat_volume_m3"]=m.signed_volume(X,fat)
(out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
