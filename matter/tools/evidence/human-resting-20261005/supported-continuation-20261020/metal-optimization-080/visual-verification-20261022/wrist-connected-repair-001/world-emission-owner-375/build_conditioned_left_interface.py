from pathlib import Path
import sys,types,importlib.util,json,time,numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"left-world-float32-conditioned-envelope-373";D.mkdir(exist_ok=True)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
b=load("b373","/tmp/build_bilateral_hand_envelope_054.py");b.OUT=D;field=load("field373",b.FIELD);b.field=field;geom=load("geom373",D.parent/"world-emission-owner-375/tools/human_skin_patch_geometry.py");conditioner=load("condition373",b.CONDITIONER)
parent=E/"left-connected-expanded-forearm-envelope-369/conditioned-field.npz";saved=np.load(parent);phi=saved["phi_m"];old=phi.copy();xs=saved["x_local_m"];ys=saved["y_local_m"];zs=saved["z_local_m"];cap=float(saved["field_cap_z_local_m"]);Q=saved["basis_world_rows"];origin=saved["origin_world_m"];grid=np.stack(np.meshgrid(xs,ys,zs,indexing="ij"),axis=-1).reshape(-1,3);start=time.monotonic()
phi,conditioning=b.condition_p1_cut(phi,field,conditioner,"left")
_,tets=field.grid_tets_for_active_cubes(phi);v,f,edge,alpha=geom.extract_p1_tetrahedral_interface(grid,tets,phi.reshape(-1),0.)
reference_world=v.astype(float)@Q+origin;world=reference_world.astype(np.float32);geom.validate_float32_surface_emission(reference_world,world,f);tri=world[f].astype(float);area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);bad=np.flatnonzero(area==0);capmask=np.all(v[f,2]==np.float32(cap),axis=1)
raw=b.topology(v,f)[0]
try:top=b.topology(v,f[~capmask])[0]
except Exception as exc:top={"error":repr(exc)}
np.savez_compressed(D/"conditioned-field.npz",**{**{k:saved[k] for k in saved.files},"phi_m":phi})
np.savez_compressed(D/"extracted-interface.npz",vertices_local_float32=v,vertices_current_world_m=world,faces=f,cap_face_mask=capmask,crossing_edge_keys=edge,edge_interpolation=alpha)
report={"parent_sha256":b.sha(parent),"conditioning":conditioning,"changed_nodes":int(np.count_nonzero(phi!=old)),"max_phi_raise_m":float((phi-old).max()),"world_float32_zero_area_face_rows":bad.tolist(),"min_world_area2_m2":float(area.min()),"raw_topology":raw,"topology":top,"cap_faces":int(capmask.sum()),"physical_admission":False,"elapsed_seconds":time.monotonic()-start}
(D/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:report[k] for k in ("changed_nodes","max_phi_raise_m","world_float32_zero_area_face_rows","min_world_area2_m2","raw_topology","topology","cap_faces","elapsed_seconds")}),flush=True)

