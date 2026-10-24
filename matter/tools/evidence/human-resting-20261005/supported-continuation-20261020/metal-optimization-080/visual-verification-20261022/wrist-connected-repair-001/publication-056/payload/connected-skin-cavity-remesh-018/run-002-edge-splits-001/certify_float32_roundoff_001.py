import numpy as np, pathlib, sys, json
E=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]);
base=np.load(E/'connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz')
c=np.load(out/'edge-split-geometry.npz')
X0=base['points_world_m'].astype(np.float32); F0=base['physical_outer_faces'].astype(np.int64)
X=c['points_world_m'].astype(np.float32); F=F0.copy(); origin=np.arange(len(F0),dtype=np.int64)
edges=c['mechanical_midpoint_edge_endpoints'].astype(np.int64); mids=c['mechanical_midpoint_global_node_rows'].astype(np.int64)
edge_mid={tuple(map(int,e)):int(m) for e,m in zip(edges,mids)}
for (u,v),mid in sorted(edge_mid.items()):
 newF=[];newO=[]
 for f,o in zip(F,origin):
  hit=None
  for k in range(3):
   a,b,d=map(int,(f[k],f[(k+1)%3],f[(k+2)%3]))
   if {a,b}=={u,v}:hit=(a,b,d);break
  if hit is None:newF.append(f);newO.append(o)
  else:
   a,b,d=hit;newF.extend(([a,mid,d],[mid,b,d]));newO.extend((o,o))
 F=np.asarray(newF,np.int64);origin=np.asarray(newO,np.int64)
ideal=X.astype(np.float64).copy()
for (u,v),mid in edge_mid.items():ideal[mid]=(X0[u].astype(np.float64)+X0[v].astype(np.float64))*0.5
def area(P,Q):
 z=P[Q].astype(np.float64);return .5*np.linalg.norm(np.cross(z[:,1]-z[:,0],z[:,2]-z[:,0]),axis=1)
a0=area(X0,F0); aout=area(X,F); aid=area(ideal,F)
tri=ideal[F]; actual=X[F].astype(np.float64); e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
d1=(actual[:,1]-actual[:,0])-e1;d2=(actual[:,2]-actual[:,0])-e2
bound=.5*(np.linalg.norm(e1,axis=1)*np.linalg.norm(d2,axis=1)+np.linalg.norm(e2,axis=1)*np.linalg.norm(d1,axis=1)+np.linalg.norm(d1,axis=1)*np.linalg.norm(d2,axis=1))
sumout=np.bincount(origin,weights=aout,minlength=len(F0));sumid=np.bincount(origin,weights=aid,minlength=len(F0));sumb=np.bincount(origin,weights=bound,minlength=len(F0))
err=np.abs(sumout-a0); iberr=np.abs(sumid-a0)
pass_bound=err<=sumb+2e-14*np.maximum(a0,1e-30)
mid_err=[];ulp=[]
for (u,v),mid in edge_mid.items():
 exact=(X0[u].astype(np.float64)+X0[v].astype(np.float64))*0.5
 mid_err.append(float(np.max(np.abs(X[mid].astype(np.float64)-exact))))
 ulp.append(float(np.max(np.abs(np.spacing(exact.astype(np.float32))))))
print(json.dumps({'parent_face_count':len(F0),'child_face_count':len(F),'all_parent_faces_present':bool(np.all(np.bincount(origin,minlength=len(F0))>0)),'all_ideal_child_areas_partition_parent':bool(np.all(iberr<=2e-14*np.maximum(a0,1e-30))),'float32_roundoff_area_bound_pass':bool(np.all(pass_bound)),'bound_fail_count':int(np.count_nonzero(~pass_bound)),'max_parent_area_abs_error_m2':float(err.max()),'max_parent_relative_area_error':float((err/np.maximum(a0,1e-300)).max()),'max_parent_roundoff_bound_m2':float(sumb.max()),'max_parent_error_over_bound':float((err/np.maximum(sumb,1e-300)).max()),'total_parent_area_m2':float(a0.sum()),'total_emitted_area_m2':float(aout.sum()),'total_area_rel_error':float(abs(aout.sum()-a0.sum())/a0.sum()),'midpoint_max_component_rounding_error_m':max(mid_err),'midpoint_max_component_ulp_m':max(ulp),'midpoint_error_over_half_ulp':max(np.max(np.abs(X[mid].astype(np.float64)-((X0[u].astype(np.float64)+X0[v].astype(np.float64))*0.5))/np.maximum(np.abs(np.spacing(((X0[u].astype(np.float64)+X0[v].astype(np.float64))*0.5).astype(np.float32))),1e-300)) for (u,v),mid in edge_mid.items()),'edges':edges.tolist()},indent=2))
