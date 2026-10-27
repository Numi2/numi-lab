#!/usr/bin/env python3
"""One sparse connected fit for all valid top-three exit witnesses; construction only."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

w=Path(__file__).resolve().parent;s=w.parent;c=w/"combined-43-57-65smooth-109-124-001";out=w/"batch-top3-selfguard-round4-lp653-001";out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def mod(n,p):
 spec=importlib.util.spec_from_file_location(n,p);o=importlib.util.module_from_spec(spec);sys.modules[n]=o;spec.loader.exec_module(o);return o
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
start=time.monotonic();base=load(c/"candidate.npz");x=base["points_world_m"];faces=base["source_O_faces"];ref=base["area_reference_037_points_world_m_f64"]*1000.;anchors=base["typed_coarse_anchor_output_rows_018"];assert sha(c/"candidate.npz")=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
tp=w/"targets-top-three-from653-001/nearest-union-exits.npz";tr=json.loads((tp.parent/"nearest-union-exits.json").read_text());targets=load(tp);assert tr["status"]=="PASS_PARTIAL_FIRST_UNION_EXIT_TARGETS_ONLY" and tr["inputs"]["candidate_sha256"]==sha(c/"candidate.npz") and targets["pass_mask"].all() and targets["target_exactly_outside_all185"].all();selected=targets["query_rows"];assert len(selected)==55 and not np.intersect1d(selected,anchors).size
qpath=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py");assert sha(qpath)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2";q=mod("batch_q_653",qpath);area_owner=mod("batch_area_653",q.DEFAULT_AREA_OWNER);q.verify_area_linearization_layout(area_owner)
source=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz");assert sha(source)=="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
with np.load(c/"candidate.npz",allow_pickle=False) as z:refsrc,_=q.compose_source_reference(z,source)
bm=x.astype(float)*1000.;srcmm=refsrc*1000.;dp=w/"depth-from653-001/point-depth-rows.npz";dr=json.loads((dp.parent/"365-compatible-report.json").read_text());nodes,normals,lower_m,ids,records,_=q.validate_point_depth_rows_payload(load(dp),dr,base_candidate_sha256=sha(c/"candidate.npz"),base_points=x,rows_npz_path=dp,owner_path=q.DEFAULT_POINT_DEPTH_OWNER,candidate_path=c/"candidate.npz")
selfpairs_path=w/"batch-top3-self-witness-109-round4-001/merged-pairs.npy";assert sha(selfpairs_path)=="8f615832e5e62d574ea735b28b3e3b3140a48d637420254668c1f15c69a64f5d";self_pairs=np.load(selfpairs_path,allow_pickle=False);pair_owner=mod("self_pair_batch109_653",q.DEFAULT_PAIR_OWNER);self_rows,self_records=pair_owner.build_pair_rows(bm,faces,self_pairs,margin_fraction=.25);print("self witness pairs",len(self_pairs),"separation rows",len(self_rows),flush=True)
edge=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);adj=sparse.csr_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(x),len(x)));support=set(map(int,selected))|set(map(int,faces[self_pairs.reshape(-1)].reshape(-1)));front=set(support)
for _ in range(3):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
support=np.array(sorted(support),dtype=np.int64);local={int(v):i for i,v in enumerate(support)};affected=np.flatnonzero(np.any(np.isin(faces,support),axis=1));normalall,_,_,req=q.fixed_reference_area_requirements(bm,ref,faces);prows,_=q.build_scoped_output_node_direction_rows(len(x),support,anchors,nodes,normals,lower_m,ids);prows=prows+self_rows;G,rhs,cur,_=area_owner.area_linearization(bm,faces,affected,normalall[affected],req[affected],minimum_ratio=.1);rhs=np.maximum(rhs,-.35*cur);A,b=q._owner_matrix(prows,G,rhs,support);EG,eb,_=q.build_edge_gradient_constraints(bm,faces,support,.5);A=sparse.vstack((A,EG),format="csr");b=np.r_[b,eb]
# Linearized local normal cone and area ceiling, relative to the current
# face rather than a whole-body displacement scale. Exact F32 gates follow.
ns=normalall[affected];axis=np.zeros_like(ns);axis[np.arange(len(ns)),np.argmin(np.abs(ns),axis=1)]=1.;t1=np.cross(ns,axis);t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.cross(ns,t1);cols=q.interleaved_coordinate_columns(support);Gn=G[:,cols].tocsr();cone=.35;blocks=[-Gn];limits=[-.75*cur]
for tangent in (t1,t2):
 Gt,_,_,_=area_owner.area_linearization(bm,faces,affected,tangent,req[affected],minimum_ratio=.1);Gt=Gt[:,cols].tocsr();blocks.extend((cone*Gn+Gt,cone*Gn-Gt));limits.extend((-cone*cur,-cone*cur))
H=sparse.vstack(blocks,format="csr");hb=np.concatenate(limits);nn=np.sqrt(np.asarray(H.multiply(H).sum(axis=1)).ravel());assert np.all(nn>1e-14);H=sparse.diags(1./nn)@H;hb=hb/nn;A=sparse.vstack((A,H),format="csr");b=np.r_[b,hb];lo,hi,_=q.source_safe_trust_bounds(bm,srcmm,support,anchors,4.,12.)
# L1 distance to selected exact exit goals + an edge-difference regularizer.
# Every variable remains a real coordinate of the connected patch.
n=len(support)*3;m=len(selected)*3;coords=np.array([3*local[int(v)]+k for v in selected for k in range(3)]);F=sparse.csr_matrix((np.ones(m),(np.arange(m),coords)),shape=(m,n)); er=[];ec=[];ev=[];ei=0
for u,v in edge:
 iu=local.get(int(u));iv=local.get(int(v))
 if iu is None and iv is None:continue
 weight=1./max(float(np.linalg.norm(bm[u]-bm[v])),.5)
 if iu is not None:er.append(ei);ec.append(iu);ev.append(weight)
 if iv is not None:er.append(ei);ec.append(iv);ev.append(-weight)
 ei+=1
E=sparse.kron(sparse.coo_matrix((ev,(er,ec)),shape=(ei,len(support))).tocsr(),sparse.eye(3),format="csr");k=E.shape[0];I1=sparse.eye(m,format="csr");I2=sparse.eye(k,format="csr");Z=lambda r,col:sparse.csr_matrix((r,col))
AA=sparse.vstack((sparse.hstack((-A,Z(A.shape[0],m),Z(A.shape[0],k))),sparse.hstack((F,-I1,Z(m,k))),sparse.hstack((-F,-I1,Z(m,k))),sparse.hstack((E,Z(k,m),-I2)),sparse.hstack((-E,Z(k,m),-I2))),format="csr");obj=np.r_[np.zeros(n),np.ones(m),np.full(k,.02)];bounds=list(zip(lo,hi))+[(0,None)]*(m+k);target=(targets["target_world_f32_m"].astype(float)-x[selected].astype(float)).reshape(-1)*1000.;results=[]
print(json.dumps({"support_nodes":len(support),"affected_faces":len(affected),"target_nodes":len(selected),"depth_rows":len(prows),"LP_variables":len(obj),"LP_rows":AA.shape[0],"target_max_mm":float(np.linalg.norm(target.reshape(-1,3),axis=1).max())}),flush=True)
for alpha in (.5,):
 ub=np.r_[-b,alpha*target,-alpha*target,np.zeros(2*k)];res=linprog(obj,A_ub=AA,b_ub=ub,bounds=bounds,method="highs",options={"time_limit":25.});item={"alpha":alpha,"success":bool(res.success),"solver_status":int(res.status),"message":str(res.message),"iterations":int(res.nit)}
 if not res.success:item["status"]="INFEASIBLE_OR_SOLVE_FAILURE";results.append(item);continue
 d=res.x[:n].reshape(-1,3);y=x.copy();y[support]=(x[support].astype(float)+d/1000.).astype(np.float32);actual=(y.astype(float)-x.astype(float))*1000.;area=q.full_f32_area_gate(y,bm,ref,faces,.1);ls=float((A@actual[support].reshape(-1)-b).min(initial=1e30));es=float((EG@actual[support].reshape(-1)-eb).min(initial=1e30));mx=float(np.linalg.norm(actual,axis=1).max());sm=float((np.linalg.norm(y.astype(float)-refsrc,axis=1)*1000.).max());changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1));u=x.astype(float)[faces[affected]];v=y.astype(float)[faces[affected]];n0=np.cross(u[:,1]-u[:,0],u[:,2]-u[:,0]);n1=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);m0=np.linalg.norm(n0,axis=1);m1=np.linalg.norm(n1,axis=1);ratio=m1/m0;cos=np.sum(n0*n1,axis=1)/(m0*m1);item.update(area_gate=area,linear_slack=ls,edge_slack_mm=es,max_step_mm=mx,max_source_mm=sm,changed_count=len(changed),target_residual_l1_mm=float(np.abs(F@actual[support].reshape(-1)-alpha*target).sum()),local_min_area=float(ratio.min()),local_max_area=float(ratio.max()),local_min_normal_cos=float(cos.min()))
 if not area["pass"] or ls< -1e-4 or es< -1e-4 or mx>4.000001 or sm>12.000001 or not np.array_equal(y[anchors],x[anchors]):item["status"]="POST_EMISSION_GATE_FAIL";results.append(item);continue
 payload={k:v.copy() for k,v in base.items()};payload["points_world_m"]=y;p=out/("candidate-alpha-"+str(alpha)+".npz")
 with p.open("xb") as fp:np.savez_compressed(fp,**payload)
 item.update(status="CHEAP_GATE_PASS_ONLY",candidate=str(p),candidate_sha256=sha(p));results.append(item)
report={"baseline_sha256":sha(c/"candidate.npz"),"target_sha256":sha(tp),"support_nodes":len(support),"affected_faces":len(affected),"results":results,"elapsed_s":time.monotonic()-start,"native_admission":False,"accepted_physical_steps":0,"limitations":"soft targets and linearized area only; exact skin/bone/self/containment, skin-fat transfer and native gates remain"};(out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(report,indent=2),flush=True)
