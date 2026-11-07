#!/usr/bin/env python3
"""FP64 evaluation of the fixed fixture equations. No integration or fallback."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def evaluate(data,h=.0005,arm="ipc",rho=1000.,target=5e-7,capture="trace"):
    a=SimpleNamespace(h=h,arm=arm,rho=rho,target=target,capture=capture)
    def buf(name,dtype="<f4",width=4):
        return np.frombuffer(data[name],dtype).reshape(-1,width).astype(np.float64)
    old=buf("fem_accepted_nodes",width=24);cur=buf("fem_candidate_nodes",width=24);N=len(cur);h=float(np.float32(a.h));tetf=buf("tetrahedra_accepted",width=24);tetu=buf("tetrahedra_accepted",dtype="<u4",width=24).astype(np.uint32)
    samples=buf("contact_samples",width=40);sample_ids=buf("contact_samples",dtype="<u4",width=40).astype(np.uint32)
    mass=cur[:,3];v=cur[:,4:7];points=cur[:,:3];direction=buf("fem_solution_correction")[:N,:3];native_b=buf("full_rhs_before_fgmres")[:N,:3];native_jv=buf("center_production_jdelta_action")[:N,:3]
    mu,lam=2000.,5000.;slop=float(np.float32(1e-4));gravity=np.array([0.,0.,float(np.float32(-9.81))]);internal=np.zeros((N,3));elastic_action=np.zeros((N,3));minJ=1.
    for tf,ti in zip(tetf,tetu):
     ids=ti[:4].astype(int);B=tf[4:16].reshape(3,4)[:,:3];volume=tf[7];z=points[ids]
     if np.all(cur[ids,19]==1):
      u=cur[ids,16:19]+cur[ids,20:23];F=np.eye(3)+(u[1:]-u[0]).T@B
     else:F=(z[1:]-z[0]).T@B
     J=np.linalg.det(F);assert J>0;minJ=min(minJ,J);invT=np.linalg.inv(F).T;c=lam*np.log(J)-mu;P=mu*F+c*invT
     H=-volume*P@B.T;forces=np.vstack([-H.sum(axis=1),H.T]);np.add.at(internal,ids,forces)
     d=direction[ids];dF=h*(d[1:]-d[0]).T@B
     dP=mu*dF+lam*np.sum(invT*dF)*invT-c*invT@dF.T@invT
     dH=-volume*dP@B.T;np.add.at(elastic_action,ids,np.vstack([-dH.sum(axis=1),dH.T]))
    contact=np.zeros((N,3));contact_action=np.zeros((N,3));force_errors=[];tangent_errors=[];impact_dominant=0
    for s,si in zip(samples,sample_ids):
     if not si[3]&1:continue
     i=int(si[0]);assert i<N;n=s[8:11];gap=s[7]+(s[31] if si[3]&512 else 0.);normal_v=s[11]
     if a.arm=="al":
      force=max(0.,s[27]-a.rho*(gap-a.target));curvature=a.rho if force>0 else 0.
     elif gap>=slop:force=curvature=0.
     else:
      distance=max(gap,1e-4*slop);D=distance*distance;T=slop*slop;off=D-T;log=np.log(D/T);k=s[27]
      first=-k*(2*off*log+off*off/D);second=-k*(2*log+4*off/D-off*off/(D*D))
      force=max(0.,-2*distance*first);curvature=max(0.,4*D*second+2*first) if gap>=1e-4*slop else 0.
      inertial=10./max(cur[i,7]*h*h*T,1e-20);impact=max(-normal_v,0.)/max(cur[i,7]*h*max(slop-distance,1e-3*slop)*T,1e-20)
      impact_dominant+=impact>inertial
     contact[i]+=h*force*n;contact_action[i]+=h*h*curvature*n*np.dot(n,direction[i])
     force_errors.append(abs(force*h-s[19]));native_H=s[28:40].reshape(3,4)[:,:3];tangent_errors.append(np.linalg.norm(h*h*curvature*np.outer(n,n)-native_H))
    residual=h*(internal+mass[:,None]*gravity)-mass[:,None]*(v-old[:,4:7])+contact
    jv=mass[:,None]*direction-h*elastic_action+contact_action
    result={"schema":"numi.fem-plane.fp64-diagnostic.v1","capture":str(a.capture),"numpy":np.__version__,"nodes":N,"h_float32":h,"arm":a.arm,"native_rhs_l2_Ns":float(np.linalg.norm(native_b)),"fp64_rhs_l2_Ns":float(np.linalg.norm(residual)),"rhs_difference_l2_Ns":float(np.linalg.norm(residual-native_b)),"rhs_difference_max_node_Ns":float(np.max(np.linalg.norm(residual-native_b,axis=1))),"native_max_local_gravity_scaled_residual":float(np.max(np.linalg.norm(native_b,axis=1)/(mass*9.81*h))),"fp64_max_local_gravity_scaled_residual":float(np.max(np.linalg.norm(residual,axis=1)/(mass*9.81*h))),"native_linear_residual_l2_Ns":float(np.linalg.norm(native_b-native_jv)),"fp64_linear_residual_l2_Ns":float(np.linalg.norm(residual-jv)),"jv_relative_difference":float(np.linalg.norm(jv-native_jv)/max(np.linalg.norm(native_jv),1e-30)),"contact_impulse_max_difference_Ns":max(force_errors,default=0.),"contact_tangent_max_difference_kg":max(tangent_errors,default=0.),"impact_stiffness_dominant_contacts":int(impact_dominant),"position_velocity_defect_max_m":float(np.max(np.abs(points-old[:,:3]-h*v))),"min_J":minJ,"limits":["Same compiled rest matrices and captured working state; no time integration is performed.","FP64 evaluates log-J neo-Hookean stress and analytic Jv independently; the runtime remains Metal FP32.","IPC tangent comparison freezes its captured center stiffness, as the production linearization does."]}
    return result

def read_capture(path):
    x=json.loads(path.read_text());payload=Path(x["payload"]["file"])
    if not payload.exists():payload=path.parent/payload.name
    raw=payload.read_bytes();assert hashlib.sha256(raw).hexdigest()==x["payload"]["sha256"]
    data={}
    for v in x["arrays"]:
        z=raw[v["offset"]:v["offset"]+v["byte_count"]]
        assert hashlib.sha256(z).hexdigest()==v["sha256"]
        data[v["name"]]=z
    return data

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("capture",type=Path)
    p.add_argument("--h",type=float,default=.0005);p.add_argument("--arm",choices=["ipc","al"],default="ipc")
    a=p.parse_args()
    print(json.dumps(evaluate(read_capture(a.capture),a.h,a.arm,capture=str(a.capture)),indent=2))
