#!/usr/bin/env python3
"""Bounded exact-pose source-mesh search for posterior lumbar interspinous proxy paths."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"tools"))
import audit_registered_ligament_geometry as ag
import audit_actual_pose_registered_ligament_catalog as ap
import audit_ligament_regional_paths as lp
import audit_final_registered_ligament_catalog as base

def tri_normals(mesh, vertex_normals):
    tris=np.asarray(mesh[1],dtype=np.int32); ns=np.asarray(vertex_normals,dtype=np.float64)[tris].mean(axis=1)
    n=np.linalg.norm(ns,axis=1);return ns/np.maximum(n[:,None],1e-30)

def candidates(mesh, normals, upper, limit=48):
    v=np.asarray(mesh[0],dtype=np.float64);t=np.asarray(mesh[1],dtype=np.int32);c=v[t].mean(axis=1)
    x,y,z=c.T
    posterior_cut=float(np.quantile(x,.18))
    # The source lumbar vertebrae use +X anterior, +Y superior, and Z lateral
    # in the selected child-entry joint frame; retain the central posterior
    # spinous-process patch and the opposed inferior/superior edge by level.
    roi=(x<=posterior_cut)&(np.abs(z)<=.018)
    idx=np.flatnonzero(roi)
    if not len(idx):return [],{"posteriorCutM":posterior_cut,"roiCount":0}
    yr=y[idx]
    ytarget=float(np.quantile(yr,.25 if upper else .75))
    xtarget=float(np.median(x[idx]))
    target=np.array([xtarget,ytarget,0.0])
    score=np.sum((c[idx]-target)**2,axis=1)
    keep=idx[np.argsort(score)[:limit]]
    out=[(np.asarray(c[i],dtype=np.float32).astype(np.float64),int(i),float(np.linalg.norm(c[i]-target)),normals[i]) for i in keep]
    info={"posteriorCutM":posterior_cut,"roiCount":int(len(idx)),"roiBoundsMm":[(np.min(c[idx],axis=0)*1000).tolist(),(np.max(c[idx],axis=0)*1000).tolist()],"targetMm":(target*1000).tolist(),"targetRule":"posterior 18th-percentile triangle-centroid cut, central |Z|<=18mm; cranial bone y=25th percentile, caudal bone y=75th percentile"}
    return out,info

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--invocation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();outpath=a.output.resolve()
    invpath=a.invocation.resolve();ib=invpath.read_bytes();inv=json.loads(ib);argv=inv["argv"]
    if len(argv)<4:raise RuntimeError("exact invocation lacks rigid/bone argv")
    ag.RIGID=Path(argv[1]);ag.BONES=Path(argv[3])
    rb,bb,rh,bh,joints,dq,sourcepose,records,_,vertex0,index0=ag.read_payloads()
    q=list(map(float,dq))
    for i,x in enumerate(argv):
        if x=="--pose-q":q[int(argv[i+1])]=float(argv[i+2])
    nheq=Path(argv[argv.index("--joint-equality-payload")+1]);nheb,neh,q,eqcorr=ap.parse_nheq(nheq,q)
    poses,roots,loops=ap.forward_fk(joints,sourcepose,q);posemap={i:x for i,x in enumerate(poses) if x is not None}
    if sorted(records)!=list(range(1,186)):raise RuntimeError(f"expected stable IDs 1..185, got {len(records)}")
    missing=sorted(s for s,r in records.items() if r["body"] not in posemap)
    if missing:raise RuntimeError(f"source records missing articulated body poses: {missing}")
    segments=[
      {"name":"interspinous_L1_L2","cranialBody":19,"cranialStableId":164,"caudalBody":16,"caudalStableId":165,"framePair":[15,16],"frameJoint":15},
      {"name":"interspinous_L2_L3","cranialBody":16,"cranialStableId":165,"caudalBody":13,"caudalStableId":166,"framePair":[12,13],"frameJoint":12},
      {"name":"interspinous_L3_L4","cranialBody":13,"cranialStableId":166,"caudalBody":10,"caudalStableId":167,"framePair":[9,10],"frameJoint":9},
      {"name":"interspinous_L4_L5","cranialBody":10,"cranialStableId":167,"caudalBody":4,"caudalStableId":168,"framePair":[3,4],"frameJoint":3},
    ]
    out={"audit":"exact_pose_lumbar_interspinous_candidate_search_v1","classification":"screening of inferred posterior spinous-process path proxies; not validated ligament entheses or clinical calibration","source":{"invocationPath":str(invpath),"invocationSha256":hashlib.sha256(ib).hexdigest(),"sourceInvocationSha256":inv.get("source_sha256"),"binarySha256":inv.get("binary_sha256"),"rigidPath":str(ag.RIGID),"rigidSha256":hashlib.sha256(rb).hexdigest(),"bonesPath":str(ag.BONES),"bonesSha256":hashlib.sha256(bb).hexdigest(),"boneAbi":bh[1],"boneRecordCount":bh[2],"vertexCount":bh[3],"indexCount":bh[4],"nheqPath":str(nheq),"nheqSha256":hashlib.sha256(nheb).hexdigest(),"poseQOverrides":[[int(argv[i+1]),float(argv[i+2])] for i,x in enumerate(argv) if x=="--pose-q"],"nheqMaxCorrection":eqcorr,"sourcePoseRecords":len(sourcepose)},"criteria":{"endpoint":"triangle centroid on registered vertebral source surface; source-local IDs verified by pinned NHBONES payload","region":"frame-local posterior-most 18% of triangle-centroid X, |Z|<=18mm, cranial caudal-edge and caudal cranial-edge quartile targets; candidates remain inference","normal":"transformed registered vertex normals averaged at centroid; require each endpoint non-inward cosine >=0.05","path":"float32 anchor coordinates tested against all 185 registered bone meshes by scale-relative segment/triangle intersection with AABB culling; reject any interior hit","limits":"open source meshes and triangle-clearance cannot establish closed-volume nonpenetration or exact tissue insertions; no mass/calibration is inferred"},"segments":[]}
    meshes_cache={};normals_cache={}
    for seg in segments:
        ji,frame=ap.body_frame(joints,poses,tuple(seg["framePair"]))
        if ji!=seg["frameJoint"]:raise RuntimeError(f"lumbar frame identity changed {seg[name]} {ji}")
        key=ji
        if key not in meshes_cache:
            meshes_cache[key]=lp.all_meshes(frame,bb,posemap,records,vertex0,index0)
            if len(meshes_cache[key])!=185:raise RuntimeError(f"all-mesh transform count={len(meshes_cache[key])}")
        allm=meshes_cache[key]
        sid0,sid1=seg["cranialStableId"],seg["caudalStableId"]
        if (sid0,key) not in normals_cache:normals_cache[(sid0,key)]=base.load_vertex_normals(sid0,frame,bb,posemap,records,vertex0)
        if (sid1,key) not in normals_cache:normals_cache[(sid1,key)]=base.load_vertex_normals(sid1,frame,bb,posemap,records,vertex0)
        m0,m1=allm[sid0],allm[sid1]
        n0,n1=tri_normals(m0,normals_cache[(sid0,key)]),tri_normals(m1,normals_cache[(sid1,key)])
        c0,i0=candidates(m0,n0,True);c1,i1=candidates(m1,n1,False)
        pairs=sorted(((a[2]**2+b[2]**2+.02*np.linalg.norm(a[0]-b[0])**2,a,b) for a in c0 for b in c1),key=lambda x:x[0])
        accepted=[];tested=0;bestRejected=None
        for score,a,b in pairs:
            tested+=1;p0,p1=a[0],b[0];d=p1-p0;length=float(np.linalg.norm(d))
            if length<.002 or length>.060:continue
            axis=d/length;f0=float(np.dot(a[3],axis));f1=float(np.dot(b[3],-axis))
            if min(f0,f1)<.05:
                if bestRejected is None:bestRejected={"reason":"endpoint normal points inward or nearly tangent","anchorsMm":[(p0*1000).tolist(),(p1*1000).tolist()],"normalCos":[f0,f1],"lengthMm":length*1000}
                continue
            lo=np.minimum(p0,p1)-1e-9;hi=np.maximum(p0,p1)+1e-9;near=[];hits=[]
            for sid,(vv,tt) in allm.items():
                if np.all(np.max(vv,axis=0)>=lo) and np.all(np.min(vv,axis=0)<=hi):
                    near.append(sid);hh=lp.vector_hits(p0,p1,lp.mesh_tuple(vv,tt))
                    if hh:hits.append({"stableId":sid,"sourceRecord":records[sid]["sourceRecord"],"body":records[sid]["body"],"hits":[{"t":round(t,8),"triangle":ti} for t,ti in hh]})
            candidate={"score":float(score),"anchorsFrameMm":[(p0*1000).tolist(),(p1*1000).tolist()],"sourceTriangles":[a[1],b[1]],"targetResidualMm":[a[2]*1000,b[2]*1000],"referenceLengthMm":length*1000,"outwardNormalCos":[f0,f1],"nearbyStableIds":near,"segmentHits":hits}
            if not hits:accepted.append(candidate);break
            if bestRejected is None:bestRejected={"reason":"segment intersects registered source mesh","hits":hits,"anchorsFrameMm":candidate["anchorsFrameMm"],"referenceLengthMm":candidate["referenceLengthMm"],"normalCos":[f0,f1]}
        result={**seg,"frameAnchorResidualM":frame["anchorResidualM"],"allRegisteredBoneMeshesTested":len(allm),"candidateCounts":[len(c0),len(c1)],"candidateRegionCranial":i0,"candidateRegionCaudal":i1,"candidatePairsTested":tested,"endpointTopology":{"cranial":lp.mesh_topology(np.asarray(m0[1],dtype=np.int32)),"caudal":lp.mesh_topology(np.asarray(m1[1],dtype=np.int32))},"accepted":bool(accepted),"witness":accepted[0] if accepted else None,"bestRejected":bestRejected}
        out["segments"].append(result)
        print(seg["name"],"accepted",bool(accepted),"pairs",tested,"candidates",[len(c0),len(c1)],"witness",accepted[0] if accepted else bestRejected)
    out["sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    outpath.parent.mkdir(parents=True,exist_ok=True);outpath.write_text(json.dumps(out,indent=2)+"\n")
    print("receipt",outpath,"sha256",out["sha256"],"accepted",sum(x["accepted"] for x in out["segments"]),"segments",len(out["segments"]))
    if not all(x["accepted"] for x in out["segments"]):raise SystemExit(1)
if __name__=="__main__":main()
