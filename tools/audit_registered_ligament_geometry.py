import argparse, hashlib, json, math, struct
from pathlib import Path

ROOT=Path('/Users/numi/numi-human-aortic-contact-opt-20261019')
RIGID=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
BONES=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.nhbones')
INVOCATION=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-003/invocation.json')

def qmul(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)
def qrot(q,v):
    x,y,z,w=q
    uv=(y*v[2]-z*v[1],z*v[0]-x*v[2],x*v[1]-y*v[0])
    uuv=(y*uv[2]-z*uv[1],z*uv[0]-x*uv[2],x*uv[1]-y*uv[0])
    return tuple(v[k]+2*w*uv[k]+2*uuv[k] for k in range(3))
def qinvrot(q,v): return qrot((-q[0],-q[1],-q[2],q[3]),v)
def add(a,b): return tuple(a[i]+b[i] for i in range(3))
def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
def norm2(v): return dot(v,v)
def closest(p,a,b,c):
    ab=sub(b,a); ac=sub(c,a); ap=sub(p,a); d1=dot(ab,ap); d2=dot(ac,ap)
    if d1<=0 and d2<=0: return a
    bp=sub(p,b); d3=dot(ab,bp); d4=dot(ac,bp)
    if d3>=0 and d4<=d3: return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        v=d1/(d1-d3); return add(a,tuple(v*x for x in ab))
    cp=sub(p,c); d5=dot(ab,cp); d6=dot(ac,cp)
    if d6>=0 and d5<=d6: return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        w=d2/(d2-d6); return add(a,tuple(w*x for x in ac))
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); return add(b,tuple(w*x for x in sub(c,b)))
    den=1.0/(va+vb+vc); v=vb*den; w=vc*den
    return add(a,add(tuple(v*x for x in ab),tuple(w*x for x in ac)))
def project(p, mesh):
    best=None; bd=float('inf'); besttri=-1
    verts,tris=mesh
    for ti,(ia,ib,ic) in enumerate(tris):
        q=closest(p,verts[ia],verts[ib],verts[ic]); d=norm2(sub(q,p))
        if d<bd: best,bd,besttri=q,d,ti
    return best,math.sqrt(bd),besttri
def segment_hits(a,b,mesh):
    verts,tris=mesh; d=sub(b,a); hits=[]; dlen=math.sqrt(norm2(d))
    for ti,(ia,ib,ic) in enumerate(tris):
        v0,v1,v2=verts[ia],verts[ib],verts[ic]; e1=sub(v1,v0); e2=sub(v2,v0)
        h=(d[1]*e2[2]-d[2]*e2[1],d[2]*e2[0]-d[0]*e2[2],d[0]*e2[1]-d[1]*e2[0]); det=dot(e1,h)
        # det has cubic-length units; use a dimensionless, scale-aware threshold.
        scale=dlen*math.sqrt(norm2(e1))*math.sqrt(norm2(e2))
        if scale==0 or abs(det)<=32.0*2.220446049250313e-16*scale: continue
        inv=1.0/det; ss=sub(a,v0); u=inv*dot(ss,h)
        if u < -1e-8 or u > 1.0+1e-8: continue
        q=(ss[1]*e1[2]-ss[2]*e1[1],ss[2]*e1[0]-ss[0]*e1[2],ss[0]*e1[1]-ss[1]*e1[0]); v=inv*dot(d,q)
        if v < -1e-8 or u+v > 1.0+1e-8: continue
        t=inv*dot(e2,q)
        if 1e-5 < t < 1.0-1e-5: hits.append((t,ti))
    return hits

def read_payloads():
    rb=RIGID.read_bytes(); h=struct.unpack_from('<8s10I32s',rb,0)
    if h[0]!=b'NHRIGID2' or h[1]!=1: raise ValueError('unexpected NHRIGID2')
    src,nb,nj,nq,nv=h[3:8]
    joint0=224+160*nb
    js=[struct.unpack_from('<8I28f',rb,joint0+144*i) for i in range(nj)]
    q0=joint0+144*nj+64*nv
    default_q=struct.unpack_from(f'<{nq}f',rb,q0)
    map0=q0+4*nq+4*nv
    source_to_core=struct.unpack_from(f'<{src}I',rb,map0)
    pose0=map0+4*src
    bodypose={source_to_core[i]:struct.unpack_from('<7f',rb,pose0+28*i) for i in range(src)}
    if pose0+28*src!=len(rb): raise ValueError('NHRIGID2 trailing/truncated payload')
    bb=BONES.read_bytes(); bh=struct.unpack_from('<8s5I32s',bb,0)
    if bh[0]!=b'NHBONES1' or bh[1]!=3: raise ValueError('unexpected NHBONES payload')
    nrec,nverts,nindices=bh[2:5]
    record0=60; vertex0=record0+nrec*60; index0=vertex0+nverts*24
    if index0+4*nindices!=len(bb): raise ValueError('NHBONES payload dimensions mismatch')
    records={}
    for ri in range(nrec):
        t=struct.unpack_from('<6I8fI',bb,record0+60*ri)
        body,fv,nv,fi,ni,sid=t[:6]; tx,ty,tz,qx,qy,qz,qw,scale=t[6:14]; source_record=t[14]
        if sid in records: raise ValueError(f'duplicate stable id {sid}')
        records[sid]={'record':ri,'body':body,'firstVertex':fv,'vertexCount':nv,'firstIndex':fi,'indexCount':ni,
                       'translation':(tx,ty,tz),'boneQ':(qx,qy,qz,qw),'scale':scale,'sourceRecord':source_record}
    return rb,bb,h,bh,js,default_q,bodypose,records,bb,vertex0,index0

def make_frame(js,default_q,bodypose,parent,child):
    found=[(i,j) for i,j in enumerate(js) if j[0]==parent and j[1]==child]
    if len(found)!=1: raise ValueError(f'joint {parent}->{child} occurrences={len(found)}')
    ji,j=found[0]; qo,nq=j[4:6]
    if any(abs(default_q[k])>1e-8 for k in range(qo,qo+nq)): raise ValueError(f'frame joint {ji} has nonzero default q')
    pos=bodypose[child][:3]; bq=bodypose[child][3:]
    anchor=j[24:27]; cq=j[32:36]
    origin=add(pos,qrot(bq,anchor)); q=qmul(bq,cq)
    axes=[qrot(q,e) for e in ((1,0,0),(0,1,0),(0,0,1))]
    return {'joint':ji,'parent':parent,'child':child,'origin':origin,'q':q,'axes':axes}

def build_mesh(sid, frame, records, bb, vertex0, index0, bodypose):
    r=records[sid]; px,py,pz,bx,by,bz,bw=bodypose[r['body']]; bodyq=(bx,by,bz,bw)
    verts=[]
    for n in range(r['vertexCount']):
        x,y,z=struct.unpack_from('<3f',bb,vertex0+24*(r['firstVertex']+n))
        boneLocal=add(r['translation'],qrot(r['boneQ'],(x*r['scale'],y*r['scale'],z*r['scale'])))
        world=add((px,py,pz),qrot(bodyq,boneLocal))
        verts.append(qinvrot(frame['q'],sub(world,frame['origin'])))
    inds=struct.unpack_from(f'<{r["indexCount"]}I',bb,index0+4*r['firstIndex'])
    tris=[]
    for i in range(0,len(inds),3):
        tri=tuple(inds[i+k]-r['firstVertex'] for k in range(3))
        if min(tri)<0 or max(tri)>=len(verts): raise ValueError(f'bad bone index stable={sid}')
        tris.append(tri)
    return (verts,tris)

# One right-sided seed definition per row; side 0 is mirrored across local Z by the cooker.
ROWS=[
 ('iliofemoral_lateral','hip',128,18,131,2,130,131,(-.010,.012,.006),(.015,-.005,.006)),
 ('iliofemoral_medial','hip',128,18,131,2,130,131,(.010,.012,-.006),(-.015,-.005,-.006)),
 ('ischiofemoral','hip',128,18,131,2,130,131,(0,-.015,.010),(0,.015,-.010)),
 ('pubofemoral','hip',128,18,131,2,130,131,(0,-.015,-.010),(0,.015,.010)),
 ('coracohumeral_capsuloligament','shoulder',34,10,41,12,40,41,(.012,.010,0),(-.010,-.010,0)),
 ('superior_glenohumeral_capsuloligament','shoulder',34,10,41,12,40,41,(-.010,.006,0),(.010,-.006,0)),
 ('anterior_talofibular_superior','ankle',136,20,137,22,136,137,(0,0,.006),(0,0,-.006)),
 ('anterior_talofibular_inferior','ankle',136,20,137,22,136,137,(0,0,.003),(0,0,-.003)),
 ('calcaneofibular','ankle',136,20,138,6,137,138,(.005,0,.012),(-.005,0,-.012)),
 ('posterior_talofibular','ankle',136,20,137,22,136,137,(-.005,0,-.010),(.005,0,.010)),
 ('anterior_tibiotalar','ankle',136,4,137,22,136,137,(0,.005,.012),(0,-.005,-.012)),
 ('tibiocalcaneal','ankle',136,4,138,6,137,138,(0,-.010,.012),(0,.010,-.012)),
 ('posterior_tibiotalar','ankle',136,4,137,22,136,137,(0,.005,-.012),(0,-.005,.012)),
]
# Region proxy targets in the joint frame (mm). These are not claimed entheses;
# the source mesh witness and residual are reported so a reviewer can reject them.
REGION_TARGETS={
 'iliofemoral_lateral':((24,42,18),(12,-16,28)),
 'iliofemoral_medial':((18,32,-20),(12,-16,-20)),
 'ischiofemoral':((-28,44,8),(-18,-24,28)),
 'pubofemoral':((34,8,-18),(18,-30,-22)),
 'coracohumeral_capsuloligament':((22,2,-24),(18,-18,12)),
 'superior_glenohumeral_capsuloligament':((8,0,-8),(18,-10,4)),
 'anterior_talofibular_superior':((10,-3,25),(16,-3,12)),
 'anterior_talofibular_inferior':((7,-5,25),(12,-8,12)),
 'calcaneofibular':((-8,-4,25),(-15,-25,15)),
 'posterior_talofibular':((-12,-4,25),(-18,-4,12)),
 'anterior_tibiotalar':((12,12,-20),(12,0,-12)),
 'tibiocalcaneal':((0,12,-20),(0,-20,-15)),
 'posterior_tibiotalar':((-12,12,-20),(-18,0,-12)),
}

def mirror(v,side): return (v[0],v[1],-v[2]) if side==0 else v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',default=str(ROOT/'tools/audit_registered_ligament_geometry.json'));args=ap.parse_args()
    invb=INVOCATION.read_bytes(); inv=json.loads(invb); argv=inv['argv']
    if Path(argv[1]) != RIGID or Path(argv[3]) != BONES: raise ValueError('audit payloads do not match native-supported-003 argv[1]/argv[3]')
    rb,bb,rh,bh,js,dq,bodypose,records,raw,vertex0,index0=read_payloads()
    frames={}
    for side in (0,1):
        label='L' if side==0 else 'R'
        for region,(parent,child) in {'hip':((144,145) if side==0 else (130,131)), 'shoulder':((90,91) if side==0 else (40,41)), 'ankle':((150,151) if side==0 else (136,137))}.items():
            frames[(region,side)]=make_frame(js,dq,bodypose,parent,child)
    meshes={}
    out={'audit':'registered_ligament_source_geometry_v1','classification':'inferred regional reference fibers; these surface witnesses do not validate named subject entheses',
         'source':{'invocationPath':str(INVOCATION),'invocationSha256':hashlib.sha256(invb).hexdigest(),'sourceInvocationSha256':inv.get('source_sha256'),'argvRigidPath':argv[1],'argvBonesPath':argv[3],'rigidPath':str(RIGID),'rigidSha256':hashlib.sha256(rb).hexdigest(),'bonesPath':str(BONES),'bonesSha256':hashlib.sha256(bb).hexdigest(),
                   'rigidHeader':{'sourceBodies':rh[3],'engineBodies':rh[4],'joints':rh[5],'nq':rh[6],'nv':rh[7]},
                   'bonesHeader':{'payloadAbi':bh[1],'boneCount':bh[2],'vertexCount':bh[3],'indexCount':bh[4]},
                   'catalogHeaderSha256':hashlib.sha256((ROOT/'apps/NumiHumanRegisteredLigaments.hpp').read_bytes()).hexdigest()},'frames':{},'rows':[],'endpointMeshTopology':{}}
    for (region,side),f in frames.items():
        lab=('L' if side==0 else 'R');out['frames'][region+lab]={'joint':f['joint'],'parentBody':f['parent'],'childBody':f['child'],'originWorldM':f['origin'],'axesLocalInWorld':f['axes']}
    for name,region,b0,s0,b1,s1,fp,fc,seed0,seed1 in ROWS:
      for side in (0,1):
        # mirror stable IDs/body/joint pairs with laterality; all provided pairs are right-side templates.
        if side==0:
            b0s={128:128,34:84,136:150,137:151}[b0]; b1s={131:145,41:91,137:151,138:152}[b1]
            s0s={18:19,10:11,20:21,4:5}[s0]; s1s={2:3,12:13,22:23,6:7}[s1]
            fpS,fcS=({130:144,40:90,136:150,137:151}[fp],{131:145,41:91,137:151,138:152}[fc])
        else: b0s,b1s,s0s,s1s,fpS,fcS=b0,b1,s0,s1,fp,fc
        frame=frames[(region,side)]
        meshes[(s0s,region,side)]=meshes.get((s0s,region,side)) or build_mesh(s0s,frame,records,bb,vertex0,index0,bodypose)
        meshes[(s1s,region,side)]=meshes.get((s1s,region,side)) or build_mesh(s1s,frame,records,bb,vertex0,index0,bodypose)
        cur0,cur1=mirror(seed0,side),mirror(seed1,side)
        p0,d0,t0=project(cur0,meshes[(s0s,region,side)]);p1,d1,t1=project(cur1,meshes[(s1s,region,side)])
        target0,target1=REGION_TARGETS[name];target0=mirror(tuple(x/1000 for x in target0),side);target1=mirror(tuple(x/1000 for x in target1),side)
        mesh0=meshes[(s0s,region,side)];mesh1=meshes[(s1s,region,side)]
        rp0,rd0,rt0=project(target0,mesh0);rp1,rd1,rt1=project(target1,mesh1)
        hits0=segment_hits(rp0,rp1,mesh0);hits1=segment_hits(rp0,rp1,mesh1)
        def mm(v):return [round(x*1000,3) for x in v]
        out['rows'].append({'name':name,'side':('L' if side==0 else 'R'),'frameJoint':frame['joint'],'framePair':[frame['parent'],frame['child']],'catalogFramePair':[fpS,fcS],
          'surface0':{'stableId':s0s,'sourceRecord':records[s0s]['sourceRecord'],'body':b0s,'catalogSeedLocalMmUnderAuditFrame':mm(cur0),'catalogSeedProjectionLocalMmUnderAuditFrame':mm(p0),'residualMm':round(d0*1000,3),'triangle':t0,
                      'regionTargetLocalMm':mm(target0),'regionWitnessLocalMm':mm(rp0),'regionResidualMm':round(rd0*1000,3),'regionTriangle':rt0},
          'surface1':{'stableId':s1s,'sourceRecord':records[s1s]['sourceRecord'],'body':b1s,'catalogSeedLocalMmUnderAuditFrame':mm(cur1),'catalogSeedProjectionLocalMmUnderAuditFrame':mm(p1),'residualMm':round(d1*1000,3),'triangle':t1,
                      'regionTargetLocalMm':mm(target1),'regionWitnessLocalMm':mm(rp1),'regionResidualMm':round(rd1*1000,3),'regionTriangle':rt1},
          'regionTargetBasis':'experimental local-frame regional proxy; witness is nearest registered source-mesh triangle, not an enthesis label',
          'straightFiberEndpointBoneHits':{'surface0':[[round(t,7),tri] for t,tri in hits0],'surface1':[[round(t,7),tri] for t,tri in hits1],
              'crossesEitherEndpointBone':bool(hits0 or hits1),'interpretation':'interior segment/triangle intersections excluding endpoint tolerance; other bones are not tested'}})
    # Surface records are open patches; record boundary topology so no-hit is not a closed-volume proof.
    for sid in sorted({r['surface0']['stableId'] for r in out['rows']} | {r['surface1']['stableId'] for r in out['rows']}):
        rec=records[sid]
        inds=struct.unpack_from(f"<{rec['indexCount']}I",bb,index0+4*rec['firstIndex'])
        ec={}
        for k in range(0,len(inds),3):
            tri=[inds[k+i]-rec['firstVertex'] for i in range(3)]
            for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                edge=(a,b) if a<b else (b,a); ec[edge]=ec.get(edge,0)+1
        hist={}
        for n in ec.values(): hist[str(n)]=hist.get(str(n),0)+1
        out['endpointMeshTopology'][str(sid)]={'sourceRecord':rec['sourceRecord'],'body':rec['body'],'uniqueEdges':len(ec),'edgeIncidenceHistogram':hist,'boundaryEdges':hist.get('1',0),'nonManifoldEdges':sum(v for k,v in hist.items() if int(k)>2),'closed2Manifold':hist.get('1',0)==0 and hist.get('2',0)==len(ec)}
    out['sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    Path(args.receipt).write_text(json.dumps(out,indent=2)+'\n')
    print('receipt',args.receipt,'sha256',out['sha256'],'rows',len(out['rows']))
    for r in out['rows']:
      a,b=r['surface0'],r['surface1']
      print(f"{r['name']:42s} {r['side']} ids={a['stableId']},{b['stableId']} current={a['residualMm']:.2f}/{b['residualMm']:.2f}mm tri={a['triangle']}/{b['triangle']} local={a['catalogSeedProjectionLocalMmUnderAuditFrame']}/{b['catalogSeedProjectionLocalMmUnderAuditFrame']} region={a['regionResidualMm']:.2f}/{b['regionResidualMm']:.2f}mm tri={a['regionTriangle']}/{b['regionTriangle']} wit={a['regionWitnessLocalMm']}/{b['regionWitnessLocalMm']}")
if __name__=='__main__': main()
