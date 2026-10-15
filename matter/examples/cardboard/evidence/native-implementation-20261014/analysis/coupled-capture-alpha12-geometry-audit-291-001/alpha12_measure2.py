import sys,json,math,struct,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/home/cardboard-implementation-20261014/analysis/coupled-free-body-candidate-capture-reader-v3-001')
from coupled_capture_reader import Capture
p=Path('/Users/home/cardboard-implementation-20261014/native/strict-json-full-contact-arena-newton12-291/candidate-capture.coupled.json')
c=Capture(p); dt=np.float32(c.manifest['dt_seconds']); ac=c.fem_nodes_f64('fem_accepted_nodes'); ca=c.fem_nodes_f64('fem_candidate_nodes')
accepted_u=ac['reference_displacement_and_mode'][:,:3].astype(np.float32); rest=ca['rest_and_fixed'][:,:3].astype(np.float32); xcenter=ca['position_and_mass'][:,:3].astype(np.float32); vcenter=ca['velocity_and_inverse_mass'][:,:3].astype(np.float32); ucenter=ca['reference_displacement_and_mode'][:,:3].astype(np.float32); corr=c.numeric('fem_solution_correction')[:36,:3].astype(np.float32)
def f32(x): return np.asarray(x,dtype=np.float32)
def mul(a,b): return f32(f32(a)*f32(b))
def add(a,b): return f32(f32(a)+f32(b))
def fma(a,b,c): return f32(f32(a).astype(np.float64)*f32(b).astype(np.float64)+f32(c).astype(np.float64))
def state(a, fused=True):
    av=np.float32(a)
    va=fma(av,corr,vcenter) if fused else add(vcenter,mul(av,corr))
    ua=fma(dt,va,accepted_u) if fused else add(accepted_u,mul(dt,va))
    xa=add(rest,ua)
    start=add(xa,-mul(dt,va))
    return va,ua,xa,start
for fused in (False,True):
    va,ua,xa,sa=state(0,fused)
    print('center-replay',fused,'veldiff',np.count_nonzero(va.view('u4')!=vcenter.view('u4')),'udiff',np.count_nonzero(ua.view('u4')!=ucenter.view('u4')),'xdiff',np.count_nonzero(xa.view('u4')!=xcenter.view('u4')),'max_xerr',np.max(np.abs(xa-xcenter)),'start-vs-accepted',np.count_nonzero(sa.view('u4')!=ac['position_and_mass'][:,:3].astype(np.float32).view('u4')),'starterrmax',np.max(np.abs(sa-ac['position_and_mass'][:,:3].astype(np.float32))))
for a in [2**-20,2**-22,2**-23,2**-24,2**-26,0.0]:
 for fused in (False,True):
  va,ua,xa,sa=state(a,fused)
  print('alpha',a,'fused',fused,'vchg',np.count_nonzero(va.view('u4')!=vcenter.view('u4')),'uchg',np.count_nonzero(ua.view('u4')!=ucenter.view('u4')),'xchg',np.count_nonzero(xa.view('u4')!=xcenter.view('u4')),'startchg',np.count_nonzero(sa.view('u4')!=state(0,fused)[3].view('u4')),'startAcceptedChanged',np.count_nonzero(sa.view('u4')!=ac['position_and_mass'][:,:3].astype(np.float32).view('u4')),'maxStartDelta',np.max(np.abs(sa-state(0,fused)[3])),'maxXdelta',np.max(np.abs(xa-xcenter)))
# Contact rows use raw records and compute CP on reconstructed start in F64.
raw=np.frombuffer(c.raw('deformable_contacts'),dtype='<u4').reshape(-1,27,4); fl=raw.view('<f4'); active=c.numeric('deformable_contact_active_indices')[:int(c.numeric('deformable_contact_active_counts')[0,0])]
def dot64(a,b):return float(np.dot(np.asarray(a,np.float64),np.asarray(b,np.float64)))
def cp_tri(point,a,b,d):
 # Ericson closest point; FP64 on exact FP32 node words.
 ab=b-a; ac=d-a; ap=point-a; d1=dot64(ab,ap);d2=dot64(ac,ap)
 if d1<=0 and d2<=0:return a.copy(),np.array([1.,0,0]),'v0'
 bp=point-b;d3=dot64(ab,bp);d4=dot64(ac,bp)
 if d3>=0 and d4<=d3:return b.copy(),np.array([0.,1,0]),'v1'
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/(d1-d3);return a+v*ab,np.array([1-v,v,0]),'e01'
 cp=point-d;d5=dot64(ab,cp);d6=dot64(ac,cp)
 if d6>=0 and d5<=d6:return d.copy(),np.array([0.,0,1]),'v2'
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/(d2-d6);return a+w*ac,np.array([1-w,0,w]),'e02'
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:
  w=(d4-d3)/((d4-d3)+(d5-d6));return b+w*(d-b),np.array([0.,1-w,w]),'e12'
 den=va+vb+vc;v=vb/den;w=vc/den;return a+v*ab+w*ac,np.array([1-v-w,v,w]),'face'

def contacts(positions):
 rows=[]
 for idx in active:
  if int(raw[idx,1,3])&8:continue
  if raw[idx,1,2]!=0:continue
  ids=raw[idx,0].astype(int)
  pnt=positions[ids[0]].astype(np.float64); tri=np.array([positions[j] for j in ids[1:4]],np.float64)
  q,bary,branch=cp_tri(pnt,*tri)
  delta=pnt-q; gap=np.linalg.norm(delta); n=delta/max(gap,1e-300)
  rows.append((int(idx),ids,branch,bary,gap,n,q))
 return rows
accx=ac['position_and_mass'][:,:3].astype(np.float32)
for a in [0,2**-22,2**-23,2**-26]:
 va,ua,xa,sa=state(a,True)
 rows=contacts(sa); rc=contacts(state(0,True)[3])
 ds=[]; dn=[]; db=[]; branchchg=[]; centers=[]
 for old,new in zip(rc,rows):
  if old[0]!=new[0]: print('order mismatch')
  ds.append(new[4]-old[4]); dn.append(np.linalg.norm(new[5]-old[5])); db.append(np.linalg.norm(new[3]-old[3])); branchchg.append(old[2]!=new[2])
 print('contacts alpha',a,'n',len(rows),'branchchange',sum(branchchg),'gapdelta max/median',max(map(abs,ds)),np.median(np.abs(ds)),'normaldelta max',max(dn),'barydelta max',max(db),'gapdelta nonzero',sum(x!=0 for x in ds))
# compare captured geometry record against independent closest point on center reconstructed starts and accepted start
for label,pos in [('accepted',accx),('recon_center',state(0,True)[3]),('finish_center',xcenter)]:
 errs=[]; nerrs=[]; berrs=[]; branchct={}
 for idx in active:
  if int(raw[idx,1,3])&8: continue
  ids=raw[idx,0].astype(int);pnt=pos[ids[0]].astype(float);tri=np.array([pos[j] for j in ids[1:4]],float)
  q,b,br=cp_tri(pnt,*tri); gap=np.linalg.norm(pnt-q); n=(pnt-q)/max(gap,1e-300)
  errs.append(gap-float(fl[idx,5,3]));nerrs.append(min(np.linalg.norm(n-fl[idx,4,:3]),np.linalg.norm(n+fl[idx,4,:3]))); berrs.append(np.linalg.norm(b+fl[idx,2,1:4]))
  branchct[br]=branchct.get(br,0)+1
 print('record compare',label,'n',len(errs),'gap err max/median',max(map(abs,errs)),np.median(np.abs(errs)),'normal err max',max(nerrs),'bary err max',max(berrs),'branches',branchct)
# F delta using exact double F from u words and captured inverseRest rows
trs=np.frombuffer(c.raw('tetrahedra_accepted'),dtype='<u4').reshape(-1,24); tf=trs.view('<f4'); totals=[]
va,ua,xa,sa=state(2**-22,True)
ideal_u=ucenter.astype(np.float64)+float(dt)*2**-22*corr.astype(np.float64)
def F(u,rec):
 ids=rec[:4].astype(int); rf=rec.view('<f4'); inv=np.stack([rf[4:7],rf[8:11],rf[12:15]],axis=0).astype(np.float64)
 du=np.stack([u[ids[1]]-u[ids[0]],u[ids[2]]-u[ids[0]],u[ids[3]]-u[ids[0]]],axis=1)
 return np.eye(3)+du@inv
Fcenter=np.array([F(ucenter.astype(np.float64),r) for r in trs])
Ftrial=np.array([F(ua.astype(np.float64),r) for r in trs])
Fideal=np.array([F(ideal_u,r) for r in trs])
dFa=np.linalg.norm(Ftrial-Fcenter,axis=(1,2));dFi=np.linalg.norm(Fideal-Fcenter,axis=(1,2))
print('materialF alpha',2**-22,'changed_tets',sum(dFa>0),'unchanged',sum(dFa==0),'ideal changed',sum(dFi>0),'actualMax',max(dFa),'idealMax',max(dFi),'actualMedianNonzero',np.median(dFa[dFa>0]),'ratioMax',max(np.divide(dFa,dFi,out=np.zeros_like(dFa),where=dFi>0)))
print('F values center/trial for changed top')
for i in np.argsort(dFa)[-5:][::-1]: print(i,'nodes',trs[i,:4].tolist(),'dActual',dFa[i],'dIdeal',dFi[i], 'Fcent',Fcenter[i].tolist(),'Ftrial',Ftrial[i].tolist())

# Compare finish-position self-contact geometry changes (versus start-position replay above)
for a in [0,2**-22,2**-23,2**-26]:
 va,ua,xa,sa=state(a,True); rows=contacts(xa); rc=contacts(xcenter); ds=[];dn=[];db=[];bc=[]
 for old,new in zip(rc,rows):
  ds.append(new[4]-old[4]);dn.append(np.linalg.norm(new[5]-old[5]));db.append(np.linalg.norm(new[3]-old[3]));bc.append(old[2]!=new[2])
 print('finishcontact alpha',a,'n',len(rows),'branchesChanged',sum(bc),'gapmax',max(map(abs,ds)),'gapnonzero',sum(x!=0 for x in ds),'normmax',max(dn),'barymax',max(db))
