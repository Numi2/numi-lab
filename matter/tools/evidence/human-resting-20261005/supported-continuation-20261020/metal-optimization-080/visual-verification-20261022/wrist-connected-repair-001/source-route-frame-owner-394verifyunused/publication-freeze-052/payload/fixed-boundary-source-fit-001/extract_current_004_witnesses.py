import pathlib,importlib.util,json,sys,time
import numpy as np
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused';F=D/'fixed-boundary-source-fit-001';RUN=F/'run-004';BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue');CAND=RUN/'candidate-8874-local-position-fit.nhtissue';IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json';FIT=F/'build_source_fit_004.py';GATE=RUN/'exact-local-source-fit-131-allposes/exact-local-gate-report.json';CANON=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py';EXACT=E/'exact-plane-filter-owner-264/src'
def sha(p):import hashlib;return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bary(p,T):
 M=np.vstack([T.T,np.ones(3)]);q=np.r_[p,1.0];w=np.linalg.lstsq(M,q,rcond=None)[0];return w,float(np.linalg.norm(w@T-p))
def main():
 t=time.time();sys.path.insert(0,str(EXACT));from numilab_human import cardiac_cavity_intersections as ci
 sp=importlib.util.spec_from_file_location('f',FIT);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);sp=importlib.util.spec_from_file_location('c',CANON);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
 dec=c.decode_nhtiss(BASE.read_bytes());base_dec=dec;cand_dec=c.decode_nhtiss(CAND.read_bytes());ns=f.load_emitter();ident=json.loads(IDENT.read_text());pose_specs={'current191':f.pose_map(ident,'current'),'rest003':f.pose_map(ident,'rest')}
 for fn in f.FK:pose_specs[fn[:-5]]=f.pose_map(json.loads((f.POSE_DIR/fn).read_text()),'fk')
 base={};cand={}
 for pose_name,poses in pose_specs.items():ns['poses']=poses;base[pose_name]=ns['emit_selected'](BASE);ns['poses']=poses;cand[pose_name]=ns['emit_selected'](CAND)
 report=json.loads(GATE.read_text());pair_checks=[x for x in report['exact_checks'] if x['target_stable_id']==133 and x['new_pairs']];r131=next(r for r in dec['records'] if int(r[6])==131);fv,vc,fi,ic=map(int,(r131[2],r131[3],r131[4],r131[5]));ix=dec['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);Afaces=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix;r133=next(r for r in dec['records'] if int(r[6])==133);fv2,vc2,fi2,ic2=map(int,(r133[2],r133[3],r133[4],r133[5]));ix2=dec['indices'][fi2:fi2+ic2].astype(np.int64).reshape(-1,3);Bfaces=ix2-fv2 if ix2.min()>=fv2 and ix2.max()<fv2+vc2 else ix2
 out=[]
 for check in pair_checks:
  pose_name=check['pose'];pose_out=[]
  for af,bf in check['new_pairs']:
   ai=Afaces[int(af)];bi=Bfaces[int(bf)];ta=cand[pose_name][131][0][ai].astype(np.float32);tb=cand[pose_name][133][0][bi].astype(np.float32);
   def exact_lattice(v):return [tuple(__import__('fractions').Fraction(int(k),1<<149) for k in ci.float32_point_lattice_key(row)) for row in np.asarray(v,np.float32)]
   points=ci.triangle_intersection_points(exact_lattice(ta),exact_lattice(tb));items=[]
   for p in points:
    pf=np.asarray([float(x) for x in p],np.float64);wa,ea=bary(pf,ta.astype(np.float64));wb,eb=bary(pf,tb.astype(np.float64));ba=base[pose_name][131][0][ai].astype(np.float64);bb=base[pose_name][133][0][bi].astype(np.float64);pba=wa@ba;pbb=wb@bb;delta=pba-pbb;gap=float(np.linalg.norm(delta)*1000)
    items.append({'point_m':[float(x) for x in pf],'a_bary':wa.tolist(),'b_bary':wb.tolist(),'candidate_bary_error_m':max(ea,eb),'baseline_support_delta_m':delta.tolist(),'baseline_support_distance_mm':gap})
   pose_out.append({'a_face':int(af),'b_face':int(bf),'a_vertices':ai.tolist(),'b_vertices':bi.tolist(),'hit_count':len(points),'hits':items})
  out.append({'pose':pose_name,'pairs':pose_out})
 path=RUN/'current191-new-pair-witnesses.json';doc={'schema':'numi.human.source-fit.exact-current-witnesses.v1','baseline_sha256':sha(BASE),'candidate_sha256':sha(CAND),'gate_sha256':sha(GATE),'poses':list(pose_specs),'target_stable_id':133,'pose_checks':out,'elapsed_s':time.time()-t};path.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({'path':str(path),'sha256':sha(path),'pose_pair_counts':{x['pose']:len(x['pairs']) for x in out},'hit_points':sum(p['hit_count'] for x in out for p in x['pairs']),'elapsed_s':doc['elapsed_s'],'first':out[:1]},indent=2))
if __name__=='__main__':main()
