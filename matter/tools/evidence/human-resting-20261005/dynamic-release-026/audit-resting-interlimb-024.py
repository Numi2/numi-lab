from pathlib import Path
import importlib.util,json,sys,time,hashlib
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005');R=E/sys.argv[1];steps=[int(x) for x in sys.argv[2:]]
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools');import accepted_mrvpack_surface_audit as a
D=E/'heart-swept-source-inverse-native017-001/actual-rib-pair-audit/optimized_driver-002.py';spec=importlib.util.spec_from_file_location('exact_driver',D);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);pred=a.predicate_module(d.PRED)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=R/'resting-human.mrvpack';raw,st,off,s=a.read_pack(ref);f=np.asarray(s[(51007,1)]['faces']);lo=int(f.min());hi=int(f.max());v=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+lo*80)[:,:3].copy();fref=f-lo;valid=v[fref];right=np.flatnonzero((valid[:,:,1]<-.92).all(1)&(valid[:,:,0]<-.025).all(1));left=np.flatnonzero((valid[:,:,1]<-.92).all(1)&(valid[:,:,0]>-.025).all(1));raw.close();st.close()
for step in steps:
 pack=R/'accepted-geometry'/f'step-{step}.mrvpack';receipt=pack.with_suffix('.receipt.json');raw,st,off,s=a.read_pack(pack);accepted=a.validate_accepted_receipt(pack,receipt,step,raw,off,s);gf=s[(51007,1)]['faces'];f=np.asarray(gf);assert np.array_equal(f-f.min(),fref)
 sets=[[gf[i] for i in ids] for ids in [right,left]];pts=[a.surface_points(raw,off,x) for x in sets];scale=max(x[2] for x in pts);records=[a.make_records(p,scale,f,pred) for p,f in zip(pts,sets)]
 t=time.monotonic();nc,hits=d.exact_all(records[0],records[1],pred)
 result={'scope':'Exact accepted native left/right lower-limb skin subset audit, excluding adjacent groin. Not whole-skin or organ clearance.','accepted_receipt':accepted,'script_sha256':sha(__file__),'reference_pack_sha256':sha(ref),'predicate_sha256':sha(d.PRED),'pair_driver_sha256':sha(D),'partition':{'initial_supine_world_y_max_m':-.92,'initial_midline_x_m':-.025,'right_face_count':len(right),'left_face_count':len(left)},'step':step,'aabb_candidates':nc,'exact_intersecting_triangle_pairs':len(hits),'complete':True,'elapsed_s':time.monotonic()-t,'coordinate_scale_per_m':scale,'first_witnesses':hits[:16]}
 (R/f'interlimb-step-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'step':step,'hits':len(hits),'candidates':nc,'seconds':result['elapsed_s']}),flush=True);raw.close();st.close()
