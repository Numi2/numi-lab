#!/usr/bin/env python3
"""Reconstruct both mechanical cap/transition patches on actual registered skin.
No radial offset, muscle target projection, normals smoothing or physical admission.
"""
from pathlib import Path
import numpy as np,json,csv,hashlib,ast,runpy,time,sys,collections
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');A=E/'ankle-parent-tet-refinement-026';O=E/'bilateral-source-skin-cap-reconstruction-211';O.mkdir(exist_ok=False);S=A/'generated-223-bilateral-source-oriented-cap-actual007-frame/candidate-222.npz';G=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin.csv';H=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
ns={'__name__':'frozen','MAX_TET_VOL':1.5e-7};p=A/'build_common_volume_126.py';tr=ast.parse(p.read_text());exec(compile(ast.Module(body=[n for n in tr.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),str(p),'exec'),ns)
g=runpy.run_path(str(E/'right-commonfield-layer-reconstruction-202/tools/human_skin_patch_geometry.py'));decode=runpy.run_path(str(E/'registered-rest-normal-owner-199/tools/rebind_human_skin_reference.py'))['decode'];h=decode(H.read_bytes());GF=h['inds'].astype(np.int64);rows=list(csv.DictReader(G.open()));V=np.zeros((h['nv'],3))
for r in rows:V[int(r['vertex'])]=[float(r[k]) for k in ('wx','wy','wz')]
if len(rows)!=len(V) or not np.isfinite(V).all():raise ValueError('registered guide incomplete')
z=np.load(S);X=z['outer_world_actual007_m'].astype(np.float32).astype(float);F=z['surface_faces'].astype(np.int64);parent=np.arange(len(F));kind=z['surface_face_kind'];oldX=X.copy();oldF=F.copy();selected=np.isin(kind,[1,3]);patchnodes=np.unique(F[selected]);protected=set();protectededges=set()
for side in ['left','right']:
 d=np.load(A/f'generated-225-actual007-cut-stitch-domain-preflight/{side}-closed-domain.npz');cut=d['candidate223_vertex_global_ids'][d['stitch_O_local_ids']];protected.update(map(int,cut));protectededges.update(tuple(sorted((int(a),int(b)))) for a,b in zip(cut,np.roll(cut,-1)))
# Verify registered fine guide matches the retained source mesh's actual frame.
fine=np.flatnonzero(kind==2);fid=z['surface_face_detailed_source_ids'][fine];valid=fid>=0;cent=X[F[fine[valid]]].mean(1);tri=V[GF[fid[valid]]];b,ds=ns['closest_bary'](cent,tri[:,None]);err=np.sqrt(ds[:,0]);
if err.max()>2e-6:raise ValueError('fine guide source identity/frame mismatch '+str(err.max()))
fine_nodes=set(map(int,np.unique(F[kind==2]))); move=np.array(sorted(set(map(int,patchnodes))-protected-fine_nodes),np.int64);dd,ff,bb,qq=ns['closest_surface'](X[move],V,GF,k=64)
if dd.max()>0.02:raise ValueError('projection exceeded20mm diagnostic bound '+str(dd.max()))
X[move]=qq;oldguide=np.full(len(X),-1,np.int64);oldbary=np.zeros((len(X),3));oldguide[move]=ff;oldbary[move]=bb;history=[];nodeparents=[]
for rnd in range(6):
 active=np.isin(kind[parent],[1,3]);tri=F[active];ed=np.unique(np.sort(np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(X[ed[:,0]]-X[ed[:,1]],axis=1);cut=[tuple(map(int,e)) for e,l in zip(ed,length) if l>0.006 and tuple(map(int,e)) not in protectededges]
 if not cut:break
 prevn=len(X);Y,FF,PP,EE,WW=g['split_surface_edges'](X,F,{e:.5 for e in cut});d,f,b,q=ns['closest_surface'](Y[prevn:],V,GF,k=64)
 if d.max()>0.02:raise ValueError('new guide sample exceeded20mm')
 Y[prevn:]=q;parent=parent[PP];F=FF;X=Y;oldguide=np.r_[oldguide,f];oldbary=np.vstack([oldbary,b]);nodeparents.append({'start':prevn,'edges':EE.tolist(),'weights':WW.tolist()});history.append({'round':rnd,'split_edges':len(cut),'max_unprojected_guide_error_m':float(d.max()),'vertices':len(X),'faces':len(F)});print(json.dumps(history[-1]),flush=True)
X32=X.astype(np.float32);tri=X32[F].astype(float);area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);changed=np.isin(kind[parent],[1,3])|np.any(np.isin(F,move),axis=1)
# Preserve identity/classification of source facet rows, with independent guide ancestry per node.
np.savez_compressed(O/'candidate-surface-unadmitted.npz',outer_world_actual007_m=X32,surface_faces=F,surface_face_parent_223=parent,surface_face_kind=kind[parent],surface_face_source_side=z['surface_face_source_side'][parent],surface_face_parent_surface_ids=z['surface_face_parent_surface_ids'][parent],surface_face_detailed_source_ids=z['surface_face_detailed_source_ids'][parent],surface_face_original_d36_row=z['surface_face_original_d36_row'][parent],source_guide_face_ids=oldguide,source_guide_barycentric=oldbary,changed_face_ids=np.flatnonzero(changed),changed_old_vertex_ids=move,protected_cut_vertex_ids=np.array(sorted(protected)))
(O/'affine-split-construction-ancestry.json').write_text(json.dumps(nodeparents,separators=(',',':'))+'\n')
report={'physical_admission':False,'method':'bilateral conforming refinement of every authored cap and transition strip; sample exact registered detailed source skin, preserve both13-node cutrings; no skin-volume or mechanical-admission claim','source_sha256':sha(S),'guide_csv_sha256':sha(G),'guide_nhskin_sha256':sha(H),'script_sha256':sha(__file__),'source_fine_frame_parity_max_m':float(err.max()),'old_vertices':len(oldX),'vertices':len(X),'faces':len(F),'changed_faces':int(changed.sum()),'original_patch_vertices_refit':len(move),'original_vertex_max_move_m':float(np.linalg.norm(X[:len(oldX)]-oldX,axis=1).max()),'rounds':history,'nonpositive_area_faces':int((area<=0).sum()),'minimum_face_double_area_m2':float(area.min()),'cut_vertices_exact':bool(np.array_equal(X32[list(protected)],oldX.astype(np.float32)[list(protected)])),'wall_seconds':time.monotonic()-start,'remaining':['exact source self and all-nearby tissue crossings','retained motion and sourcerest embedding','independent inner/deep layers','conservative physical mass and transfer','native mechanics']};(O/'report-211.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
