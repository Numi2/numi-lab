import pathlib, importlib.util, csv, collections, json, numpy as np
D=pathlib.Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
spec=importlib.util.spec_from_file_location('b',D/'build_whole_reference_plc_083.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
R,X,gids,T0,M0,O,I,Df,inc0,detR0,detX0,qR0,qX0,J0=b.load_inputs()
# TetGen node output: exact coordinates identify the 7,233 input points; markers distinguish generated rows only secondarily.
L=(D/'whole-skin-fat-reference.1.node').read_text().splitlines(); n,dim,nattr,nmark=map(int,L[0].split()[:4]); P={}; marks={}
for s in L[1:]:
 q=s.split()
 if not q or not q[0][0].isdigit(): continue
 k=int(q[0]); P[k]=np.asarray(list(map(float,q[1:4])),float); marks[k]=int(q[4])
coord_to_src={tuple(map(float,p)):i for i,p in enumerate(R)}; src={k:coord_to_src.get(tuple(map(float,p))) for k,p in P.items()}; extra=[k for k in P if src[k] is None]
assert len(P)==n and sum(v is not None for v in src.values())==7233 and len(extra)<=128, (len(P), len(extra), sum(v is not None for v in src.values()))
# output cells and region attributes
L=(D/'whole-skin-fat-reference.1.ele').read_text().splitlines(); nt,nv,nca=map(int,L[0].split()[:3]); T=np.empty((nt,4),np.int64); M=np.empty(nt,np.int8)
for s in L[1:]:
 q=s.split()
 if not q or not q[0][0].isdigit():continue
 r=int(q[0])-1; T[r]=list(map(int,q[1:5])); M[r]=int(round(float(q[5])))
assert set(M.tolist())=={1,2}
used=set(map(int,T.ravel())); extra_used=[k for k in extra if k in used]; extra_unused=[k for k in extra if k not in used]
# input facet marker -> exact original parent triangle
Fmap={}
with (D/'facet-marker-map.csv').open(newline='') as f:
 for r in csv.DictReader(f): Fmap[int(r['marker'])]=np.array(list(map(int,r['vertex_ids_zero_based'].split())),int)
# TetGen surface-face records and parent marker lineage
L=(D/'whole-skin-fat-reference.1.face').read_text().splitlines(); nf,face_marked=map(int,L[0].split()[:2]); Fs=[]; Fm=[]
for s in L[1:]:
 q=s.split()
 if not q or not q[0][0].isdigit():continue
 Fs.append(list(map(int,q[1:4])));Fm.append(int(q[4]))
Fs=np.asarray(Fs,int); Fm=np.asarray(Fm,int); assert len(Fs)==nf
face_marker={tuple(sorted(map(int,f))):int(m) for f,m in zip(Fs,Fm)}
assert len(face_marker)==len(Fs)
marker_counts=collections.Counter(Fm.tolist())
# source parent area/planarity/barycentric coverage per exact input facet marker
children=collections.defaultdict(list);bad_children=[]; max_plane=0.;max_exc=0.; max_area_err=0.; split_markers={}
for f,m in zip(Fs,Fm):
 if int(m) not in Fmap: bad_children.append({'face':f.tolist(),'marker':int(m),'reason':'unknown parent'});continue
 children[int(m)].append(f)
area_ratios=[]
for m,par in Fmap.items():
 P0=R[par]; u=P0[1]-P0[0];v=P0[2]-P0[0]; uu=u@u;uv=u@v;vv=v@v;den=uu*vv-uv*uv
 old_area=np.linalg.norm(np.cross(u,v))/2; area=0.
 for f in children.get(m,[]):
  tri=np.asarray([P[int(k)] for k in f]); cr=np.cross(tri[1]-tri[0],tri[2]-tri[0]); area+=np.linalg.norm(cr)/2
  w=tri-P0[0]; a=(w@u*vv-w@v*uv)/den;bb=(w@v*uu-w@u*uv)/den; bary=np.stack([1-a-bb,a,bb],axis=1)
  residual=np.linalg.norm(P0[0]+a[:,None]*u+bb[:,None]*v-tri,axis=1)
  max_plane=max(max_plane,float(residual.max()));max_exc=max(max_exc,float(max(0.,-bary.min(),bary.max()-1)))
 ratio=float(area/old_area);area_ratios.append(ratio);max_area_err=max(max_area_err,abs(ratio-1.))
 if len(children.get(m,[]))>1: split_markers[str(m)]=len(children[m])
# source cell barycentric witnesses for each *used* generated node, by output incident material
E=np.transpose(R[T0[:,1:]]-R[T0[:,0],None,:],(0,2,1)); invE=np.linalg.inv(E); p0=R[T0[:,0]]
incident_materials=collections.defaultdict(set)
for tet,mat in zip(T,M):
 for k in tet: incident_materials[int(k)].add(int(mat))
Xbyout={k:X[src[k]].astype(float) for k in P if src[k] is not None}; ancestry={}; unmapped=[]; max_reconstruct=0.; max_cur_ref=0.; f32_mismatch=[]
for k in extra_used:
 p=P[k]; coeff=np.einsum('tij,tj->ti',invE,p-p0); bary=np.column_stack((1-coeff.sum(axis=1),coeff)); inside=(bary.min(axis=1)>=-2e-8)&(bary.max(axis=1)<=1+2e-8); candidates=np.flatnonzero(inside); bymat={}; cur_positions=[]
 for mat in sorted(incident_materials[k]):
  rows=[int(ti) for ti in candidates if M0[ti]==mat]
  if not rows: bymat[str(mat)]=None;continue
  ti=max(rows,key=lambda j:float(bary[j].min())); w=bary[ti]; rec=w@R[T0[ti]]; cur=w@X[T0[ti]]; cur_positions.append(cur)
  bymat[str(mat)]={'source_tet_row':int(ti),'barycentric':w.tolist(),'max_abs_reference_reconstruction_m':float(np.max(np.abs(rec-p)))}
  max_reconstruct=max(max_reconstruct,float(np.max(np.abs(rec-p))))
 if not cur_positions:unmapped.append(k)
 else:
  cur=cur_positions[0]
  if any(np.max(np.abs(v-cur))>1e-12 for v in cur_positions[1:]):unmapped.append(k)
  Xbyout[k]=cur;max_cur_ref=max(max_cur_ref,float(np.max(np.abs(cur-p))))
  if not np.array_equal(cur.astype(np.float32),np.asarray(p,dtype=np.float32)):f32_mismatch.append(k)
 ancestry[str(k)]={'marker':marks[k],'boundary_face_incidence':int(sum(k in f for f in Fs)),'output_materials':sorted(incident_materials[k]),'source_tet_by_material':bymat,'output_xyz_m':p.tolist(),'interpolated_current_xyz_m':Xbyout[k].tolist() if k in Xbyout else None}
# Geometry chain: use actual output tet incidence orientation, matched to face marker rows.
ids=sorted(P); dense={k:i for i,k in enumerate(ids)}; Td=np.vectorize(dense.get)(T)
Pr=np.asarray([P[k] for k in ids],float).astype(np.float32).astype(np.float64)
Pc=np.asarray([Xbyout[k] if k in Xbyout else np.full(3,np.nan) for k in ids],float).astype(np.float32).astype(np.float64)
# all cells use only original or ancestry-mapped nodes
used_dense=set(map(int,Td.ravel())); assert all(np.isfinite(Pc[i]).all() for i in used_dense)
inc=b.build_incidence(Td,M,Pr); oriented_children=collections.defaultdict(list); unknown_inc=[]
for key,entries in inc.items():
 if len(entries)==1: tri=entries[0][2]
 elif len(entries)==2 and entries[0][1]!=entries[1][1]: tri=next(e[2] for e in entries if e[1]==1)
 else:continue
 outids=tuple(ids[int(v)] for v in tri); keyout=tuple(sorted(outids)); mark=face_marker.get(keyout)
 if mark is None: unknown_inc.append({'face':list(outids),'key':list(keyout)});continue
 oriented_children[mark].append(outids)
signed_ratios=[]; signed_bad=0; min_signed=1.
for m,parent in Fmap.items():
 pp=R[parent]; pn=np.cross(pp[1]-pp[0],pp[2]-pp[0]); denom=float(pn@pn); val=0.
 for f in oriented_children.get(m,[]):
  q=np.asarray([P[k] for k in f]); cr=np.cross(q[1]-q[0],q[2]-q[0]); s=float(cr@pn/denom);val+=s;min_signed=min(min_signed,s)
  if s<=0:signed_bad+=1
 signed_ratios.append(val)
# F32 reference/current cell gates and per-material volume/J
Dr=b.dets(Pr,Td);Qr=b.qualities(Pr,Td);Dc=b.dets(Pc,Td);Qc=b.qualities(Pc,Td);J=Dc/Dr
by_mat={}
for mat in (1,2):
 z=M==mat;by_mat[str(mat)]={'cells':int(z.sum()),'ref_f32_nonpositive':int(np.count_nonzero(Dr[z]<=0)),'ref_f32_min_q':float(Qr[z].min()),'ref_f32_below_q1e-5':int(np.count_nonzero(Qr[z]<1e-5)),'current_f32_nonpositive':int(np.count_nonzero(Dc[z]<=0)),'current_f32_min_q':float(Qc[z].min()),'current_f32_below_q1e-5':int(np.count_nonzero(Qc[z]<1e-5)),'J_min':float(J[z].min()),'J_max':float(J[z].max()),'J_outside_0_35_2_5':int(np.count_nonzero((J[z]<.35)|(J[z]>2.5))),'reference_volume_m3':float(Dr[z].sum()/6),'current_volume_m3':float(Dc[z].sum()/6)}
oldvol={str(m):float(b.dets(R,T0[M0==m]).sum()/6) for m in (1,2)}
# report
rep={'status':'S128_BOUNDARY_SUBDIVISION_DIAGNOSTIC_ONLY_NOT_ADMITTED','input':{'points':len(R),'native_tets':len(T0),'native_material_counts':{str(m):int(np.count_nonzero(M0==m)) for m in (1,2)},'external_interface_facets':len(Fmap)},'tetgen_run':json.loads((D/'tetgen-run.json').read_text()),'output':{'node_records':n,'new_node_records':len(extra),'new_points_used_by_output_tets':len(extra_used),'new_point_rows_unused_by_output_tets':len(extra_unused),'used_new_boundary_nodes':int(sum(sum(k in f for f in Fs)>0 for k in extra_used)),'used_new_interior_nodes':int(sum(sum(k in f for f in Fs)==0 for k in extra_used)),'all_7233_original_point_coordinates_preserved_exactly':sum(v is not None for v in src.values())==7233,'output_cell_attribute_count':nca,'tet_count':nt,'material_counts':{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},'face_count':nf,'all_material_attrs_are_1_or_2':set(M.tolist())=={1,2}},'used_new_node_ancestry':ancestry,'unmapped_used_new_nodes':unmapped,'source_tet_witnesses':{'all_used_additions_mapped':not unmapped,'max_ref_reconstruction_error_m':max_reconstruct,'max_current_vs_output_ref_point_error_m':max_cur_ref,'used_node_F32_mismatch_count':len(f32_mismatch),'f32_mismatch_node_ids':f32_mismatch,'explanation':'Each used new point gets barycentric ancestry in its incident material source tetrahedra; generated current points are interpolated from frozen 035 source coordinates. This is construction lineage only.'},'surface_domain':{'all_input_facet_markers_have_output_faces':all(marker_counts.get(m,0)>0 for m in Fmap),'split_parent_markers':split_markers,'split_surface_counts':{str(s):sum(1 for k in split_markers if int(k)//100000==s) for s in (1,2,3,4)},'max_parent_area_ratio_error':max_area_err,'max_child_plane_distance_m':max_plane,'max_child_barycentric_excursion':max_exc,'oriented_child_faces_from_output_tet_incidence':len(oriented_children),'unknown_output_boundary_or_interface_faces':len(unknown_inc),'minimum_child_signed_area_ratio':min_signed,'negative_or_zero_oriented_child_count':signed_bad,'max_parent_signed_area_ratio_error':float(max(abs(x-1) for x in signed_ratios)),'oriented_parent_signed_area_ratio_min':float(min(signed_ratios)),'oriented_parent_signed_area_ratio_max':float(max(signed_ratios)),'output_nonmanifold_tet_faces_gt2':int(sum(len(v)>2 for v in inc.values())),'output_external_faces':int(sum(len(v)==1 for v in inc.values())),'output_material_interface_faces':int(sum(len(v)==2 and v[0][1]!=v[1][1] for v in inc.values())),'boundary_marker_child_area_matches_parent':max_area_err<1e-12,'oriented_parent_chain_matches':len(unknown_inc)==0 and signed_bad==0 and max(abs(x-1) for x in signed_ratios)<1e-12},'cell_gates':{'reference_float32_current_float32_by_material':by_mat,'all_cells_positive_both':bool(np.all(Dr>0) and np.all(Dc>0)),'all_q_ge_1e-5_both':bool(np.all(Qr>=1e-5) and np.all(Qc>=1e-5)),'all_J_0_35_2_5':bool(np.all((J>=.35)&(J<=2.5))),'duplicate_tet_sets':int(nt-len(np.unique(np.sort(Td,axis=1),axis=0))),'old_reference_volume_m3_by_material':oldvol,'new_reference_volume_rel_change':{str(m):float((by_mat[str(m)]['reference_volume_m3']-oldvol[str(m)])/oldvol[str(m)]) for m in (1,2)}},'interpretation':'TetGen stopped with code 200 under -YY, so this is a diagnostic postmortem, not an admitted mesh. Boundary evidence tests exact geometric subdivision on each marked parent triangle; it does not establish mass transfer, mechanics, collision clearance, or physical admission.'}
(D/'s128-ancestry-and-cell-audit.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in rep.items() if k!='used_new_node_ancestry'},indent=2,sort_keys=True))
