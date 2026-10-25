import sys, pathlib, importlib.util, json, numpy as np
D=pathlib.Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-083')
spec=importlib.util.spec_from_file_location('b',D/'build_whole_reference_plc_083.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
R,X,gids,Told,Mold,O,I,Df,inc,detR,detX,qR,qX,J=b.load_inputs()
# Parse output nodes. TetGen IDs are 1-based; input marker is exact row+1, new points have marker 0.
nodefile=D/'whole-skin-fat-reference.1.node'; lines=nodefile.read_text().splitlines(); n,dim,attrs,markers=map(int,lines[0].split()[:4]); nodes={}; outmark={}
for line in lines[1:]:
 q=line.split()
 if not q or q[0].startswith('#'): continue
 k=int(q[0]); nodes[k]=np.array(list(map(float,q[1:4])),dtype=np.float64); outmark[k]=int(q[4])
if len(nodes)!=n: raise RuntimeError(('nodecount',len(nodes),n))
# provisional markers are not authoritative; exact coordinates establish retained input identity

coord_to_in={tuple(map(float,p)):i for i,p in enumerate(R)}
coord_match={k:coord_to_in.get(tuple(map(float,p))) for k,p in nodes.items()}
orig=[k for k,v in coord_match.items() if v is not None]
new=[k for k,v in coord_match.items() if v is None]
assert len(orig)==7233
# Parse output cells/attrs
el=D/'whole-skin-fat-reference.1.ele'; elines=el.read_text().splitlines(); nt,nv,na=map(int,elines[0].split()[:3]); outT=np.empty((nt,4),np.int64); outMat=np.empty(nt,np.int64)
for line in elines[1:]:
 q=line.split()
 if not q or q[0].startswith('#'):continue
 i=int(q[0])-1; outT[i]=[int(v) for v in q[1:5]]; outMat[i]=int(round(float(q[5])))
# parse original input PLC facets to exact parent marker and vertex ids
poly=D/'whole-skin-fat-reference.poly'; plines=poly.read_text().splitlines(); j=0
def nxt():
 global j
 while j<len(plines) and (not plines[j].strip() or plines[j].lstrip().startswith('#')): j+=1
 s=plines[j].strip();j+=1;return s
nn,dim,na,nmark=map(int,nxt().split()); assert nn==len(R)
for _ in range(nn): nxt()
nf,facmark=map(int,nxt().split()); facets={}; marker_desc={}
for _ in range(nf):
 hdr=list(map(int,nxt().split())); nPoly,nHole,mark=hdr
 assert nPoly==1 and nHole==0
 trirow=list(map(int,nxt().split())); assert trirow[0]==3
 tri=np.array(trirow[1:])-1
 facets[mark]=tri
# parse output faces and markers
ff=D/'whole-skin-fat-reference.1.face'; flines=ff.read_text().splitlines(); nfout,nfm=map(int,flines[0].split()[:2]); outfaces=[]; outfm=[]
for line in flines[1:]:
 q=line.split()
 if not q or q[0].startswith('#'): continue
 outfaces.append([int(x) for x in q[1:4]]);outfm.append(int(q[4]))
outfaces=np.asarray(outfaces,np.int64);outfm=np.asarray(outfm,np.int64)
# point->parent facet closest plane/barycentric, exhaustive 43 x 14511 (broadcast)
triids=np.array(list(facets.values()),np.int64); triXYZ=R[triids]
new_rows=[]
for oid in new:
 p=nodes[oid]
 A=triXYZ[:,0]; u=triXYZ[:,1]-A; v=triXYZ[:,2]-A; w=p-A
 uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v);wu=np.einsum('ij,ij->i',w,u);wv=np.einsum('ij,ij->i',w,v)
 den=uu*vv-uv*uv; a=(wu*vv-wv*uv)/den; bb=(wv*uu-wu*uv)/den; bary=np.stack([1-a-bb,a,bb],axis=1)
 proj=A+a[:,None]*u+bb[:,None]*v; dist=np.linalg.norm(proj-p,axis=1)
 good=(bary.min(axis=1)>=-1e-10)&(bary.max(axis=1)<=1+1e-10)
 poss=np.where(good)[0]; idx=poss[np.argmin(dist[poss])] if len(poss) else int(np.argmin(dist))
 # also identify nearest marker, parent surface
 marker=list(facets.keys())[idx]
 new_rows.append({'output_node_id':oid,'marker':outmark[oid],'xyz_m':p.tolist(),'best_marker':int(marker),'best_barycentric':bary[idx].tolist(),'plane_distance_m':float(dist[idx]),'inside_parent_barycentric':bool(good[idx]),'closest_any_distance_m':float(dist.min()),'best_inside_distance_m':float(dist[poss].min()) if len(poss) else None})
# Face marker and input parent subdivision summary
face_marker_counts={str(int(m)):int(np.count_nonzero(outfm==m)) for m in np.unique(outfm)}
# map all output faces by marker, calculate area and compare original parent triangle area; also test coordinates lie in parent triangle
face_area=[]; facet_child_count={}; facet_area={}; bad_child=[]
nodearr=np.zeros((max(nodes)+1,3),np.float64)
for k,p in nodes.items():nodearr[k]=p
for f,mark in zip(outfaces,outfm):
 tri=nodearr[f]; area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2);face_area.append(area)
 facet_child_count[int(mark)]=facet_child_count.get(int(mark),0)+1;facet_area[int(mark)]=facet_area.get(int(mark),0.0)+area
 if int(mark) not in facets: bad_child.append({'face':f.tolist(),'marker':int(mark),'reason':'unknown_marker'});continue
 origtri=R[facets[int(mark)]]; u=origtri[1]-origtri[0];v=origtri[2]-origtri[0]; uu=float(u@u);uv=float(u@v);vv=float(v@v);den=uu*vv-uv*uv
 # each child vertex barycentric into original facet
 for vid,p in zip(f,tri):
  w=p-origtri[0]; wu=float(w@u);wv=float(w@v);aa=(wu*vv-wv*uv)/den;bb=(wv*uu-wu*uv)/den;bc=np.array([1-aa-bb,aa,bb]); plane=float(np.linalg.norm(origtri[0]+aa*u+bb*v-p))
  if bc.min() < -1e-8 or bc.max()>1+1e-8 or plane>2e-10:
   bad_child.append({'face':f.tolist(),'marker':int(mark),'vertex':int(vid),'bary':bc.tolist(),'plane_distance_m':plane})
area_ratios=[]; missing=[]
for mark,tri in facets.items():
 ar=float(np.linalg.norm(np.cross(R[tri[1]]-R[tri[0]],R[tri[2]]-R[tri[0]]))/2)
 ca=facet_area.get(mark,0.0)
 area_ratios.append(ca/ar if ar else float('nan'))
 if mark not in facet_child_count:missing.append(int(mark))
# output cells evaluated at F32 emission, reference/current. Current = baseline X exactly and TetGen output doubles lie? Inputs retained, new points may added, treat current same as reference as no displacement; diagnostics only.
P=np.array([nodes[k] for k in sorted(nodes)],np.float64).astype(np.float32).astype(np.float64); idmap={k:i for i,k in enumerate(sorted(nodes))}; tout=np.vectorize(idmap.get)(outT)
d=b.dets(P,tout);q=b.qualities(P,tout); bymat={str(m):{'cells':int((outMat==m).sum()),'nonpositive_det':int(np.count_nonzero(d[outMat==m]<=0)),'min_det':float(d[outMat==m].min()),'q_below_1e-5':int(np.count_nonzero(q[outMat==m]<1e-5)),'min_q':float(q[outMat==m].min()),'volume_m3':float(d[outMat==m].sum()/6)} for m in sorted(set(outMat.tolist()))}
report={'input_counts':{'nodes':len(R),'facets':nf,'tets':len(Told),'external_interface_faces':len(facets)},'generated_counts':{'nodes':n,'new_steiner_nodes':len(new),'preserved_original_coordinate_nodes':len(orig),'marker_counts':{str(int(m)):int(sum(v==m for v in outmark.values())) for m in set(outmark.values())},'new_node_marker_counts':{str(int(outmark[k])):int(sum(outmark[k2]==outmark[k] for k2 in new)) for k in new},'tets':nt,'attrs':na,'materials':{str(m):int(np.count_nonzero(outMat==m)) for m in sorted(set(outMat.tolist()))},'faces':nfout,'face_marker_counts':face_marker_counts},'new_point_facet_witnesses':new_rows,'new_point_on_input_facets':{'within_bary_tolerance_1e-10_and_plane_2e-10m':int(sum(r['inside_parent_barycentric'] and r['plane_distance_m']<=2e-10 for r in new_rows)),'max_best_inside_plane_distance_m':max((r['best_inside_distance_m'] for r in new_rows if r['best_inside_distance_m'] is not None),default=None)},'facet_subdivision_coverage':{'input_facets_with_children':len(facet_child_count),'missing_input_facets':len(missing),'child_faces_outside_parent_or_nonplanar':len(bad_child),'max_parent_area_ratio_error':float(max(abs(x-1) for x in area_ratios)),'min_parent_area_ratio':float(min(area_ratios)),'max_parent_area_ratio':float(max(area_ratios)),'bad_examples':bad_child[:12]},'float32_cell_diagnostic':{'by_material':bymat,'duplicate_tet_sets':int(nt-len(np.unique(np.sort(tout,axis=1),axis=0))),'all_attributes_in_1_2':bool(set(outMat.tolist())<= {1,2})}}
(D/'steiner-postmortem.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
