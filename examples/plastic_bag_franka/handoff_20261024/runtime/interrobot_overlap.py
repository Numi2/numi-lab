"""Saved articulated mesh surface crossings across the two robots; no physics."""
exec(__import__('pathlib').Path(__file__).with_name('robot_film_overlap.py').read_text().split('events=[];narrow_links=0')[0])
events=[];narrow=0
for f in fs:
 robots=[]
 for key in ['robot','robot2']:
  groups_world={}
  for idx,(va,triangles,bvh,corners) in (static2 if key=='robot2' else static).items():
   pose=f[key]['links'][idx];rot=np.asarray(Quaternion((pose[6],*pose[3:6])).to_matrix(),dtype=float);pos=np.array(pose[:3]);bounds=corners@rot.T+pos
   groups_world[idx]=(va@rot.T+pos,triangles,bounds.min(axis=0),bounds.max(axis=0),rot,pos)
  robots.append(groups_world)
 hits=[]
 for i,(av,at,alo,ahi,ar,ap) in robots[0].items():
  for j,(bv,bt,blo,bhi,br,bp) in robots[1].items():
   if np.any(ahi<blo-1e-7) or np.any(bhi<alo-1e-7):continue
   narrow+=1;other=(bv-ap)@ar;query=BVHTree.FromPolygons([Vector(v) for v in other],bt,all_triangles=True,epsilon=1e-7)
   own=static[i]
   for a,b in own[2].overlap(query):
    t1=own[0][list(at[a])];t2=other[list(bt[b])]
    if crossing(t1,t2) or crossing(t2,t1):hits.append((i,j));break
 if hits:events.append({'step':f['step'],'time_s':f['time_s'],'link_pairs':hits})
print('INTERROBOT_AUDIT '+json.dumps({'saved_states':len(fs),'narrow_link_pairs':narrow,'intersection_state_count':len(events),'events':events[:50],'scope':'Native saved articulated meshes across arms, excluding compliant rubber patches; no whole-robot CCD or motor dynamics.'}))
