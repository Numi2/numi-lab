from pathlib import Path
import csv,json,hashlib,itertools
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=R/'elbow-refinement-current-main-015-layer-local-anchor-repair-001'; O=R/'d15-canonical-parent-boundary-check-001'
def rows(n):return list(csv.DictReader((D/'generated'/n).open()))
faces=rows('faces.csv'); parents=rows('parent-tetrahedra.csv'); nodes=rows('nodes.csv'); n0=2332
xyz=np.asarray([[[float(r[f'{l}{c}']) for c in 'xyz'] for r in nodes[:n0]] for l in ['o','i','d']])
lookup={tuple(sorted(int(r[k]) for k in 'abc')):i for i,r in enumerate(faces)}
ptets=np.asarray([[int(r[k]) for k in 'abcd'] for r in parents]); mats=np.asarray([int(r['material']) for r in parents])
facets=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
records=[]
for material,layer,label in [(1,0,'skin_outer'),(1,1,'skin_inner'),(2,1,'fat_inner'),(2,2,'fat_deep')]:
    c={'material':material,'layer':layer,'name':label,'matched':0,'unmatched_refined_parent_facets':0,'dot_positive':0,'dot_negative':0,'dot_zero':0,'min_cos':None,'max_cos':None,'max_abs_area_rel_error':0.,'examples':[]}
    for ti,tet in enumerate(ptets):
        if mats[ti]!=material: continue
        for fids in facets:
            fg=[int(tet[j]) for j in fids]
            lays={v//n0 for v in fg}
            if lays!={layer}:continue
            local=[v%n0 for v in fg]; key=tuple(sorted(local)); row=lookup.get(key)
            if row is None:
                c['unmatched_refined_parent_facets']+=1;continue
            sid=[int(faces[row][k]) for k in 'abc']
            p=xyz[layer]
            va=p[local]; vb=p[sid]
            npv=np.cross(va[1]-va[0],va[2]-va[0]); nsv=np.cross(vb[1]-vb[0],vb[2]-vb[0])
            den=np.linalg.norm(npv)*np.linalg.norm(nsv)
            cos=float(np.dot(npv,nsv)/den) if den>0 else 0.
            c['matched']+=1
            if cos>1e-12:c['dot_positive']+=1
            elif cos< -1e-12:c['dot_negative']+=1
            else:c['dot_zero']+=1
            c['min_cos']=cos if c['min_cos'] is None else min(c['min_cos'],cos)
            c['max_cos']=cos if c['max_cos'] is None else max(c['max_cos'],cos)
            a0=np.linalg.norm(npv)*0.5; a1=np.linalg.norm(nsv)*0.5
            c['max_abs_area_rel_error']=max(c['max_abs_area_rel_error'],abs(a1-a0)/a0)
            if len(c['examples'])<3:c['examples'].append({'parent_tet':ti,'candidate_face_row':row,'stable_face':int(faces[row]['stable_source_face']),'oriented_parent_nodes':local,'registered_face_nodes':sid,'normal_cosine':cos,'parent_tet_signed_volume_convention':'positive'})
    records.append(c)
report={'schema':'numi.canonical_parent_boundary_orientation.v1','candidate_manifest_sha256':hashlib.sha256((D/'package-manifest-015.json').read_bytes()).hexdigest(),'method':'Derive outward-oriented boundary facets from unchanged positive parent tetrahedra using the owner facet order {123,032,013,021}; compare against candidate registered face orientation on exactly matching unrefined parent triangles. Refined parent facets that do not have a same-triangle candidate face are counted separately.','parent_tetrahedra_sha256':hashlib.sha256((D/'generated/parent-tetrahedra.csv').read_bytes()).hexdigest(),'registered_surface_faces_sha256':hashlib.sha256((D/'generated/faces.csv').read_bytes()).hexdigest(),'results':records}
out=O/'canonical-parent-boundary-orientation.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
