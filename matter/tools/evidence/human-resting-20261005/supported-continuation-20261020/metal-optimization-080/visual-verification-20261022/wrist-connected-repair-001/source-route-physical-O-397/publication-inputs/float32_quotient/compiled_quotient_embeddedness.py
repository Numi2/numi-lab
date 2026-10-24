"""Exact spatial-quotient embeddedness census of compiled Human source meshes.

Duplicate authored vertex indices at identical compiled Float32 coordinates
are welded topologically without moving any points or changing face support.
This is distinct from the original indexed-mesh census: both outcomes are
reported, and neither is clinical anatomy or physical volume qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from . import model as human
from . import lung_envelope as lung
from . import surface_topology_repair as repair
from . import whole_body_embeddedness as indexed

SCHEMA='numi.human.compiled-quotient-embeddedness.v1'
ROW_SCHEMA='numi.human.compiled-quotient-surface-embeddedness.v1'
CODE_FILES=('compiled_quotient_embeddedness.py','whole_body_embeddedness.py',
            'surface_topology_audit.py','cardiac_cavity_intersections.py',
            'cardiac_cavity_geometry.py')


def require(ok,message):
    if not ok:raise human.ImportError('compiled quotient embeddedness: '+message)


def coordinate_quotient(vertices,faces):
    """Collapse only exactly equal executed positions; retain all triangles."""
    unique={};points=[];source_to_quotient=[]
    for p in vertices:
        key=tuple(p)
        if key not in unique:
            unique[key]=len(points);points.append(list(p))
        source_to_quotient.append(unique[key])
    qfaces=[[source_to_quotient[i] for i in f] for f in faces]
    used=sorted({i for f in qfaces for i in f})
    compact={old:new for new,old in enumerate(used)}
    compact_points=[points[i] for i in used]
    compact_faces=[[compact[i] for i in f] for f in qfaces]
    require(all([tuple(points[source_to_quotient[i]]) for i in f]==[tuple(vertices[i]) for i in f]
                for f in faces), 'quotient changed source triangle support')
    mapping=[compact.get(i,-1) for i in source_to_quotient]
    return compact_points,compact_faces,mapping


def classify_quotient(vertices,faces):
    qv,qf,mapping=coordinate_quotient(vertices,faces)
    checked=indexed.classify_surface(qv,qf)
    require(checked['topology']['vertex_count']==len(qv)
            and checked['topology']['face_count']==len(faces), 'quotient cardinality')
    return {'compiled_coordinate_quotient_vertex_count':len(qv),
            'compiled_exact_duplicate_vertex_count':len(vertices)-len(set(tuple(p) for p in vertices)),
            'unused_quotient_vertex_count':sum(i<0 for i in mapping),
            'source_vertex_to_compiled_quotient_sha256':hashlib.sha256(json.dumps(mapping,separators=(',',':')).encode()).hexdigest(),
            'source_face_support_unchanged':True,'new_points_added':False,
            'face_count_unchanged':True,**checked}


def scan(payload:Path,source_repair_manifest:Path,output:Path)->dict:
    import numpy as np
    payload=payload.resolve();source_repair_manifest=source_repair_manifest.resolve();output=output.resolve()
    require(human.sha256(payload)==repair.BASE_SHA,'published 579-surface payload identity')
    specs,provenance=repair.source_specs(payload)
    declared=human.read_json(source_repair_manifest)
    require(declared['base_payload_sha256']==repair.BASE_SHA
            and len(declared['source_outcomes'])==579, 'authored exact-quotient diagnostic identity')
    _,header,records,vertices,indices=lung.decode(payload)
    require(header[:2]==(5,579),'complete source payload')
    code={name:human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
    identity={'schema':SCHEMA,'payload_sha256':repair.BASE_SHA,
              'source_repair_manifest_sha256':human.sha256(source_repair_manifest),
              'source_manifest_hashes':provenance,'predicate_source_sha256':code,
              'python':__import__('sys').version.split()[0],'numpy':np.__version__,
              'surface_count':579,
              'boundary':'Compiled Float32 exact-coordinate quotient preserves every source triangle spatially. It does not prove component containment, cross-surface disjointness, clinical anatomy, tissue volume or mechanics.'}
    output.mkdir(parents=True,exist_ok=True)
    identity_path=output/'identity.json'
    if identity_path.exists():require(human.read_json(identity_path)==identity,'resume identity changed')
    else:indexed.atomic_json(identity_path,identity)
    rows_dir=output/'rows';rows_dir.mkdir(exist_ok=True)
    t0=time.monotonic();new=0
    for spec,record,source in zip(specs,records,declared['source_outcomes'],strict=True):
        owner,fv,nv,fi,ni,stable,layer,reserved=map(int,record)
        require(stable==spec['stable_id']==source['stable_id']
                and owner==spec['core_body_index']==source['core_body_index']
                and layer==spec['layer_code']==source['layer_code'] and reserved==0 and ni%3==0,
                'source semantic/record identity')
        require(0<=fv<fv+nv<=len(vertices) and 0<=fi<fi+ni<=len(indices),'compiled geometry ranges')
        v=vertices[fv:fv+nv,:3]
        global_faces=indices[fi:fi+ni]
        require(bool(np.isfinite(v).all()) and bool(((global_faces>=fv)&(global_faces<fv+nv)).all()),
                'nonfinite or escaped source geometry')
        mesh_hash=indexed.geometry_hash(vertices[fv:fv+nv],global_faces)
        row_path=rows_dir/f'{stable:03}.json'
        if row_path.exists():
            row=human.read_json(row_path)
            require(row['schema']==ROW_SCHEMA and row['stable_id']==stable
                    and row['geometry_sha256']==mesh_hash and row['predicate_source_sha256']==code,
                    'resume quotient provenance changed')
            continue
        f=(global_faces-fv).reshape(-1,3).tolist()
        t=time.monotonic();checked=classify_quotient(v.astype(float).tolist(),f)
        row={'schema':ROW_SCHEMA,'stable_id':stable,'provider':spec['provider'],
             'source_member_id':spec.get('member_id'),'source_object_name':spec.get('object_name'),
             'source_body_index':owner,'source_layer_code':layer,'vertex_count':nv,
             'triangle_count':ni//3,'geometry_sha256':mesh_hash,
             'predicate_source_sha256':code,
             'authored_source_exact_quotient_closed':source['source_topology_before']['closed_oriented_manifold_candidate'],
             'wall_seconds':time.monotonic()-t,**checked,
             'physical_volume':False,'clinical_anatomy':False,'mechanics':False}
        indexed.atomic_json(row_path,row);new+=1
        print(json.dumps({'stable_id':stable,'faces':ni//3,
                          'exact_duplicate_vertices':row['compiled_exact_duplicate_vertex_count'],
                          'status':row['status'],'intersection_pairs':row['exact_intersection_pairs'],
                          'seconds':row['wall_seconds']}),flush=True)
    paths=sorted(rows_dir.glob('*.json'))
    require(len(paths)==579 and [p.stem for p in paths]==[f'{i:03}' for i in range(1,580)],
            'incomplete or unexpected quotient rows')
    require({name:human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}==code,
            'predicate source changed during execution')
    rows=[human.read_json(p) for p in paths]
    counts=dict(sorted(Counter(row['status'] for row in rows).items()))
    result={'schema':SCHEMA,'identity_sha256':human.sha256(identity_path),
            'payload_sha256':repair.BASE_SHA,'surface_count':579,
            'triangles_audited':sum(r['triangle_count'] for r in rows),
            'exact_predicate_rows':sum(r['self_intersection']=='exact_checked' for r in rows),
            'status_counts':counts,'closed_embedded_surface_candidate_count':counts.get('closed_embedded_source_candidate',0),
            'total_exact_intersection_pairs':sum(r['exact_intersection_pairs'] or 0 for r in rows),
            'unqualified_source_ids':[r['stable_id'] for r in rows if not r['closed_embedded_surface_candidate']],
            'authored_exact_quotient_closed_count':sum(r['authored_source_exact_quotient_closed'] for r in rows),
            'compiled_quotient_differs_from_authored_closed_ids':[r['stable_id'] for r in rows
                if r['topology']['closed_oriented_manifold_candidate']!=r['authored_source_exact_quotient_closed']],
            'row_files':{p.name:human.sha256(p) for p in paths},
            'new_rows_this_execution':new,'wall_seconds_this_execution':time.monotonic()-t0,
            'cross_surface_disjointness':False,'component_containment':False,
            'physical_volume':False,'clinical_anatomy':False,'mechanics':False,
            'boundary':'Exact compiled-coordinate quotient and per-surface self-intersection only. It does not establish organ volume, anatomical placement, cross-surface disjointness, physical ownership or mechanics.'}
    indexed.atomic_json(output/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload',type=Path,required=True)
    p.add_argument('--source-repair-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=scan(a.payload,a.source_repair_manifest,a.output)
    print(json.dumps({'status_counts':result['status_counts'],
                      'triangles_audited':result['triangles_audited'],
                      'new_rows_this_execution':result['new_rows_this_execution']}),flush=True)


if __name__=='__main__':main()
