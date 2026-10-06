"""Locate actual source-bound colon wall interfaces, without a whole-pair waiver."""
from pathlib import Path
import hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'taenia-motion-interface-classification-108';out.mkdir(exist_ok=False)
audit_path=E/'taenia-motion-margin-audit-103/report.json';report=json.loads(audit_path.read_text())
candidate=E/'taenia-motion-margin-audit-103/surface-457.npz';d=np.load(candidate);v=d['vertices'];f=d['faces']
base=E/'passive-neighbor-depth-062/source-surfaces.npz';data=np.load(base)
receipt_path=E/'passive-organ-native-input-074/resting-anatomy-receipt.json';receipt=json.loads(receipt_path.read_text());source_map=receipt['provenance']['source_id_map']
expected={454:'FJ2566',455:'FJ2567',456:'FJ2568',457:'FJ2569',458:'FJ2570',459:'FJ2571',460:'FJ2572'}
assert all(source_map[str(k)]['source_member']==member for k,member in expected.items())
assert report['self']['count']==0 and len(report['pairs'])==78
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools');import accepted_mrvpack_surface_audit as audit
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py');pred=audit.predicate_module(predicate)
scale=audit.coordinate_lattice_scale(float(c) for sid in expected for p in (v if sid==457 else data[f'v{sid}']) for c in p)
points={457:[tuple(audit.lattice_integer(float(c),scale) for c in p) for p in v]}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for row in report['pairs']:
    sid=row['ids'][1]
    if not row.get('count'):continue
    assert sid in expected
    points[sid]=[tuple(audit.lattice_integer(float(c),scale) for c in p) for p in data[f'v{sid}']]
    g=data[f'f{sid}'];witnesses=[]
    for a,b in row['triangle_pairs']:
        p=pred.triangle_intersection_points(tuple(points[457][i] for i in f[a]),tuple(points[sid][i] for i in g[b]))
        assert p
        witnesses.extend([[float(c)/scale for c in point] for point in p])
    xyz=np.asarray(witnesses);lo=xyz.min(0);hi=xyz.max(0)
    detail={'stable_ids':[457,sid],'source_members':[expected[457],expected[sid]],'exact_crossing_pair_count':row['count'],
            'exact_witness_count':len(xyz),'witness_bounds_m':[lo.tolist(),hi.tolist()],
            'source_triangle_pairs':row['triangle_pairs']}
    if sid in (454,455,460):
        detail.update(interface='longitudinal_muscle_component_of_colon_wall',
                      interpretation='The source colon organ envelope includes its muscular wall; the named taenia is a constituent inspection layer, not a second independent organ volume.')
    elif sid in (456,458):
        lower=float(data['v454'][:,1].min());span=[float(lo[1]-lower),float(hi[1]-lower)]
        assert 0<=span[0]<=span[1]<=.020
        detail.update(interface='taenia_convergence_at_caudal_ascending_colon',
                      source_anchor_stable_id=454,source_anchor_member='FJ2566',
                      distance_from_caudal_ascending_extent_m=span,reference_localization_bound_m=.020,
                      interpretation='All actual band-to-band witnesses lie in the caudal end region of the source ascending colon, consistent with taenia convergence at the caecal/appendiceal end. This does not exempt band crossings elsewhere.')
    else:
        upper=float(data['v459'][:,1].max());span=[float(upper-hi[1]),float(upper-lo[1])]
        assert 0<=span[0]<=span[1]<=.005
        detail.update(interface='longitudinal_muscle_continuation_at_rectal_entry',
                      distance_from_cranial_rectal_extent_m=span,reference_localization_bound_m=.005,
                      interpretation='All actual witnesses lie at the cranial source rectal entry, consistent with the bands joining the rectal longitudinal muscle layer. No distal rectal taenia is inferred.')
    rows.append(detail)
result={'scope':'Explicit interpretation and localization of actual reference intersections; not a general collision whitelist. Numerical witness checks and source ontology support this inference, not measured wall thickness or clinical validation.',
        'driver_sha256':sha(__file__),'audit_sha256':sha(audit_path),'candidate_sha256':sha(candidate),'source_surfaces_sha256':sha(base),
        'source_receipt_sha256':sha(receipt_path),'predicate_sha256':sha(predicate),'coordinate_lattice_scale':scale,
        'non_colon_neighbor_crossings':0,'interfaces':rows,
        'sources':[{'url':'https://openstax.org/books/anatomy-and-physiology/pages/23-5-the-small-and-large-intestines','supports':'Taeniae are the longitudinal muscle layer of the large intestinal wall.'},
                   {'url':'https://www.ncbi.nlm.nih.gov/books/NBK537245/','supports':'The bands coalesce into the rectal outer longitudinal muscle layer.'},
                   {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7271214/','supports':'The taeniae unite at the appendix base and flare into the rectal muscular layer.'}],
        'native_requirement':'Check the corrected candidate through accepted breathing-cycle geometry, retaining any witness outside these localized source interfaces as an error.'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='source_triangle_pairs'} for r in rows]),flush=True)
