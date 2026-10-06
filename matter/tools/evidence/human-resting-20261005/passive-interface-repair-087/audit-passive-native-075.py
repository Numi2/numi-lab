from pathlib import Path
import hashlib,json,sys,time,numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
run=E/'cardiac-map-multiregion-affine-repair-011/native-fullchain073-6s-001'
out=E/'passive-organ-native-audit-075';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools')
import accepted_mrvpack_surface_audit as audit
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py');pred=audit.predicate_module(predicate)
anatomy=E/'passive-organ-native-input-073/resting-anatomy-receipt.json'
metadata=json.loads(anatomy.read_text());ids=metadata['functional_bindings']['passive_viscera_geometry_binding']['stable_ids']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Exact native accepted-frame self and passive-neighbor audits for pancreas and spleen; frames are exported by the combined cardiac/passive scene. Invalid source neighbors remain unresolved.',
        'driver_sha256':sha(__file__),'predicate_sha256':sha(predicate),'anatomy_receipt_sha256':sha(anatomy),'frames':[]}
for step in [0,63,511,639,1951,2207,2783,2999]:
    start=time.monotonic();pack=run/'accepted-geometry'/f'step-{step}.mrvpack';receipt=pack.with_suffix('.receipt.json')
    mm,stream,offset,surfaces=audit.read_pack(pack);provenance=audit.validate_accepted_receipt(pack,receipt,step,mm,offset,surfaces)
    points={s:audit.surface_points(mm,offset,surfaces[(51010,s)]['faces']) for s in ids}
    scale=max(x[2] for x in points.values());boxes={s:np.array(list(p[1].values())) for s,p in points.items()};records={}
    def rec(s):
        if s not in records:records[s]=audit.make_records(points[s],scale,surfaces[(51010,s)]['faces'],pred)
        return records[s]
    frame={'step':step,'pack_sha256':sha(pack),'accepted_receipt':provenance,'self':{},'pairs':[]}
    for s in [3,13]:frame['self'][str(s)]=pred._audit_pair(rec(s),rec(s),same_surface=True)
    for a,b in sorted({tuple(sorted((a,b))) for a in [3,13] for b in ids if a!=b}):
        v,w=boxes[a],boxes[b]
        if np.any(v.max(0)<w.min(0)) or np.any(w.max(0)<v.min(0)):
            result={'count':0,'aabb_disjoint':True,'audit_complete':True}
        else:
            try:result=pred._audit_pair(rec(a),rec(b),same_surface=False)
            except ValueError as err:result={'invalid_geometry':str(err),'audit_complete':False}
        frame['pairs'].append({'ids':[a,b],**result})
    frame['elapsed_seconds']=time.monotonic()-start;report['frames'].append(frame)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'step':step,'self':{s:x['count'] for s,x in frame['self'].items()},'crossings':[(p['ids'],p['count']) for p in frame['pairs'] if p.get('count')],
                      'invalid':[p['ids'] for p in frame['pairs'] if p.get('invalid_geometry')],'elapsed_seconds':frame['elapsed_seconds']}),flush=True)
    mm.close();stream.close()
