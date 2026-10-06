"""Exact accepted-cycle checks and material-localized passive interfaces."""
from pathlib import Path
import hashlib,json,sys,time,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');run=E/'bladder-taenia-native-114'
out=E/'bladder-taenia-native-audit-115';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools')
import accepted_mrvpack_surface_audit as audit
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
from numilab_human.resting_pleura_proxy import _parse_payload
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py')
pred=audit.predicate_module(predicate)
receipt_path=E/'bladder-cardiac-input-113/resting-anatomy-receipt.json';receipt=json.loads(receipt_path.read_text())
ids=receipt['functional_bindings']['passive_viscera_geometry_binding']['stable_ids']
payload=receipt_path.with_name('resting-thorax.nhanatomy');_,_,source_rows=_parse_payload(payload.read_bytes())
paths={457:E/'taenia-motion-margin-audit-103/report.json',462:E/'bladder-interface-audit-109/report.json'}
source={s:json.loads(p.read_text()) for s,p in paths.items()}
source_pairs={s:{r['ids'][1]:set(map(tuple,r.get('triangle_pairs',[]))) for r in d['pairs']} for s,d in source.items()}
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
report={'scope':'Exact accepted native bladder and taenia self/neighbor checks. Each nonzero interface is localized in source material coordinates using barycentric coordinates of actual native witnesses. No family-wide exemption.',
        'driver_sha256':sha(__file__),'predicate_sha256':sha(predicate),'anatomy_receipt_sha256':sha(receipt_path),
        'source_audit_sha256':{str(s):sha(p) for s,p in paths.items()},'frames':[]}
for step in [0,639,2783,2999]:
    start=time.monotonic();pack=run/'accepted-geometry'/f'step-{step}.mrvpack';rp=pack.with_suffix('.receipt.json')
    mm,stream,offset,surfaces=audit.read_pack(pack)
    provenance=audit.validate_accepted_receipt(pack,rp,step,mm,offset,surfaces)
    points={s:audit.surface_points(mm,offset,surfaces[(51010,s)]['faces']) for s in ids}
    for sid in ids:
        native_faces=np.asarray(surfaces[(51010,sid)]['faces'],dtype=np.int64)
        original_faces=source_rows[sid]['faces']
        assert native_faces.shape==original_faces.shape
        assert np.ptp(native_faces-original_faces)==0,('source face correspondence changed',sid)
    scale=max(x[2] for x in points.values());boxes={s:np.array(list(p[1].values())) for s,p in points.items()};records={}
    def rec(s):
        if s not in records:records[s]=audit.make_records(points[s],scale,surfaces[(51010,s)]['faces'],pred)
        return records[s]
    def material_localization(a,b,pairs):
        native_faces={s:surfaces[(51010,s)]['faces'] for s in (a,b)}
        mapped={a:[],b:[]};max_residual=0.
        for fa,fb in pairs:
            p=pred.triangle_intersection_points(rec(a)[fa][0],rec(b)[fb][0])
            assert p
            world=np.array([[float(c)/scale for c in q] for q in p])
            for s,face in ((a,fa),(b,fb)):
                tri=np.array([points[s][1][i] for i in native_faces[s][face]])
                matrix=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]))
                uv=np.linalg.lstsq(matrix,(world-tri[0]).T,rcond=None)[0].T
                weights=np.column_stack((1-uv.sum(1),uv))
                max_residual=max(max_residual,float(np.max(np.linalg.norm(weights@tri-world,axis=1))))
                assert weights.min()>=-1e-6 and weights.max()<=1+1e-6
                original=source_rows[s]['vertices6'][source_rows[s]['faces'][face],:3].astype(float)
                mapped[s].extend((weights@original).tolist())
        assert max_residual<1e-9
        info={'source_material_witness_bounds_m':{str(s):[np.min(x,axis=0).tolist(),np.max(x,axis=0).tolist()] for s,x in mapped.items()},
              'maximum_native_barycentric_reconstruction_error_m':max_residual}
        xyz=np.array(mapped[a]);ylo=float(xyz[:,1].min());yhi=float(xyz[:,1].max())
        if a==457 and b in (454,455,460):
            info.update(interface='longitudinal_muscle_component_of_colon_wall',within_declared_interface=True)
        elif a==457 and b in (456,458):
            lower=float(source_rows[454]['vertices6'][:,1].min());span=[ylo-lower,yhi-lower]
            info.update(interface='taenia_convergence_at_caudal_ascending_colon',source_localization_span_m=span,within_declared_interface=0<=span[0]<=span[1]<=.020)
        elif a==457 and b==459:
            upper=float(source_rows[459]['vertices6'][:,1].max());span=[upper-yhi,upper-ylo]
            info.update(interface='longitudinal_muscle_continuation_at_rectal_entry',source_localization_span_m=span,within_declared_interface=0<=span[0]<=span[1]<=.005)
        elif a==462 and b==463:
            lower=float(source_rows[462]['vertices6'][:,1].min());upper=float(source_rows[463]['vertices6'][:,1].max())
            prostate=np.array(mapped[b]);spans=[[ylo-lower,yhi-lower],[upper-float(prostate[:,1].max()),upper-float(prostate[:,1].min())]]
            info.update(interface='preserved_source_bladder_prostate_neck',source_localization_spans_m=spans,
                        within_declared_interface=all(0<=x[0]<=x[1]<=.005 for x in spans))
        else:info.update(interface='unregistered',within_declared_interface=False)
        return info
    frame={'step':step,'pack_sha256':sha(pack),'accepted_receipt':provenance,'organs':[]}
    for organ in (457,462):
        row={'stable_id':organ,'self':pred._audit_pair(rec(organ),rec(organ),same_surface=True),'pairs':[]}
        for sid in ids:
            if sid==organ:continue
            a,b=boxes[organ],boxes[sid]
            if np.any(a.max(0)<b.min(0)) or np.any(b.max(0)<a.min(0)):
                result={'count':0,'aabb_disjoint':True,'audit_complete':True}
            else:
                try:result=pred._audit_pair(rec(organ),rec(sid),same_surface=False)
                except ValueError as err:result={'invalid_geometry':str(err),'audit_complete':False}
            native=set(map(tuple,result.get('triangle_pairs',[])));old=source_pairs[organ][sid]
            result.update(source_crossing_count=len(old),new_triangle_pairs=sorted(native-old),removed_source_pair_count=len(old-native))
            if native:result['localization']=material_localization(organ,sid,result['triangle_pairs'])
            row['pairs'].append({'ids':[organ,sid],**result})
        frame['organs'].append(row)
    frame['elapsed_seconds']=time.monotonic()-start;report['frames'].append(frame)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'step':step,'organs':[{'id':x['stable_id'],'self':x['self']['count'],
        'nonzero':[(p['ids'],p.get('count'),p.get('localization',{}).get('within_declared_interface')) for p in x['pairs'] if p.get('count')],
        'invalid':[p['ids'] for p in x['pairs'] if p.get('invalid_geometry')]} for x in frame['organs']],
        'elapsed_seconds':frame['elapsed_seconds']}),flush=True)
    mm.close();stream.close()
