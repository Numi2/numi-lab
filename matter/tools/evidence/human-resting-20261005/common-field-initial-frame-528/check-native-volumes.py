"""Independent volume regression on captured native accepted cardiac surfaces."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--tools',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('accepted_mrvpack_surface_audit',a.tools/'accepted_mrvpack_surface_audit.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
receipt=json.loads(a.receipt.read_text());common=receipt['provenance']['cardiac_geometry_binding']['common_field']
rows={int(r['step']):r for r in csv.DictReader((a.run/'resting-surface-audit.csv').open())}
inv=json.loads((a.run/'invocation.json').read_text());meta=json.loads((a.run/'run-metadata.json').read_text())
assert sha(a.receipt)==inv['asset_sha256'][str(a.receipt.resolve())]
assert meta['exit_code']==0 and meta['loaded_metal_runtime']['verified'] and not meta['source_files_changed_during_run']
out={'scope':'Independent Float64 integration of actual GPU-deformed accepted triangle buffers, not physiological validation. Runtime threshold is retained at 2e-4 relative volume error.','driver_sha256':sha(__file__),'reader_sha256':sha(reader.__file__),'receipt_sha256':sha(a.receipt),'run_metadata_sha256':sha(a.run/'run-metadata.json'),'frames':[]}
for pack in sorted((a.run/'accepted-geometry').glob('step-*.mrvpack'),key=lambda x:int(x.stem.split('-')[1])):
    step=int(pack.stem.split('-')[1]);row=rows[step]
    mapped,stream,offset,surfaces=reader.read_pack(pack)
    valid=reader.validate_accepted_receipt(pack,pack.with_suffix('.receipt.json'),step,mapped,offset,surfaces)
    frame={'step':step,'time_s':float(row['time_s']),'accepted_receipt':valid,'surfaces':[]}
    try:
        for sid,key in zip([318,319,320,321],['ra_target_ml','rv_target_ml','la_target_ml','lv_target_ml']):
            frame['surfaces'].append({'stable_id':sid,'target_ml':float(row[key]),'target_source':'synchronized accepted native trace'})
        for i,sid in enumerate([1,23,24]):
            frame['surfaces'].append({'stable_id':sid,'target_ml':float(np.float32(common['material_target_volumes_m3'][i]))*1e6,'target_source':'source-bound conserved material target, serialized to native Float32'})
        for result in frame['surfaces']:
            sid=result['stable_id'];key=(51025 if sid in [318,319,320,321] else 51010,sid)
            gf=np.asarray(surfaces[key]['faces'],dtype=np.int64)
            used,invf=np.unique(gf,return_inverse=True)
            vertexview=np.ndarray((max(int(used.max())+1,1),3),dtype='<f4',buffer=mapped,offset=offset,strides=(80,4))
            v=vertexview[used].astype(float);f=invf.reshape(-1,3);del vertexview
            assert np.isfinite(v).all()
            xyz=v[f];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
            zero=int(np.count_nonzero(np.all(normal==0,axis=1)))
            xyz-=v.mean(axis=0)
            volume=math.fsum(np.einsum('ij,ij->i',xyz[:,0],np.cross(xyz[:,1],xyz[:,2]))/6)*1e6
            error=abs(volume-result['target_ml'])/result['target_ml']
            result.update(vertices=len(v),triangles=len(f),signed_volume_ml=volume,relative_error=error,zero_area_triangles=zero,pass_=error<=2e-4 and zero==0 and volume>0)
        frame['pass']=all(x['pass_'] for x in frame['surfaces']) and int(row['functional_geometry_status'])==0 and int(row['common_coordinate_solver_status'])==0
        frame['common_coordinates']=[float(row['common_coordinate_'+key]) for key in ['RA','RV','LA','LV','RA_material','ventricular_material','LA_material']]
    finally:
        mapped.close();stream.close()
    out['frames'].append(frame)
assert out['frames'] and out['frames'][0]['step']==0 and out['frames'][0]['time_s']==0
out['pass']=all(f['pass'] for f in out['frames'])
out['maximum_relative_error']=max(s['relative_error'] for f in out['frames'] for s in f['surfaces'])
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'pass':out['pass'],'frames':[f['step'] for f in out['frames']],'max_error':out['maximum_relative_error']}))
raise SystemExit(0 if out['pass'] else 1)
