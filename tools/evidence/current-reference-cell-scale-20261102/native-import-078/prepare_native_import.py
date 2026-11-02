#!/usr/bin/env python3
"""Bind the actual 1061 mass allocation to the existing direct-O native scene."""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys,os,subprocess,time
S=Path(__file__).resolve().parent.parent
P=S/'postpass-005'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p,name):
    q=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(q);sys.modules[q.name]=m;q.loader.exec_module(m);return m
def prepare(converter,build,stage_name):
    cfg=read(P/'identity-003.json');ad=load(P/'adapter_current_regional_1061.py','ad1061_native');owner=ad._ORIGINAL
    route=ad.route_module(cfg);pins=ad.base.merged_pins(cfg);route.verify_input_pins(pins)
    for raw,digest in cfg['baseline_input_pins'].items():assert sha(raw)==digest,raw
    baseline_path=Path(cfg['static_pins']['baseline_launch_snapshot_path']);assert sha(baseline_path)==cfg['static_pins']['baseline_launch_sha256']
    fractional=ad.verify_h706_fractional_env(cfg)
    receipt=P/'mass-receipt-007.json';mass=read(receipt);ad.verify_fractional_mass_contract(route,mass,cfg)
    assert read(converter/'execution.json')['exit_code']==0
    assert 'PASS_SOLVER_FREE_PAIRED_CURRENT_ROLE_FLOW_CONVERSION' in (converter/'stdout.log').read_text()
    stems=['CHILD_MASS','TRANSPORT_FLOW','SOURCE_CARRIER_FLOW','CHILD_SOURCE_DONOR','SOURCE_PARENT','SOURCE_PARENT_DONOR']
    names=['child-mass.csv','transport-flow.csv','source-carrier-flow.csv','child-source-donor.csv','source-parent.csv','source-parent-donor.csv']
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_'
    recorded={prefix+'RECORDED_CARRIER_ALLOCATION_CSV':'1','NUMI_H706_FRACTIONAL_RECORDED_FLOW_ADMISSION_RECEIPT_JSON':str(receipt),
        'NUMI_H706_FRACTIONAL_RECORDED_FLOW_ADMISSION_RECEIPT_SHA256':sha(receipt)}
    role_pins={}
    for role in ['skin','fat']:
        folder=converter/'role-output';rp=folder/(role+'-recorded-flow-receipt.json');doc=read(rp)
        assert doc['paired_solver_run_sha256']==mass['solver']['transport_receipt_sha256']
        assert doc['complete_allocation'] is True and doc['physical_admission'] is False and doc['native_steps']==0
        rprefix=prefix+'RECORDED_ALLOCATION_'+role.upper()+'_'
        recorded[rprefix+'RECEIPT_JSON']=str(rp);recorded[rprefix+'RECEIPT_SHA256']=sha(rp);role_pins[str(rp)]=sha(rp)
        for stem,name in zip(stems,names):
            f=folder/(role+'-'+name);digest=sha(f)
            refs=[v for k,v in doc.items() if isinstance(v,dict) and f.name in v]
            if not refs:raise ValueError('role receipt omits CSV map '+f.name)
            assert refs[0][f.name]['sha256']==digest
            recorded[rprefix+stem+'_CSV']=str(f);recorded[rprefix+stem+'_CSV_SHA256']=digest;role_pins[str(f)]=digest
    bm=read(build);assert bm['exit_code']==0 and bm['source']['source_unchanged_during_build'] is True
    binary=Path(bm['binary']['path']);assert sha(binary)==bm['binary']['sha256']
    for raw,ref in bm['runtime_dependencies'].items():assert sha(raw)==ref['sha256'],raw
    sourcepins=read(bm['source']['pins']['path']);assert sha(bm['source']['pins']['path'])==bm['source']['pins']['sha256']
    for raw,ref in sourcepins.items():assert sha(raw)==ref['sha256'],raw
    baseline=read(baseline_path);static={'baseline':baseline,'build':{'build':{'binary':str(binary)}}}
    root=Path(__file__).resolve().parent/stage_name
    assert not root.exists()
    argv,scene,stale=ad.base.safe_scene(cfg,static,route,root,2)
    suffix=owner.direct_env(route,cfg,pins)
    source_only=stage_name=='source-preflight'
    environment=owner.apply_contact_capacity_trial(owner.apply_direct_env(route,scene,suffix,source_only,False,fractional,recorded))
    environment['NUMI_HUMAN_SKIN_HARD_ATTACHMENT_REFERENCE']='0'
    # Current registered replacement owns installation; legacy identity is not a second install.
    environment['NUMI_HUMAN_UNPAIRED_IDENTITY_INSTALL']='0'
    environment['NUMI_HUMAN_UNPAIRED_IDENTITY_PREFLIGHT']='0'
    # The fractional route imports the current physical volume directly.
    # Historical separated-reference refinement is a different, source-only owner.
    assert environment[prefix+'OPERATIONAL_REFERENCE_SPLIT']=='0'
    environment[prefix+'PREFLIGHT_OPERATIONAL_CONFORMING_REFINEMENT_CHAIN']='0'
    restored=read(S/'current-reference-import-031/registration-restoration.json')
    original_registration=environment[prefix+'PREFLIGHT_NODE_REGISTRATION_CSV']
    assert sha(original_registration)==restored['incomplete_sidecar']['sha256']
    assert restored['source']['sha256']==environment[prefix+'PREFLIGHT_COMBINED_NODE_MAP_SHA256']
    assert sha(restored['output']['path'])==restored['output']['sha256']
    assert restored['rows']=={'O':76193,'I':118197,'D':103509}
    environment[prefix+'PREFLIGHT_NODE_REGISTRATION_CSV']=restored['output']['path']
    environment[prefix+'PREFLIGHT_NODE_REGISTRATION_SHA256']=restored['output']['sha256']
    attachment=read(S/'attachment-remap-039/remap-receipt.json')
    assert attachment['status']=='PASS_REFERENCE_ATTACHMENT_REMAP_ONLY'
    assert attachment['geometry_changed'] is False and attachment['mass_changed'] is False
    assert len(attachment['rows'])==14 and attachment['stable_attachment_ids_preserved'] is True
    for raw,digest in attachment['inputs'].items():assert sha(raw)==digest,raw
    assert sha(attachment['output']['path'])==attachment['output']['sha256']
    environment[prefix+'PREFLIGHT_DEEP_ATTACHMENT_REQUESTS_CSV']=attachment['output']['path']
    environment[prefix+'PREFLIGHT_DEEP_ATTACHMENT_REQUESTS_SHA256']=attachment['output']['sha256']
    reframe=read(S/'dominant-frame-registration-042/receipt.json')
    assert reframe['status']=='PASS_CURRENT_DOMINANT_FRAME_REGISTRATION_ONLY'
    assert reframe['changed_boundary_nodes']==14
    for key in ['world_geometry_changed','source_ids_changed','full86_weights_changed',
                'physical_cell_semantics_changed','mass_allocation_changed','historical_registration_changed']:
        assert reframe[key] is False,key
    for raw,digest in reframe['inputs'].items():assert sha(raw)==digest,raw
    for raw,item in reframe['outputs'].items():assert sha(raw)==item['sha256'],raw
    frame_root=S/'dominant-frame-registration-042'
    for role in 'OID':
        rp=prefix+'PREFLIGHT_ROLE_'+role+'_'
        sf=frame_root/('registered-'+role+'.csv');asset=frame_root/(role+'-source.json')
        environment[prefix+'PREFLIGHT_SURFACE_'+role+'_NODES_CSV']=str(sf)
        environment[prefix+'PREFLIGHT_SURFACE_'+role+'_NODES_SHA256']=sha(sf)
        environment[rp+'SOURCE_ASSET']=str(asset);environment[rp+'SOURCE_ASSET_SHA256']=sha(asset)
    environment[prefix+'PREFLIGHT_COMBINED_NODE_MAP_CSV']=str(frame_root/'combined-node-map.csv')
    environment[prefix+'PREFLIGHT_COMBINED_NODE_MAP_SHA256']=sha(frame_root/'combined-node-map.csv')
    # Registry counts describe the typed source ID space, not compact boundary rows.
    import numpy as np
    source_domains={}
    for role in 'OID':
        rp=prefix+'PREFLIGHT_ROLE_'+role+'_'
        asset=Path(environment[rp+'SOURCE_ASSET'])
        assert sha(asset)==environment[rp+'SOURCE_ASSET_SHA256']
        doc=read(asset);raw=doc['derivation']['current_role_asset_pins'][role]
        source=Path(raw['path']);assert sha(source)==raw['sha256']
        with np.load(source,allow_pickle=False) as z:
            points=z['points_world_m_f32'];faces=z['faces']
            rows=doc['vertices'];ids=np.asarray([v['source_vertex_row'] for v in rows],dtype=np.int64)
            assert len(set(ids.tolist()))==len(ids) and np.all(ids>=0) and np.all(ids<len(points))
            actual=np.asarray([v['current_world_m'] for v in rows],dtype=np.float64)
            assert np.array_equal(actual,points[ids].astype(np.float64)),role
            vertices_count=len(points);face_count=len(faces)
        environment[rp+'SOURCE_VERTEX_COUNT']=str(vertices_count)
        environment[rp+'SOURCE_FACE_COUNT']=str(face_count)
        source_domains[role]={'asset':pin(asset),'typed_source':raw,'registered_vertices':len(rows),
            'source_vertex_count':vertices_count,'source_face_count':face_count,
            'exact_registered_to_source_coordinate_match':True}
    if not source_only:
        source_receipt=read(Path(__file__).resolve().parent/'source-preflight.execution.json')
        assert source_receipt['exit_code']==0
        environment.update({'NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT':'1','NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS':'1',
            'NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT':'1','NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT':'1',
            'NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP':'1','NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP':'2',
            'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS':'0,1,2'})
    if not source_only:route.validate_full_scene_mode(argv[1:],environment)
    plan={'schema':'numi.human.current-native-launch-1025-direct-o-recorded-flow.v1','status':'DIRECT_O_SOURCE_PREFLIGHT_READY' if source_only else 'DIRECT_O_MOTION_PLAN_READY_AFTER_SOURCE_PREFLIGHT',
        'execution_performed':False,'physical_admission':False,'GPU_executed':False,'native_runtime_executed':False,
        'manifest_path':str(P/'identity-003.json'),'manifest_sha256':sha(P/'identity-003.json'),
        'native_binary':bm['binary'],'runtime_dependencies':bm['runtime_dependencies'],'build_manifest':pin(build),'geometry_admission':cfg['geometry_admission'],'mass_admission':pin(receipt),
        'converter_run':pin(converter/'run-manifest.json'),'converter_execution':pin(converter/'execution.json'),
        'input_pins':pins,'role_input_pins':role_pins,'baseline_launch':pin(baseline_path),'source_qualifier':pin(__file__),
        'current_source_domain_audit':source_domains,
        'registration_restoration':pin(S/'current-reference-import-031/registration-restoration.json'),
        'attachment_remap':pin(S/'attachment-remap-039/remap-receipt.json'),
        'current_frame_registration':pin(S/'dominant-frame-registration-042/receipt.json'),
        'measurement_scope':'release-motion integration probe; not supported rest or sustained motion qualification',
        'stages':[{'name':stage_name,'argv':argv,'environment':environment,'cwd':baseline['cwd'],
            'expected_accepted_steps':0 if source_only else 2,'expected_runtime':not source_only,'expected_gpu':True}]}
    path=Path(__file__).resolve().parent/(stage_name+'.plan.json');assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n')
    shell=path.with_suffix('.sh');shell.write_text(ad.base.stage_shell(route,plan['stages'][0],baseline['cwd']));shell.chmod(0o755)
    print('plan_ready',path,'sha256',sha(path))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--converter',type=Path,required=True);a.add_argument('--build',type=Path,required=True);a.add_argument('--stage',choices=['source-preflight','install-motion'],required=True);v=a.parse_args();prepare(v.converter,v.build,v.stage)
