#!/usr/bin/env python3
"""Prepare the existing resting owner for one retired liver alias regression arm."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
    if not v:raise ValueError(m)
def save(p,j):p.write_text(json.dumps(j,separators=(',',':'),sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser()
    for k in ('parent','build','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--candidate',type=Path)
    a=ap.parse_args();parent=a.parent.resolve();out=a.output.resolve();build=a.build.resolve()
    req(not out.exists(),'preserve output')
    d=json.loads((parent/'baseline/run-declaration.json').read_text())
    ex=json.loads((parent/'baseline/execution.json').read_text())
    req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'parent unsuccessful')
    pins=dict(d['immutable_assets'])
    req(all(sha(p)==h for p,h in pins.items()),'parent pin drift')
    br=json.loads((build/'build-report.json').read_text())
    req(not br['changed_inputs'] and len(br['commands'])==2 and all(r['returncode']==0 for r in br['commands']),'build failed')
    req(sha(build/'bin/numi-human-native')==br['binary_sha256'],'binary drift')
    req(all(sha(p)==h for p,h in br['input_sha256'].items()),'build source drift')
    src=d['argv'];mi=src.index('numilab_human.resting_run');tail=src[mi+1:].copy()
    oldreceipt=Path(tail[tail.index('--anatomy-receipt')+1]).resolve()
    receipt=json.loads(oldreceipt.read_text())
    oldorgans=Path(receipt['payload']['path']).resolve();oldbuild=Path(tail[tail.index('--build')+1]).resolve()
    if a.candidate:
        cr=json.loads((a.candidate/'candidate-report.json').read_text())
        req(cr['exact_regression']['all_non22_rows_byte_identical_after_decode_and_repack'],'candidate regression failed')
        req(cr['candidate']['stable_id_22_absent'] and cr['candidate']['runtime_stable_ids_match_current_receipt_source_map'],'candidate identities failed')
        req(sha(cr['candidate']['payload'])==cr['candidate']['payload_sha256'],'payload drift')
        req(sha(cr['candidate']['receipt'])==cr['candidate']['receipt_sha256'],'receipt drift')
        receipt=json.loads(Path(cr['candidate']['receipt']).read_text())
        common=receipt['provenance']['cardiac_geometry_binding']['common_field']
        req(common['anatomy_payload_sha256']==cr['inputs']['source_payload_sha256'],'different common source')
        common['anatomy_payload_sha256']=receipt['payload']['sha256']
        # Compact map ranges index the sidecar in stable-ID/local source order.
        # Candidate proof preserves every cardiac/passive source row verbatim.
        receipt['provenance']['retired_liver_alias_native_replay']={
            'candidate_report_path':str((a.candidate/'candidate-report.json').resolve()),
            'candidate_report_sha256':sha(a.candidate/'candidate-report.json'),
            'source_anatomy_sha256':cr['inputs']['source_payload_sha256'],
            'common_map_bytes_unchanged':True,'common_map_rebound_payload_sha256':receipt['payload']['sha256'],
            'native_qualification':'pending'}
        for k in ('payload','receipt'):pins[cr['candidate'][k]]=cr['candidate'][k+'_sha256']
        pins[str((a.candidate/'candidate-report.json').resolve())]=sha(a.candidate/'candidate-report.json')
    out.mkdir();(out/'baseline').mkdir();(out/'assets').mkdir()
    for p in (parent/'assets').iterdir():
        if p.is_file():os.link(p,out/'assets'/p.name)
    common=receipt['provenance']['cardiac_geometry_binding']['common_field']
    for k in ('map','polynomials','domain_boxes'):
        rec=common[k];rel=Path(rec['path']);req(not rel.is_absolute() and '..' not in rel.parts,'invalid map path')
        source=parent/rel;req(sha(source)==rec['sha256'],'map drift')
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    receipt['qualification']='Native latest-source positive anatomy-parent regression with preserved back412, repaired skin and retired liver alias; forearm and later axillary defects remain. Complete anatomy qualification pending.'
    rp=out/'resting-anatomy-receipt.json';save(rp,receipt)
    tail[tail.index('--anatomy-receipt')+1]=str(rp)
    tail[tail.index('--build')+1]=str(build)
    oldout=Path(tail[tail.index('--output')+1]).resolve();tail[tail.index('--output')+1]=str(out/'baseline/native-run')
    env=dict(x.split('=',1) for x in src[2:mi-2])
    env['DYLD_LIBRARY_PATH']=str(build/'lib')+':'+str(build/'matter')
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'baseline/native-run/common-field-failure.json')
    sys.path.insert(0,env['PYTHONPATH'])
    from numilab_human import resting_run as rr
    parser=argparse.ArgumentParser();rr.add_arguments(parser);native,assets=rr.command(parser.parse_args(tail))
    old=d['owner_cli_preview']['native_argv']
    replacement={str(oldbuild/'bin/numi-human-native'):str(build/'bin/numi-human-native'),
        str(oldreceipt):str(rp),str(oldorgans):str(Path(receipt['payload']['path']).resolve())}
    expected=[replacement.get(v,v.replace(str(oldout),str(out/'baseline/native-run'))) for v in old]
    req(native==expected,'native arguments differ beyond reviewed binary/receipt/payload/output substitutions')
    launch=out/'launch_arm.py';shutil.copy2(parent/'launch_arm.py',launch)
    pins.update(br['input_sha256']);pins.update(br['reused_native_runtime_sha256']);pins.update(assets)
    for p in (Path(__file__),rp,launch,build/'build-report.json',build/'native-dependencies.d'):
        pins[str(p.resolve())]=sha(p)
    d.update(immutable_assets=pins,argv=['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],src[mi-2],'-m','numilab_human.resting_run',*tail],
        qualification=receipt['qualification'],launch_adapter={'path':str(launch),'sha256':sha(launch)},
        owner_cli_preview={'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':rr.invocation_environment(env)},
        exact_build_report={'path':str(build/'build-report.json'),'sha256':sha(build/'build-report.json')})
    dp=out/'baseline/run-declaration.json';save(dp,d)
    save(out/'launch-guard.json',{'launcher_sha256':sha(launch),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(out/'baseline/native-run')}}})
    save(out/'preparation.json',{'parent':str(parent),'candidate':str(a.candidate.resolve()) if a.candidate else None,'build':str(build),'binary_sha256':br['binary_sha256'],'physiology_or_force_owner_changes':False,'native_pending':True})
    req(all(sha(p)==h for p,h in pins.items()),'input drift')
    print(json.dumps({'output':str(out),'pins':len(pins),'candidate':bool(a.candidate)}))
if __name__=='__main__':main()
