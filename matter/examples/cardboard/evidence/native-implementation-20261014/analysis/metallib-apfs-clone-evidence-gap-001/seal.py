#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,time
p=pathlib.Path(__file__).parent
report=json.loads((p/'current-inventory.json').read_text())
report['historical_operation']={
 'transcript_reported_replacements':378,
 'transcript_reported_errors':0,
 'transcript_reported_skips':0,
 'transcript_reported_df_available_kib_before':1665916,
 'transcript_reported_df_available_kib_after':2623616,
 'transcript_reported_net_df_kib':957700,
 'transcript_values_independently_recovered':False,
 'per_path_before_after_manifest_recovered':False,
 'operation_script_recovered':False,
 'historical_metadata_verification':'unverified',
 'scope_note':'Concurrent cleanup may have contributed to df change; exact changed path set is unavailable.'
}
(p/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
files=['README.md','census.py','current-inventory.json','report.json']
manifest={
 'schema':1,
 'package':'metallib-apfs-clone-evidence-gap-001',
 'created_utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
 'evidence_status':'historical operation is not independently reproducible; explicit gap retained',
 'files':{name:{'sha256':hashlib.sha256((p/name).read_bytes()).hexdigest(),'bytes':(p/name).stat().st_size} for name in files}
}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({
 'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
 'files':manifest['files']
},indent=2))
