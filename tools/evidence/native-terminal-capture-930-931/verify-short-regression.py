from pathlib import Path
import json,csv,hashlib,struct,mmap
E=Path('/Users/n/numi-human-resting-evidence-20261005')
A=E/'native-terminal-reference-929';B=E/'native-terminal-candidate-930'
OUT=E/'native-terminal-capture-review-930'
H=struct.Struct('<8sIIQQ32s24s');D=struct.Struct('<IIQQQII32s')
def sha(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        while x:=f.read(8*1024*1024):h.update(x)
    return h.hexdigest()
def sections(p):
    with p.open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
            head=H.unpack_from(m);assert head[0]==b'MRVPACK2' and head[1]==2
            out={};metadata=None
            for i in range(head[2]):
                d=D.unpack_from(m,H.size+D.size*i)
                data=m[d[2]:d[2]+d[3]]
                out[d[0]]={'directory':list(d[:7]),'sha256':hashlib.sha256(data).hexdigest()}
                if d[0]==1:metadata=data
            return out,metadata
def normalize_metadata(raw,r):
    content=r['pack_content_hash'].encode()
    root=r['accepted_root_fingerprint_hex'].encode()
    # Only the explicitly bound content identity and accepted program/root ID
    # may differ. All remaining metadata bytes are compared unchanged.
    assert raw.count(content)==1 and raw.count(root)==1
    return raw.replace(content,b'sha256:'+b'0'*64).replace(root,b'0x'+b'0'*16)
report={'scope':'Exact accepted-state terminal presentation; not anatomical or physiological qualification',
        'reference':'65 physical roots, existing presentation captures accepted64 before root65',
        'candidate':'64 physical roots, query-only GPU FK and presentation of terminal accepted64',
        'runs':{},'captures':{},'traces':{}}
for p in [A,B]:
    m=json.loads((p/'run-metadata.json').read_text())
    assert m['exit_code']==0 and m['loaded_metal_runtime']['verified'] and not m['source_files_changed_during_run']
    report['runs'][p.name]={'metadata_sha256':sha(p/'run-metadata.json'),'wall_seconds':m['wall_seconds'],'argv':m['argv']}
for step in [0,63,64]:
    ra=json.loads((A/f'accepted-geometry/step-{step}.receipt.json').read_text())
    rb=json.loads((B/f'accepted-geometry/step-{step}.receipt.json').read_text())
    for p,r in [(A,ra),(B,rb)]:
        assert sha(p/f'accepted-geometry/step-{step}.mrvpack')==r['pack_file_sha256']
        assert r['accepted_step']==step and r['accepted_root_fingerprint']==r['accepted_transaction_fingerprint']
    keys=['accepted_body_state_sha256','accepted_registered_body_poses','accepted_respiration_state_sha256',
          'accepted_respiratory_motion','accepted_step','accepted_time_s','accepted_timestamp_microseconds',
          'captured_vertex_buffer_sha256','surface_audit','surface_audit_endpoint','physical_endpoint']
    assert all(ra[k]==rb[k] for k in keys)
    x,mx=sections(A/f'accepted-geometry/step-{step}.mrvpack')
    y,my=sections(B/f'accepted-geometry/step-{step}.mrvpack')
    assert set(x)==set(y)
    assert all(x[k]==y[k] for k in x if k!=1)
    assert normalize_metadata(mx,ra)==normalize_metadata(my,rb)
    report['captures'][step]={'identical_state_fields':keys,'identical_pack_section_ids':[k for k in x if k!=1],
      'metadata_difference':'Only accepted root fingerprint and dependent pack content hash; new shader changes program identity',
      'pack_sha256':rb['pack_file_sha256'],'receipt_sha256':sha(B/f'accepted-geometry/step-{step}.receipt.json'),
      'accepted_time_s':rb['accepted_time_s'],'accepted_timestamp_microseconds':rb['accepted_timestamp_microseconds']}
for name in ['resting-coupled.csv','resting-com-q-integration.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv']:
    ra=list(csv.DictReader((A/name).open()));rb=list(csv.DictReader((B/name).open()))
    assert ra[:len(rb)]==rb
    report['traces'][name]={'reference_rows':len(ra),'candidate_rows':len(rb),'shared_row_differences':0,'candidate_sha256':sha(B/name)}
log=(B/'native.log').read_text()
assert 'resting_terminal_presentation=accepted step=64' in log
assert 'physical_steps_advanced=0 controller_steps_advanced=0' in log
report['pass']=True
report['initial_comparison_correction']='The first report expected identical whole-pack hashes. That was overly strict because the new shader changes the accepted program/root identity in metadata. This report compares every non-metadata section byte-for-byte and permits exactly the two explicitly bound metadata identity strings; the original report is retained.'
(OUT/'verification-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pass':True,'report':str(OUT/'verification-v2.json'),'sha256':sha(OUT/'verification-v2.json')}))
