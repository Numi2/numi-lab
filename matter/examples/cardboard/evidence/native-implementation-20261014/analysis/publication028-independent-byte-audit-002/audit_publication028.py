#!/usr/bin/env python3
"""Independent read-only byte audit for staged cardboard publication 028."""
from __future__ import annotations
import gzip, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path('/Users/home/numi-cardboard-20261006')
BASE = Path('/Users/home/cardboard-implementation-20261014')
EVID = ROOT / 'matter/examples/cardboard/evidence/native-implementation-20261014'
OUT = BASE / 'analysis/publication028-independent-byte-audit-002'
EXPECTED = {
    'stage_paths_sha256': '639fe00ea24320f0264d47e1fafaa21efd9959ed2e0fce38e54d7a79c1769bf4',
    'summary_sha256': 'd025f2ce80dda69efb980ed24a21001cc8e2c6ac41a68299768bb25c26182196',
    'binding_sha256': 'd720819e837db8e503b8cc06c3f9664709d10089ccf05da24f97fbfb0bf02f0d',
    '027_pathspec_sha256': '8d14076a060be308f86e2dca697f975d6ebab5d152ccae6e24f22880693a9f0c',
    '028_pathspec_sha256': '2014e68375cec1a720edc04feca77f64d2a8d5bc18c5be1949ca1802dba84662',
    'prior_summary_sha256': '855f5b9c193720744ee0b49cd5d5b7a26be53b21509369da7dc408533ae756c6',
    'external_recovery_index_sha256': 'ac63ec0a',
}

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def hash_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256(); n = 0
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b: break
            h.update(b); n += len(b)
    return n, h.hexdigest()

def readnul(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\0'):
        raise ValueError(f'{path} lacks terminal NUL')
    vals = [x.decode('utf-8', 'surrogateescape') for x in raw.split(b'\0') if x]
    if len(set(vals)) != len(vals): raise ValueError(f'duplicate NUL paths in {path}')
    return vals

def git(args: list[str], *, check=True) -> bytes:
    p = subprocess.run(['git', *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f'git {args}: {p.returncode}: {p.stderr.decode(errors="replace")}')
    return p.stdout

def stream_equal_blob(path: Path, oid: str) -> tuple[bool, int]:
    proc = subprocess.Popen(['git', 'cat-file', 'blob', oid], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    total = 0; equal = True
    try:
        if path.is_symlink():
            local = os.fsencode(os.readlink(path))
            pos = 0
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk: break
                if local[pos:pos+len(chunk)] != chunk: equal = False
                pos += len(chunk); total += len(chunk)
            if pos != len(local): equal = False
        else:
            with path.open('rb') as f:
                while True:
                    blob = proc.stdout.read(1024 * 1024)
                    local = f.read(1024 * 1024)
                    if not blob and not local: break
                    if blob != local: equal = False
                    total += len(blob)
        stderr = proc.stderr.read() if proc.stderr else b''
        rc = proc.wait()
        if rc: raise RuntimeError(f'git cat-file {oid} failed rc={rc}: {stderr!r}')
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
    return equal, total

def main() -> int:
    checks = {}
    failures: list[str] = []
    def check(name: str, ok: bool, detail=None):
        checks[name] = {'pass': bool(ok), 'detail': detail}
        if not ok: failures.append(name)

    binding_path = EVID / 'publication-binding-reference028.json'
    summary_path = EVID / 'summary.json'
    prior_summary_path = EVID / 'summary-reference027.json'
    binding = json.loads(binding_path.read_text())
    summary = json.loads(summary_path.read_text())
    prior_summary = json.loads(prior_summary_path.read_text())
    prepared = json.loads((BASE / 'publication028-cases-prepared.json').read_text())
    stage_list_path = BASE / 'publication028-staged-paths.nul'
    staged_expected = readnul(stage_list_path)
    raw_stage_list = stage_list_path.read_bytes()
    path027_path = BASE / 'publication-reference027-pathspec.nul'
    path028_path = BASE / 'publication-reference028-pathspec.nul'
    raw027, raw028 = path027_path.read_bytes(), path028_path.read_bytes()
    list027, list028 = readnul(path027_path), readnul(path028_path)

    # Core publication metadata and unchanged historical prefix.
    sbytes, shash = hash_file(summary_path)
    bbytes, bhash = hash_file(binding_path)
    pbytes, phash = hash_file(prior_summary_path)
    check('summary_hash', shash == EXPECTED['summary_sha256'] == binding.get('summary_sha256'), {'bytes': sbytes, 'sha256': shash})
    check('binding_hash', bhash == EXPECTED['binding_sha256'], {'bytes': bbytes, 'sha256': bhash})
    check('historical_summary_hash', phash == EXPECTED['prior_summary_sha256'] == binding.get('historical_prefix', {}).get('sha256'), {'bytes': pbytes, 'sha256': phash})
    nold = binding.get('previous_prefix_records')
    check('summary_record_counts', len(summary.get('records', [])) == 1480 and nold == 1460 and len(prior_summary.get('records', [])) == 1460 and binding.get('new_case_records') == 20,
          {'current': len(summary.get('records', [])), 'prior': len(prior_summary.get('records', [])), 'declared_old': nold, 'declared_new': binding.get('new_case_records')})
    old_prefix_equal = summary.get('records', [])[:1460] == prior_summary.get('records', [])
    check('summary_old_prefix_exact', old_prefix_equal, {'records': 1460})
    case_names = [r.get('case') for r in summary.get('records', [])[1460:]]
    check('case_list_exact', case_names == binding.get('cases') == prepared.get('cases') and len(case_names) == 20,
          {'count': len(case_names), 'cases_sha256': sha_bytes(('\n'.join(case_names)+'\n').encode())})
    qualification_text = summary.get('qualification', {})
    verdicts = summary.get('verdicts', {})
    axis_names = {'solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly_in_simulation','crush_impact_prediction'}
    status_map = {k: v.get('status') for k,v in verdicts.items() if isinstance(v,dict)}
    expected_status_map = {k: v.get('status') for k,v in binding.get('six_axis_verdicts',{}).items() if isinstance(v,dict)}
    statuses_bounded = set(status_map)==axis_names and status_map==expected_status_map and all(s not in ('qualified','passed','complete') for s in status_map.values())
    required_limit_phrases = {
        'solver_correctness': 'Complete nonlinear solve',
        'numerical_convergence': 'active native468 case and launch guard are deliberately excluded',
        'component_calibration': 'remain open',
        'analogue_validation': 'no new full-board analogue validation pass',
        'physical_assembly_in_simulation': 'no fully scored',
        'crush_impact_prediction': 'No qualified'
    }
    text_bounded = set(qualification_text)==axis_names and all(isinstance(v,str) and required_limit_phrases[k].lower() in v.lower() for k,v in qualification_text.items())
    check('qualification_claims_bounded', binding.get('physical_validation') is False and statuses_bounded and text_bounded,
          {'physical_validation': binding.get('physical_validation'), 'qualification_axis_count':len(qualification_text), 'summary_statuses':status_map, 'binding_statuses':expected_status_map, 'required_limit_phrases':required_limit_phrases,'descriptive_qualification':qualification_text})

    # New case payloads: check source bytes, copied bytes, and decompressed bytes.
    file_rows = [(r, f) for r in summary.get('records', [])[1460:] for f in r.get('files', [])]
    raw_rows = [f for _, f in file_rows if 'repository_copy' in f]
    gz_rows = [f for _, f in file_rows if 'repository_compressed_copy' in f]
    payload_failures=[]; raw_ok=0; gz_ok=0; total_copy_bytes=0
    for rec, row in file_rows:
        src = Path(row['external_path'])
        try:
            sn, sh = hash_file(src)
            if sn != row['bytes'] or sh != row['sha256']:
                payload_failures.append({'case':rec['case'],'name':row['name'],'layer':'external','bytes':sn,'sha256':sh})
                continue
            if 'repository_copy' in row:
                dst = EVID / row['repository_copy']
                dn, dh = hash_file(dst)
                if dn != row['bytes'] or dh != row['sha256']:
                    payload_failures.append({'case':rec['case'],'name':row['name'],'layer':'repository_copy','bytes':dn,'sha256':dh})
                else: raw_ok += 1; total_copy_bytes += dn
            elif 'repository_compressed_copy' in row:
                dst = EVID / row['repository_compressed_copy']
                dn, dh = hash_file(dst)
                if dn != row['compressed_bytes'] or dh != row['compressed_sha256']:
                    payload_failures.append({'case':rec['case'],'name':row['name'],'layer':'gzip_payload','bytes':dn,'sha256':dh})
                else:
                    with gzip.open(dst, 'rb') as gf:
                        h=hashlib.sha256(); un=0
                        while True:
                            chunk=gf.read(1024*1024)
                            if not chunk: break
                            h.update(chunk); un += len(chunk)
                    if un != row['bytes'] or h.hexdigest() != row['sha256']:
                        payload_failures.append({'case':rec['case'],'name':row['name'],'layer':'gzip_decompressed','bytes':un,'sha256':h.hexdigest()})
                    else: gz_ok += 1; total_copy_bytes += dn
            else:
                payload_failures.append({'case':rec['case'],'name':row['name'],'layer':'missing_repository_destination'})
        except Exception as e:
            payload_failures.append({'case':rec.get('case'),'name':row.get('name'),'layer':'exception','error':repr(e)})
    check('payload_file_totals', len(file_rows) == 715 and len(raw_rows) == 707 and len(gz_rows) == 8 and binding.get('copied_files') == 715 and binding.get('raw_files') == 707 and binding.get('gzip_files') == 8,
          {'payloads':len(file_rows),'raw':len(raw_rows),'gzip':len(gz_rows)})
    check('payload_source_copy_and_decompression_hashes', not payload_failures and raw_ok == 707 and gz_ok == 8,
          {'raw_verified':raw_ok,'gzip_verified':gz_ok,'copy_bytes':total_copy_bytes,'mismatches':payload_failures[:20],'mismatch_count':len(payload_failures)})

    # Pathspec ancestry and exact delta.
    check('pathspec_hashes_and_counts', len(list027)==4849 and len(list028)==5566 and sha_bytes(raw027)==EXPECTED['027_pathspec_sha256'] and sha_bytes(raw028)==EXPECTED['028_pathspec_sha256'] and binding.get('prior_pathspec_entries')==4849 and binding.get('prior_pathspec_sha256')==EXPECTED['027_pathspec_sha256'],
          {'027_count':len(list027),'027_sha256':sha_bytes(raw027),'028_count':len(list028),'028_sha256':sha_bytes(raw028)})
    prefix_exact = raw028.startswith(raw027)
    delta = set(list028[len(list027):]) if prefix_exact else set()
    evidence_prefix='matter/examples/cardboard/evidence/native-implementation-20261014/'
    expected_delta = {evidence_prefix+(f.get('repository_copy') or f.get('repository_compressed_copy')) for _,f in file_rows}
    expected_delta |= {evidence_prefix+'summary-reference027.json',evidence_prefix+'publication-binding-reference028.json'}
    # published pathspec is relative to evidence root
    check('pathspec_old_prefix_and_new_delta', prefix_exact and delta == expected_delta and len(delta)==717,
          {'prefix_exact':prefix_exact,'delta_count':len(delta),'expected_delta_count':len(expected_delta),'missing':sorted(expected_delta-delta)[:20],'unexpected':sorted(delta-expected_delta)[:20]})

    # External recovery index and every published recovery pointer/archive reference.
    idx_ext = BASE/'analysis/publication028-external-recovery-index-001/external-recovery-index.json'
    idx_sha_expected='ac63ec0a0f4cba8832709024c1d3ecc56dc54e5b97d65e880292b33c77ebca25'
    idx_bytes, idx_sha = hash_file(idx_ext)
    idx=json.loads(idx_ext.read_text())
    pointer_rows=idx.get('pointer_files',[])
    archive_descriptors=[]
    def walk(o, ctx='root'):
        if isinstance(o,dict):
            if isinstance(o.get('path'),str) and isinstance(o.get('bytes'),int) and isinstance(o.get('sha256'),str):
                archive_descriptors.append((ctx,o))
            for k,v in o.items(): walk(v,ctx+'/'+k)
        elif isinstance(o,list):
            for i,v in enumerate(o): walk(v,ctx+f'/{i}')
    walk(idx)
    dedup={item[1]['path']:item for item in archive_descriptors}
    archive_failures=[]
    for pth, (ctx,row) in dedup.items():
        p=Path(pth)
        if not p.exists():
            archive_failures.append({'path':pth,'context':ctx,'error':'missing'}); continue
        n,h=hash_file(p)
        if n!=row['bytes'] or h!=row['sha256']:
            archive_failures.append({'path':pth,'context':ctx,'bytes':n,'sha256':h,'expected_bytes':row['bytes'],'expected_sha256':row['sha256']})
    pointer_failures=[]; pointer_copies_ok=0; source_pointers_present=0
    for row in pointer_rows:
        src=Path(row['source_path']); dst=EVID/'analysis/publication028-external-recovery-index-001'/row['published_copy']
        if not dst.exists(): pointer_failures.append({'published_copy':row['published_copy'],'error':'missing_copy'}); continue
        n,h=hash_file(dst)
        if n!=row['bytes'] or h!=row['sha256']:
            pointer_failures.append({'published_copy':row['published_copy'],'bytes':n,'sha256':h,'expected_bytes':row['bytes'],'expected_sha256':row['sha256']})
        else:
            pointer_copies_ok += 1
            if src.exists():
                source_pointers_present += 1; sn,sh=hash_file(src)
                if sn!=n or sh!=h: pointer_failures.append({'source_path':str(src),'error':'source_copy_mismatch'})
    check('external_recovery_index_hash_and_case', idx_sha == idx_sha_expected and len(pointer_rows)==128 and binding.get('external_recovery_archives') is not None,
          {'bytes':idx_bytes,'sha256':idx_sha,'pointer_rows':len(pointer_rows),'archive_descriptors_unique':len(dedup)})
    check('external_archive_and_inventory_hashes', not archive_failures,
          {'descriptors':len(dedup),'verified':len(dedup)-len(archive_failures),'mismatches':archive_failures[:20]})
    check('external_recovery_pointer_copies', pointer_copies_ok==128 and not pointer_failures,
          {'copies_verified':pointer_copies_ok,'source_pointers_still_present':source_pointers_present,'mismatches':pointer_failures[:20]})
    index_receipt = json.loads((BASE/'analysis/publication028-external-recovery-index-001/build-receipt.json').read_text())
    receipt_index_rows=[r for r in index_receipt.get('files',[]) if r.get('path')=='external-recovery-index.json']
    check('external_recovery_receipt', len(receipt_index_rows)==1 and receipt_index_rows[0].get('sha256')==idx_sha and receipt_index_rows[0].get('bytes')==idx_bytes and index_receipt.get('archive_count')==128,
          {'archive_count':index_receipt.get('archive_count'),'index_sha256':idx_sha,'receipt_index_rows':receipt_index_rows})
    status_note=(BASE/'analysis/publication028-external-recovery-index-001/README-current-status.md').read_text()
    recovery_flags=idx.get('verification',{})
    check('active468_status_and_exclusion_explicit', 'Command 003 was refused' in status_note and 'Command 004 was later launched' in status_note and 'active native468 run' in status_note and recovery_flags.get('native468_active_case_or_launch_guard_included') is False,
          {'status_note_sha256':hash_file(BASE/'analysis/publication028-external-recovery-index-001/README-current-status.md')[1], 'active_flag':recovery_flags.get('native468_active_case_or_launch_guard_included')})

    # Command003 refusal is historical; command004 is executable-bound and prepared separately.
    cmd3=EVID/'analysis/native468-source157-dt-half-cycle-command-003/native-468-command.json'
    cmd4=EVID/'analysis/native468-source157-dt-half-cycle-command-004/native-468-command.json'
    c3=json.loads(cmd3.read_text()); c4=json.loads(cmd4.read_text())
    refuse=json.loads((EVID/'analysis/native468-executable-preflight-refusal-001/report.json').read_text())
    bind4=json.loads((EVID/'analysis/native468-source157-dt-half-cycle-command-004/command-binding.json').read_text())
    buildrec=json.loads((BASE/'analysis/source157-telemetry-native-build-001/build-metadata/source-build.json').read_text())
    # argv is an array in these command files; locate the argument after '--'.
    def after_dashdash(argv):
        i=argv.index('--'); return argv[i+1]
    exe3=after_dashdash(c3); exe4=after_dashdash(c4)
    build_artifacts=buildrec.get('artifacts_sha256',{})
    expected_exe_hash=build_artifacts.get(after_dashdash(c4))
    exe_bytes, exe_hash=hash_file(Path(exe4)) if Path(exe4).exists() else (None,None)
    cmd3_bytes,cmd3_sha=hash_file(cmd3)
    cmd4_bytes,cmd4_sha=hash_file(cmd4)
    cmd3_manifest=json.loads((EVID/'analysis/native468-source157-dt-half-cycle-command-003/manifest.json').read_text())
    cmd4_manifest=json.loads((EVID/'analysis/native468-source157-dt-half-cycle-command-004/manifest.json').read_text())
    check('command003_refusal_preserved', cmd3_sha==refuse.get('command_sha256')==cmd3_manifest.get('command_sha256') and refuse.get('native_launched') is False and refuse.get('case_created') is False and refuse.get('launch_guard_created') is False and refuse.get('exit_code')==1 and refuse.get('native_executable')==exe3,
          {'command003_executable':exe3,'refusal':{k:refuse.get(k) for k in ('exit_code','native_launched','case_created','launch_guard_created','command_sha256')}})
    check('command004_executable_exact_binding', cmd4_sha==cmd4_manifest.get('command_sha256') and bind4.get('source_build_provenance_sha256')==hash_file(BASE/'analysis/source157-telemetry-native-build-001/build-metadata/source-build.json')[1] and exe_hash==expected_exe_hash and exe4==bind4.get('argv_after_separator_executable') and exe4.endswith('/build-cardboard-provenance-157/numi-matter-cardboard-probe') and exe_hash==bind4.get('source157_expected_executable_sha256')==bind4.get('source157_executable_sha256') and bind4.get('all_external_inputs_hash_verified') is True,
          {'command004_executable':exe4,'bytes':exe_bytes,'actual_sha256':exe_hash,'binding_sha256':bind4.get('source157_expected_executable_sha256'),'build_record_artifact':expected_exe_hash})
    # Ensure the live solver run itself and guard are not in published/cached pathspecs.
    forbidden_tokens=['/solver/cgls-dt-half-cycle-468','/native468-launch-guard','launch-guard-468']
    pathspec_text='\n'.join(list028); staged_text='\n'.join(staged_expected)
    no_active=all(tok not in pathspec_text and tok not in staged_text for tok in forbidden_tokens)
    check('active468_run_and_guard_excluded', no_active,
          {'forbidden_tokens':forbidden_tokens,'active_case_records_in_summary':sum('cgls-dt-half-cycle-468' in str(r) for r in summary['records'][1460:])})

    # Metallib evidence gap is explicit and remains an unresolved historic operation.
    gapdir=EVID/'analysis/metallib-apfs-clone-evidence-gap-001'
    gap_manifest=json.loads((gapdir/'manifest.json').read_text())
    gap_report=gzip.decompress((gapdir/'report.json.gz').read_bytes())
    gap=json.loads(gap_report)
    gap_readme=(gapdir/'README.md').read_text()
    check('metallib_history_gap_explicit', 'not independently reproducible' in gap_manifest.get('evidence_status','') and 'not a reproducible proof' in gap_readme and 'unverified' in gap_readme.lower(),
          {'evidence_status':gap_manifest.get('evidence_status'),'report_sha256':sha_bytes(gap_report),'report_status':gap.get('evidence_status') or gap.get('verdict')})

    # Source verification package count and exact no-qualification boundary.
    source_report=json.loads((EVID/'analysis/source157-root-integration-001/report.json').read_text())
    build_record_path=BASE/'analysis/source157-telemetry-native-build-001/build-metadata/source-build.json'
    build_record=json.loads(build_record_path.read_text())
    source_root=Path(build_record['source_snapshot'])
    source_map=build_record.get('source_sha256',{})
    source_mismatches=[]; source_verified=0
    for rel, expected_sha in source_map.items():
        fp=source_root/rel
        if not fp.is_file(): source_mismatches.append({'path':rel,'error':'missing'}); continue
        nn,hh=hash_file(fp)
        if hh!=expected_sha: source_mismatches.append({'path':rel,'bytes':nn,'sha256':hh,'expected':expected_sha})
        else: source_verified+=1
    check('source157_787_hashes', len(source_map)==787 and source_verified==787 and not source_mismatches and source_report.get('selected_source_hashes_verified')==787 and source_report.get('mismatches')==[] and source_report.get('physical_qualification') is False,
          {'source_map_entries':len(source_map),'independently_verified':source_verified,'mismatches':source_mismatches[:20],'source_report_count':source_report.get('selected_source_hashes_verified'),'physical_qualification':source_report.get('physical_qualification')})

    # Staging set, worktree/index byte equality and unrelated untracked files.
    stage_actual=git(['diff','--cached','--name-only','-z']).split(b'\0')
    stage_actual=[x.decode('utf-8','surrogateescape') for x in stage_actual if x]
    stage_hash=sha_bytes(raw_stage_list)
    check('staged_list_hash_and_count', stage_hash==EXPECTED['stage_paths_sha256'] and len(staged_expected)==722 and set(stage_actual)==set(staged_expected),
          {'sha256':stage_hash,'count_expected_file':len(staged_expected),'count_git_index':len(stage_actual),'git_only':sorted(set(stage_actual)-set(staged_expected))[:20],'list_only':sorted(set(staged_expected)-set(stage_actual))[:20]})
    staged_outside_publication=sorted(set(staged_expected)-set(list028))
    expected_staged_integration=[
      'matter/examples/cardboard/implementation-20261014.md',
      'matter/examples/cardboard/solver-milestone-source157.md',
      'matter/examples/cardboard/solver-progress-20261014.md',
      'matter/tools/cardboard_probe.mm'
    ]
    check('staged_integration_paths_outside_evidence_pathspec', staged_outside_publication==expected_staged_integration,
          {'count':len(staged_outside_publication),'paths':staged_outside_publication})
    staged_rows=git(['ls-files','--stage','-z']).split(b'\0')
    index_map={}
    for item in staged_rows:
        if not item: continue
        meta,path=item.split(b'\t',1)
        mode,oid,stage=meta.decode().split()
        if int(stage)!=0: failures.append('index_stage_conflict'); continue
        index_map[path.decode('utf-8','surrogateescape')]=(mode,oid)
    byte_mismatches=[]; staged_n=0; compared_blob_bytes=0
    for p in staged_expected:
        staged_n += 1
        if p not in index_map:
            byte_mismatches.append({'path':p,'error':'missing_index_entry'}); continue
        mode,oid=index_map[p]
        fs=ROOT/p
        if not fs.exists() and not fs.is_symlink():
            byte_mismatches.append({'path':p,'error':'missing_worktree'}); continue
        equal,n=stream_equal_blob(fs,oid)
        compared_blob_bytes += n
        if not equal: byte_mismatches.append({'path':p,'error':'index_blob_differs_from_worktree','index_oid':oid,'bytes_compared':n})
    check('all_staged_bytes_equal_worktree', not byte_mismatches and staged_n==722,
          {'files_compared':staged_n,'bytes_compared':compared_blob_bytes,'mismatches':byte_mismatches[:30],'mismatch_count':len(byte_mismatches)})
    unstaged=git(['diff','--name-only','-z']).split(b'\0'); unstaged=[x.decode('utf-8','surrogateescape') for x in unstaged if x]
    check('no_unstaged_tracked_changes', not unstaged, {'count':len(unstaged),'paths':unstaged[:30]})
    untracked=git(['ls-files','--others','--exclude-standard','-z']).split(b'\0'); untracked=[x.decode('utf-8','surrogateescape') for x in untracked if x]
    expected_untracked={
       'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/CMakeOBJCXXCompilerId.mm',
       'matter/tools/evidence/cohesive-scratch-alias/source-audit.json'
    }
    check('unrelated_untracked_preserved_outside_stage', set(untracked)==expected_untracked and not (set(untracked)&set(staged_expected)),
          {'untracked_count':len(untracked),'paths':untracked,'expected':sorted(expected_untracked),'staged_overlap':sorted(set(untracked)&set(staged_expected))})
    staged_status=git(['diff','--cached','--name-status']).decode().splitlines()
    deleted=[line for line in staged_status if line.startswith('D\t')]
    check('no_staged_deletions', not deleted, {'deleted':deleted[:30]})

    report={
      'schema':'numi.cardboard.publication028.independent_byte_audit.v1',
      'repo':str(ROOT),'head':git(['rev-parse','HEAD']).decode().strip(),
      'binding_path':str(binding_path),'binding_sha256':bhash,
      'summary_sha256':shash,'summary_records':len(summary.get('records',[])),
      'new_cases':len(case_names),'payloads':{'total':len(file_rows),'raw':raw_ok,'gzip':gz_ok,'copy_bytes':total_copy_bytes},
      'pathspec':{'027_entries':len(list027),'028_entries':len(list028),'new_entries':len(delta),'old_prefix_exact':prefix_exact,'028_sha256':sha_bytes(raw028)},
      'source157':{'source_map_entries':len(source_map),'source_files_verified':source_verified,'mismatch_count':len(source_mismatches)},
      'external_recovery':{'index_sha256':idx_sha,'pointer_count':len(pointer_rows),'unique_indexed_files':len(dedup),'pointer_copies_verified':pointer_copies_ok,'pointer_sources_present':source_pointers_present},
      'staging':{'paths':len(staged_expected),'stage_list_sha256':stage_hash,'outside_evidence_pathspec':staged_outside_publication,'worktree_byte_equal':not byte_mismatches,'compared_bytes':compared_blob_bytes,'unstaged_tracked':len(unstaged),'untracked_paths':untracked},
      'checks':checks,'failures':failures,'verdict':'PASS' if not failures else 'FAIL'
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'verdict':report['verdict'],'failures':failures,'payloads':report['payloads'],'staged':report['staging'],'checks':{k:v['pass'] for k,v in checks.items()}},indent=2))
    return 0 if not failures else 1

if __name__ == '__main__':
    raise SystemExit(main())
