from pathlib import Path
import hashlib
import json
import shlex
import subprocess

B=Path('/Users/home/cardboard-implementation-20261014')
S=B/'snapshots/source-right-grip-telemetry-157'
O=B/'analysis/source163-rejected-constitutive-export-overlay-001'
old=O/'validation-attempt002'
(O/'validation').rename(old)
(old/'validator.py').write_bytes((B/'validate_source163_attempt002.py').read_bytes())
(old/'runner.stdout').write_bytes((B/'source163-validation002.stdout').read_bytes())
(old/'runner.stderr').write_bytes((B/'source163-validation002.stderr').read_bytes())
(old/'failure.json').write_text(json.dumps({'exit_code':1,'stage':'standalone candidate probe syntax command',
    'error':'Candidate file is outside the frozen source tools directory, so quoted cardboard_glue_mesh.hpp include was not found. The compile command needed the frozen tools include directory.',
    'writer_checks_completed_before_failure':4,'native_run':False,'candidate_changed':False},indent=2)+'\n')
V=O/'validation'
V.mkdir()
checks=[]
for opt in ['0','3']:
    for kind in ['full','empty']:
        directory=old/('output-O'+opt+'-'+kind)
        metadata=json.loads((directory/'manifest.json').read_text())
        names=['candidate_nodes.bin','linearization_accepted_material_history.bin',
               'linearization_scheduler_exponents.bin','linearization_F_high.bin','linearization_F_low.bin']
        sizes=[80,16,12,48,48] if kind=='full' else [0]*5
        assert [r['name'] for r in metadata['files']]==names
        for row,expected in zip(metadata['files'],sizes):
            payload=(directory/row['name']).read_bytes()
            assert len(payload)==row['bytes']==expected
            assert hashlib.sha256(payload).hexdigest()==row['sha256']
            assert payload==(old/('output-O0-'+kind)/row['name']).read_bytes()
        checks.append({'optimization':'O'+opt,'fixture':kind,'arrays':5,
                       'all_bytes_and_sha256_verified':True,'source_directory':str(directory)})
cc=json.loads(Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157/compile_commands.json').read_text())
rows=[r for r in cc if r['file'].endswith('/matter/tools/cardboard_probe.mm')
      and 'numi-matter-cardboard-probe.dir/' in r.get('output','')]
assert len(rows)==1
clean=[]
skip=False
for arg in shlex.split(rows[0]['command']):
    if skip:
        skip=False
        continue
    if arg=='-o': skip=True
    elif arg=='-c': clean.append('-fsyntax-only')
    elif arg==rows[0]['file']: clean.append(str(O/'candidate/matter/tools/cardboard_probe.mm'))
    else: clean.append(arg)
clean.extend(['-I'+str(S/'matter/tools')])
commands=[]
for args,name in [(clean,'probe-syntax'),
    (['patch','--dry-run','--batch','-p1','-i',str(O/'overlay.patch'),'-d',str(S)],'patch-dry-run')]:
    q=subprocess.run(args,capture_output=True,text=True)
    (V/(name+'.stdout')).write_text(q.stdout)
    (V/(name+'.stderr')).write_text(q.stderr)
    commands.append({'argv':args,'exit_code':q.returncode})
    assert q.returncode==0,(name,q.stderr)
report={'schema':'numi.cardboard.rejected_constitutive_export_cpu_validation.v1',
    'checks':checks,'writer_extracted_from_candidate':True,'probe_syntax_exit_code':0,
    'patch_dry_run_exit_code':0,'extra_gpu_readback_bytes':0,'native_physics_transactions':0,
    'failed_validation_setup_attempts_preserved':['validation-attempt001','validation-attempt002'],
    'limits':['Source-derived actual writer exercised with synthetic bytes, plus full probe TU syntax only; no app build or native capture parity.',
              'Exported history is accepted history at AP linearization, not current projected constitutive state.',
              'Exact Newton correction and perturbed projected states remain unavailable in this export.']}
(V/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(V/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(O/'finalize.py').write_bytes(Path(__file__).read_bytes())
(O/'README.md').write_text('# Preserve rejected AP constitutive inputs (source163)\n\n'
 'The probe writes four arrays already present in its bounded RuntimeStateSnapshot: accepted history at AP linearization, '
 'scheduler rate exponents, and material-local F high/low words. The prior writer exported candidate nodes and AP/RAP while '
 'dropping these arrays at process exit. New metadata labels accepted history explicitly and marks exact Newton corrections '
 'and perturbed projected states unavailable. Old failures cannot be retroactively filled.\n\n'
 'No solver, target generation, constitutive program or Metal source change; zero added GPU readback bytes. '
 'Runtime already copies and size-validates the vectors under its regional64MiB snapshot bound. The overlay only adds disk export.\n\n'
 'Four CPU O0/O3 full/empty fixtures exercise the source-derived actual production writer and verify emitted byte counts and SHA-256 values; '
 'O0/O3 output bytes match. Full probe TU syntax and source157-relative patch dry-run pass. No native build, capture or trajectory parity is claimed. '
 'Two validation setup failures are retained: ambiguous probe/box-driver compile commands, then a missing include path for an external candidate file. '
 'They did not change native evidence or the candidate patch.\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'schema':'numi.cardboard.rejected_constitutive_export_overlay.v1','base_snapshot':str(S),
    'base_probe_sha256':sha(S/'matter/tools/cardboard_probe.mm'),
    'candidate_probe_sha256':sha(O/'candidate/matter/tools/cardboard_probe.mm'),
    'patch_sha256':sha(O/'overlay.patch'),'physics_changed':False,'native_run':False,
    'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
# Include fixture manifests too; only the top-level manifest is excluded.
manifest['files']={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*'))
                   if p.is_file() and p!=O/'manifest.json'}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(O/'manifest.json'),'probe_syntax':True,'writer_checks':len(checks)}))
