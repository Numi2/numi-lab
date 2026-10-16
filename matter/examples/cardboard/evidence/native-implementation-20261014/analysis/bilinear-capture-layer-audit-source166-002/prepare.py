from pathlib import Path
import difflib
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
P = B / 'analysis/bilinear-capture-layer-audit-source166-001'
O = B / 'analysis/bilinear-capture-layer-audit-source166-002'
O.mkdir(exist_ok=False)
shutil.copytree(P/'candidate',O/'candidate',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(P/'reference-provenance.json',O/'reference-provenance.json')
path = O/'candidate/matter/tools/cardboard_bilinear_capture_layers.py'
source = path.read_text().replace('import json\n','import json\nimport math\n',1)
helper = '''def finite_report(value, path='$'):
    """Keep every unresolved scalar visible while writing strict JSON.

    Raw numerical arrays retain NaN/Inf in the NPZ. JSON uses null plus an
    explicit location/value ledger, so a failed reference projection cannot
    silently disappear or prevent the remaining diagnostic from being saved.
    """
    if isinstance(value,float) and not math.isfinite(value):
        return None,[{'path':path,'value':repr(value),'classification':'unresolved_nonfinite_diagnostic'}]
    if isinstance(value,dict):
        result={};missing=[]
        for key,item in value.items():
            child_path=path+'/'+str(key).replace('~','~0').replace('/','~1')
            result[key],rows=finite_report(item,child_path);missing.extend(rows)
        return result,missing
    if isinstance(value,(list,tuple)):
        result=[];missing=[]
        for index,item in enumerate(value):
            clean,rows=finite_report(item,path+'/'+str(index));result.append(clean);missing.extend(rows)
        return result,missing
    return value,[]

'''
source = source.replace('def project(p,F,old):',helper+'def project(p,F,old):',1)
needle = "    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\\n')"
assert source.count(needle)==1
source = source.replace(needle,"    report,nonfinite=finite_report(report)\n    report['nonfinite_diagnostic_fields']=nonfinite\n"+needle)
path.write_text(source)
test = O/'candidate/matter/tools/test_cardboard_bilinear_capture_layers.py'
source = test.read_text()
needle = 'class CaptureLayerContracts(unittest.TestCase):\n'
assert source.count(needle)==1
source = source.replace(needle,needle+'''    def test_nonfinite_report_fields_retain_explicit_locations_and_do_not_change_gates(self):
        import json
        raw={'nested':[1.25,float('nan'),{'a/b':float('inf'),'~key':-float('inf')}],
             'all_rows_at_relative_gate':False,'failed_rows':3,'optional':None}
        clean,missing=layer.finite_report(raw)
        self.assertEqual(clean['nested'],[1.25,None,{'a/b':None,'~key':None}])
        self.assertEqual([r['path'] for r in missing],['$/nested/1','$/nested/2/a~1b','$/nested/2/~0key'])
        self.assertEqual([r['value'] for r in missing],['nan','inf','-inf'])
        self.assertFalse(clean['all_rows_at_relative_gate'])
        self.assertEqual(clean['failed_rows'],3)
        self.assertIsNone(clean['optional'])
        json.dumps(clean,allow_nan=False)

''')
test.write_text(source)
(O/'README.md').write_text('''# FP64 capture reader: explicit unresolved report values

This supersedes source166-001 after native469 replay001 computed and retained its
NPZ but failed strict JSON serialization on a nonfinite reference intermediate.
The failed command, stderr, receipt and partial arrays remain unchanged. The
constitutive equations, projection iteration policy and comparison thresholds
are unchanged. Every NaN/Inf JSON value becomes null with its exact field path
and original value in `nonfinite_diagnostic_fields`; raw NPZ arrays are retained.
This is a reporting fix and cannot turn an inconclusive row into a passed gate.
''')
patch=''
for path in sorted((O/'candidate/matter/tools').glob('*.py')):
    rel=str(path.relative_to(O/'candidate'))
    patch+=''.join(difflib.unified_diff([],path.read_text().splitlines(keepends=True),
                                      fromfile='/dev/null',tofile='b/'+rel))
(O/'overlay.patch').write_text(patch)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'source-overlay.json').write_text(json.dumps(dict(
    schema='numi.cardboard.source-overlay.v1',base_snapshot=str(B/'snapshots/source-right-grip-telemetry-157'),
    overlay_id=O.name,patch_sha256=sha(O/'overlay.patch'),
    supersedes=dict(package=str(P),manifest_sha256=sha(P/'manifest.json'),
                    failed_command=str(B/'analysis/native469-fp64-replay-command-001')),
    files=[dict(path=str(p.relative_to(O/'candidate')),base_sha256=None,overlay_sha256=sha(p))
           for p in sorted((O/'candidate/matter/tools').glob('*.py'))],
    native_execution_performed=False,qualification=False),indent=2)+'\n')
shutil.copy2(Path(__file__),O/'prepare.py')
print(O)
