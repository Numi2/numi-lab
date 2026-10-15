from pathlib import Path
import ast, datetime, hashlib, json, subprocess, sys

B=Path('/Users/home/cardboard-implementation-20261014')
S=B/'analysis/dimensionless-material674-point-390-strict-reference-addendum-001'
V=B/'analysis/cardboard-source380-point-regression-verifier-001'
O=B/'analysis/dimensionless-material674-root-verification-003'
O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for folder in (S,V) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
# Redirect only generated output files, preserving source and the sealed package.
tree=ast.parse((S/'strict_reference.py').read_text())
replaced=0
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PKG' for t in n.targets):
  n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(O))],keywords=[]);replaced+=1
assert replaced==1
sys.path.insert(0,str(S))
with (O/'reference.log').open('x') as f:
 old=sys.stdout;sys.stdout=f
 try:exec(compile(ast.fix_missing_locations(tree),str(S/'strict_reference.py'),'exec'),{'__name__':'__main__','__file__':str(S/'strict_reference.py')})
 finally:sys.stdout=old
for name in ('report.json','fp64-reference-vectors-674.json'):
 assert (O/name).read_bytes()==(S/name).read_bytes(),name
cmd=[sys.executable,'-m','unittest','-v','test_verify_point']
with (O/'verifier-unittest.log').open('x') as f:
 code=subprocess.call(cmd,cwd=V,stdout=f,stderr=subprocess.STDOUT,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
assert code==0
assert before=={str(p):sha(p) for folder in (S,V) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(O/'report-verification.json').write_text(json.dumps(dict(schema='numi.cardboard.point390.root-reexecution.v1',finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),strict_reference_report_and_vectors_byte_identical=True,opt_in_verifier_unittests_passed=6,source_packages_unchanged=True,source_files_sha256=before,physical_steps=0,physical_validation=False),indent=2)+'\n')
(O/'run.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()},indent=2)+'\n')
print('strict reference and all six verifier tests passed')
