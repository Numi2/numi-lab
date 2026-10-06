import datetime,hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path('/Users/home/cardboard-gap-evidence-20261006')
repo=pathlib.Path('/Users/home/numi-cardboard-20261006')
name=sys.argv[1]; args=sys.argv[2:]
if not name or '/' in name: raise SystemExit('invalid trial name')
receipt=root/'receipts'/name
receipt.mkdir(parents=True,exist_ok=False)
binary=repo/'build-cardboard/numi-matter-cardboard-probe'
out=root/name
if out.exists(): raise SystemExit('native output already exists')
argv=[str(binary),'--output',str(out),*args]
preset='literature2009' if '--preset' not in args else args[args.index('--preset')+1]
roles={'liner':repo/'matter/materials'/('hajali2009_liner_hill_ideal.nmatter' if preset=='literature2009' else 'nagasawa2013_liner_orthotropic.nmatter'),
       'medium':repo/'matter/materials'/('hajali2009_medium_hill_ideal.nmatter' if preset=='literature2009' else 'nagasawa2013_medium_orthotropic.nmatter')}
if preset=='literature2009':roles['glue']=repo/'matter/materials/starch2007_finite_glue_elastic.nmatter'
for i,arg in enumerate(args[:-1]):
 if arg in {'--liner-material','--medium-material','--glue-material'}:
  path=pathlib.Path(args[i+1]);roles[arg[2:].split('-')[0]]=path if path.is_absolute() else repo/path
 elif arg=='--material':
  role,value=args[i+1].split('=',1);path=pathlib.Path(value);roles[role]=path if path.is_absolute() else repo/path
files=[binary,repo/'build-cardboard/shaders/NumiMatter.metallib',*roles.values(),repo/'matter/tools/cardboard_glue_mesh.hpp',repo/'matter/tools/cardboard_probe.mm']
files += [repo/'matter/src/metal/fem.metalinc',repo/'matter/src/runtime.mm',repo/'matter/include/numi/matter/matter.hpp']
files += [repo/'matter/tools/cardboard_tooling.hpp',repo/'matter/src/validation_impl.cpp']
files=list(dict.fromkeys(path.resolve() for path in files))
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
record={'schema':'numi.cardboard.native-invocation.v1','argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'purpose':'bounded numerical instrument qualification; no physical validation claim','timeout_seconds':900}
for input_path in files:
 if input_path.suffix in {'.mm','.hpp','.nmatter','.metalinc','.cpp'}:
  (receipt/input_path.name).write_bytes(input_path.read_bytes())
(receipt/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=repo))
record['source_patch_sha256']=hashlib.sha256((receipt/'source.patch').read_bytes()).hexdigest()
(receipt/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
t=time.monotonic()
with (receipt/'stdout.log').open('w') as stdout,(receipt/'stderr.log').open('w') as stderr:
 try:
  p=subprocess.run(argv,stdout=stdout,stderr=stderr,cwd=repo,timeout=900)
  record['return_code']=p.returncode
 except subprocess.TimeoutExpired:
  record['return_code']=None;record['status']='timeout'
record['wall_seconds']=time.monotonic()-t
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['inputs_unchanged']=all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
(receipt/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['return_code','wall_seconds','inputs_unchanged']}))
print((receipt/'stdout.log').read_text()[-5000:])
print((receipt/'stderr.log').read_text()[-5000:])
if record['return_code']!=0 or not record['inputs_unchanged']:sys.exit(1)
