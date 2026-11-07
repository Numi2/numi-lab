import hashlib,json,pathlib,subprocess,time
p=pathlib.Path(__file__).resolve().parent
root=p.parents[2]; binary='/tmp/numi_momentum_bins_20261107'
def digest(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
m={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
   'source_sha256':digest(root/'tests/contact/fem_momentum_bins_diagnostic.cpp'),
   'model_sha256':digest(root/'tests/contact/fem_plane_surface_diagnostic.cpp'),
   'binary_sha256':digest(binary),'timeout_seconds':180,'runs':[]}
for n in (16,32):
 name=f'n{n}-h1us';cmd=[binary,str(n)];start=time.monotonic()
 with (p/(name+'.csv')).open('wb') as out,(p/(name+'.stderr')).open('wb') as err:
  try:result=subprocess.run(cmd,stdout=out,stderr=err,timeout=180);status=result.returncode
  except subprocess.TimeoutExpired:status='timeout'
 m['runs'].append({'name':name,'argv':cmd,'status':status,'elapsed_seconds':time.monotonic()-start,'output_sha256':digest(p/(name+'.csv'))})
 (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(name,status,m['runs'][-1]['elapsed_seconds'],flush=True)
 if status!=0:raise SystemExit('retained failure; stopping')
