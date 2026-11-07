import hashlib,json,pathlib,subprocess,time
p=pathlib.Path(__file__).resolve().parent
root=p.parents[2]
binary='/tmp/numi_contact_wave_20261107'
def digest(f): return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
manifest={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
  'source_sha256':digest(root/'tests/contact/fem_standing_wave_diagnostic.cpp'),
  'model_sha256':digest(root/'tests/contact/fem_plane_surface_diagnostic.cpp'),
  'binary_sha256':digest(binary),'timeout_seconds_per_run':180,'runs':[]}
for n in (8,16):
 for mode in (2,8):
  for h in ('0.000002','0.000001'):
   amps=('0.00000001','0.000000005') if (n,mode,h)==(16,8,'0.000001') else ('0.00000001',)
   for amp in amps:
    name=f'n{n}_m{mode}_h{float(h)*1e6:g}us_a{float(amp)*1e9:g}nm'
    cmd=[binary,str(n),str(mode),h,'0.01',amp]
    start=time.monotonic()
    with open(p/(name+'.csv'),'wb') as out, open(p/(name+'.stderr'),'wb') as err:
     try:
      result=subprocess.run(cmd,stdout=out,stderr=err,timeout=180)
      status=result.returncode
     except subprocess.TimeoutExpired: status='timeout'
    manifest['runs'].append({'name':name,'argv':cmd,'status':status,'elapsed_seconds':time.monotonic()-start,'output_sha256':digest(p/(name+'.csv'))})
    (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(name,status,manifest['runs'][-1]['elapsed_seconds'],flush=True)
    if status!=0: raise SystemExit('retained failure; stopping')
