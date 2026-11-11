from pathlib import Path
import gzip,hashlib,json,subprocess
repo=Path('/tmp/crow-restore-check-20261110')
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
prefix='artifacts/crow-v15-start-perturbation-controls-20261111/'
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','FETCH_HEAD'],text=True).strip()
manifest=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+prefix+'SHA256SUMS'],text=True)
expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in manifest.splitlines()}
rows=[]
for name in ('single','none'):
 for kind in ('replay.json','trace.tsv'):
  relative=name+'-20261126/'+kind+'.gz'
  blob=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+prefix+relative])
  compressed_sha=hashlib.sha256(blob).hexdigest()
  assert compressed_sha==expected[relative],relative
  restored=gzip.decompress(blob)
  raw=root/'build/crow-v15-start-perturbation-controls-20261111'/(name+'-20261126-'+kind)
  rawsha=hashlib.sha256(raw.read_bytes()).hexdigest()
  assert rawsha==hashlib.sha256(restored).hexdigest(),relative
  rows.append({'committed_compressed_file':prefix+relative,'compressed_sha256':compressed_sha,'raw_duplicate':str(raw),'raw_sha256':rawsha,'bytes':len(restored),'restore_matches':True})
out={'remote':'git@github.com:Numi2/numi-lab.git','fetched_commit':commit,'checks':rows,'scope':'Fetched into an independent filtered clone from GitHub; verified committed manifest bytes and decompressed payload against existing build outputs.'}
Path('artifacts/crow-v15-start-perturbation-archive-check-20261111/restore-check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
