import hashlib,json,pathlib,subprocess,urllib.request,zlib
root=pathlib.Path(open('/tmp/crow-continuation-path').read().strip())/'repo';pre=json.load(open('/tmp/crow-plain-preflight.json'));token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for row in pre['rows']:
 path=pathlib.PurePosixPath(row['github_path']);manifest=str(path.parent/'SHA256SUMS');data=subprocess.check_output(['git','show',row['commit']+':'+manifest],cwd=root,text=True);expected=next(line.split()[0] for line in data.splitlines() if line.endswith('  '+path.name));url='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(path)+'?ref='+row['commit'];req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});compressed=hashlib.sha256();plain=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=90) as f:
  while True:
   b=f.read(1048576)
   if not b:break
   compressed.update(b);out=z.decompress(b);n+=len(out);plain.update(out)
 assert z.eof and not z.unused_data and compressed.hexdigest()==expected and n==row['bytes'] and plain.hexdigest()==row['sha256'],row['path'];rows.append({**row,'github_gzip_sha256':expected,'restored_sha256':plain.hexdigest(),'gzip_eof_verified':True});print('verified',row['path'],flush=True)
pathlib.Path('/tmp/crow-plain-verification.json').write_text(json.dumps({'rows':rows,'scope':'Independently streamed exact immutable GitHub gzip blobs; compressed and restored hashes match; no deletion'},indent=2)+'\n')
