#!/usr/bin/env python3
"""Restore verified evidence to a chosen directory; never overwrite."""
import argparse, hashlib, json, os, shutil, subprocess, tempfile, urllib.parse, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--destination',required=True);p.add_argument('--path',action='append',help='exact archived path; omit to restore all verified records');a=p.parse_args()
root=Path(a.destination).resolve();root.mkdir(parents=True,exist_ok=True);records=[json.loads(line) for line in Path(a.manifest).read_text().splitlines()];records=[r for r in records if r.get('status')=='verified' and (not a.path or r['path'] in a.path)]
if a.path and {r['path'] for r in records}!=set(a.path):raise SystemExit('Requested path not found in verified manifest')
token=os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
if not token:
 try:token=subprocess.check_output(['gh','auth','token'],text=True).strip()
 except Exception:raise SystemExit('Set GITHUB_TOKEN or authenticate gh')
for r in records:
 rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
 dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.parent.resolve().is_relative_to(root):raise RuntimeError('Destination parent escapes root')
 if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
 url='https://raw.githubusercontent.com/Numi2/numi-lab/'+r['commit']+'/'+urllib.parse.quote(r['path']);req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token});h=hashlib.sha256();size=0
 fd,tmp=tempfile.mkstemp(prefix='.crow-restore-',dir=dest.parent)
 try:
  with os.fdopen(fd,'wb') as out,urllib.request.urlopen(req,timeout=120) as src:
   while b:=src.read(1048576):out.write(b);h.update(b);size+=len(b)
  if h.hexdigest()!=r['sha256'] or size!=r['bytes']:raise ValueError('Restored bytes failed verification: '+r['path'])
  os.chmod(tmp,r['mode'] & 0o7777);os.utime(tmp,ns=(r['mtime_ns'],r['mtime_ns']))
  os.link(tmp,dest);os.unlink(tmp)
  print('Restored',r['path'],flush=True)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
