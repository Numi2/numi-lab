from pathlib import Path
import hashlib,json,shutil,sys
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-witness-20261108';B=R/'build/crow-line-search-witness-build-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
entries=archive.tree();P=B/'reference';P.mkdir();runtime=B/'runtime';(runtime/'metalrobo').mkdir(parents=True)
with archive.Objects() as objects:
 m=json.loads(objects.read(entries['artifacts/crow-rl-quality-refinement-20261108/payload/manifest.json'][1]))
 for f in m['files']:
  src=f['path'];dest=None
  if src.endswith('/reference/crow-fixture'):dest=P/'crow-fixture'
  if src.endswith('/integration-observations/strict-capture.bin'):dest=P/'strict-gold.bin'
  if '/integration-observations/api-0/' in src and not src.endswith('runtime.json'):dest=P/'api-gold'/Path(src).name
  if dest:
   raw=b''.join(archive.content(f,entries,objects));assert hashlib.sha256(raw).hexdigest()==f['sha256']
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);dest.chmod(f['mode'])
assert (P/'crow-fixture').is_file() and (P/'strict-gold.bin').is_file() and len(list((P/'api-gold').iterdir()))>=10
for src,dest in [('build/lib/libmetalrobo.dylib',runtime/'libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib',runtime/'metalrobo/MetalRobo.metallib'),
 ('build/bin/metalrobo_metal_unified_quality_probe',runtime/'quality-owner'),('build/bin/metalrobo_metal_world_contact_probe',runtime/'contact-owner')]:shutil.copy2(R/src,dest)
seal={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for root in [P,runtime] for p in root.rglob('*') if p.is_file()}
(A/'runtime.json').write_text(json.dumps(dict(files=seal),indent=2,sort_keys=True)+'\n')
