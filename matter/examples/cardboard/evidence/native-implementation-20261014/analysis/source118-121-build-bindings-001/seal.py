from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');O=B/'analysis/source118-121-build-bindings-001';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
for n in ['118','118-full','119','120','121']:
 T=R/('build-cardboard-provenance-'+n);p=T/'source-build.json';j=json.loads(p.read_text());assert j['success'];artifacts=[]
 for name,h in j['artifacts_sha256'].items():
  a=Path(name);assert sha(a)==h,a;artifacts.append(dict(path=str(a),sha256=h,bytes=a.stat().st_size))
 for name,h in j['build_metadata_sha256'].items():assert sha(T/name)==h,name
 d=O/n;d.mkdir();shutil.copy2(p,d/'source-build.json')
 for p in sorted(T.glob('provenance-command-*.log')):shutil.copy2(p,d/p.name)
 rows.append(dict(build=n,receipt_sha256=sha(T/'source-build.json'),selected_sources=len(j['source_sha256']),artifacts=artifacts,success=True))
(O/'report.json').write_text(json.dumps(dict(schema='numi.cardboard.source_build_bindings.v1',builds=rows,source118full_cpu='74selectedPASS; zero physicalsteps',source119_cpu='74selectedPASS; zero physicalsteps',source121='3native diagnostic targetsPASS; newC1hosttracecapacity safety and summaryserializer must independently qualify native captures',physical_box_qualification=False),indent=2)+'\n');(O/'seal.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2)+'\n');print('PASS 5 build receipts and all retained compiled artifacts/metadata')
