"""Independently download the published archive, verify all payload hashes, and restore core files."""
from pathlib import Path
import subprocess,json,hashlib,tarfile
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');P=ROOT/'work/plastic_bag_empty_preservation_20261024';V=P/'download_verification';V.mkdir(parents=True,exist_ok=True)
subprocess.run(['/opt/homebrew/bin/gh','release','download','plastic-bag-empty-franka-2026-10-24','--repo','Numi2/numi-lab','--dir',str(V),'--pattern','numi_plastic_bag_empty_franka_20261024.tar.gz','--pattern','SHA256SUMS'],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=V/'numi_plastic_bag_empty_franka_20261024.tar.gz';expected=(V/'SHA256SUMS').read_text().split()[0];assert sha(a)==expected
with tarfile.open(a) as t:
 m=json.load(t.extractfile('empty_20261024/manifest_sha256.json'));verified=0
 for name,h in m.items():
  member=t.getmember('empty_20261024/'+name);assert member.isfile();assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==h,name;verified+=1
 restored=[]
 for name in ['solver/source/PlasticBag.metal','solver/source/plastic_bag_coins_scene.mm','solver/native_runtime/numi_classic_bag_drop','solver/grip.csv','evidence/combined/latest_checkpoint.bin']:
  data=t.extractfile('empty_20261024/'+name).read();p=V/'restored'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==m[name];restored.append(name)
r={'release':'https://github.com/Numi2/numi-lab/releases/tag/plastic-bag-empty-franka-2026-10-24','independently_downloaded':True,'archive_sha256':expected,'all_manifest_files_verified':True,'verified_payload_files':verified,'restoration_hash_verified':restored,'local_evidence_retired':False}
(P/'github_download_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
