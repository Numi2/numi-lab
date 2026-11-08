"""Independently recover the published handoff archive and verify every payload."""
from pathlib import Path
import subprocess,json,hashlib,tarfile
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');P=ROOT/'work/plastic_bag_handoff_preservation_20261024';V=P/'download_verification';V.mkdir(parents=True,exist_ok=True)
subprocess.run(['/opt/homebrew/bin/gh','release','download','plastic-bag-franka-handoff-2026-10-24','--repo','Numi2/numi-lab','--dir',str(V),'--pattern','numi_plastic_bag_franka_handoff_20261024.tar.gz','--pattern','SHA256SUMS','--pattern','empty_bag_two_franka_handoff_8s_normal.mp4'],check=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
expected={line.split()[1]:line.split()[0] for line in (V/'SHA256SUMS').read_text().splitlines()}
for n,h in expected.items():assert sha(V/n)==h,n
archive=V/'numi_plastic_bag_franka_handoff_20261024.tar.gz'
with tarfile.open(archive) as t:
 m=json.load(t.extractfile('handoff_20261024/manifest_sha256.json'))
 for n,h in m.items():
  member=t.getmember('handoff_20261024/'+n);assert member.isfile();assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==h,n
 restored=[]
 for n in ['solver/source/PlasticBag.metal','solver/source/plastic_bag_coins_scene.mm','solver/native_runtime/numi_classic_bag_handoff','solver/native_runtime/PlasticBag.metallib','solver/grip.csv','solver/grip2.csv','evidence/capture/latest_checkpoint.bin','evidence/finish/latest_checkpoint.bin','evidence/trajectory.json']:
  p=V/'restored'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile('handoff_20261024/'+n).read());assert sha(p)==m[n];restored.append(n)
r={'release':'https://github.com/Numi2/numi-lab/releases/tag/plastic-bag-franka-handoff-2026-10-24','independently_downloaded':True,'all_release_asset_hashes_verified':True,'archive_sha256':sha(archive),'all_manifest_files_verified':True,'verified_payload_files':len(m),'restoration_hash_verified':restored,'local_evidence_retired':False}
(P/'github_download_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
