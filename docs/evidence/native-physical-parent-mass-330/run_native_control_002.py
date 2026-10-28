from pathlib import Path
import json,subprocess,hashlib,time,os,gzip,shutil
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-prepared-identity-replacement-326')
old=S/'attempts/008-parent-reference-identity-control'
new=Path(__file__).parent/'native-source-identity-control-002'
new.mkdir()
launch=json.loads((old/'launch.json').read_text())
def change(v):
 if isinstance(v,str):return v.replace(str(old),str(new))
 if isinstance(v,list):return [change(x) for x in v]
 if isinstance(v,dict):return {k:change(x) for k,x in v.items()}
 return v
launch=change(launch)
repo=Path('/Users/numi/Work/skin-fat-native-integration-209')
binary=repo/'build-native/bin/numi-human-native'
frozen=new/'numi-human-native';shutil.copy2(binary,frozen);launch['argv'][0]=str(frozen)
(new/'output').mkdir()
(new/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={}
source=new/'source';source.mkdir()
for f in ['NumiHumanPhysicalSkinChildIntegration.hpp','NumiHumanPhysicalSkinChildMassMapping.hpp',
          'NumiHumanPhysicalSkinChildMassMappingCheck.cpp','NumiHumanPhysicalSkinReplacement.hpp','NumiHumanPhysicalSkinReplacementCheck.cpp','numilab_human_myosim_visual_probe.mm']:
 data=(repo/'apps'/f).read_bytes()
 (source/(f+'.gz')).write_bytes(gzip.compress(data,mtime=0))
 sources[f]=hashlib.sha256(data).hexdigest()
manifest=dict(binary_sha256=sha(frozen),source_sha256=sources,
 shaders_sha256={p.name:sha(p) for p in [
  repo/'build-native/matter/shaders/NumiMatter.metallib',
  repo/'build-native/matter/shaders/HumanRespiration.metallib',
  repo/'build-native/shaders/MetalRobo.metallib']})
(new/'launch-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
t=time.perf_counter()
try:
 r=subprocess.run(launch['argv'],cwd=repo,env=launch['environment'],capture_output=True,timeout=60)
 code=r.returncode;out=r.stdout;err=r.stderr;timed_out=False
except subprocess.TimeoutExpired as e:
 code=None;out=e.stdout or b'';err=e.stderr or b'';timed_out=True
(new/'stdout.txt').write_bytes(out);(new/'stderr.txt').write_bytes(err)
text=out.decode(errors='replace')
result=dict(exit_code=code,timed_out=timed_out,wall_s=time.perf_counter()-t,
 source_identity_import_pass='whole_body_skin_fat_physical_child_identity_import=PASS' in text,
 mass_mapping_pass='whole_body_skin_fat_physical_child_identity_mass_mapping=PASS' in text,
 source_identity_remap_pass='whole_body_skin_fat_unpaired_identity_remap=PASS' in text,
 source_prepared_replacement_pass='whole_body_skin_fat_unpaired_identity_install=PASS' in text,
 source_capture_only='skin_fat_parent_mass_capture_only=PASS' in text,
 physical_steps=0,repaired_volume_admitted=False,motion_qualified=False,**manifest)
(new/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
print('\n'.join(line for line in text.splitlines() if line.startswith('whole_body_skin_fat_physical_child_identity') or line.startswith('whole_body_skin_fat_unpaired_identity') or line.startswith('skin_fat_parent_mass_capture_only')))
print(err.decode(errors='replace')[-1500:])
