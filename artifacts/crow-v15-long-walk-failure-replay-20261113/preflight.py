from pathlib import Path
import hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-long-walk-failure-replay-20261113';P=R/'artifacts/crow-v15-walk-hold-slower-cycle-20261111/autonomous-20261126';receipt=json.loads((P/'inputs.json').read_text());cmd=json.loads((P/'command.json').read_text());paths={'host_sha256':R/'build/crow-v15-autonomous-flight-fold-isolation-20261111/libmetalrobo.dylib','metallib_sha256':Path(cmd[cmd.index('--metallib')+1]),'frozen_flight_sha256':R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal','landing_policy_sha256':Path(cmd[cmd.index('--policy-pack')+1]),'route_policy_sha256':Path(cmd[cmd.index('--base-policy-pack')+1])}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={k:{'path':str(v),'expected':receipt[k],'actual':sha(v),'match':sha(v)==receipt[k]} for k,v in paths.items()}
part=P/'run.stderr.gz.part000';manifest=json.loads((P/'stderr-parts.json').read_text());checks['original_log']={'path':str(part),'actual':sha(part),'expected':manifest['compressed_sha256'],'match':sha(part)==manifest['compressed_sha256'] and part.stat().st_size==manifest['bytes']}
(A/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2));assert all(x['match'] for x in checks.values())
