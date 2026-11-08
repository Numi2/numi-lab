from pathlib import Path
import hashlib,json
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-bounded-drive-20261108'
p=G/'recover_git.py';s=p.read_text()
needle="write(G/'temporary-inventory.json',inventory)"
assert s.count(needle)==1
s=s.replace(needle,"assert sum(f['bytes'] for f in inventory)<=plan['temporary_budget_bytes']\n"+needle)
p.write_text(s)
manifest='artifacts/crow-rl-bounded-drive-integration-20261108/payload/manifest.json'
raw=(R/manifest).read_bytes();m=json.loads(raw)
plan={'status':'sealed_before_restored_native_execution','manifest':manifest,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'archive_summary':m['summary'],'restore_max_bytes':m['summary']['source_bytes'],'temporary_budget_bytes':m['summary']['source_bytes']+40*1024*1024,'native_commands':9,'timeout_seconds_per_command':300,'stop_rule':'Stop on the first unexpected exit code, loader identity or byte comparison; preserve all failures.','scope':'Execute original capacity and motor failures, v2 and final constitutive checks, strict quantization failure, current arithmetic, bad-shader rejection, final physical tilt and the default physical replay from restored committed Git bytes.','excluded_reexecution':'Invalid early SIMD crow trajectories and the full object-push integration check are restored and Git-byte verified, not re-executed.','cleanup_requirement':'Verify every temporary byte from actual committed and pushed Git content before deletion; require exact inventory, hashes, modes, mtimes and zero open handles.','policy_or_profile_promoted':False,'inputs':{str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [G/'recover_git.py',G/'cleanup_git_recovery.py']}}
(G/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
(G/'README.md').write_text('''# Git recovery and bounded native replay

The manifest in `plan.json` retains every raw build, calibration, runtime and
crow trial file for the bounded-drive integration slice, including failed
instruments and rejected physical observations. Recovery reads the actual Git
objects, restores all manifest files into a temporary directory and runs nine
scoped native commands with relocated libraries and explicitly selected shaders.

The sealed plan fixes hashes, memory-on-disk budget, expected exits and a stop
rule before execution. Exact physical, numerical and protected-failure byte
comparisons are reported in `verification.json`. This establishes scoped recovery,
not scientific qualification or a successful learning environment.

Every restored or newly produced temporary file is inventoried. Original bytes
must resolve to the committed archive; new bytes are compressed into this Git
directory. Cleanup requires pushed HEAD parity, actual Git readback, unchanged
hash/mode/mtime inventories and zero open handles. The raw source archive is
pruned only after the restored execution and temporary cleanup succeed.

The experimental motor mode is not promoted. The final crow trial fails on tilt
and penetration; the default-mode regression remains byte-exact. See the
[physical study](../crow-rl-bounded-drive-crow-20261108/README.md) and
[requested optimizer checks](../crow-rl-post-step-rate-20261108/README.md).
''')
print(json.dumps(plan,indent=2))
