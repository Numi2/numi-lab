import hashlib, json, os, pathlib, subprocess, time
root=pathlib.Path("/Users/n/numi-human-separation-20261018/C/outer-contact-mpm-261")
build=root/"build"
run=root/"native-mixed-init-001"
binary=run/"mixed_fem_mpm_runtime_init_check"
msl=pathlib.Path("/Users/n/numi-human-separation-20261018/C/aortic-wall-outer-contact-260/production-outer-contact-001/MatterSource.metal")
msha="1f91b715ebbff704e1fef071386618410c359a60aab19c6980816bc75b396e4b"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(msl)==msha
env=os.environ.copy()
env["DYLD_LIBRARY_PATH"]=str(build)
env["DYLD_PRINT_LIBRARIES"]="1"
env["NUMI_MATTER_TEST_SOURCE"]=str(msl)
env["NUMI_MATTER_TEST_SOURCE_SHA256"]=msha
started=time.monotonic()
try:
    p=subprocess.run([str(binary),str(build/"source/matter")],env=env,capture_output=True,text=True,timeout=300)
    status=p.returncode
    stdout=p.stdout
    stderr=p.stderr
    timeout=False
except subprocess.TimeoutExpired as e:
    status=None
    stdout=e.stdout.decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or "")
    stderr=e.stderr.decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")
    timeout=True
elapsed=time.monotonic()-started
(run/"stdout.log").write_text(stdout)
(run/"stderr.log").write_text(stderr)
result={
 "exit_code":status,"timeout":timeout,"elapsed_s":elapsed,
 "binary":str(binary),"binary_sha256":sha(binary),
 "fixture_source":str(run/"mixed_fem_mpm_runtime_init_check.mm"),
 "fixture_sha256":sha(run/"mixed_fem_mpm_runtime_init_check.mm"),
 "library":str(build/"libmetalrobo.dylib"),
 "library_sha256":sha(build/"libmetalrobo.dylib"),
 "matter_source":str(msl),"matter_source_sha256":sha(msl),
 "matter_source_expected_sha256":msha,
 "dyld_path":str(build),
 "loaded_candidate_library":str(build/"libmetalrobo.dylib") in stderr,
 "stdout":str(run/"stdout.log"),"stderr":str(run/"stderr.log"),
}
(run/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if status not in (0,): raise SystemExit(124 if timeout else (status or 1))
if not result["loaded_candidate_library"]: raise SystemExit("candidate dylib load not confirmed")
