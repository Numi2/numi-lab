from pathlib import Path
import gzip,hashlib,json,subprocess
root=Path("/Users/n/Developer/numi-crow-rl-20261107")
restore=Path("/tmp/crow-restore-f219c956-20261111")
artifacts=[
 "crow-v15-toes-perch-teacher-integration-20261111",
 "crow-v15-teacher-landing-flare-control-20261111",
 "crow-v15-contact-gated-flight-leg-tuck-20261111",
]
files=[
 ("crow-v15-toes-perch-teacher-integration-20261111","teacher-20261126/replay.json.gz","teacher-20261126-replay.json"),
 ("crow-v15-toes-perch-teacher-integration-20261111","teacher-20261126/trace.tsv.gz","teacher-20261126-trace.tsv"),
 ("crow-v15-teacher-landing-flare-control-20261111","teacher-20261126/replay.json.gz","teacher-20261126-replay.json"),
 ("crow-v15-teacher-landing-flare-control-20261111","teacher-20261126/trace.tsv.gz","teacher-20261126-trace.tsv"),
 ("crow-v15-contact-gated-flight-leg-tuck-20261111","flight-replay.json.gz","flight-replay.json"),
 ("crow-v15-contact-gated-flight-leg-tuck-20261111","flight-trace.tsv.gz","flight-trace.tsv"),
]
def digest(file):
 h=hashlib.sha256()
 for chunk in iter(lambda:file.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
checks=[]
for name in artifacts:
 p=restore/"artifacts"/name
 manifest=dict(l.split("  ",1)[::-1] for l in (p/"SHA256SUMS").read_text().splitlines())
 selected=[relative for selected_name,relative,_ in files if selected_name==name]
 for relative in selected:
  with (p/relative).open("rb") as f:actual=digest(f)
  if actual!=manifest[relative]:raise RuntimeError("restored manifest mismatch: "+str(p/relative))
 checks.append({"artifact":name,"restored_selected_files":selected,"selected_manifest_entries":"verified"})
retire=[]
for name,relative,rawname in files:
 source=restore/"artifacts"/name/relative;raw=root/"build"/name/rawname
 with gzip.open(source,"rb") as f:expanded=digest(f)
 with raw.open("rb") as f:rawhash=digest(f)
 if expanded!=rawhash:raise RuntimeError("raw and restored archive differ: "+str(raw))
 proc=subprocess.run(["lsof","-t","--",str(raw)],capture_output=True,text=True)
 if proc.returncode not in (0,1) or proc.stdout.strip():raise RuntimeError("raw file has open handles or lsof failed: "+str(raw))
 retire.append({"path":str(raw),"bytes":raw.stat().st_size,"sha256":rawhash,"restored_gzip":str(source),"open_handles":False})
out={"independent_fetched_commit":"f219c9569a6c0694688666d0b5c4a27befd6bff3","manifest_checks":checks,"verified_redundant_uncompressed_files":retire,"retired":False}
(root/"artifacts/crow-v15-independent-archive-retirement-20261111/verification.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
