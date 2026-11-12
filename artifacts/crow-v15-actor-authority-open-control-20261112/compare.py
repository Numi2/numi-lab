from pathlib import Path
import gzip,hashlib,io,json,subprocess
A=Path("artifacts/crow-v15-actor-authority-open-control-20261112")
C=Path("artifacts/crow-v15-actor-authority-launch-diagnostic-20261112")
name="candidate-20261126-rebound-hip"
manifest=json.loads((C/name/"stderr-parts.json").read_text())
class GitParts(io.RawIOBase):
 def __init__(self):
  self.files=iter(manifest["parts"]);self.proc=None
 def readable(self):return True
 def readinto(self,buffer):
  while True:
   if self.proc is None:
    part=next(self.files,None)
    if part is None:return 0
    self.proc=subprocess.Popen(["git","show","HEAD:"+str(C/name/part["name"])],stdout=subprocess.PIPE)
   n=self.proc.stdout.readinto(buffer)
   if n:return n
   self.proc.stdout.close()
   if self.proc.wait()!=0:raise RuntimeError("git could not restore archived part")
   self.proc=None
def read(stream):
 states={}
 for line in stream:
  if line.startswith(b"CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(b" ",1)[1]);states[(x["lane"],x["step"])]=hashlib.sha256(line).hexdigest()
 return states
with gzip.GzipFile(fileobj=io.BufferedReader(GitParts()),mode="rb") as stream:course=read(stream)
with gzip.open(A/name/"run.stderr.gz","rb") as stream:opened=read(stream)
ends=json.loads((C/name/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"]
rows=[]
for lane,end in enumerate(ends):
 keys=[(lane,t) for t in range(2,end+2)]
 missing=[t for e,t in keys if (e,t) not in course or (e,t) not in opened]
 diffs=[t for e,t in keys if (e,t) in course and (e,t) in opened and course[e,t]!=opened[e,t]]
 rows.append({"lane":lane,"course_episode_transitions":end,"first_different_trace_submission":min(diffs) if diffs else None,"missing_trace_submissions":missing})
(A/"state-parity.json").write_text(json.dumps(rows,indent=2)+"\n")
for row in rows:print(row)
