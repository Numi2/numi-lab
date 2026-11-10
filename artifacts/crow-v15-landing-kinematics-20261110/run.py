from pathlib import Path
import gzip,json,subprocess
r=Path.cwd();a=r/"artifacts/crow-v15-landing-kinematics-20261110"
cases=[("old","artifacts/crow-v15-perch-toe-damped-20261110/diagnostic-reserved-final-j-velocity08c.stderr.gz"),("new","artifacts/crow-v15-jnew-force-observer-20261110/diagnostic-reserved-final-j-jnew-observer.stderr.gz")]
lines=[]
for name,p in cases:
 with gzip.open(r/p,"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(line.split(" ",1)[1])
    if z["step"] in (2450,2480,2500,2513,2525,2550):lines.append(name+"_"+str(z["step"])+" "+" ".join(map(str,z["q"]))+"\n")
data="".join(lines);(a/"input.txt").write_text(data)
p=subprocess.run([str(r/"build/crow-v15-landing-kinematics-20261110/probe")],input=data,text=True,capture_output=True,check=True);(a/"kinematics.txt").write_text(p.stdout)
for line in p.stdout.splitlines():
 if " -1 " in line and ("old_2500 " in line or "new_2500 " in line or "new_2513 " in line):
  print(line)
