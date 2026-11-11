from pathlib import Path
import gzip,json,subprocess,os
R=Path.cwd();A=R/"artifacts/crow-v15-release-hysteresis-control-20261111";D=A/"autonomous-20261126"
with gzip.open(D/"replay.json.gz","rt") as f:d=json.load(f)["payload"]
assert d["environment"]==5 and d["body_names"][20]=="crow_left_front_toe_1_distal" and d["body_names"][40]=="crow_course_perch"
txt=""
for frame in d["frames"]:
 t=frame["step"]+1
 if t not in (3500,3540,3560,3570,3580,3590,3600,3650,3800):continue
 b=frame["body_states"];txt+=str(t)+" "+" ".join(map(str,b[20*13:20*13+7]+b[40*13:40*13+7]))+"\n"
p=subprocess.run([str(R/"build/crow-v15-perch-geometry-correction-20261111/collision-oracle")],input=txt,text=True,capture_output=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")},check=True);(R/"artifacts/crow-v15-perch-geometry-correction-20261111/collision-oracle.stdout").write_text(p.stdout);print(p.stdout)
