from pathlib import Path
import gzip,json,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-perch-toe-collision-restart-20261111";P=R/"artifacts/crow-v15-release-hysteresis-control-20261111/autonomous-20261126/replay.json.gz"
with gzip.open(P,"rt") as f:d=json.load(f)["payload"]
assert d["environment"]==5 and d["body_names"][20]=="crow_left_front_toe_1_distal" and d["body_names"][40]=="crow_course_perch"
x=next(x for x in d["frames"] if x["step"]==3539);s=x["body_states"];values=x["q"]+x["v"]+sum((s[i*13:i*13+13] for i in range(35,41)),[]);(A/"snapshot.txt").write_text(" ".join(map(str,values))+"\n");(A/"inputs.json").write_text(json.dumps({"source_replay_sha256":hashlib.sha256(P.read_bytes()).hexdigest(),"snapshot_sha256":hashlib.sha256((A/"snapshot.txt").read_bytes()).hexdigest(),"accepted_observer_step":3540,"replay_frame_step":3539,"lane":5,"scope":"Fresh-cache one-substep diagnostic, not a replay of the original cached world or a qualification."},indent=2)+"\n")
