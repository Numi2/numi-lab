from pathlib import Path
p=Path('/tmp/crow-48s-multiview-20261114/run_round_perch.py');source=p.read_text()
assert source.count("a/'round-perch-400'")==1 and source.count("b/'restart-round-perch.metallib'")==1
source=source.replace("a/'round-perch-400'","a/'round-aligned-matched-repeat-400'")
source=source.replace("b/'restart-round-perch.metallib'","b/'restart-round-midpoint-yaw-front.metallib'")
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
