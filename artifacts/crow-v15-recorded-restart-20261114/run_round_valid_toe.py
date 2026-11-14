from pathlib import Path
import sys
name='aligned-valid-toe'
p=Path('/tmp/crow-48s-multiview-20261114/run_round_perch.py');source=p.read_text()
assert source.count("a/'round-perch-400'")==1 and source.count("b/'restart-round-perch.metallib'")==1
source=source.replace("a/'round-perch-400'","a/'round-position-"+name+"-400'")
source=source.replace("b/'restart-round-perch.metallib'","b/'restart-round-"+name+".metallib'")
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
