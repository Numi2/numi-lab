from pathlib import Path
p=Path('/tmp/crow-48s-multiview-20261114/run_continuation.py');source=p.read_text()
assert source.count("a/'continuation-250'")==1 and source.count("b/'restart.metallib'")==2
source=source.replace("a/'continuation-250'","a/'near-perch-250'").replace("b/'restart.metallib'","b/'restart-near-perch.metallib'")
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
