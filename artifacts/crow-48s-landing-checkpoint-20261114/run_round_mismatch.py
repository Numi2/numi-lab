from pathlib import Path
p=Path('/tmp/crow-48s-multiview-20261114/run_continuation.py');source=p.read_text()
assert source.count("a/'continuation-250'")==1 and source.count("b/'restart.metallib'")==2 and source.count("('--steps','250')")==1
source=source.replace("a/'continuation-250'","a/'round-perch-policy-mismatch'")
source=source.replace("b/'restart.metallib'","b/'restart-round-perch.metallib'")
source=source.replace("b/'reset-observer'","b/'round-perch-observer'")
source=source.replace("('--steps','250')","('--steps','1')")
source=source.replace('five-second exploratory reconstructed-state continuation','one-step round-perch policy-contract mismatch probe')
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
