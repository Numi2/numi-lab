from pathlib import Path
p=Path('/tmp/crow-48s-multiview-20261114/run_continuation.py')
source=p.read_text()
assert source.count("a/'continuation-250'")==1 and source.count("b/'restart.metallib'")==2
assert source.count("('--steps','250')")==1
source=source.replace("a/'continuation-250'","a/'pitch-balance-negative-1000'")
source=source.replace("b/'restart.metallib'","b/'restart-pitch-balance-negative.metallib'")
source=source.replace("('--steps','250')","('--steps','1000')")
source=source.replace('five-second exploratory reconstructed-state continuation','20-second reconstructed-state extended pitch balance diagnostic')
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
