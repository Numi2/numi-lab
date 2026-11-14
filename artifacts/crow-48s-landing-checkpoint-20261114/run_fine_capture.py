from pathlib import Path
import sys
name=sys.argv[1];assert name in ('x1730','x1745')
p=Path('/tmp/crow-48s-multiview-20261114/run_continuation.py');source=p.read_text()
assert source.count("a/'continuation-250'")==1 and source.count("b/'restart.metallib'")==2 and source.count("('--steps','250')")==1
source=source.replace("a/'continuation-250'","a/'capture-fine-"+name+"-400'")
source=source.replace("b/'restart.metallib'","b/'restart-capture-"+name+".metallib'")
source=source.replace("b/'reset-observer'","b/'full-contact-observer'")
source=source.replace("('--steps','250')","('--steps','400')")
source=source.replace('five-second exploratory reconstructed-state continuation','eight-second fine capture diagnostic; first episode only')
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
