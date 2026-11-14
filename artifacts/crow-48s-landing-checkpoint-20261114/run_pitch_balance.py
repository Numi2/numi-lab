from pathlib import Path
import sys
sign = sys.argv[1]
assert sign in ('negative', 'positive')
p = Path('/tmp/crow-48s-multiview-20261114/run_continuation.py')
source = p.read_text()
assert source.count("a/'continuation-250'") == 1 and source.count("b/'restart.metallib'") == 2
assert source.count("('--steps','250')") == 1
source = source.replace("a/'continuation-250'", "a/'pitch-balance-"+sign+"-600'")
source = source.replace("b/'restart.metallib'", "b/'restart-pitch-balance-"+sign+".metallib'")
source = source.replace("('--steps','250')", "('--steps','600')")
source = source.replace('five-second exploratory reconstructed-state continuation',
                        '12-second reconstructed-state pitch balance diagnostic')
exec(compile(source, str(p), 'exec'), {'__name__': '__main__'})
