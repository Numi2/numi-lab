from pathlib import Path
s=Path('/Users/numi/mixed-bounded-20261108/candidate-64.metal').read_text();assert s.count('0.002f')==3
for label,tol in [('2e-4','0.0002f'),('2e-5','0.00002f')]:Path('candidate-'+label+'.metal').write_text(s.replace('0.002f',tol))
