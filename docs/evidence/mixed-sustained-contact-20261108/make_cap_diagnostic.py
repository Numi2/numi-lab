from pathlib import Path
for label in ('2e-4','2e-5'):
 s=Path('candidate-'+label+'-reporting.metal').read_text();a='packetHasRod ? 64u : requestedIterations';b='packetHasRod ? 128u : requestedIterations';assert s.count(a)==1;s=s.replace(a,b);Path('candidate-'+label+'-cap128.metal').write_text(s)
