from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-candidate-replay-fk-20261114');P=Path('artifacts/crow-v15-flight-release-latch-20261114/turn-20261127-rebound-latch27/replay.json.gz');x=json.load(gzip.open(P,'rt'))['payload'];assert x['environment']==7
steps=(1,170,376,500,898,918,1200,1550,1800);frames={f['step']:f for f in x['frames']};assert all(t in frames for t in steps)
(A/'input.tsv').write_text(''.join(str(t)+' '+' '.join(map(str,frames[t]['q']+frames[t]['v']))+'\n' for t in steps)); print('extracted',len(steps),'frames')
