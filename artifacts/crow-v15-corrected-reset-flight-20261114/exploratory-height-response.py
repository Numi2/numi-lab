from pathlib import Path
import gzip,json,statistics
A=Path('artifacts/crow-v15-corrected-reset-flight-20261114');s={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'straight-reference.json.gz','rt'))};rows=[]
for e,command in [(6,[.55,.80]),(7,[.80,.55])]:
 means=[statistics.mean(s[e,t]['q'][2] for t in range(a,b+1)) for a,b in [(1050,1150),(1650,1750)]]
 rows.append({'lane':e,'scripted_height_targets_m':command,'early_and_late_window_mean_height_m':means,'late_minus_early_m':means[1]-means[0]})
out={'rows':rows,'windows_inclusive':[[1050,1150],[1650,1750]],'scope':'post-hoc exploratory window choice on one used seed; not a preregistered climb or descent qualification'};(A/'exploratory-height-response.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
