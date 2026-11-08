import collections,json,re,sys
from pathlib import Path
w=Path(sys.argv[1]); src=Path(sys.argv[2]); p=re.compile(r'\.(?:tar|tgz|tar\.gz|tar\.zst|tzst|zip|7z)$',re.I)
tip=json.loads((src/'docs/evidence/suture-branch-tip-search-20261108/archive-inventory.json').read_text())
tip_paths={name for row in tip['archive_blobs'] for name in row['paths']}
seen=collections.defaultdict(set); commit=None
for line in (w/'broad.txt').read_text().splitlines():
 if line.startswith('commit '): commit=line.split()[1]
 elif commit and p.search(line): seen[line].add(commit)
new={path:sorted(commits) for path,commits in sorted(seen.items()) if path not in tip_paths}
summary={'tip_paths':len(tip_paths),'history_archive_paths':len(seen),'not_at_tips':new}
(w/'archive-path-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print('tip_paths',len(tip_paths),'historical_paths',len(seen),'not_at_tips',len(new))
for path,commits in new.items(): print(commits[0],path)
