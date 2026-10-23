from pathlib import Path
import json,subprocess,re,hashlib,time
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001');O=E/'bilateral-knee-connected-weight-repair-100';asset=O/'candidate.nhskin'
old=json.loads((P/'joint-surface-audit-final074-001/report.json').read_text());poses={n:r['path'] for n,r in old['poses'].items()}
for name in ['neutral','right-thumb-ip-46','right-index-pip-49','right-middle-pip-53','right-ring-pip-57','right-little-pip-61','left-thumb-ip-84','left-index-pip-87','left-middle-pip-91','left-ring-pip-95','left-little-pip-99']:poses[name]=str(P/(name+'.json'))
for name in ['knee_0p5','knee_1p0','knee_1p5','knee_2p0','mtp_m0p5','mtp_0p5']:poses[name]=str(E/'joint-regional-quality-088-expanded-poses'/(name+'.json'))
baseline=json.loads((E/'middle-pip-singular-strain-repair-080/all-joint-poses-001/report.json').read_text())
counts={r['name']:r['crossings'] for r in baseline['results']}
for name in poses:
 if name not in counts:counts[name]=0
counts.update({'knee_1p0':10,'knee_1p5':92,'knee_2p0':155})
report={'candidate_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'scope':'Source surface only; preexisting 2rad bilateral limb contacts and shoulder folds retained; no physical admission','results':[]}
for name,path in poses.items():
 csv=O/(name+'.csv');t=time.perf_counter()
 z=subprocess.run([old['inputs']['pose_exporter']['path'],str(asset),path,str(P/'final074-neutral-reference-002/registered-skin.csv'),str(csv)],text=True,capture_output=True);assert z.returncode==0,z.stderr
 z=subprocess.run([old['inputs']['probe']['path'],str(asset),old['inputs']['fixture']['path'],str(csv)],text=True,capture_output=True)
 hit=re.search(r'source_crossing_count=(\d+)',z.stdout);count=int(hit.group(1)) if hit else None
 report['results'].append(dict(pose=name,pose_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),baseline_crossings=counts[name],crossings=count,returncode=z.returncode,stderr=z.stderr,stdout=z.stdout,seconds=time.perf_counter()-t));csv.unlink()
 print(name,counts[name],count,flush=True)
report['no_new_crossing_count_regression']=all(r['crossings'] is not None and r['crossings']<=r['baseline_crossings'] for r in report['results'])
(O/'all-pose-regression.json').write_text(json.dumps(report,indent=2)+'\n')
