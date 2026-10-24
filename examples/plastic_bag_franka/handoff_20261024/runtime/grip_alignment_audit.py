"""Bind each native force channel to that arm's real FK fingertip faces."""
from pathlib import Path
import json,sys,subprocess,tempfile
W=Path(__file__).resolve().parents[1];frames=json.loads(Path(sys.argv[1]).read_text())['frames'];reports=[]
for arm,key in enumerate(['robot','robot2']):
 fs=[]
 for f in frames:
  g=dict(f);g['robot']=f[key];g['grip_contacts']=[c for c in f['grip_contacts'] if c.get('robot',0)==arm];fs.append(g)
 with tempfile.TemporaryDirectory(prefix='numi_handoff_grip_') as tmp:
  p=Path(tmp)/'states.json';p.write_text(json.dumps({'frames':fs},separators=(',',':')))
  script=(W/'runtime/grip_alignment_single.py').read_text();path=W/('grip2.csv' if arm else 'grip.csv');script=script.replace("Path(__file__).resolve().parent.parent/'grip.csv'",repr(str(path)));script=script.replace('path_rows=np.loadtxt(path_file','path_file=Path(path_file)\npath_rows=np.loadtxt(path_file');q=Path(tmp)/'audit.py';q.write_text(script)
  x=json.loads(subprocess.check_output([sys.executable,str(q),str(p)],text=True));x['robot']=arm;reports.append(x)
print(json.dumps({'arms':reports,'all_forces_at_actual_pad_faces':all(r['all_forces_at_visible_authored_pads'] for r in reports)},indent=2))
