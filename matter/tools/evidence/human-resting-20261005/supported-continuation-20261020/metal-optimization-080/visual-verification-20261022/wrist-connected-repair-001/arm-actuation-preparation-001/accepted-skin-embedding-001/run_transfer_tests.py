#!/usr/bin/env python3
"""Run existing and embedded accepted-skin transfer CPU regressions on SSH Numi.
This tests registration/presentation, not physical FEM evolution.
"""
import argparse,hashlib,json,platform,subprocess,time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 root=next(p for p in Path(__file__).resolve().parents if (p/'apps/NumiHumanAcceptedFEMSkinTransfer.hpp').is_file())
 a.output.mkdir(parents=True,exist_ok=False)
 report={'scope':'CPU registration/presentation tests only; no FEM/contact or continuous physical qualification','revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'platform':platform.platform(),'header_sha256':sha(root/'apps/NumiHumanAcceptedFEMSkinTransfer.hpp'),'tests':[]}
 for stem in ['human_accepted_fem_skin_transfer','human_accepted_fem_skin_transfer_mixed_donor','human_accepted_fem_skin_transfer_embedding']:
  src=root/'tests'/f'{stem}.cpp';binary=a.output/stem
  compile_cmd=['xcrun','clang++','-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(root/'apps'),str(src),'-o',str(binary)]
  t=time.monotonic();c=subprocess.run(compile_cmd,cwd=root,capture_output=True,text=True)
  (a.output/f'{stem}.compile.stdout').write_text(c.stdout);(a.output/f'{stem}.compile.stderr').write_text(c.stderr)
  row={'source':str(src),'source_sha256':sha(src),'compile_command':compile_cmd,'compile_exit':c.returncode,'compile_wall_seconds':time.monotonic()-t}
  if c.returncode==0:
   t=time.monotonic();r=subprocess.run([str(binary)],cwd=root,capture_output=True,text=True)
   (a.output/f'{stem}.stdout').write_text(r.stdout);(a.output/f'{stem}.stderr').write_text(r.stderr)
   row.update(run_exit=r.returncode,run_wall_seconds=time.monotonic()-t,binary_sha256=sha(binary))
  report['tests'].append(row)
 report['passed']=all(t.get('run_exit')==0 for t in report['tests'])
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
