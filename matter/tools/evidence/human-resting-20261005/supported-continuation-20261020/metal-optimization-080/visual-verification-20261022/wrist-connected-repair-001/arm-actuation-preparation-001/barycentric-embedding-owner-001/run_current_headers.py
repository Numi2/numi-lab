#!/usr/bin/env python3
"""Test existing native source registration using an immutable 167-row fixture."""
import argparse,subprocess,hashlib,json,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 e=Path(__file__).resolve().parent;r=next(p for p in e.parents if (p/'apps/NumiHumanCoarseTissuePreparation.hpp').is_file())
 a.output.mkdir(parents=True,exist_ok=False)
 cmd=['xcrun','clang++','-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(r/'apps'),'-I'+str(r/'include'),'-I'+str(r/'matter/include'),str(e/'test_source_embedding.cpp'),'-o',str(a.output/'test')]
 report={'scope':'CPU preparation only; does not admit physical FEM or contact','revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'inputs':{str(p):sha(p) for p in [r/'apps/NumiHumanCoarseTissuePreparation.hpp',r/'apps/NumiHumanSkinFEMIntegration.wholebody.hpp',e/'test_source_embedding.cpp',e/'current-075-missing-palm-fixture.tsv']},'compile_command':cmd}
 for label,command in [('compile',cmd),('test',[str(a.output/'test'),str(e/'current-075-missing-palm-fixture.tsv')])]:
  t=time.monotonic();v=subprocess.run(command,cwd=r,capture_output=True,text=True)
  (a.output/f'{label}.stdout').write_text(v.stdout);(a.output/f'{label}.stderr').write_text(v.stderr)
  report[label]={'exit':v.returncode,'wall_seconds':time.monotonic()-t}
  if v.returncode:break
 report['passed']=report.get('test',{}).get('exit')==0
 if (a.output/'test').is_file():report['binary_sha256']=sha(a.output/'test')
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
