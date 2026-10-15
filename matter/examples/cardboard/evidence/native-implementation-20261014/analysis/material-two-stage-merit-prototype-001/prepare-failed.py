from pathlib import Path
import hashlib, json, subprocess, os
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-valid-summary-tet858-121'
O=B/'analysis/material-two-stage-merit-prototype-001';O.mkdir(exist_ok=False)
s=(B/'analysis/tet858-cpu-merit-experiments-001/replay.cpp').read_text()
s=s.replace('if(argc!=4)', 'if(argc!=5)')
s=s.replace('constexpr unsigned ti=858u;', 'const unsigned ti=std::stoul(argv[4]);')
s=s.replace('meritMode==1?scaledAccept:(scaledAccept&&rawGuard)', 'meritMode==1?scaledAccept:(meritMode==2?(scaledAccept&&rawGuard):(rn>out.tol?rawAccept:scaledAccept))')
assert 'rn>out.tol?rawAccept:scaledAccept' in s
(O/'replay.cpp').write_text(s)
cmd=['clang++','-std=c++20','-O2','-ffp-contract=off','-I'+str(S/'matter/include'),'-I'+str(S/'include'),'-I'+str(S/'matter/tools'),str(O/'replay.cpp'),str(R/'build-cardboard-provenance-121/matter/libnumi_matter_compiler.a'),'-o',str(O/'replay')]
(O/'compile-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
q=subprocess.run(cmd,capture_output=True,text=True);(O/'compile.log').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr
cases=[(858,'dimensionless-certified-coupon-392','rejected-conditioning-000001'),(674,'explicit32-certified-coupon-380','rejected-conditioning-000002')]
results=[]
for ti,case,rejected in cases:
 n=B/'solver'/case/'output';args=[str(O/'replay'),str(n/'compiled.nmatterpack'),str(n/rejected/'candidate_nodes.bin'),str(S/'matter/materials/cardboard/haj_ali_2009_nssc_medium.nmatter'),str(ti)]
 if not Path(args[3]).exists():
  candidates=list((S/'matter').rglob('*nssc*.nmatter'));assert len(candidates)==1,candidates;args[3]=str(candidates[0])
 for mode in (0,1,3):
  env=os.environ.copy();env['CARDBOARD_MERIT_MODE']=str(mode)
  q=subprocess.run(args,capture_output=True,text=True,env=env)
  stem=f'tet{ti}-mode{mode}';(O/(stem+'.stdout.log')).write_text(q.stdout);(O/(stem+'.stderr.log')).write_text(q.stderr)
  results.append(dict(tet=ti,mode=mode,command=args,exit_code=q.returncode,projection=next((x for x in q.stdout.splitlines() if x.startswith('production_projection_valid=')),None)))
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n');(O/'prepare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(results,indent=2))
