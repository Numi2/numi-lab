from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-valid-summary-tet858-121';O=B/'analysis/tet858-cpu-merit-experiments-001';O.mkdir(exist_ok=False)
old=(B/'analysis/coupon-392-status5-audit-001/replay.cpp').read_text();s=old.replace('#include "/Users/home/numi-cardboard-20261006/matter/tools/cardboard_paper_material_check.cpp"','#include "'+str(S/'matter/tools/cardboard_paper_material_check.cpp')+'"')
s=s.replace('Projection out;unsigned n=m.stateCount;', 'const int meritMode=std::getenv("CARDBOARD_MERIT_MODE")?std::atoi(std::getenv("CARDBOARD_MERIT_MODE")):1;Projection out;unsigned n=m.stateCount;')
s=s.replace('std::array<float,32> scales{};std::array<float,32> rows{};', 'float scaledNorm=0;std::array<float,32> scales{};std::array<float,32> rows{};')
s=s.replace('rows[r]=q;}', 'rows[r]=q;scaledNorm=std::max(scaledNorm,std::abs(residual[r])/q);}')
s=s.replace('float alpha=1;bool accepted=false;', 'std::cerr<<std::setprecision(17)<<"ITER "<<it<<" raw "<<rn<<" bound "<<out.tol<<" corr "<<cn<<" scaled "<<scaledNorm<<"\\n";float alpha=1;bool accepted=false;')
s=s.replace('float tr=0;bool good=true;', 'float tr=0,ts=0;bool good=true;')
s=s.replace('good=good&&ok;tr=std::max(tr,std::abs(v));}', 'good=good&&ok;tr=std::max(tr,std::abs(v));if(scaled)ts=std::max(ts,std::abs(v)/rows[row]);}')
s=s.replace('if(good&&tr<=(1-1e-4f*alpha)*rn){', 'const bool rawAccept=tr<=(1-1e-4f*alpha)*rn;const bool scaledAccept=ts<scaledNorm&&ts<=(1-1e-4f*alpha)*scaledNorm;const bool rawGuard=rn<=out.tol?tr<=out.tol:rawAccept;const bool accept=good&&(meritMode==0?rawAccept:(meritMode==1?scaledAccept:(scaledAccept&&rawGuard)));std::cerr<<"TRIAL "<<it<<" "<<bt<<" alpha "<<alpha<<" raw "<<tr<<" scaled "<<ts<<" rawAccept "<<rawAccept<<" scaledAccept "<<scaledAccept<<" rawGuard "<<rawGuard<<" accepted "<<accept<<"\\n";if(accept){')
assert s!=old
(O/'replay.cpp').write_text(s)
cmd=['clang++','-std=c++20','-O2','-ffp-contract=off','-I'+str(S/'matter/include'),'-I'+str(S/'include'),'-I'+str(S/'matter/tools'),str(O/'replay.cpp'),str(R/'build-cardboard-provenance-121/matter/libnumi_matter_compiler.a'),'-o',str(O/'replay')]
(O/'compile-command.json').write_text(json.dumps(cmd,indent=2)+'\n');q=subprocess.run(cmd,capture_output=True,text=True);(O/'compile.log').write_text(q.stdout+q.stderr);print('compile',q.returncode);assert q.returncode==0,q.stderr
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
