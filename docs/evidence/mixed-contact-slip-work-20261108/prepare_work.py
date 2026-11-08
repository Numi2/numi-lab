from pathlib import Path
s=Path('probe.cpp').read_text()
a='auto& r=a.contactEvidence.evaluatedRows.at(3*i);'
b='auto& r=a.contactEvidence.evaluatedRows.at(3*i);auto& ru=a.contactEvidence.evaluatedRows.at(3*i+1);auto& rv=a.contactEvidence.evaluatedRows.at(3*i+2);'
assert s.count(a)==1;s=s.replace(a,b)
a='<<"],\\\"target\\\":"<<r.targetVelocity'
b='<<"],\\\"pre_relative\\\":["<<r.preSolveVelocity<<","<<ru.preSolveVelocity<<","<<rv.preSolveVelocity<<"],\\\"target\\\":"<<r.targetVelocity'
assert s.count(a)==1;s=s.replace(a,b)
Path('work-probe.cpp').write_text(s)
