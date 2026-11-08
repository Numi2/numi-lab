from pathlib import Path
p=Path("make_probe.py");s=p.read_text();lines=s.splitlines()
lines[14]="""r("statePass=std::isfinite(xl)&&std::isfinite(vl)&&", "const auto &p=result.states[0].positions[1], &v=result.states[0].velocities[1]; const auto &f0=result.reactions[0].averageForceOnTarget, &f1=result.reactions[1].averageForceOnTarget; const bool transverse=std::hypot(p[0]*ay-p[1]*ax,p[2])<=1e-6 && std::hypot(v[0]*ay-v[1]*ax,v[2])<=.001 && std::hypot(f0[0]*ay-f0[1]*ax,f0[2])<=1e-6 && std::hypot(f1[0]*ay-f1[1]*ax,f1[2])<=1e-6; forcePass=forcePass&&transverse; statePass=std::isfinite(xl)&&std::isfinite(vl)&&")"""
lines.pop(15);p.write_text("\n".join(lines)+"\n")
