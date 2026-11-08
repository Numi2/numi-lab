from pathlib import Path
s=Path("baseline-probe.mm").read_text()
def r(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
r("if (argc != 2) return 2;", "if (argc != 3) return 2; const bool rotated=std::string_view(argv[2])==\"y\"; if (!rotated && std::string_view(argv[2])!=\"x\") return 2; const double ax=rotated?0:1, ay=rotated?1:0;")
r("state.positions[0]={left,0,0}; state.positions[1]={test.x,0,0};", "state.positions[0]={left*ax,left*ay,0}; state.positions[1]={test.x*ax,test.x*ay,0};")
r("state.positions[2]={right,0,0}; state.velocities[1]={test.v,0,0};", "state.positions[2]={right*ax,right*ay,0}; state.velocities[1]={test.v*ax,test.v*ay,0};")
r("attachments[0].nodeIndex=0; attachments[0].targetPosition={left,0,0};", "attachments[0].nodeIndex=0; attachments[0].targetPosition={left*ax,left*ay,0};")
r("attachments[1].nodeIndex=2; attachments[1].targetPosition={right,0,0};", "attachments[1].nodeIndex=2; attachments[1].targetPosition={right*ax,right*ay,0};")
r("xl=result.states[0].positions[1][0]; vl=result.states[0].velocities[1][0];", "xl=result.states[0].positions[1][0]*ax+result.states[0].positions[1][1]*ay; vl=result.states[0].velocities[1][0]*ax+result.states[0].velocities[1][1]*ay;")
r("al=result.reactions[0].averageForceOnTarget[0];", "al=result.reactions[0].averageForceOnTarget[0]*ax+result.reactions[0].averageForceOnTarget[1]*ay;")
r("ar=result.reactions[1].averageForceOnTarget[0];", "ar=result.reactions[1].averageForceOnTarget[0]*ax+result.reactions[1].averageForceOnTarget[1]*ay;")
r("statePass=std::isfinite(xl)&&std::isfinite(vl)&&", "const auto &p=result.states[0].positions[1], &v=result.states[0].velocities[1]; const auto &f0=result.reactions[0].averageForceOnTarget, &f1=result.reactions[1].averageForceOnTarget; const bool transverse=std::hypot(p[0]*ay-p[1]*ax,p[2])<=1e-6 && std::hypot(v[0]*ay-v[1]*ax,v[2])<=.001 && std::hypot(f0[0]*ay-f0[1]*ax,f0[2])<=1e-6 && std::hypot(f1[0]*ay-f1[1]*ax,f1[2])<=1e-6; forcePass=forcePass&&transverse; statePass=std::isfinite(xl)&&std::isfinite(vl)&&")
Path("probe.mm").write_text(s)
