from pathlib import Path
s=Path('candidate-64.metal').read_text();a='''        laneNormalResidual = max(
            laneNormalResidual,
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f)
        );''';b=a+'''
        contact.friction.xyz = float3(dot(localRows[0].direction.xyz, relative),
            dot(localRows[1].direction.xyz, relative), dot(localRows[2].direction.xyz, relative));'''
assert s.count(a)==1;s=s.replace(a,b);Path('candidate-trace.metal').write_text(s)
p=Path('/Users/numi/mixed-slip-validation-20261108/probe.cpp').read_text();a='<<",\\\"normal_residual\\\":"<<st.residuals.y';b='<<",\\\"performed_sweeps\\\":"<<st.solverIterations<<",\\\"normal_residual\\\":"<<st.residuals.y';assert p.count(a)==1;p=p.replace(a,b);Path('trace-probe.cpp').write_text(p)
