from pathlib import Path
p=Path('/Users/numi/frictionless-mixed-rod-20261108');s=(p/'probe.cpp').read_text()
a='''<<"],\\"target\\":"<<r.targetVelocity'''
b='''<<"],\\"relaxation_trace\\":["<<c.targetVelocityAndPreSolveNormal.x<<","<<c.targetVelocityAndPreSolveNormal.y<<","<<c.targetVelocityAndPreSolveNormal.z<<"],\\"target\\":"<<r.targetVelocity'''
assert s.count(a)==1;s=s.replace(a,b);(p/'trace_probe.cpp').write_text(s)
