"""Source-checked FP32 impulse refinement and diagnostic context; retire with integration."""
from pathlib import Path
import hashlib
import os
os.chdir(Path(__file__).resolve().parents[1])
p=Path('src/metal/NumiHumanStand.metal');s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='e4893837e19aa0d7cde7969106a9d3d4f893c7b13b342d10bb5ce1c85a17adfc'
s=s.replace('candidateV[dof] += seed * normalResponse[dof];','candidateV[dof] = fma(seed, normalResponse[dof], candidateV[dof]);')
s=s.replace('candidateV[dof] += applied[axis] * response[dof];','candidateV[dof] = fma(applied[axis], response[dof], candidateV[dof]);')
assert hashlib.sha256(s.encode()).hexdigest()=='f4c7318fea9dea01d7141d7edab4872ed8b39f632b73eb4fe9c3e9c8620a1822'
start='''                if (gap > support.frictionSlopAndStabilization.y) continue;
                float3 velocity{0.0f};'''
assert s.count(start)==1
s=s.replace(start,'''                if (gap > support.frictionSlopAndStabilization.y) continue;
                // Re-evaluate the actual J*v after a block update. In FP32,
                // retracting a large warm start can lose the last few bits of
                // its small solved impulse. A second local residual correction
                // uses the same normal/friction equations and response columns;
                // no extra force, compliance, or relaxed acceptance is added.
                for (uint contactRefinement = 0u; contactRefinement < 2u;
                     ++contactRefinement) {
                float3 velocity{0.0f};''',1)
end='''                        candidateV[dof] = fma(applied[axis], response[dof], candidateV[dof]);
                    }
                }
            }
'''
assert s.count(end)==1
s=s.replace(end,'''                        candidateV[dof] = fma(applied[axis], response[dof], candidateV[dof]);
                    }
                }
                }
            }
''',1)
p.write_text(s)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2b464343138b9687cf4f7df6f04e36a3db4b69d79520c3c803cd689292533d68'
p=Path('tests/numi_human_bilateral_metal_test.mm');s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='9bd466ca739f03e01b3041ef798d027689a6c0cd237b06c8217c698a0c99da69'
s=s.replace('''            require(normalVelocity>=-2e-6,"equality correction invalidated unilateral contact");++checks;''','''            if(normalVelocity < -2e-6) {
                std::cerr.precision(10);
                std::cerr<<"normal residual env="<<e<<" h="<<h<<" sweeps="<<sweeps
                         <<" contactBody="<<contactBody<<" seed="<<seedImpulse
                         <<" loaded="<<loaded<<" closing="<<closing
                         <<" normalVelocity="<<normalVelocity
                         <<" expectedImpulse="<<expectedNormal[e]<<'\\n';
            }
            require(normalVelocity>=-2e-6,"equality correction invalidated unilateral contact");++checks;''')
p.write_text(s)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='af5bce481e89f8a7a5271b7f12b4610c8e1591e6671ee86d58d3de3db8c14e50'
