"""Source-checked FP32 impulse updates and diagnostic context; retire with integration."""
from pathlib import Path
import hashlib
import os
os.chdir(Path(__file__).resolve().parents[1])
p=Path('src/metal/NumiHumanStand.metal');s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='e4893837e19aa0d7cde7969106a9d3d4f893c7b13b342d10bb5ce1c85a17adfc'
s=s.replace('candidateV[dof] += seed * normalResponse[dof];','candidateV[dof] = fma(seed, normalResponse[dof], candidateV[dof]);')
s=s.replace('candidateV[dof] += applied[axis] * response[dof];','candidateV[dof] = fma(applied[axis], response[dof], candidateV[dof]);')
p.write_text(s)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='f4c7318fea9dea01d7141d7edab4872ed8b39f632b73eb4fe9c3e9c8620a1822'
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
