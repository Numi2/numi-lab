from pathlib import Path
import subprocess
r=Path.cwd();a=r/'artifacts/crow-v15-preimpact-wide-perch-grip-screen-20261112';b=r/'build/crow-v15-preimpact-wide-perch-grip-screen-20261112';p=r/'build/crow-v15-hip-roll-substep-observer-20261111';w=r/'build/crow-v15-wide-contact-isolation-20261111';b.mkdir(parents=True,exist_ok=True)
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(w/'include'),'-Iinclude',str(a/'probe.cpp'),'-L'+str(p),'-lmetalrobo','-Wl,-rpath,'+str(p),'-o',str(b/'probe')],check=True)
