from pathlib import Path
import subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-wide-leg-response-20261110';B=R/'build/crow-v15-wide-leg-response-20261110';B.mkdir(parents=True,exist_ok=True)
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-Iinclude','-Iartifacts/crow-v15-wing-inertia-prototype-20261110',str(A/'probe.cpp'),'-Lbuild/crow-v15-ccd-budget-20261110','-lmetalrobo','-Wl,-rpath,@loader_path/../crow-v15-ccd-budget-20261110','-o',str(B/'probe')],check=True)
