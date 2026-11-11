from pathlib import Path
import subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-recenter-after-grip-screen-20261111";B=R/"build/crow-v15-recenter-after-grip-screen-20261111";P=R/"build/crow-v15-hip-roll-substep-observer-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111";B.mkdir(parents=True,exist_ok=True)
subprocess.run(["xcrun","clang++","-std=c++20","-O2","-I"+str(W/"include"),"-Iinclude",str(A/"probe.cpp"),"-L"+str(P),"-lmetalrobo","-Wl,-rpath,"+str(P),"-o",str(B/"probe")],check=True)
