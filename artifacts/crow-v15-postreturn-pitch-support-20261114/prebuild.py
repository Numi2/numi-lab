from pathlib import Path
import gzip,hashlib,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-postreturn-pitch-support-20261114'; B=R/'build/crow-v15-postreturn-pitch-support-20261114'; P=R/'build/crow-v15-perch-toe-substep-velocity-20261114'; W=R/'build/crow-v15-wide-contact-isolation-20261111'
B.mkdir(parents=True,exist_ok=True)
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(W/'include'),'-Iinclude',str(A/'probe.cpp'),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(B/'probe')],check=True)
b=(B/'probe').read_bytes();(A/'prebuild-observed.sha256').write_text(hashlib.sha256(b).hexdigest()+'  '+str((B/'probe').relative_to(R))+'\n')
with (A/'prebuild-observed.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(b)
