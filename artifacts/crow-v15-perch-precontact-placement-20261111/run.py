from pathlib import Path
import gzip,os,subprocess,threading,json
R=Path.cwd();A=R/'artifacts/crow-v15-perch-precontact-placement-20261111';B=R/'build/crow-v15-perch-precontact-placement-20261111';P=R/'build/crow-v15-toe-velocity-substep-observer-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(W/'include'),'-Iinclude',str(A/'probe.cpp'),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(B/'probe')],check=True)
results=[]
for gain in ('0.02','0.04','0.06','0.08'):
 D=A/('offset-'+gain);D.mkdir(exist_ok=False);p=subprocess.Popen([str(B/'probe'),gain],cwd=R,env={**os.environ,'CROW_SUBSTEP_TRACE':'1'},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def pump(src,path):
  with path.open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:
    for chunk in iter(lambda:src.read(1<<16),b''):out.write(chunk)
 ts=[threading.Thread(target=pump,args=(p.stdout,D/'trace.log.gz')),threading.Thread(target=pump,args=(p.stderr,D/'substeps.log.gz'))]
 for t in ts:t.start()
 rc=p.wait()
 for t in ts:t.join()
 results.append({'hip_offset_rad':float(gain),'exit_code':rc});(D/'run-result.json').write_text(json.dumps(results[-1],indent=2)+'\n');print('hip offset',gain,'exit',rc,flush=True)
(A/'run-results.json').write_text(json.dumps(results,indent=2)+'\n')
