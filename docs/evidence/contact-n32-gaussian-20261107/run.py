import json,os,pathlib,signal,subprocess,time
e=pathlib.Path('/Users/numi/numi-lab-contact-n32-gaussian-evidence-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-surface-n32'
rows=[]
for h,limit in ((2e-6,240),(1e-6,360)):
 t=time.monotonic()
 p=subprocess.Popen(['/usr/bin/time','-l',exe,'32',str(h),'.01','gaussian'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 timed_out=False
 try:out,err=p.communicate(timeout=limit)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
 name='n32-h'+str(h)+'-gaussian'
 (e/(name+'.json')).write_text(out);(e/(name+'.stderr')).write_text(err)
 row={'h':h,'timeout_seconds':limit,'timed_out':timed_out,'exit':p.returncode,
      'wall_seconds':time.monotonic()-t}
 rows.append(row);(e/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(row,flush=True)
 if timed_out or p.returncode:raise SystemExit(1)
