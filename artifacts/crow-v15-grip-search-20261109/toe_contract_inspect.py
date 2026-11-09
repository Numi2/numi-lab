from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';orig=json.loads((A/'toe-integration-neutral-bindings.json').read_text())['argv']
for mode in ('original','toe'):
 cmd=orig.copy()
 for arg in ('--policy-pack','--base-policy-pack'):
  i=cmd.index(arg);del cmd[i:i+2]
 for k,v in [('--steps','1'),('--state-trace',str(B/('contract-'+mode+'.tsv'))),('--crow-replay-pack',str(B/('contract-'+mode+'.json')))]:cmd[cmd.index(k)+1]=v
 lib=B/('complete-observer' if mode=='original' else 'toe-integration');env={**os.environ,'DYLD_LIBRARY_PATH':str(lib),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1'};p=subprocess.run(cmd,env=env,cwd=R,text=True,capture_output=True,timeout=120)
 (A/('contract-'+mode+'.stdout')).write_text(p.stdout);(A/('contract-'+mode+'.stderr')).write_text(p.stderr);print(mode,'return',p.returncode,'error',p.stderr[-500:])
