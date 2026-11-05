from pathlib import Path
import json,hashlib,subprocess,os,time,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104'); C=W/'joint-boundary-rebuild-64c/compiler';P=W/'prescribed-local-mass-315'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');B=H/'regional-compensation-1061-root'
out=P/'converter';out.mkdir(exist_ok=True);sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def put(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
old=json.loads((B/'converter-020/run-manifest.json').read_text());new=json.loads((P/'allocation/run-manifest.json').read_text());cfg=json.loads((P/'admission-002/identity.json').read_text());mr=P/'admission-002/mass-receipt.json';assert mr.is_file()
argv=old['argv'];en=old['environment'].copy()
for a in new['argv']:
 if a.startswith('NUMI_H706_FRACTIONAL_'):
  k,v=a.split('=',1)
  if k in en:en[k]=v
pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
for k in en:
 if k.startswith(pre) and k[len(pre):] in cfg['native_preflight_env']:en[k]=cfg['native_preflight_env'][k[len(pre):]]
en['NUMI_H706_FRACTIONAL_OUTPUT_DIR']=str(P/'allocation/allocator-output')
args=dict(zip(argv[1::2],argv[2::2]));assert len(args)*2+1==len(argv)
amap={'--child-node':C/'volume-export/volume/mesh.node','--child-elements':C/'volume-export/volume/mesh.ele','--child-labels':C/'volume-export/volume/child.labels.csv','--p1-skin':C/'p1-Skin/skin-current-p1-143-support.npz','--p1-fat':C/'p1-Fat/fat-current-p1-143-support.npz','--carrier-skin':C/'mass-support/skin-source-carriers.npz','--carrier-fat':C/'mass-support/fat-source-carriers.npz','--projection-csv':C/'mass-support/exact-projection-support.csv','--centroid-csv':C/'mass-support/source-centroid-slots.csv','--carrier-witness-skin':C/'mass-support/skin-carrier-witnesses.csv.gz','--carrier-witness-fat':C/'mass-support/fat-carrier-witnesses.csv.gz','--flow-csv':P/'allocation/allocator-output/flow.csv','--semantic-csv':P/'allocation/allocator-output/semantic-groups.csv','--source-donor-csv':P/'allocation/allocator-output/source-donors.csv','--carrier-trace-csv':P/'allocation/allocator-output/carrier-trace.csv','--perf-log':P/'allocation/stderr.log','--admission-receipt':mr}
for k,p in amap.items():
 args[k]=str(p)
 hk=k+'-sha256'
 # Existing interface uses abbreviated hash flag spellings.
 hk={'--projection-csv':'--projection-sha256','--centroid-csv':'--centroid-sha256','--flow-csv':'--flow-sha256','--semantic-csv':'--semantic-sha256','--source-donor-csv':'--source-donor-sha256','--carrier-trace-csv':'--carrier-trace-sha256','--admission-receipt':'--admission-sha256'}.get(k,hk)
 if hk in args:args[hk]=sha(p)
for role in ['skin','fat']:
 for key in ['geometry','volume']:args['--'+key+'-'+role+'-sha256']=en['NUMI_H706_FRACTIONAL_'+key.upper()+'_'+role.upper()+'_SHA256']
 args['--support-sha-'+role]=sha(amap['--p1-'+role]);args['--carrier-packet-sha-'+role]=sha(amap['--carrier-'+role])
args['--state-sha256']=en['NUMI_H706_FRACTIONAL_STATE_SHA256'];args['--output-dir']=str(out/'role-output')
cmd=[argv[0]]+[a for k,v in args.items() for a in [k,v]]
put(out/'run-manifest.json',{'argv':cmd,'environment':en,'existing_converter':{'path':argv[0],'sha256':sha(argv[0])},'physical_admission':False,'native_steps':0})
t=time.time()
with (out/'stdout.log').open('w') as f,(out/'stderr.log').open('w') as e:r=subprocess.run(cmd,env=dict(os.environ,**en),stdout=f,stderr=e)
put(out/'execution.json',{'exit_code':r.returncode,'elapsed_s':time.time()-t,'manifest_sha256':sha(out/'run-manifest.json'),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'physical_admission':False,'native_steps':0})
print('converter_exit',r.returncode);print((out/'stdout.log').read_text()[-1500:]);print((out/'stderr.log').read_text()[-2500:]);sys.exit(r.returncode)
