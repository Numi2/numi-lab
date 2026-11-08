import hashlib,json,os,subprocess
from pathlib import Path
base=Path('/Users/numi/numi-verified-final-offline-20261108')
prefix=base/'installed'; source=base/'extracted/source'
verifier=Path('/Users/numi/numi-lab-autoload-package-source-20261108/python/metalrobo/numi_source_provenance.py')
archive=Path('/Users/numi/numi-verified-final-bundle-20261108.tar.gz')
rev='a182448ec9d987ef9446fea384560d61cb5ed4ee'; sha=hashlib.sha256(archive.read_bytes()).hexdigest()
common=['--prefix',str(prefix),'--expected-revision',rev,'--expected-archive-sha256',sha]
audit=['python3',str(verifier),'audit',*common]
stamp=['python3',str(verifier),'stamp','--source',str(source),'--archive',str(archive),*common]
def call(argv):
 r=subprocess.run(argv,text=True,capture_output=True)
 return {'returncode':r.returncode,'stderr':r.stderr.strip()}
result={'independent_audit':call(audit)}
receipt=prefix/'share/numi/SOURCE-PROVENANCE.json'; revision=prefix/'share/numi/REVISION'; before=(receipt.read_bytes(),revision.read_bytes())
extra=source/'unexpected-proof-test-file'
assert not extra.exists()
try:
 extra.write_text('extra')
 result['extra_source']=call(stamp)
finally: extra.unlink()
assert (receipt.read_bytes(),revision.read_bytes())==before
artifact=prefix/'libexec/numi/train'; mode=artifact.stat().st_mode&0o777
try:
 artifact.chmod(mode ^ 0o100)
 result['changed_installed_mode']=call(audit)
finally: artifact.chmod(mode)
result['restored_audit']=call(audit)
assert result['independent_audit']['returncode']==0 and result['extra_source']['returncode']!=0 and result['changed_installed_mode']['returncode']!=0 and result['restored_audit']['returncode']==0
(base/'mutation-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
