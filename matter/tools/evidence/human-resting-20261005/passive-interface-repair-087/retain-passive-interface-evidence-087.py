from pathlib import Path
import gzip, hashlib, json, shutil

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'passive-interface-publication-087';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
names=['passive-neighbor-depth-062.py','passive-neighbor-containment-065.py','passive-organ-overlap-volumes-066.py',
       'repair-passive-organ-interfaces-067.py','condition-passive-organ-interfaces-068.py','passive-interface-helper-069.py',
       'condition-passive-organ-star-070.py','open-passive-organ-sliver-071.py','audit-passive-organ-candidate-072.py',
       'audit-passive-native-075.py','reproduce-passive-flips-079.py','passive-organ-native-input-074-invocation.json',
       'passive-interfaces-074-tests.log','passive-organ-native-audit-075.log',
       'passive-neighbor-depth-062/input.json','passive-neighbor-depth-062/report.json','passive-neighbor-depth-062/source-topology.json',
       'passive-neighbor-containment-065/input.json','passive-neighbor-containment-065/report.json','passive-organ-overlap-volumes-066/report.json',
       'passive-organ-interface-candidate-067-25um/report.json','passive-organ-interface-candidate-067-50um/report.json',
       'passive-organ-interface-candidate-067-100um/report.json','passive-organ-interface-conditioned-068/report.json',
       'passive-organ-interface-flipped-069/report.json','passive-organ-interface-star-070/report.json',
       'passive-organ-interface-conditioned-071/report.json','passive-organ-candidate-audit-072/report.json',
       'passive-organ-native-input-074/resting-anatomy-receipt.json','passive-organ-native-input-074/resting-anatomy-manifest.json',
       'passive-organ-native-audit-075/report.json','passive-flip-reproduction-079/report.json']
run=E/'cardiac-map-multiregion-affine-repair-011/native-fullchain073-6s-001'
manifest={'scope':'Inferred passive pancreas/spleen interface repair, source audit and eight accepted-frame native checks in a six-second integrated cardiac scene. Not final full-body or five-minute acceptance. The original taenia neighbor remains unresolved.',
          'human_implementation_revision':'3d706b8','retained_files':[],'retained_on_macmini':[],'packaging_driver_sha256':sha(__file__)}
def retain(path,relative):
    target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
    original=sha(path)
    if path.suffix=='.json' and path.stat().st_size>65536:
        target=target.with_suffix(target.suffix+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
        assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()==original
    else:shutil.copyfile(path,target)
    manifest['retained_files'].append({'source':str(path),'source_sha256':original,'retained':str(target.relative_to(out)),'retained_sha256':sha(target)})
for name in names:retain(E/name,Path(name))
for name in ['invocation.json','run.log','resting-coupled.csv','resting-surface-audit.csv']:
    retain(run/name,Path('native073')/name)
retain(Path(__file__),Path('retain-passive-interface-evidence-087.py'))
paths=[E/'costal-current003-full-chain-003/resting-thorax.nhanatomy',E/'passive-organ-native-input-074/resting-thorax.nhanatomy',
       E/'passive-organ-native-input-073/resting-anatomy-receipt.json',run/'native-viewer.mov',run/'resting-human.mrvpack']
paths += [run/'accepted-geometry'/f'step-{step}.mrvpack' for step in [0,63,511,639,1951,2207,2783,2999]]
paths += [E/f'passive-organ-interface-conditioned-071/surface-{sid}.npz' for sid in [3,13]]
paths += [E/'passive-neighbor-depth-062/source-surfaces.npz',E/'passive-neighbor-containment-065/source-surfaces.npz']
for p in paths:manifest['retained_on_macmini'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(out/'manifest.json'),'files':len(manifest['retained_files']),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}))
