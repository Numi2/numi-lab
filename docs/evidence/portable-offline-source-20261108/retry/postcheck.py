import hashlib,json,pathlib,subprocess
base=pathlib.Path('/Users/numi/numi-offline-bundle-retry-20261108')
run=base/'workspace/.numi/runs/window-train-smoke'
selection=json.loads((run/'selection/selection.json').read_text())
label=selection['selected'] if selection['selected']=='incumbent' else selection['selected_candidate_label']
meta=json.loads((run/f'selection/{label}.evidence.json.meta.json').read_text())
identity=meta['catalog_input_identity']
assert identity['schema']=='numi.catalog-input-identity.v1' and identity['complete'] is True
world=base/'fixture/world.mrworld'; original=world.read_bytes()
assert identity['files']['--world-pack']['sha256']==hashlib.sha256(original).hexdigest()
def decide():
    return json.loads(subprocess.check_output([str(base/'decision-check'),str(run)], text=True))
result={'metadata_complete':True,'initial':decide()}
try:
    world.write_bytes(original+b'modified')
    result['changed_world']=decide()
    world.unlink()
    result['missing_world']=decide()
finally:
    world.write_bytes(original)
result['restored']=decide()
for k in ('initial','changed_world','missing_world','restored'):
    assert result[k]['balance']=='guided held-out balance status unknown' and not result[k]['should_load']
assert world.read_bytes()==original
(base/'postcheck.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
