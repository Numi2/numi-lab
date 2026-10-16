from pathlib import Path
import difflib
import hashlib
import json
import shutil

B=Path('/Users/home/cardboard-implementation-20261014')
S=B/'snapshots/source-right-grip-telemetry-157'
O=B/'analysis/source167-test-registration-001'
O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=(S/'matter/CMakeLists.txt').read_text()
needle='foreach(cardboard_helper shipping-driver shipping-memory shipping-closure-mesh shipping-snapshot)'
assert source.count(needle)==1
source=source.replace(needle,needle[:-1]+' coupon-resume inside-clearance shipping-clearance-memory regional-consumer-capture)')
start=source.index('foreach(cardboard_helper ')
end=source.index('endforeach()',start)
part=source[start:end]
assert part.count('-Wall -Wextra -Wpedantic -Werror)')==1
source=source[:start]+part.replace('-Wall -Wextra -Wpedantic -Werror)',
    '-Wall -Wextra -Wpedantic -Werror -UNDEBUG)')+source[end:]
files={'matter/CMakeLists.txt':source}
origins=[]
for package,old_name,new_name,old_include,new_include in [
    ('coupon-accepted-resume-overlay-002','coupon_resume_contract_test.cpp','coupon_resume',
     '../candidate/matter/tools/cardboard_coupon_resume.hpp','cardboard_coupon_resume.hpp'),
    ('source159-inside-surface-role-map-overlay-003','inside_clearance_check.cpp','inside_clearance',None,None),
    ('source159-inside-surface-role-map-overlay-003','shipping_clearance_memory_check.cpp','shipping_clearance_memory',
     '../candidate/matter/tools/cardboard_shipping_memory.hpp','cardboard_shipping_memory.hpp'),
    ('selector5-restart-consumer-capture-source160-002','regional_consumer_capture_contract_test.cpp','regional_consumer_capture',None,None)]:
    origin=B/'analysis'/package/'tests'/old_name
    text=origin.read_text()
    if old_include:
        assert text.count(old_include)==1
        text=text.replace(old_include,new_include)
    relative='matter/tools/cardboard_'+new_name+'_check.cpp'
    files[relative]=text
    origins.append(dict(origin=str(origin),origin_sha256=sha(origin),path=relative,
                        transformation='checkout-local include only' if old_include else 'byte-identical'))
patch='';rows=[]
for relative,text in files.items():
    before=S/relative
    old=before.read_text() if before.is_file() else ''
    target=O/'candidate'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(text)
    patch+=''.join(difflib.unified_diff(old.splitlines(keepends=True),text.splitlines(keepends=True),
        fromfile='a/'+relative if before.is_file() else '/dev/null',tofile='b/'+relative))
    rows.append(dict(path=relative,base_sha256=sha(before) if before.is_file() else None,overlay_sha256=sha(target)))
(O/'overlay.patch').write_text(patch)
(O/'source-overlay.json').write_text(json.dumps(dict(schema='numi.cardboard.source-overlay.v1',
    overlay_id=O.name,base_snapshot=str(S),patch_sha256=sha(O/'overlay.patch'),files=rows,
    test_origins=origins,
    scope='Register retained CPU contract checks as portable native CMake targets; keep assertions active in Release checks.',
    native_execution_performed=False,physical_qualification=False),indent=2)+'\n')
shutil.copy2(Path(__file__),O/'prepare.py')
print(json.dumps(dict(source_overlay_sha256=sha(O/'source-overlay.json'),patch_sha256=sha(O/'overlay.patch'))))
