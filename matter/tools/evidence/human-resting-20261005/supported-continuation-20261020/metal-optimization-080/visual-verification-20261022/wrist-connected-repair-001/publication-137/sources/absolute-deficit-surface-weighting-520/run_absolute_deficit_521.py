"""Thin path adapter for frozen 520; preflight can stop before any fit work."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
ENTRY_520=D/'D/D/absolute-deficit-surface-weighting-520/run_absolute_deficit_520.py'
FROZEN_513_HELPER=D/'D/D/anchor-compatible-finite-triangle-513/anchor_finite_triangle_513.py'
PREFLIGHT_FLAG='--preflight-only-521'
EXPECTED_ENTRY_520_SHA='ed8f875fb20f37c8326047f131476ce9e364851afe557d21958617a894d15feb'
EXPECTED_513_HELPER_SHA='a6aacc460e71e85ccee2efd9a9183237eb5a832daa2c32a6990ab5003ba18241'

def load(path):
    spec=importlib.util.spec_from_file_location('absolute_deficit_520_for_521',path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load frozen 520 entry: {path}')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

class _ReachedFitCall(Exception):
    pass

def main():
    preflight = sys.argv[1:].count(PREFLIGHT_FLAG)==1
    if sys.argv[1:].count(PREFLIGHT_FLAG)>1:
        raise ValueError(f'{PREFLIGHT_FLAG} may appear at most once')
    if not ENTRY_520.is_file() or not FROZEN_513_HELPER.is_file():
        raise FileNotFoundError('521 adapter dependency is missing')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(ENTRY_520)!=EXPECTED_ENTRY_520_SHA:
        raise ValueError('521 frozen 520 entry hash mismatch')
    if sha(FROZEN_513_HELPER)!=EXPECTED_513_HELPER_SHA:
        raise ValueError('521 frozen 513 helper hash mismatch')
    module=load(ENTRY_520)
    # The 520 copy derived 513's helper path from its own directory. Resolve
    # that dependency to the frozen, hash-pinned 513 owner before main runs.
    module.HELPER_513=FROZEN_513_HELPER
    if not preflight:
        module.main()
        return
    args=[arg for arg in sys.argv[1:] if arg != PREFLIGHT_FLAG]
    sys.argv[:]=[sys.argv[0],*args]
    original_load=module.load
    holder={}
    def load_intercept(name,path):
        loaded=original_load(name,path)
        if name=='fit505_for_508':
            holder['entry505']=loaded
            def stop_before_fit():
                raise _ReachedFitCall()
            loaded.main=stop_before_fit
        return loaded
    module.load=load_intercept
    try:
        module.main()
    except _ReachedFitCall:
        entry505=holder.get('entry505')
        if entry505 is None:
            raise RuntimeError('preflight reached fit sentinel without loading 505')
        print(json.dumps({
            'status':'CLI_PREFLIGHT_PASSED_NO_FIT',
            'entry_521':str(Path(__file__).resolve()),
            'delegated_entry_520':str(ENTRY_520.resolve()),
            'delegated_entry_520_sha256':sha(ENTRY_520),
            'finite_triangle_helper':str(module.HELPER_513.resolve()),
            'finite_triangle_helper_sha256':sha(module.HELPER_513),
            'all_pair_mode':'--anchor-compatible-finite-triangle-all-pairs-513' in args,
            'fit_started':False,
            'global_surface_clearance_certified':False,
        },sort_keys=True),flush=True)
        return
    raise RuntimeError('521 preflight did not reach the intercepted pre-fit boundary')

if __name__=='__main__':
    main()
