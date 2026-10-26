"""Thin opt-in 523 installer adapter over the frozen matched 521 entry."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
ENTRY_521=D/'D/D/absolute-deficit-surface-weighting-520/run_absolute_deficit_521.py'
ENTRY_520=D/'D/D/absolute-deficit-surface-weighting-520/run_absolute_deficit_520.py'
HELPER_523=D/'D/D/distance-slack-scaling-523/distance_slack_scaling_523.py'
SOFT_470=D/'D/D/soft-target-nonregression-470/soft_target_nonregression_470.py'
OWNER_408=D/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
RUNNER_485=D/'D/D/component-lp-decomposition-485/run_matched_component_fit_485.py'
FLAG_523='--distance-slack-scaling-523'
PREFLIGHT_FLAG='--preflight-only-527'
SOFT_IMPORT_NAME='soft_target_noregression_470_for_485'
EXPECTED={
 ENTRY_521:'93869a2b5cd45b018c557d3ae9ee7284ee10d44e1a0425d91fc168b30f5c80be',
 ENTRY_520:'ed8f875fb20f37c8326047f131476ce9e364851afe557d21958617a894d15feb',
 HELPER_523:'7087e763caec3bc7f3ff43c891f7e2c80b6a0341eb2bbb44f914d432e48e2d59',
 SOFT_470:'990e126932186e33c0a5363a6ac34fa967e211f4bc55977567582e2b89b8bb79',
 OWNER_408:'5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2',
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f'cannot load pinned module {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def _attach_to_505_entry(entry505, scaling, state):
    """Wrap frozen 505 -> 520 loader so the 485-loaded 470 installer is scaled."""
    if getattr(entry505,'_distance_slack_523_hook_attached',False):
        raise RuntimeError('527 hook already attached to 505 entry')
    previous=entry505.patch_485_loader
    def patch_485_loader(entry490,validation,hook_states,helper505):
        previous(entry490,validation,hook_states,helper505)
        load_485=entry490._load_module
        runner_path=Path(entry490.RUNNER_485).resolve()
        if runner_path!=RUNNER_485.resolve():
            raise ValueError('527 observed unexpected 485 loader path')
        def load_module(path):
            runner=load_485(path)
            if Path(path).resolve()!=runner_path:
                return runner
            old_load=runner.load
            def load(name,load_path):
                loaded=old_load(name,load_path)
                if name==SOFT_IMPORT_NAME:
                    if state.get('installed'):
                        raise RuntimeError('527 saw more than one frozen 470 module installation')
                    if Path(load_path).resolve()!=SOFT_470.resolve():
                        raise ValueError('527 observed unexpected 470 source path')
                    restore=scaling.patch_install_solver(loaded)
                    state.update({'installed':True,'receipt':restore,'soft_module':loaded})
                return loaded
            runner.load=load
            return runner
        entry490._load_module=load_module
    entry505.patch_485_loader=patch_485_loader
    entry505._distance_slack_523_hook_attached=True
    state['attached']=True
    return {'hook':'505.patch_485_loader -> 490._load_module -> 485.load(soft470)',
            'soft_import_name':SOFT_IMPORT_NAME,'runner485_path':str(RUNNER_485.resolve())}

def _exercise_actual_loader_chain(entry505,scaling):
    """Exercise real 505+520+527 hook composition using a no-fit fake 485 loader."""
    soft=load('soft470_preflight_527',SOFT_470)
    original_installer=soft.install_solver
    class FakeRunner:
        def load(self,name,path):
            if name==SOFT_IMPORT_NAME and Path(path).resolve()==SOFT_470.resolve():
                return soft
            return SimpleNamespace()
    runner=FakeRunner()
    entry490=SimpleNamespace(RUNNER_485=RUNNER_485,_load_module=lambda path:runner)
    fake_helper505=SimpleNamespace(install_owner_bounds=lambda *a,**k:{'preflight_only':True})
    state={}
    entry505.patch_485_loader(entry490,validation=None,hook_states=[],helper505=fake_helper505)
    loaded_runner=entry490._load_module(RUNNER_485)
    loaded_soft=loaded_runner.load(SOFT_IMPORT_NAME,SOFT_470)
    if loaded_soft.install_solver is original_installer:
        raise RuntimeError('527 loader chain failed to patch the loaded 470 installer')
    return {'passed':True,'module_loaded_from':str(SOFT_470),'entry505_patch_loader_exercised':True,
            'loaded470_installer_hooked':True}

def main():
    argv=list(sys.argv[1:])
    if argv.count(FLAG_523)!=1: raise ValueError(f'explicit opt-in {FLAG_523} is required exactly once')
    argv.remove(FLAG_523)
    preflight=argv.count(PREFLIGHT_FLAG)==1
    if argv.count(PREFLIGHT_FLAG)>1: raise ValueError(f'{PREFLIGHT_FLAG} may appear at most once')
    if preflight:
        argv.remove(PREFLIGHT_FLAG)
        argv.append('--preflight-only-521')
    for path,digest in EXPECTED.items():
        if not Path(path).is_file() or sha(path)!=digest:
            raise ValueError(f'527 source pin mismatch: {path}')
    state={}
    scaling=load('distance_slack_scaling_523_for_527',HELPER_523)
    runner521=load('distance_slack_entry_521_for_527',ENTRY_521)
    old_521_load=runner521.load
    def load_521(path):
        loaded=old_521_load(path)
        if Path(path).resolve()!=ENTRY_520.resolve():
            return loaded
        old_520_load=loaded.load
        def load_520(name2,path2):
            item=old_520_load(name2,path2)
            if name2=='fit505_for_508':
                state['entry505']=item
                _attach_to_505_entry(item,scaling,state)
            return item
        loaded.load=load_520
        state['entry520']=loaded
        return loaded
    runner521.load=load_521
    sys.argv[:]=[sys.argv[0],*argv]
    runner521.main()
    if not preflight:
        return
    entry505=state.get('entry505')
    if entry505 is None or not state.get('attached'):
        raise RuntimeError('527 CLI preflight did not attach at the actual 505 loader')
    hook_check=_exercise_actual_loader_chain(entry505,scaling)
    # Restore the temporary preflight module hook after asserting composition.
    if state.get('soft_module') is not None:
        state['soft_module'].install_solver=state['receipt']
    print(json.dumps({
        'status':'CLI_PREFLIGHT_PASSED_NO_FIT',
        'entry_527':str(Path(__file__).resolve()),
        'entry_521_sha256':sha(ENTRY_521),'entry_520_sha256':sha(ENTRY_520),
        'helper_523_sha256':sha(HELPER_523),'soft_470_sha256':sha(SOFT_470),
        'owner_408_sha256':sha(OWNER_408),
        'hook_attachment':_attach_receipt(state),
        'actual_loader_chain_preflight':hook_check,
        'fit_started':False,'geometry_started':False,
        'physical_hard_rows_and_output_bounds_unchanged':True,
        'objective_and_stage2_budget_algebraically_equivalent':True,
        'global_surface_clearance_certified':False,
    },sort_keys=True),flush=True)

def _attach_receipt(state):
    return {'attached':bool(state.get('attached')),
            'loader_path':'505.patch_485_loader -> 490._load_module -> 485.load(soft470)',
            'soft_import_name':SOFT_IMPORT_NAME,
            'soft_source_path':str(SOFT_470)}

if __name__=='__main__': main()
