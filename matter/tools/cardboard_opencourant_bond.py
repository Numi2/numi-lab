#!/usr/bin/env python3
"""Cohesive birth with a deformable solid adherend and free interface DOFs.

Synthetic solver instrument, not a starch or hot-melt material calibration.
The existing all-dormant one-element case is retained as a negative control.
"""
from pathlib import Path
import json
import re
from cardboard_opencourant_adhesion import _starter, _engine, END_TIME, ACTIVATION_TIME


def export_bond_cases(output: Path) -> list[dict]:
    cases = []
    motion = [[(0.,0.),(.0002,.05),(.0006,.05),(.001,.15)]]
    for implicit in (False, True):
        for birth in (False, True):
            stem = ('imp' if implicit else 'exp') + ('_bond_birth' if birth else '_bond_active')
            root = output / stem
            root.mkdir(parents=True, exist_ok=False)
            text = _starter(stem, 'I', motion, activation_time=ACTIVATION_TIME if birth else None)
            # Existing nodes 5-8 are the free inner adherend face. New 9-12
            # are the outer gripper face. Only the outer face is driven.
            extra = '\n'.join(f'{i+9:10d}{x:20g}{y:20g}{1.:20g}' for i,(x,y) in enumerate(((0,0),(1,0),(1,1),(0,1))))
            text = text.replace('/GRNOD/NODE/1', extra+'\n/GRNOD/NODE/1')
            text = text.replace('/IMPDISP/1\nPrescribed Z separation\n', '/IMPDISP/1\nPrescribed outer adherend displacement\n')
            pattern = r'(/IMPDISP/1\n[^\n]*\n#[^\n]*\n)([^\n]*)'
            def outer(m):
                row=m[2];return m[1]+row[:40]+f'{3:10d}'+row[50:]
            text,n = re.subn(pattern,outer,text);assert n==1
            addition=(
                '/MAT/LAW1/2\nSynthetic elastic adherend\n'+f'{1e-6:20g}\n{10000.:20g}{0.:20g}\n'
                '/PROP/TYPE14/2\nFully integrated adherend\n'
                +f'{14:10d}{0:10d}{"":10}{0:10d}{"":10}{222:10d}{"":10}{0:10d}{0.:20g}\n'
                +f'{0.:20g}'*5+'\n'+f'{0.:20g}'*5+'\n'+f'{0:10d}{0:10d}{2:10d}\n'
                '/PART/2\nElastic solid adherend\n         2         2         0\n'
                '/BRICK/2\n'+''.join(f'{v:10d}' for v in (2,5,6,7,8,9,10,11,12))+'\n'
                '/GRNOD/NODE/3\nOuter boundary gripper\n'+''.join(f'{v:10d}' for v in (9,10,11,12))+'\n'
                '/BCS/3\nRestrain outer in-plane axes\n'+f'{"110 000":>10}{0:10d}{3:10d}\n'
                '/TH/NODE/104\nouter gripper history\n'+''.join(f'{v:<10}' for v in ('DZ','VZ','REACZ'))+'\n'
                +''.join(f'{i:10d}{0:10d}\n' for i in range(9,13))
                +'/TH/BRIC/105\nsolid adherend state\nIE        STRESS    STRAIN\n         2\n')
            text=text.replace('/END',addition+'/END')
            text='\n'.join(l for l in text.splitlines() if l.strip())+'\n'
            starter=stem+'_0000.rad';engines=[stem+'_0001.rad']
            (root/starter).write_text(text)
            engine=_engine(stem,1,END_TIME,interval=2e-5)
            if implicit:
                # Fixed maximum increment through growth factor 1.0. The
                # physical time and motion are identical to explicit controls.
                engine=re.sub(r'/DT/NODA/STOP\n[^\n]*\n','',engine)
                engine=('/IMPL/NONL/01\n20 12 1e-6 1e-6\n'
                        '/IMPL/DTINI\n0.00001\n/IMPL/DT/1\n4 1.0 15 0.6666667\n'
                        '/IMPL/SOLV/01\n1 0 3 1e-10\n/IMPL/PRINT/NONLIN/1\n'+engine)
            (root/engines[0]).write_text(engine)
            spec={'schema':'cardboard.opencourant.bond.v1','id':stem,'starter':starter,'engines':engines,
                  'expected_end_time_s':END_TIME,'implicit':implicit,'activation':birth,
                  'purpose':'Native equilibrium and accepted-geometry cohesive birth with an active solid adherend',
                  'prediction':{'birth_time_s':ACTIVATION_TIME if birth else None,
                                'pre_birth_gap_mm':.05 if birth else 0.,
                                'birth_hold_force_N':0. if birth else 5./1.01,
                                'final_force_N':10./1.01 if birth else None,
                                'cohesive_stiffness_N_per_mm':100.,'adherend_stiffness_N_per_mm':10000.,
                                'OFF_active':1,'OFF_inactive':0},
                  'gates':['Complete pre-birth, hold, post-birth histories with finite free interface displacement',
                           'Zero dormant/activation-hold force; post-birth traction follows incremental measured interface gap',
                           'Native residual convergence for implicit; no added mass; retain energy and restart histories'],
                  'classification':'synthetic instrument, not physical box qualification','mass_scaling':False}
            (root/'case.json').write_text(json.dumps(spec,indent=2)+'\n')
            cases.append(spec)
    return cases


def export_staged_birth(source: Path, output: Path) -> dict:
    """Native restart around bond setting, without resetting any mesh/history.

    Absence of /IMPL in the second Engine selects explicit integration; this
    is the same transition documented in Radioss example RD-E:5501.
    """
    stem = 'staged_bond_birth'
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((source / 'case.json').read_text())
    starter = stem + '_0000.rad'
    text = (source / original['starter']).read_text().replace(original['id'], stem)
    (output / starter).write_text(text)
    engines = []
    # Keep the explicit/implicit transition strictly inside the stationary
    # hold: explicit time integration may finish one stable step beyond RUN.
    for stage, end, implicit in ((1, .00049, True), (2, .00055, False), (3, .001, True)):
        name = f'{stem}_{stage:04d}.rad'
        engine = _engine(stem, stage, end, interval=1e-5)
        if implicit:
            engine = re.sub(r'/DT/NODA/STOP\n[^\n]*\n', '', engine)
            # Native force residual remains 1e-6. Relative increment-energy
            # is undefined at the stress-free hold (both work and residual
            # energy are roundoff). Absolute energy continuity is measured
            # across the checkpoint independently, not waived.
            engine = ('/IMPL/NONL/01\n20 2 1e-6 1e-6\n'
                      '/IMPL/DTINI\n0.00001\n/IMPL/DT/1\n4 1.0 15 0.6666667\n'
                      '/IMPL/SOLV/01\n1 0 3 1e-10\n/IMPL/PRINT/NONLIN/1\n' + engine)
        (output / name).write_text(engine)
        engines.append(name)
    original.update(id=stem, starter=starter, engines=engines,
                    implicit='staged implicit-explicit-implicit',
                    stage_end_times_s=[.00049, .00055, .001],
                    transition_reference='https://2022.help.altair.com/2022.3/hwsolvers/rad/topics/solvers/rad/fan_blade_rotation_initialization_example_r.htm')
    original['gates'].append('Native restart preserves active bond, stress-free birth gap and energy across both scheme changes')
    (output / 'case.json').write_text(json.dumps(original, indent=2)+'\n')
    return original


def export_rotated_birth(source: Path, output: Path) -> dict:
    """Held-out frame covariance: cyclic XYZ rotation of the staged fixture."""
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((source / 'case.json').read_text())
    old, stem = spec['id'], spec['id'] + '_rotated'
    text = (source / spec['starter']).read_text().replace(old, stem)
    start, end = text.index('/NODE\n')+len('/NODE\n'), text.index('/GRNOD/NODE/1')
    rows = []
    for row in text[start:end].splitlines():
        if not row.strip() or row.startswith('#'):
            rows.append(row)
        else:
            nid, x, y, z = row.split()
            rows.append(f'{int(nid):10d}{float(z):20g}{float(x):20g}{float(y):20g}')
    text = text[:start]+'\n'.join(rows)+'\n'+text[end:]
    text = text.replace('110 000', '011 000')
    text = text.replace(f'{"Z":>10}', f'{"X":>10}')
    # All-axis top and bottom histories are retained. Only the outer gripper
    # requests a single axis; make that request match the rotated normal.
    text = text.replace('DZ        VZ        REACZ', 'DX        VX        REACX')
    starter = spec['starter'].replace(old, stem)
    (output / starter).write_text(text)
    engines = []
    for name in spec['engines']:
        new = name.replace(old, stem)
        (output/new).write_text((source/name).read_text().replace(old, stem))
        engines.append(new)
    spec.update(id=stem, starter=starter, engines=engines,
                rigid_frame_map='(x,y,z) -> (z,x,y); det R=+1', normal_axis='X',
                held_out_of_patch_diagnosis=True)
    (output/'case.json').write_text(json.dumps(spec, indent=2)+'\n')
    return spec


if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    print(json.dumps(export_bond_cases(root),indent=2))
