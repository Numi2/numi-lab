#!/usr/bin/env python3
"""Exact source151 array-shape admission arithmetic for the planned 464 capture."""
from __future__ import annotations
import hashlib,json,struct
from pathlib import Path
D=Path(__file__).resolve().parent
inv=json.loads((D/'compiled-package-topology-inventory.json').read_text())
layout=json.loads((D/'graph-layout.json').read_text())
E=inv['dispatch']['environment_count']; O=inv['dispatch']['object_count']; N=inv['dispatch']['node_count']; T=inv['dispatch']['tetrahedron_count']; S=inv['dispatch']['material_state_stride']; K=26; KS=27
assert (E,O,N,T,S)==(2,1,882,1920,15)
sh={str(k):v['element_count'] for k,v in inv['sections'].items()}
# Payload array insertion order mirrors source runtime.mm append calls. Entries=(name,count,element bytes).
state=[]
def add(dst,name,count,width): dst.append((name,count,width))
add(state,'accepted_nodes',E*N,80); add(state,'candidate_nodes',E*N,80)
add(state,'accepted_tetrahedra',E*T,96); add(state,'candidate_tetrahedra',E*T,96)
add(state,'accepted_material_history',E*T*S,4); add(state,'candidate_material_state_buffer',E*T*S,4)
add(state,'objects',max(sh['10'],1),176); add(state,'materials',max(sh['3'],1),160)
add(state,'scalar_programs',max(sh['9'],1),16); add(state,'expression_instructions',max(sh['8'],1),32)
add(state,'environment_parameters',E*inv['dispatch']['parameter_count'],4)
add(state,'schedulers',E*O,80); add(state,'adaptive_state',E*O,160)
U=2*E*N
add(state,'original_newton_correction',U,16); add(state,'base_residual_rhs',U,16)
add(state,'algorithmic_tangent_cache',E*T*81,4); add(state,'algorithmic_tangent_cache_valid',E*T,4)
add(state,'fgmres_state',E,32); add(state,'mixed_solver',1,96)
add(state,'armijo_metrics',E,16); add(state,'armijo_statuses',E,48); add(state,'production_statuses',E,48)
add(state,'fem_line_search',E*O,16); add(state,'environment_line_search',E,4)
regional=[]
add(regional,'P',K*N,16); add(regional,'effective_P',E*KS*N,16); add(regional,'AP',E*KS*N,16)
add(regional,'base_RAP_lu',E*K*K,4); add(regional,'base_RAP_pivots',E*K,4); add(regional,'base_RAP_valid',E,4)
add(regional,'base_coarse_rhs',E*K,4); add(regional,'base_coarse_coefficients',E*K,4)
# Augmentation buffers are included by the actual 462 source151 settings.
for name,count,width in [('augmented_RAP_lu',E*KS*KS,4),('augmented_RAP_pivots',E*KS,4),('augmented_RAP_valid',E,4),('augmented_coarse_rhs',E*KS,4),('augmented_coarse_coefficients',E*KS,4),('restart_generation',E,16),('restart_mode_valid',E,4),('restart_tail_preserved',E,4),('restart_target_scale',E,4),('restart_raw_norm_squared',E,4),('restart_mode_scale',E,4)]: add(regional,name,count,width)
G=layout
for name,count,width in [('graph_patches',G['patch_count'],16),('graph_patch_nodes',G['patch_node_count'],4),('graph_rows',G['dof_count'],16),('graph_node_patch_offsets',N+1,4),('graph_node_patch_values',G['patch_node_count'],4),('graph_node_coverage',N,4),('graph_original_matrices',E*G['matrix_float_count'],4),('graph_patch_lu',E*G['matrix_float_count'],4),('graph_pivots',E*G['pivot_count'],4),('graph_factor_valid',E*G['patch_count'],4)]: add(regional,name,count,width)
# The CGLS primary selection word is appended only if primary policy is selected
# at the actual pre-install capture point. Budget its maximum case.
add(regional,'primary_policy_selected_environment',E,4)
def arena_bytes(arrays):
    cursor=0; records=[]
    for name,count,width in arrays:
        offset=(cursor+15)&~15; size=count*width
        records.append({'name':name,'count':count,'element_bytes':width,'offset':offset,'bytes':size})
        cursor=offset+size
    return cursor,records
state_bytes,state_records=arena_bytes(state)
regional_bytes,regional_records=arena_bytes(regional)
# Source code cgls snapshot arena: eight u32 header words, seven U*float4,
# U*u32, E*float4, E*uint4, and two E*u32 vectors; optionally E*N*3*float4 right scale.
vectorBytes=U*16
cgls_sizes=[vectorBytes]*7+[U*4,E*16,E*16,E*4,E*4]
cgls=32
for size in cgls_sizes+[E*N*3*16]:
    cgls=(cgls+15)&~15; cgls+=size
trace=E*(16+4)
aggregate=state_bytes+regional_bytes+cgls+trace
if len(state)!=24 or len(regional)!=30: raise SystemExit(f'unexpected array counts state={len(state)} regional={len(regional)}')
report={
 'schema':'numi.cardboard.native464.capture-preflight.v1',
 'source_snapshot':'source-cgls-regional-state-capture-151',
 'compiled_package_sha256':inv['compiled_package_sha256'],
 'graph_layout':G,
 'source151_expected_roles':'environment 0 active/eligible, environment 1 quiet per corrected 463 audit; actual 464 capture must validate selection/health words and must not rely on this expectation alone',
 'regional_augmentation_expected':{'enabled':True,'base_modes':K,'mode_stride':KS,'augmented_modes':KS,'cycle_currentness':'unadmitted by reader v2: raw tag cycle records producer cycle c; consumer uses c+1, and capture has no independent consumed-cycle value'},
 'arrays':{'state_count':len(state),'state_payload_bytes':state_bytes,'state':state_records,'regional_max_count':len(regional),'regional_payload_bytes':regional_bytes,'regional':regional_records},
 'additional_capture_scratch':{'cgls_snapshot_bytes':cgls,'trace_bytes':trace,'right_scale_included':True},
 'aggregate_readback_bytes':aggregate,'aggregate_cap_bytes':96*1024*1024,
 'admission':'planned checked upper bound; actual native capture must report complete and pass runtime cap; this is not a runtime or physical qualification',
}
report['aggregate_fits_96MiB']=aggregate <= 96*1024*1024
(D/'capture-capacity-preflight.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:report[k] for k in ('arrays','additional_capture_scratch','aggregate_readback_bytes','aggregate_fits_96MiB')},indent=2))
