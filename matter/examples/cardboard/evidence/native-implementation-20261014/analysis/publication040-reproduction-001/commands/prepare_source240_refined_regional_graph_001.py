from pathlib import Path
import json,shutil,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-shipping-resolution-controls-239-002';s=q/'snapshots/source-refined-regional-graph-240-001';d=q/'analysis/source240-refined-regional-graph-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir();d.mkdir()
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
n='matter/src/regional_graph_schwarz.hpp';p=s/n;t=p.read_text()
start=t.index('[[nodiscard]] inline RegionalGraphSchwarzLayout buildBudgetedRegionalGraphSchwarzLayout(')
end=t.index('\n} // namespace',start)
old=t[start:end]
new=old.replace('const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull\n','const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull,\n    const bool extendedPureFEM = false\n')
new=new.replace('    if (!resourceRejection) return original;','    if (!resourceRejection && !extendedPureFEM) return original;')
new=new.replace('    for (const std::uint32_t patchNodes : {256u, 128u, 64u, 32u}) {','    if (resourceRejection) for (const std::uint32_t patchNodes : {256u, 128u, 64u, 32u}) {')
new=new.replace('    return original;\n}', '''    // The caller must prove the static contact-free pure FEM scope before
    // enabling this path. The same checked byte budget is retained. Smaller
    // graph solves affect conditioning only; every residual and operator
    // certificate still uses the original full FEM equations.
    if (extendedPureFEM) {
        for (const std::uint32_t patchNodes : {256u, 128u, 64u, 32u, 16u, 8u, 4u}) {
            auto candidate = buildRegionalGraphSchwarzLayout(
                nodeCount, tetrahedra, freeMask, environmentCount,
                maximumDeviceBytes, 32768u, 131072u, 16384u,
                patchNodes, patchNodes / 2u, true, true);
            if (candidate.error.empty()) return candidate;
            if (candidate.error != "graph Schwarz patch count exceeds its bounded layout" &&
                candidate.error != "graph Schwarz packed device arenas, including source matrices for one refinement pass, exceed the explicit 96 MiB bound")
                return candidate;
        }
    }
    return original;
}''')
assert old!=new and 'extendedPureFEM = false' in new;t=t[:start]+new+t[end:];p.write_text(t)
n='matter/src/runtime.mm';p=s/n;t=p.read_text()
old='[[nodiscard]] RegionalGraphSchwarzHostLayout buildRegionalGraphSchwarz(\n    const CompiledWorld& world\n)'
assert t.count(old)==1;t=t.replace(old,old.replace('const CompiledWorld& world','const CompiledWorld& world, const bool extendedPureFEM'))
old='nodes, tetrahedra, result.sourceFreeMask,\n        world.dispatch.environmentCount);';assert t.count(old)==1;t=t.replace(old,'nodes, tetrahedra, result.sourceFreeMask,\n        world.dispatch.environmentCount, 96ull * 1024ull * 1024ull,\n        extendedPureFEM);')
start=t.index('        // Expanded coupon storage is confined')
end=t.index('        const detail::FEMTrueResidualArenaShape trueResidualArenaShape{',start)
block=t[start:end];t=t[:start]+t[end:]
anchor='        RegionalGraphSchwarzHostLayout graphSchwarz;';assert t.count(anchor)==1;t=t.replace(anchor,block+anchor)
old='graphSchwarz = buildRegionalGraphSchwarz(world);';assert t.count(old)==1;t=t.replace(old,'graphSchwarz = buildRegionalGraphSchwarz(\n                world, candidate->femExtendedPureTrueResidualLayout);')
p.write_text(t)
n='matter/tools/cardboard_probe.mm';p=s/n;t=p.read_text()
t=t.replace('graph_schwarz_coupon_node_cap\\\": 4096','graph_schwarz_coupon_node_cap\\\": 32768')
t=t.replace('graph Schwarz coupon <=4096 nodes / <=32768 tets','graph Schwarz pure coupon <=32768 nodes / <=131072 tets')
p.write_text(t)
changed=['matter/src/regional_graph_schwarz.hpp','matter/src/runtime.mm','matter/tools/cardboard_probe.mm']
for n in changed:
 f['source_sha256'][n]=sha(s/n)
 if n in cfg['files']:cfg['files'][n]=sha(s/n)
f['source240_refined_regional_graph']={'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'changed':changed,'pure_caps':{'nodes':32768,'tetrahedra':131072,'patches':16384,'device_bytes':96*1024*1024},'legacy_admitted_layouts_retained_exactly':True,'pure_scope_proof_reused_from_true_residual_arena':True,'physical_operator_changed':False,'acceptance_criteria_changed':False,'qualification':'CPU exact layout and native preconditioner evidence required; this declaration does not qualify mesh convergence.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
r={'source_snapshot':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'changed':changed,'gpu_execution':False}
(d/'preparation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
