from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-strict-local-projection-main-237-001';s=q/'snapshots/source-refined-coupon-arenas-238-001';d=q/'analysis/source238-refined-coupon-arena-preparation-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/'frozen-source.json')=='ab2511b4b6992854556059676aa650769a43090f504c9cc5ead6a94e45067d6f'
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
changed=[]
def edit(n,pairs):
 p=s/n;t=p.read_text()
 for a,b,count in pairs:
  assert t.count(a)==count,(n,a,t.count(a),count);t=t.replace(a,b)
 p.write_text(t);changed.append(n)
edit('matter/include/numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp',[
 ('kFEMPureArmijoMaximumScratchBytes=16ull','kFEMPureArmijoMaximumScratchBytes=64ull',1),
 ('s.femNodeCount>8192u','s.femNodeCount>32768u',1),('s.tetrahedronCount>32768u','s.tetrahedronCount>131072u',1),
 ('pure_fem_armijo_node_cap_8192','pure_fem_armijo_node_cap_32768',1),('pure_fem_armijo_tetrahedron_cap_32768','pure_fem_armijo_tetrahedron_cap_131072',1),
 ('pure_fem_armijo_cgls_scratch_exceeds_16_mib_cap','pure_fem_armijo_cgls_scratch_exceeds_64_mib_cap',1)])
edit('matter/include/numi/matter/detail/true_residual_arena_plan.hpp',[
 ('rigidGeneralizedCapacity=0u,maxBufferLength=0u;','rigidGeneralizedCapacity=0u,maxBufferLength=0u; bool extendedPureFEM=false;',1),
 ('none=0u,environment_count,','none=0u,extended_pure_layout,environment_count,',1),
 ('std::uint64_t r,FEMTrueResidualArenaPlan& p) noexcept {','std::uint64_t r,FEMTrueResidualArenaPlan& p,bool extendedPureFEM=false) noexcept {',1),
 (' if(!e||e>4u)', ' if(extendedPureFEM&&(c||r)){p.failure=FEMTrueResidualArenaFailure::extended_pure_layout;return false;}\n if(!e||e>4u)',1),
 ('if(!n||n>8192u)','if(!n||n>(extendedPureFEM?32768u:8192u))',1),
 ('if(!t||t>32768u)','if(!t||t>(extendedPureFEM?131072u:32768u))',1),
 ('FEMTrueResidualArenaPlan a{},b{};','FEMTrueResidualArenaPlan a{},b{},c{};',1),
 ('return a.aggregateBytes>b.aggregateBytes?a.aggregateBytes:b.aggregateBytes;','if(!femTrueResidualPlanSupportedCounts(4u,32768u,131072u,0u,0u,c,true))return 0u;return std::max(c.aggregateBytes,std::max(a.aggregateBytes,b.aggregateBytes));',1),
 ('s.rigidGeneralizedCapacity,p))','s.rigidGeneralizedCapacity,p,s.extendedPureFEM))',1),
 ('switch(f){case FEMTrueResidualArenaFailure::none:', 'switch(f){case FEMTrueResidualArenaFailure::extended_pure_layout:return "extended_true_residual_requires_no_cohesive_or_rigid_unknowns";case FEMTrueResidualArenaFailure::none:',1),
 ('#include <cstddef>','#include <algorithm>\n#include <cstddef>',1)])
n='matter/src/runtime.mm';p=s/n;t=p.read_text()
def rep(a,b,count=1):
 global t
 assert t.count(a)==count,(a,t.count(a),count);t=t.replace(a,b)
rep('bool femPureFEMArmijo8192ArenaAdmitted = false;','bool femPureFEMArmijo8192ArenaAdmitted = false;\n    bool femExtendedPureTrueResidualLayout = false;')
needle='        const detail::FEMTrueResidualArenaShape trueResidualArenaShape{'
extra="""        // Expanded coupon storage is confined to the same static contact-free
        // FEM layout as residual Armijo. Coupled/contact extents stay bounded
        // by their existing admission; allocation alone never admits a step.
        constexpr std::uint32_t extendedPureUnsupportedFlags =
            NM_MATTER_CONTACT | NM_MATTER_ADAPTIVE | NM_MATTER_IDENTIFICATION |
            NM_MATTER_MIXED_FEM | NM_MATTER_MULTIPHYSICS | NM_MATTER_MUTATION |
            NM_MATTER_LEARNED_MATERIAL;
        candidate->femExtendedPureTrueResidualLayout =
            candidate->objectLayout.size() == 1u &&
            candidate->objectLayout[0].representation == NM_REPRESENTATION_FEM &&
            (candidate->objectLayout[0].flags &
                (NM_OBJECT_ADAPTIVE | NM_OBJECT_IDENTIFIABLE | NM_OBJECT_MIXED_FEM |
                 NM_OBJECT_MULTIPHYSICS | NM_OBJECT_MUTABLE_TOPOLOGY)) == 0u &&
            (candidate->objectLayout[0].flags &
                (NM_OBJECT_DISABLE_SELF_CONTACT | NM_OBJECT_DISABLE_DEFORMABLE_CONTACT)) ==
                (NM_OBJECT_DISABLE_SELF_CONTACT | NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) &&
            (candidate->dispatch.flags & extendedPureUnsupportedFlags) == 0u &&
            candidate->dispatch.rigidProxyCount == 0u &&
            candidate->dispatch.contactPairCount == 0u &&
            candidate->dispatch.cohesiveFaceCount == 0u &&
            candidate->dispatch.rigidGeneralizedCapacity == 0u &&
            candidate->dispatch.rigidQCapacity == 0u &&
            candidate->dispatch.fieldBoundaryCount == 0u &&
            candidate->dispatch.punctureChannelCount == 0u &&
            candidate->dispatch.mpmActiveNodeCapacity == 0u &&
            candidate->dispatch.particleCount == 0u &&
            candidate->dispatch.mpmGridCount == 0u &&
            candidate->dispatch.mpmBlockCount == 0u;
"""
rep(needle,extra+needle)
rep('static_cast<std::uint64_t>(candidate->device.maxBufferLength)};','static_cast<std::uint64_t>(candidate->device.maxBufferLength),\n            candidate->femExtendedPureTrueResidualLayout};')
rep('candidate->dispatch.femNodeCount <= 8192u &&\n            candidate->dispatch.tetrahedronCount <= 32768u &&','candidate->dispatch.femNodeCount <= 32768u &&\n            candidate->dispatch.tetrahedronCount <= 131072u &&')
rep('return maximumFEMNodeCount > 4096u\n                        ? "fem_node_cap_8192"','return maximumFEMNodeCount > 8192u\n                        ? "fem_node_cap_32768"\n                        : maximumFEMNodeCount > 4096u\n                        ? "fem_node_cap_8192"')
rep('if (state.dispatch.tetrahedronCount > 32768u ||\n                    state.femTrueResidualElementOperator == nil ||','const bool extendedPureArena = state.femExtendedPureTrueResidualLayout &&\n                    state.femPureFEMArmijo8192ArenaAdmitted;\n                if (state.dispatch.tetrahedronCount > (extendedPureArena ? 131072u : 32768u) ||\n                    state.femTrueResidualElementOperator == nil ||')
rep('static_cast<std::uint64_t>(state.device.maxBufferLength)};\n                const auto arenaPlan','static_cast<std::uint64_t>(state.device.maxBufferLength),\n                    extendedPureArena};\n                const auto arenaPlan')
rep('state.dispatch.rigidGeneralizedCapacity == 6u, 8192u);','state.dispatch.rigidGeneralizedCapacity == 6u,\n                    extendedPureArena ? 32768u : 8192u);')
rep('femNewtonTraceScope(false, 8192u)','femNewtonTraceScope(false, 32768u)')
rep('"pure_fem_armijo_8192_checked_arena_not_admitted"','"pure_fem_armijo_checked_arena_not_admitted"')
rep('state.dispatch.femNodeCount > 4096u ||\n                               state.dispatch.tetrahedronCount > 32768u ||','state.dispatch.femNodeCount >\n                                   (state.femPureFEMArmijo8192ArenaAdmitted ? 32768u : 4096u) ||\n                               state.dispatch.tetrahedronCount >\n                                   (state.femPureFEMArmijo8192ArenaAdmitted ? 131072u : 32768u) ||')
rep('"requires_cached_static_fem_within_E2_N4096_T32768"','"requires_cached_static_fem_within_checked_pure_arena"')
p.write_text(t);changed.append(n)
edit('matter/tools/pure_fem_armijo_cgls_arena_plan_check.cpp',[
 ('s.femNodeCount=8193;s.unknownCount=32772;','s.femNodeCount=32769;s.unknownCount=131076;',1),('s.tetrahedronCount=32769;','s.tetrahedronCount=131073;',1),
 (' std::uint64_t out=0;',' s=shape();s.femNodeCount=20682;s.tetrahedronCount=89088;s.unknownCount=4*s.femNodeCount;\n s.includeCauchy=true;s.includeCGLS=true;s.includeRightScale=true;\n auto finest=planFEMPureArmijoCGLSArena(s);require(finest.admitted()&&finest.scratchBytes<64ull*1024ull*1024ull);\n s.femNodeCount=32768;s.tetrahedronCount=131072;s.unknownCount=4*s.femNodeCount;\n require(planFEMPureArmijoCGLSArena(s).admitted());\n s.femNodeCount=528444;s.tetrahedronCount=1477728;s.unknownCount=4*s.femNodeCount;\n require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::fem_node_count);\n std::uint64_t out=0;',1)])
edit('matter/tools/true_residual_arena_plan_check.cpp',[
 ('e4.aggregateBytes==e4.maximumAggregateBytes','e4.aggregateBytes<=e4.maximumAggregateBytes',1),
 (' std::cout<<',' auto refinedShape=s(2,20682,89088,0);refinedShape.extendedPureFEM=true;auto refined=planFEMTrueResidualArena(refinedShape);req(refined.admitted(),"actual finest coupon");\n auto maxShape=s(4,32768,131072,0);maxShape.extendedPureFEM=true;auto maxPure=planFEMTrueResidualArena(maxShape);req(maxPure.admitted()&&maxPure.aggregateBytes==maxPure.maximumAggregateBytes,"extended maximum aggregate");\n maxShape.cohesiveFaceCount=1;req(planFEMTrueResidualArena(maxShape).failure==FEMTrueResidualArenaFailure::extended_pure_layout,"extended cohesive rejected");\n maxShape.cohesiveFaceCount=0;maxShape.rigidGeneralizedCapacity=6;req(planFEMTrueResidualArena(maxShape).failure==FEMTrueResidualArenaFailure::extended_pure_layout,"extended rigid rejected");\n maxShape.rigidGeneralizedCapacity=0;maxShape.femNodeCount=32769;req(!planFEMTrueResidualArena(maxShape).admitted(),"extended node cap");\n maxShape.femNodeCount=32768;maxShape.tetrahedronCount=131073;req(!planFEMTrueResidualArena(maxShape).admitted(),"extended tet cap");\n std::cout<<',1)])
edit('matter/tools/pure_fem_cgls_arena_8192_check.mm',[
 ('int main(){','int main(int argc,const char* argv[]){',1),
 (' constexpr U E=2,N=8192,T=32768,UCount=2*E*N;',' need(argc==1||(argc==2&&(std::string(argv[1])=="--refined-coupon"||std::string(argv[1])=="--maximum-coupon-arena")),"usage: checker [--refined-coupon|--maximum-coupon-arena]");\n const bool refined=argc==2&&std::string(argv[1])=="--refined-coupon";\n const bool maximum=argc==2&&std::string(argv[1])=="--maximum-coupon-arena";\n const U E=2,N=refined?20682u:(maximum?32768u:8192u),T=refined?89088u:(maximum?131072u:32768u),UCount=2*E*N;',1),
 ('std::cout<<"pure_fem_cgls_arena_8192_native_check=PASS checks="','std::cout<<"pure_fem_cgls_arena_"<<N<<"_native_check=PASS checks="',1)])
edit('matter/tools/fgmres_true_residual_native_check.mm',[
 ('int main() {','int main(int argc, const char* argv[]) {',1),
 ('            checkLargeNodeResidualMeasurement(device,queue,measure,4096u);','            require(argc==1||(argc==2&&std::string(argv[1])=="--refined-coupon"),"usage: true-residual check [--refined-coupon]");\n            if(argc==2){checkLargeNodeResidualMeasurement(device,queue,measure,20682u);checkLargeNodeResidualMeasurement(device,queue,measure,32768u);}\n            checkLargeNodeResidualMeasurement(device,queue,measure,4096u);',1)])
(d/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n')
(d/'preparation.json').write_text(json.dumps({'schema':'numi.cardboard.source238-preparation.v1','base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'changed_files':changed,'snapshot_not_yet_frozen':True,'physical_operator_changed':False,'extended_scope':'contact-free static explicit FEM only; legacy coupled/contact true-residual limits retained; refined coupon scratch64MiB counted against16GiB runtime resident ceiling','requested_finest_mesh':{'nodes':20682,'tets':89088},'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print(json.dumps({'snapshot':str(s),'changed':changed,'frozen':False}))
