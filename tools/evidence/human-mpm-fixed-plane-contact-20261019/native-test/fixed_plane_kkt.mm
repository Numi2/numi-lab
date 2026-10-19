#define main legacy_floor_fixture_main
#include "fixture_local.mm"
#undef main

namespace {
constexpr std::uint32_t kFixedPlaneKKT=1u<<8u;
double dot3(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<class T> bool sameBytes(const std::vector<T>& a,const std::vector<T>& b){
    return a.size()==b.size()&&(a.empty()||
        std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
void printReactionDiff(const std::vector<NMRigidReactionGPU>& a,
                       const std::vector<NMRigidReactionGPU>& b){
    std::size_t changed=0;
    for(std::size_t i=0;i<std::max(a.size(),b.size());++i){
        if(i<a.size()&&i<b.size()&&
           std::memcmp(&a[i],&b[i],sizeof(NMRigidReactionGPU))==0)continue;
        ++changed;
        if(i>=a.size()||i>=b.size()){
            std::cerr<<"reaction_diff index="<<i<<" missing_side="
                <<(i>=a.size()?"before":"after")<<"\n";
            continue;
        }
        const auto& x=a[i];const auto& y=b[i];
        std::cerr<<std::setprecision(17)<<"reaction_diff index="<<i
            <<" before_impulse="<<x.impulseAndCount.x<<","<<x.impulseAndCount.y
            <<","<<x.impulseAndCount.z<<","<<x.impulseAndCount.w
            <<" after_impulse="<<y.impulseAndCount.x<<","<<y.impulseAndCount.y
            <<","<<y.impulseAndCount.z<<","<<y.impulseAndCount.w
            <<" before_angular="<<x.angularImpulse.x<<","<<x.angularImpulse.y
            <<","<<x.angularImpulse.z<<","<<x.angularImpulse.w
            <<" after_angular="<<y.angularImpulse.x<<","<<y.angularImpulse.y
            <<","<<y.angularImpulse.z<<","<<y.angularImpulse.w<<"\n";
    }
    std::cerr<<"reaction_diff_summary before_count="<<a.size()
        <<" after_count="<<b.size()<<" changed_entries="<<changed<<"\n";
}
void requireSameAccepted(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){
    require(a.available&&b.available,"rollback snapshot unavailable");
    require(a.controlStep==b.controlStep&&a.physicsSubstep==b.physicsSubstep,
            "rejection advanced accepted transaction cursor");
    require(sameBytes(a.particles,b.particles),"rejection changed accepted particles");
    require(sameBytes(a.vascularState,b.vascularState)&&sameBytes(a.vascularClock,b.vascularClock),
            "rejection changed accepted vascular state/clock");
    require(sameBytes(a.rigidStates,b.rigidStates)&&
            sameBytes(a.rigidGeneralizedCandidate,b.rigidGeneralizedCandidate),
            "rejection changed accepted rigid state");
    require(sameBytes(a.reactions,b.reactions),"rejection changed accepted reaction ledger");
    require(sameBytes(a.contactHistories,b.contactHistories)&&
            sameBytes(a.deformableContactHistories,b.deformableContactHistories),
            "rejection changed accepted contact history");
    require(sameBytes(a.mpmActiveNodeIndices,b.mpmActiveNodeIndices)&&
            sameBytes(a.mpmActiveNodeCounts,b.mpmActiveNodeCounts),
            "rejection changed accepted active MPM map");
}
struct Plane{std::uint32_t proxy=NM_INVALID_INDEX;Vec3 n{};double d=0;};
Plane staticPlane(const CompiledWorld& w){
    Plane out;
    for(std::uint32_t i=0;i<w.contact.rigidProxies.size();++i){
        const auto& p=w.contact.rigidProxies[i];
        if(p.shapeKind!=NM_RIGID_PLANE||p.bodyIndex!=NM_INVALID_INDEX)continue;
        require(out.proxy==NM_INVALID_INDEX,"more than one static plane");
        out.proxy=i;out.n={p.localCenterAndRadius.x,p.localCenterAndRadius.y,
                           p.localCenterAndRadius.z};
        out.d=p.localCenterAndRadius.w;
    }
    require(out.proxy!=NM_INVALID_INDEX,"static plane missing");
    const double len=norm(out.n);require(std::isfinite(len)&&len>0,"invalid plane normal");
    out.n=mul(1.0/len,out.n);out.d/=len;return out;
}
double gap(const Plane& p,Vec3 x){return dot3(p.n,x)-p.d;}
void printPlaneAdmission(const CompiledWorld& w,std::uint32_t owner,const Plane& p){
    require(owner<w.objects.size(),"admission owner outside compiled objects");
    require(p.proxy<w.contact.rigidRanges.size(),"plane rigid range missing");
    const auto& object=w.objects[owner];
    require(object.materialIndex<w.materials.size(),"blood material index invalid");
    const auto& blood=w.materials[object.materialIndex].interfaceResponse;
    const auto& range=w.contact.rigidRanges[p.proxy];
    std::size_t matching=0,nonpositive=0;
    double minGap=INFINITY,maxGap=-INFINITY,nearestAbs=INFINITY,nearestGap=INFINITY;
    std::uint32_t nearestNode=NM_INVALID_INDEX,nearestPair=NM_INVALID_INDEX;
    std::uint32_t firstInterface=NM_INVALID_INDEX;
    for(std::uint32_t i=0;i<w.contact.pairs.size();++i){
        const auto& pair=w.contact.pairs[i];
        if(pair.objectIndex!=owner||pair.rigidProxy!=p.proxy||
           pair.continuumNode>=w.dispatch.gridNodeCount)continue;
        require(pair.continuumNode<w.mpm.nodes.size(),"compiled grid node missing");
        require(pair.materialInterface<w.materials.size(),"interface material index invalid");
        ++matching;
        const auto& node=w.mpm.nodes[pair.continuumNode];
        const Vec3 x{node.positionAndMass.x,node.positionAndMass.y,node.positionAndMass.z};
        const double g=gap(p,x),a=std::abs(g);
        minGap=std::min(minGap,g);maxGap=std::max(maxGap,g);
        if(g<=0)++nonpositive;
        if(a<nearestAbs){nearestAbs=a;nearestGap=g;nearestNode=pair.continuumNode;nearestPair=i;firstInterface=pair.materialInterface;}
    }
    std::cerr<<std::setprecision(17)<<"fixed_plane_admission owner="<<owner
        <<" proxy="<<p.proxy<<" rigid_range_first="<<range.first
        <<" rigid_range_count="<<range.count<<" matching_mpm_pairs="<<matching
        <<" reference_gap_nonpositive="<<nonpositive<<" reference_gap_min_m="<<minGap
        <<" reference_gap_max_m="<<maxGap<<" nearest_pair="<<nearestPair
        <<" nearest_node="<<nearestNode<<" nearest_reference_gap_m="<<nearestGap
        <<" blood_material="<<object.materialIndex
        <<" blood_interface_response="<<blood.x<<","<<blood.y;
    if(firstInterface!=NM_INVALID_INDEX){
        const auto& planeMaterial=w.materials[firstInterface].interfaceResponse;
        const double staticMu=std::sqrt(std::max(double(blood.x)*planeMaterial.x,0.0));
        const double dynamicMu=std::sqrt(std::max(double(blood.y)*planeMaterial.y,0.0));
        std::cerr<<" interface_material="<<firstInterface
            <<" interface_response="<<planeMaterial.x<<","<<planeMaterial.y
            <<" effective_friction="<<staticMu<<","<<dynamicMu;
    }
    std::cerr<<"\n";
}
void stageRestPacket(Respiration& r,float density,Vec3 x){
    stagePacket(r,density,x,{0,0,0},kPacketVolumeM3);
    auto* s=static_cast<NMHumanRespirationState*>(r.accepted.contents);
    require(s!=nullptr,"accepted respiration unavailable");
    s->externalBlood.pendingAngularMomentum={0,0,0,0};
    s->externalBlood.energyLedger.x=0;
}
double kineticEnergy(const std::vector<NMParticleStateGPU>& ps,std::uint32_t owner,double h){
    double e=0;
    for(const auto& p:ps){
        if(p.identity.x!=owner||(p.identity.w&NM_OBJECT_ACTIVE)==0)continue;
        const double m=p.positionAndMass.w;
        const double vx=p.velocityAndReferenceVolume.x,vy=p.velocityAndReferenceVolume.y,
                     vz=p.velocityAndReferenceVolume.z;
        const double c2=
            double(p.affineRow0.x)*p.affineRow0.x+double(p.affineRow0.y)*p.affineRow0.y+
            double(p.affineRow0.z)*p.affineRow0.z+double(p.affineRow1.x)*p.affineRow1.x+
            double(p.affineRow1.y)*p.affineRow1.y+double(p.affineRow1.z)*p.affineRow1.z+
            double(p.affineRow2.x)*p.affineRow2.x+double(p.affineRow2.y)*p.affineRow2.y+
            double(p.affineRow2.z)*p.affineRow2.z;
        e+=.5*m*(vx*vx+vy*vy+vz*vz)+.125*m*h*h*c2;
    }
    return e;
}
struct Contact{std::size_t samples=0,kkt=0,active=0;double lambda=0,proxyN=0,
    sampleInward=0,sampleAbsLambdaVelocity=0,minNodeGap=INFINITY;};
Contact contactMetrics(const RuntimeStateSnapshot& s,const CompiledWorld& w,
                       std::uint32_t owner,const Plane& p){
    const auto nodes=activePositiveMassMpmNodes(s,w.dispatch.gridNodeCount);
    Contact c;
    for(const auto& q:s.contactSamples){
        if(!(q.identity.w&NM_CONTACT_VALID)||q.identity.z!=owner||q.identity.y!=p.proxy)continue;
        ++c.samples;if(!(q.identity.w&kFixedPlaneKKT))continue;++c.kkt;
        if(!isActivePositiveMassMpmNode(nodes,q,w.dispatch.gridNodeCount))continue;
        ++c.active;const double lam=q.impulseAndNormal.w,vn=q.normalAndVelocity.w;
        require(std::isfinite(lam)&&lam>=-1e-15,"invalid KKT multiplier");
        require(std::isfinite(vn),"invalid KKT normal velocity");
        c.lambda+=std::max(lam,0.0);
        c.proxyN+=dot3({q.impulseAndNormal.x,q.impulseAndNormal.y,
                        q.impulseAndNormal.z},p.n);
        c.sampleInward=std::max(c.sampleInward,std::max(-vn,0.0));c.sampleAbsLambdaVelocity+=std::abs(lam*vn);
        c.minNodeGap=std::min(c.minNodeGap,double(q.pointAndSeparation.w));
    }
    return c;
}
void runCase(const char* net,const char* config,const char* material,bool gravity){
    configureBlood(material,true,0);
    WorldSource zero;zero.environmentCount=1;zero.frameTimestep=kFrameSeconds;zero.gravity={0,0,0};
    RestingRun run(net,config,kFrameSeconds,true,gravity?nullptr:&zero);
    require(run.world.dispatch.maximumRateExponent==0,"expected one microtick per frame");
    const auto owner=bloodObjectIndex(run.world);const auto& obj=run.world.objects[owner];
    const double h=run.world.mpm.grids[obj.elementOffset].metrics.x;
    const float density=run.world.materials[obj.materialIndex].bulk.x;
    const Plane plane=staticPlane(run.world);
    printPlaneAdmission(run.world,owner,plane);
    const Vec3 start=add(mul(plane.d,plane.n),mul(h,plane.n));
    stageRestPacket(*run.respiration,density,start);
    const double clearance=std::max(
        double(run.world.mixedSolver.contactAcceptance.x)*run.world.dispatch.numericalLimits.x,
        16.0*1.1920928955078125e-7*std::max(std::abs(plane.d),1e-12));
    require(gap(plane,start)>clearance,"initial particle is not clear of plane");
    std::cerr<<std::setprecision(17)<<"fixed_plane_begin gravity="<<gravity
        <<" plane_n="<<plane.n.x<<","<<plane.n.y<<","<<plane.n.z<<" plane_d_m="<<plane.d
        <<" start_gap_m="<<gap(plane,start)<<" h_m="<<h
        <<" scope=grid_support_boundary_not_pooled_fluid\n";
    Vec3 previous=start;double lsum=0,wsum=0,rsum=0,minGap=INFINITY,minSweep=INFINITY,
        lastEnergy=0,maxSampleInward=0;bool rollback=false;
    constexpr std::uint32_t frames=4;
    for(std::uint32_t frame=0;frame<frames;++frame){
        if(frame==1){
            const auto before=run.runtime.snapshot();
            const auto* rs=static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
            require(before.available&&rs,"pre-rejection state unavailable");
            const auto respirationBefore=*rs;
            run.batch(frame,1,true,frame);
            const auto afterRejected=run.runtime.snapshot();
            if(!sameBytes(before.reactions,afterRejected.reactions))
                printReactionDiff(before.reactions,afterRejected.reactions);
            requireSameAccepted(before,afterRejected);
            require(std::memcmp(&respirationBefore,rs,sizeof(respirationBefore))==0,
                    "rejection changed accepted respiration state");
            run.respiration->dispatch.reject=0;run.batch(frame,1,false);rollback=true;
        }else run.batch(frame,1,false);
        const auto s=run.runtime.snapshot();require(s.available,"accepted snapshot unavailable");
        const auto totals=particleTotals(s.particles,owner,h);
        std::cerr<<std::setprecision(17)<<"fixed_plane_particle_totals frame="<<(frame+1)
            <<" active="<<totals.active<<" volume_m3="<<totals.volume
            <<" mass_kg="<<totals.mass<<" expected_volume_m3="<<kPacketVolumeM3
            <<" expected_mass_kg="<<double(density)*kPacketVolumeM3<<"\n";
        require(totals.active==1&&near(totals.volume,kPacketVolumeM3,1e-14,2e-5)&&
                near(totals.mass,double(density)*kPacketVolumeM3,1e-12,2e-5),
                "particle mass/reference volume changed");
        const auto* respirationState=static_cast<const NMHumanRespirationState*>(
            run.respiration->accepted.contents);
        require(respirationState!=nullptr,"accepted respiration state unavailable");
        checkInventory(*respirationState,totals,kPacketVolumeM3,density,1u);
        Vec3 current{};std::size_t count=0;
        for(const auto& q:s.particles)if(q.identity.x==owner&&(q.identity.w&NM_OBJECT_ACTIVE)){
            current=particlePosition(q);++count;
        }
        require(count==1,"ambiguous particle trajectory");
        const double g0=gap(plane,previous),g1=gap(plane,current),sweep=std::min(g0,g1);
        require(g1>=clearance&&sweep>=clearance,"particle endpoint/sweep crossed plane");
        const Contact c=contactMetrics(s,run.world,owner,plane);
        require(c.samples&&c.kkt&&c.active,"no active fixed-plane KKT sample");
        const double energy=kineticEnergy(s.particles,owner,h);
        if(!gravity)require(energy<=1e-18,"zero-g rest state gained kinetic energy");
        const auto& reaction=s.reactions[plane.proxy];
        const double ledgerN=dot3({reaction.impulseAndCount.x,reaction.impulseAndCount.y,
                                   reaction.impulseAndCount.z},plane.n);
        require(std::abs(ledgerN-c.proxyN)<=std::max(1e-14,.01*c.lambda)&&
                std::abs(ledgerN+c.lambda)<=std::max(1e-14,.01*c.lambda),
                "proxy reaction is not equal/opposite to KKT impulse");
        if(gravity)require(c.lambda>0,"gravity load produced no KKT reaction");
        else require(c.lambda<=1e-14,"zero-g rest state produced unexplained impulse");
        lsum+=c.lambda;wsum+=c.sampleAbsLambdaVelocity;rsum+=ledgerN;minGap=std::min(minGap,g1);
        minSweep=std::min(minSweep,sweep);lastEnergy=energy;maxSampleInward=std::max(maxSampleInward,c.sampleInward);
        std::cerr<<std::setprecision(17)<<"fixed_plane_frame gravity="<<gravity
            <<" frame="<<(frame+1)<<" particle_gap_m="<<g1<<" sweep_gap_m="<<sweep
            <<" mass_kg="<<totals.mass<<" volume_m3="<<totals.volume
            <<" kinetic_energy_J="<<energy<<" active_kkt_samples="<<c.active
            <<" min_grid_gap_m="<<c.minNodeGap<<" mpm_impulse_Ns="<<c.lambda
            <<" proxy_impulse_Ns="<<ledgerN<<" sample_abs_lambda_v_J="<<c.sampleAbsLambdaVelocity
            <<" sample_inward_grid_speed_m_s="<<c.sampleInward<<" final_normal_validation=accepted_transaction\n";
        previous=current;
    }
    require(rollback,"rollback/retry path not exercised");
    std::cout<<"fixed_plane_case=PASS gravity_loaded="<<gravity<<" accepted_frames="<<frames
        <<" particle_mass_kg="<<double(density)*kPacketVolumeM3
        <<" particle_volume_m3="<<kPacketVolumeM3<<" min_particle_gap_m="<<minGap
        <<" min_sweep_gap_m="<<minSweep<<" cumulative_mpm_impulse_Ns="<<lsum
        <<" cumulative_proxy_impulse_Ns="<<rsum<<" cumulative_sample_abs_lambda_v_J="<<wsum
        <<" max_sample_inward_grid_speed_m_s="<<maxSampleInward<<" final_kinetic_energy_J="<<lastEnergy
        <<" rollback_retry=PASS scope=grid_support_boundary_not_pooling\n";
}
}
int main(int argc,char** argv){
    std::cerr.setf(std::ios::unitbuf);
    @autoreleasepool{try{
        require(argc==4,"usage: fixed_plane_kkt NETWORK CONFIG MATERIAL");
        runCase(argv[1],argv[2],argv[3],false);runCase(argv[1],argv[2],argv[3],true);
        std::cout<<"fixed_plane_kkt_acceptance=PASS pooled_fluid=NOT_TESTED\n";return 0;
    }catch(const std::exception& e){
        std::cerr<<"fixed_plane_kkt_acceptance=FAIL reason="<<e.what()<<"\n";return 1;
    }}
}

