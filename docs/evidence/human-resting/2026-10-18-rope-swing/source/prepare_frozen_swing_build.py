from pathlib import Path
import shutil,json,hashlib
old=Path('/Users/n/numi-human-rope-abi41-build-20261018-013');out=Path('/Users/n/numi-human-rope-swing-build-20261018-001');out.mkdir();shutil.copytree(old/'source',out/'source');(out/'bin').mkdir()
src=out/'source'
p=src/'NumiHumanRestingRopeSupport.hpp';s=p.read_text()
s=s.replace('MRHumanRestingRopeGPU parameters_{};','MRHumanRestingRopeGPU parameters_{};\n    mr_float4 swing_{};')
s=s.replace('public:\n    static bool configuredFromEnvironment()',"""public:
    static mr_float4 swingMotion(float seconds) {
        const char* text=std::getenv("NUMI_HUMAN_ROPE_SWING");
        if(!text)return {seconds,0,1,1};
        const auto p=parse<3u>(text,"NUMI_HUMAN_ROPE_SWING must be amplitude_m,period_s,ramp_s");
        require(p[0]>=0&&p[0]<=.75f&&p[1]>=2&&p[2]>=1,"rope swing bounds invalid");
        return {seconds,p[0],p[1],p[2]};
    }
    static bool configuredFromEnvironment()""")
s=s.replace('parameters_.indices={contactIndex','swing_=swingMotion(0);\n        parameters_.indices={contactIndex')
s=s.replace('if(fingerprint_==0)', 'for(float x:{swing_.y,swing_.z,swing_.w})fingerprint_=mix(fingerprint_,std::bit_cast<std::uint32_t>(x));\n        if(fingerprint_==0)')
s=s.replace('[enc dispatchThreadgroups:', 'auto motion=swing_;motion.x=float(pass.stepIndex)*.002f;\n        [enc setBytes:&motion length:sizeof(motion) atIndex:7u];\n        [enc dispatchThreadgroups:')
p.write_text(s)
p=src/'NumiHumanRestingRopeVisual.hpp';s=p.read_text().replace('e.dispatchThreads(e.context,layout_.dynamicVertexCount(),64u);','const auto motion=NumiHumanRestingRopeSupport::swingMotion(float(lease.acceptedTimestampMicroseconds)*1e-6f);\n            e.setBytes(e.context,&motion,sizeof(motion),4u);\n            e.dispatchThreads(e.context,layout_.dynamicVertexCount(),64u);');p.write_text(s)
p=src/'NumiHumanRestingWindow.hpp';s=p.read_text()
s=s.replace('const auto frame=_advance(','''if(std::getenv("NUMI_HUMAN_ROPE_SWING")) {
            static const unsigned order[8]={0,5,1,6,2,7,3,8};
            _layers.selectedSegment=0;
            _cameras.selectedSegment=order[unsigned(_lastSimulatedSeconds/3.0)%8];
        }
        const auto frame=_advance(''')
s=s.replace('@[@"Front",@"Side",@"Chest",@"Whole body",@"Heart detail"]','(std::getenv("NUMI_HUMAN_ROPE_SWING")?@[@"Front",@"Right",@"Back",@"Left",@"Heart",@"Front-right",@"Back-right",@"Back-left",@"Front-left"]:@[@"Front",@"Side",@"Chest",@"Whole body",@"Heart detail"])')
p.write_text(s)
p=src/'NumiHumanRestingVisual.hpp';s=p.read_text()
s=s.replace('const std::array<mr_float4,4> cameras{{','std::array<mr_float4,4> cameras{{').replace('const std::array<mr_float4,4> targets{c,c,chest,c};','std::array<mr_float4,4> targets{c,c,chest,c};\n        if(std::getenv("NUMI_HUMAN_ROPE_SWING")) {\n            cameras={mr_float4{c.x,c.y-distance,c.z+.1f*distance,0},mr_float4{c.x+distance,c.y,c.z+.1f*distance,0},mr_float4{c.x,c.y+distance,c.z+.1f*distance,0},mr_float4{c.x-distance,c.y,c.z+.1f*distance,0}};targets={c,c,c,c};\n        }')
p.write_text(s)
p=src/'numilab_human_myosim_visual_probe.mm';s=p.read_text().replace('if(detailCamera)episode.sensors.push_back(*detailCamera);','''if(detailCamera)episode.sensors.push_back(*detailCamera);
    if(std::getenv("NUMI_HUMAN_ROPE_SWING")) {
        for(unsigned i=0;i<4;++i) {
            const float angle=(float(i)*.5f+.25f)*3.14159265359f;
            const mr_float4 pos{center.x+framing.distance*std::sin(angle),center.y-framing.distance*std::cos(angle),center.z+.1f*framing.distance,0};
            episode.sensors.push_back(makeCamera("rope_diagonal_"+std::to_string(i),pos,center,dimension));
        }
    }''');p.write_text(s)
metal=Path('/Users/n/numi-human-rope-runtime-source-20261018-002/visual-dispatch-test-003/rope.metal').read_text()
helper="""inline float3 nmRopeSwingOffset(float4 motion) {
    const float u=clamp(motion.x/motion.w,0.0f,1.0f);
    const float ramp=u*u*u*(10.0f+u*(-15.0f+6.0f*u));
    const float phase=6.28318530718f*motion.x/motion.z;
    return motion.y*ramp*float3(sin(phase),0.5f*(cos(phase)-1.0f),0.0f);
}
"""
metal=metal.replace('// BEGIN resting rope support',helper+'// BEGIN resting rope support')
metal=metal.replace('device MRNumiHumanStandContactGPU* contacts [[buffer(6)]],','device MRNumiHumanStandContactGPU* contacts [[buffer(6)]],\n    constant float4& motion [[buffer(7)]],')
metal=metal.replace('const float3 toward=(cfg.anchorAndLength.xyz-high)-low;','const float3 drivenAnchor=cfg.anchorAndLength.xyz+nmRopeSwingOffset(motion);\n    const float3 toward=(drivenAnchor-high)-low;').replace('const float3 plane=cfg.anchorAndLength.xyz-cfg.anchorAndLength.w*normal;','const float3 plane=drivenAnchor-cfg.anchorAndLength.w*normal;')
metal=metal.replace('device MRVisualVertexGPUV2* vertices [[buffer(3)]],','device MRVisualVertexGPUV2* vertices [[buffer(3)]],\n    constant float4& motion [[buffer(4)]],').replace('const float3 anchor = rope.anchorAndLength.xyz;','const float3 anchor = rope.anchorAndLength.xyz+nmRopeSwingOffset(motion);')
(out/'rope.metal').write_text(metal)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((old/'build-plan-abi41.json').read_text());plan['compile_argv']=[x.replace(str(old),str(out)) for x in plan['compile_argv']];plan['link_argv']=[x.replace(str(old),str(out)) for x in plan['link_argv']]
for i,x in enumerate(plan['compile_argv']):
 if x.startswith('-DNUMI_HUMAN_RESTING_ROPE_SOURCE='):plan['compile_argv'][i]='-DNUMI_HUMAN_RESTING_ROPE_SOURCE="'+str(out/'rope.metal')+'"'
 if x.startswith('-DNUMI_HUMAN_RESTING_ROPE_SOURCE_SHA256='):plan['compile_argv'][i]='-DNUMI_HUMAN_RESTING_ROPE_SOURCE_SHA256="'+sha(out/'rope.metal')+'"'
plan['application_header_sha256']={str(src/Path(x).name):sha(src/Path(x).name) for x in plan['application_header_sha256']}
plan['source_sha256']=plan['new_source_sha256']=sha(src/'numilab_human_myosim_visual_probe.mm')
(out/'build-plan-abi41.json').write_text(json.dumps(plan,indent=2))
(out/'execute_build_abi41.py').write_text((old/'execute_build_abi41.py').read_text().replace(str(old),str(out)))
print(out)
