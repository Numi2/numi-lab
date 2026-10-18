from pathlib import Path
import json,shutil,hashlib
old=Path('/Users/n/numi-human-rope-swing-build-20261018-001');out=Path('/Users/n/numi-human-rope-swing-build-20261018-002');out.mkdir();(out/'bin').mkdir();shutil.copytree(old/'source',out/'source')
p=out/'source/NumiHumanRestingRopeVisual.hpp';s=p.read_text().replace('return gpu.counts.x + gpu.counts.w;','return gpu.counts.x + gpu.counts.w + 164u;');p.write_text(s)
s=(old/'rope.metal').read_text().replace('if (vertexIndex >= ropeVertexCount + harnessVertexCount) return;','if (vertexIndex >= ropeVertexCount + harnessVertexCount + 164u) return;')
tag='    const float3 chord = endpoint - anchor;'
block="""    if(vertexIndex >= ropeVertexCount + harnessVertexCount) {
        const uint hardware=vertexIndex-ropeVertexCount-harnessVertexCount;
        auto out=vertices[visual.ranges.x+vertexIndex];
        float3 p;
        if(hardware<160u) {
            const float theta=6.28318530718f*float(hardware/8u)/20.0f;
            const float phi=6.28318530718f*float(hardware%8u)/8.0f;
            const float r=.018f+.003f*cos(phi);
            p=anchor+float3(r*cos(theta),r*sin(theta),.003f*sin(phi));
        } else {
            const uint k=hardware-160u;
            p=anchor+float3((k==0u||k==3u)?-.07f:.07f,k<2u?-.07f:.07f,.012f);
        }
        out.position=float4(p,1);vertices[visual.ranges.x+vertexIndex]=out;return;
    }
"""
assert tag in s;s=s.replace(tag,block+tag);(out/'rope.metal').write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((old/'build-plan-abi41.json').read_text())
for k in ['compile_argv','link_argv']:plan[k]=[v.replace(str(old),str(out)) for v in plan[k]]
for j,v in enumerate(plan['compile_argv']):
 if v.startswith('-DNUMI_HUMAN_RESTING_ROPE_SOURCE_SHA256='):plan['compile_argv'][j]='-DNUMI_HUMAN_RESTING_ROPE_SOURCE_SHA256="'+sha(out/'rope.metal')+'"'
plan['application_header_sha256']={str(out/'source'/Path(x).name):sha(out/'source'/Path(x).name) for x in plan['application_header_sha256']}
plan['source_sha256']=plan['new_source_sha256']=sha(out/'source/numilab_human_myosim_visual_probe.mm')
(out/'build-plan-abi41.json').write_text(json.dumps(plan,indent=2))
(out/'execute_build_abi41.py').write_text((old/'execute_build_abi41.py').read_text().replace(str(old),str(out)))
