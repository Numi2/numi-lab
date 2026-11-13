from pathlib import Path
import json,subprocess,shlex
R=Path('/Users/n/Developer/numi-crow-rl-20261107');a=R/'artifacts/crow-v15-bank-damping-sensitivity-20261113';b=R/'build/crow-v15-bank-damping-observer-contract-20261113';b.mkdir(exist_ok=True)
s=(R/'build/crow-v15-mirrored-pronation-20261113/MetalWorld.mm').read_text();old='id<MTLBuffer> before=nil; id<MTLBuffer> after=nil; id<MTLBuffer> aq=nil; id<MTLBuffer> av=nil; id<MTLBuffer> ctrl=nil;';new=old+' id<MTLBuffer> bodyParametersSnapshot=nil;';assert s.count(old)==1;s=s.replace(old,new)
old='        [blit endEncoding];\n    }\n    const bool encoded=encodeContactThreadKernel(\n        context, commandBuffer, context.flappingWingPipeline,'
new='''        if (traceSubstep==0u && traceStep%10u==0u) {
            id<MTLBuffer> params=context.buffers[kTaskBodyParameters];
            if (params==nil || params.length<environmentCount*bodyStride*sizeof(mr_float4)) throw std::runtime_error("body parameter observer size");
            bodyParametersSnapshot=[context.device newBufferWithLength:params.length options:MTLResourceStorageModeShared];
            if (bodyParametersSnapshot==nil) throw std::bad_alloc();
            [blit copyFromBuffer:params sourceOffset:0 toBuffer:bodyParametersSnapshot destinationOffset:0 size:params.length];
        }
        [blit endEncoding];
    }
    const bool encoded=encodeContactThreadKernel(
        context, commandBuffer, context.flappingWingPipeline,''';assert s.count(old)==1;s=s.replace(old,new)
old='                const auto* qin=static_cast<const float*>(aq.contents);const auto* vin=static_cast<const float*>(av.contents);'
new='''                if (bodyParametersSnapshot!=nil) {
                    const auto* p=static_cast<const mr_float4*>(bodyParametersSnapshot.contents);
                    const auto& v=p[e*bodyStride];
                    std::fprintf(stderr,"CROW_TORSO_PARAMETERS %llu %u %lu %.9g %.9g %.9g %.9g\\n",traceStep,traceSubstep,(unsigned long)e,v.x,v.y,v.z,v.w);
                }
'''+old;assert s.count(old)==1;s=s.replace(old,new);src=a/'MetalWorld-parameter-observer-contract.mm';src.write_text(s)
entries=json.loads((R/'build/compile_commands.json').read_text());c=shlex.split(next(x['command'] for x in entries if x['file'].endswith('/MetalWorld.mm')));c.insert(1,'-I'+str(R/'build/crow-v15-wide-contact-isolation-20261111/include'));c.extend(['-I',str(R/'src/metal')]);c[c.index('-o')+1]=str(b/'MetalWorld.mm.o');c[c.index('-c')+1]=str(src);subprocess.run(c,cwd=R/'build',check=True)
out=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands','lib/libmetalrobo.dylib'],text=True).splitlines()[-1];assert out.startswith(': && ') and out.endswith(' && :');args=shlex.split(out[5:-5]);args[args.index('-o')+1]=str(b/'libmetalrobo.dylib');oldobj='CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o';assert oldobj in args;args[args.index(oldobj)]=str(b/'MetalWorld.mm.o');wide=R/'build/crow-v15-wide-contact-isolation-20261111';names={0:'src/core/CompiledMetalMultiArticulatedProgram.cpp',1:'src/core/CompiledMetalMultiArticulatedContactProgram.cpp',2:'src/metal/MetalArticulatedInverseMass.mm',3:'src/metal/MetalArticulatedABA.mm',5:'src/metal/MetalMultiArticulatedContact.mm',6:'src/metal/MetalMultiArticulatedABA.mm',7:'src/metal/MetalMultiArticulatedInverseMass.mm'}
for i,n in names.items():
 k='CMakeFiles/metalrobo.dir/'+n+'.o';assert k in args,k;args[args.index(k)]=str(wide/(str(i)+'.o'))
taskobj='CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o';assert taskobj in args;args[args.index(taskobj)]=str(R/'build/crow-v15-matched-binary-gait-screen-20261112/TaskProgram.cpp.o')
capi='CMakeFiles/metalrobo.dir/src/c_api.cpp.o';assert capi in args;args[args.index(capi)]=str(R/'build/crow-v15-mirrored-pronation-20261113/c_api.cpp.o');subprocess.run(args,cwd=R/'build',check=True)
(a/'body-parameter-observer-contract-build.json').write_text(json.dumps({'compile':c,'link':args},indent=2)+'\n');print('built observational host')
