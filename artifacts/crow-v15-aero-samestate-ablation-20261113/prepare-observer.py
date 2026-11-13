from pathlib import Path
import gzip
R=Path.cwd();old='crow-v15-aero-power-audit-20261113';new='crow-v15-aero-samestate-ablation-20261113';A=R/'artifacts'/new;B=R/'build'/new
s=gzip.decompress((R/'artifacts'/old/'observer.mm.gz').read_bytes()).decode()
s=s.replace('id<MTLBuffer> before=nil; id<MTLBuffer> after=nil; id<MTLBuffer> aq=nil; id<MTLBuffer> av=nil;', 'id<MTLBuffer> before=nil; id<MTLBuffer> after=nil; id<MTLBuffer> aq=nil; id<MTLBuffer> av=nil; id<MTLBuffer> scratch=nil; id<MTLBuffer> specOriginal=nil; id<MTLBuffer> specModified=nil; id<MTLBuffer> dispCopy=nil;')
needle='''[blit copyFromBuffer:source sourceOffset:0 toBuffer:after destinationOffset:0 size:source.length];
        [blit endEncoding];
        [commandBuffer addCompletedHandler:'''
replacement='''[blit copyFromBuffer:source sourceOffset:0 toBuffer:after destinationOffset:0 size:source.length];
        id<MTLBuffer> specs=context.buffers[kFlappingWingSpecs], disp=context.buffers[kFlappingWingDispatch];
        if(specs.length<2*sizeof(MRFlappingWingGPU) || disp.length<sizeof(MRCompiledFlappingWingDispatchGPU)) throw std::runtime_error("ablation buffer size");
        scratch=[context.device newBufferWithLength:source.length options:MTLResourceStorageModeShared];
        specOriginal=[context.device newBufferWithLength:specs.length options:MTLResourceStorageModeShared];
        specModified=[context.device newBufferWithLength:specs.length options:MTLResourceStorageModeShared];
        dispCopy=[context.device newBufferWithLength:disp.length options:MTLResourceStorageModeShared];
        if(scratch==nil || specOriginal==nil || specModified==nil || dispCopy==nil) throw std::bad_alloc();
        [blit copyFromBuffer:before sourceOffset:0 toBuffer:scratch destinationOffset:0 size:source.length];
        [blit copyFromBuffer:specs sourceOffset:0 toBuffer:specOriginal destinationOffset:0 size:specs.length];
        [blit copyFromBuffer:specs sourceOffset:0 toBuffer:specModified destinationOffset:0 size:specs.length];
        [blit copyFromBuffer:disp sourceOffset:0 toBuffer:dispCopy destinationOffset:0 size:disp.length];
        for(unsigned i=0;i<2;++i) [blit fillBuffer:specModified range:NSMakeRange(i*sizeof(MRFlappingWingGPU)+offsetof(MRFlappingWingGPU,unsteadyCoefficients),sizeof(float)) value:0];
        [blit endEncoding];
        id<MTLComputeCommandEncoder> alt=[commandBuffer computeCommandEncoder];
        if(alt==nil) throw std::runtime_error("ablation encoder");
        [alt setComputePipelineState:context.flappingWingPipeline];
        [alt setBuffer:specModified offset:0 atIndex:0]; [alt setBuffer:disp offset:0 atIndex:1];
        [alt setBuffer:context.buffers[qState] offset:0 atIndex:2]; [alt setBuffer:context.buffers[vState] offset:0 atIndex:3];
        [alt setBuffer:scratch offset:0 atIndex:4]; [alt setBuffer:context.buffers[kFlappingTail] offset:0 atIndex:5]; [alt setBuffer:context.buffers[kFlappingFuselage] offset:0 atIndex:6];
        dispatchWorldThreads(alt,context.flappingWingPipeline,environmentCount); [alt endEncoding];
        [commandBuffer addCompletedHandler:'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='''const auto* b=static_cast<const MRABABodyWrenchGPU*>(after.contents);
            for (std::size_t e=0;e<environmentCount;++e) {'''
replacement='''const auto* b=static_cast<const MRABABodyWrenchGPU*>(after.contents);
            const auto* c=static_cast<const MRABABodyWrenchGPU*>(scratch.contents);
            const auto* orig=static_cast<const MRFlappingWingGPU*>(specOriginal.contents);
            const auto* modified=static_cast<const MRFlappingWingGPU*>(specModified.contents);
            const auto* dp=static_cast<const MRCompiledFlappingWingDispatchGPU*>(dispCopy.contents);
            bool specsOkay=true;
            for(unsigned i=0;i<2;++i){MRFlappingWingGPU expected=orig[i];expected.unsteadyCoefficients.x=0; if(std::memcmp(&expected,&modified[i],sizeof(expected))!=0) specsOkay=false;}
            std::fprintf(stderr,"CROW_AERO_ABLATE_CONFIG %llu %u %d %.9g %.9g %.9g %.9g %.9g %.9g\\n",traceStep,traceSubstep,int(specsOkay),orig[0].unsteadyCoefficients.x,orig[1].unsteadyCoefficients.x,dp->windVelocityAndDensity.x,dp->windVelocityAndDensity.y,dp->windVelocityAndDensity.z,dp->windVelocityAndDensity.w);
            for (std::size_t e=0;e<environmentCount;++e) {'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='''if(j==0 || j==5 || j==6 || j==7) std::fprintf(stderr,"CROW_AERO_WRENCH'''
replacement='''if(j==0 || j==5 || j==6 || j==7) { const auto& z=c[e*bodyStride+j]; std::fprintf(stderr,"CROW_AERO_ABLATE %llu %u %lu %lu %.12g %.12g %.12g %.12g %.12g %.12g\\n",traceStep,traceSubstep,(unsigned long)e,(unsigned long)j,double(z.force.x)-double(x.force.x),double(z.force.y)-double(x.force.y),double(z.force.z)-double(x.force.z),double(z.torque.x)-double(x.torque.x),double(z.torque.y)-double(x.torque.y),double(z.torque.z)-double(x.torque.z)); }
                    if(j==0 || j==5 || j==6 || j==7) std::fprintf(stderr,"CROW_AERO_WRENCH'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
(B/'MetalWorld.mm').write_text(s);(A/'observer.mm.gz').write_bytes(gzip.compress(s.encode(),mtime=0))
for f in ('build-observer.py','run.py'):(A/f).write_text((R/'artifacts'/old/f).read_text().replace(old,new))
