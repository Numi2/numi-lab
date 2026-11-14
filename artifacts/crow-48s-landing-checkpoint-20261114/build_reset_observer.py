"""Capture GPU reset q/v immediately after task reset, before physics."""
from pathlib import Path
import hashlib,json,shlex,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';w=r/'build/crow-v15-wide-contact-isolation-20261111'
original=(r/'artifacts/crow-v15-descent-observer-extension-20261114/MetalWorld-launch-aero-observer.mm').read_text()
start=original.index('bool encodeTaskObserve(');end=original.index('\nbool encodeDeviceObservationBodies(',start);section=original[start:end]
assert section.count('    return encodeContactThreadKernel(')==1
section=section.replace('    return encodeContactThreadKernel(','    const bool encoded = encodeContactThreadKernel(',1)
ending='''        environmentCount
    );
}'''
assert section.count(ending)==1
replacement='''        environmentCount
    );
    if (encoded && environmentCount==1 && std::getenv("CROW_RESTART_RESET_AUDIT")!=nullptr) {
        id<MTLBuffer> sourceQ=context.buffers[kResetQ], sourceV=context.buffers[kResetV];
        if (!sourceQ || !sourceV || sourceQ.length<41*sizeof(float) || sourceV.length<40*sizeof(float)) throw std::runtime_error("reset audit buffers invalid");
        id<MTLBuffer> savedQ=[context.device newBufferWithLength:41*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> savedV=[context.device newBufferWithLength:40*sizeof(float) options:MTLResourceStorageModeShared];
        if (!savedQ || !savedV) throw std::bad_alloc();
        id<MTLBlitCommandEncoder> blit=[commandBuffer blitCommandEncoder];
        [blit copyFromBuffer:sourceQ sourceOffset:0 toBuffer:savedQ destinationOffset:0 size:41*sizeof(float)];
        [blit copyFromBuffer:sourceV sourceOffset:0 toBuffer:savedV destinationOffset:0 size:40*sizeof(float)];
        [blit endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status!=MTLCommandBufferStatusCompleted) {std::fprintf(stderr,"CROW_RESET_AUDIT_ERROR\\n");return;}
            auto* q=static_cast<const float*>(savedQ.contents);auto* v=static_cast<const float*>(savedV.contents);
            std::fprintf(stderr,"CROW_RESET_AUDIT q");for (int i=0;i<41;i++) std::fprintf(stderr," %.9g",q[i]);
            std::fprintf(stderr," v");for (int i=0;i<40;i++) std::fprintf(stderr," %.9g",v[i]);std::fprintf(stderr,"\\n");
        }];
    }
    return encoded;
}'''
section=section.replace(ending,replacement);src=a/'MetalWorld-reset-observer.mm';src.write_text(original[:start]+section+original[end:])
entries=json.loads((r/'build/compile_commands.json').read_text());cmd=shlex.split(next(x['command'] for x in entries if x['file'].endswith('/MetalWorld.mm')));cmd.insert(1,'-I'+str(w/'include'));cmd.extend(['-I',str(r/'src/metal')]);obj=b/'MetalWorld-reset-observer.mm.o';cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(src);subprocess.run(cmd,cwd=r/'build',check=True)
observer=b/'reset-observer';observer.mkdir(exist_ok=False)
library=observer/'libmetalrobo.dylib'
link=json.loads((a/'build.json').read_text())['link'];oldobj=str(r/'build/crow-v15-descent-observer-extension-20261114/MetalWorld.mm.o');assert oldobj in link;link[link.index(oldobj)]=str(obj);link[link.index('-o')+1]=str(library);subprocess.run(link,cwd=r/'build',check=True)
(a/'reset-observer-build.json').write_text(json.dumps({'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,obj,library)}},indent=2)+'\n');print('built GPU reset observer')
