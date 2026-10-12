import Foundation
import Metal
let args=CommandLine.arguments
guard args.count==4 else { fatalError("usage: witness.swift shader.metal input.bin output.bin") }
let device=MTLCreateSystemDefaultDevice()!
let source=try String(contentsOfFile:args[1],encoding:.utf8)
let input=try Data(contentsOf:URL(fileURLWithPath:args[2]))
guard input.count>0 && input.count%(21*16)==0 else {fatalError("invalid 21-core witness input")}
let count=input.count/(21*16)
let options=MTLCompileOptions()
options.mathMode = .safe
let begin=ProcessInfo.processInfo.systemUptime
let library=try device.makeLibrary(source:source,options:options)
let pipeline=try device.makeComputePipelineState(function:library.makeFunction(name:"parent_position_witness")!)
let src=input.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:input.count,options:.storageModeShared)! }
let dst=device.makeBuffer(length:count*2*16,options:.storageModeShared)!
let queue=device.makeCommandQueue()!
let command=queue.makeCommandBuffer()!
let encoder=command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(src,offset:0,index:0)
encoder.setBuffer(dst,offset:0,index:1)
encoder.dispatchThreads(MTLSize(width:count,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:min(count,32),height:1,depth:1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
guard command.status == .completed else { fatalError(command.error?.localizedDescription ?? "Metal witness failed") }
try Data(bytes:dst.contents(),count:dst.length).write(to:URL(fileURLWithPath:args[3]),options:.withoutOverwriting)
let result:[String:Any]=["device":device.name,"vertices":count,"math_mode":"safe","gpu_completed":true,"wall_seconds":ProcessInfo.processInfo.systemUptime-begin,"scope":"isolated numerical instrument; not physical or anatomical admission"]
let report=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
print(String(data:report,encoding:.utf8)!)
