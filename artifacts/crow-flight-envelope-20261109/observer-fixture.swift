import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fatalError("Metal unavailable") }
let shader = URL(fileURLWithPath: CommandLine.arguments[1])
let library = try device.makeLibrary(URL: shader)
func buffer(_ n: Int) -> MTLBuffer {
    let b = device.makeBuffer(length: n, options: .storageModeShared)!
    memset(b.contents(), 0, n)
    return b
}
func uint(_ b: MTLBuffer, _ offset: Int, _ v: UInt32) { b.contents().storeBytes(of: v, toByteOffset: offset, as: UInt32.self) }
func flt(_ b: MTLBuffer, _ offset: Int, _ v: Float) { b.contents().storeBytes(of: v, toByteOffset: offset, as: Float.self) }
func get(_ b: MTLBuffer, _ offset: Int) -> Float { b.contents().load(fromByteOffset: offset, as: Float.self) }
let wings = buffer(384), dispatch = buffer(48), wrench = buffer(128), poses = buffer(128), tail = buffer(48), output = buffer(18 * 4)
uint(wings, 0, 1); uint(wings, 192, 2)
uint(dispatch, 0, 1); uint(dispatch, 12, 4); uint(dispatch, 24, 0)
uint(tail, 0, 3)
flt(wrench, 0 * 32 + 16 + 8, 0.2)
flt(wrench, 1 * 32 + 4, 2.0); flt(wrench, 1 * 32 + 16 + 8, 0.1)
flt(wrench, 2 * 32 + 4, -1.0); flt(wrench, 2 * 32 + 16 + 8, 0.2)
flt(wrench, 3 * 32 + 4, 0.5); flt(wrench, 3 * 32 + 16 + 8, 0.4)
flt(poses, 1 * 32, 1.0); flt(poses, 2 * 32, -1.0); flt(poses, 3 * 32, -2.0)
let inputs = [wings, dispatch, wrench, poses, tail, output]
func run(_ name: String) throws {
    let fn = library.makeFunction(name: name)!
    let pipeline = try device.makeComputePipelineState(function: fn)
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipeline)
    for (i,b) in inputs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    guard cb.status == .completed else { fatalError("GPU command failed") }
}
try run("mr_debug_wrench_before")
flt(wrench, 0 * 32 + 16 + 8, 0.6)
try run("mr_debug_wrench_after")
let got = [get(output, 15*4), get(output, 16*4), get(output, 17*4)]
let expected: [Float] = [0.2, 3.3, -0.2]
guard zip(got, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) else { fatalError("Fixture mismatch: \(got)") }
print("{\"fixture\":\"passed\",\"root_before\":\(got[0]),\"wings\":\(got[1]),\"root_plus_tail\":\(got[2])}")
